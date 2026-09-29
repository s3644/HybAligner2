# cython: language_level=3, boundscheck=False, wraparound=False
"""HybAligner2 Cython Core v2.1 — 2-bit packed DNA + CPU anchor chaining.

Pipeline:
  1. CPU: encode_2bit(ref + reads) → packed uint8 arrays
  2. GPU: build_index(packed ref)  → minimizer hash table
  3. GPU: seed_reads(packed reads) → anchor positions
  4. CPU: chain_anchors()          → 1D DP over diagonals
  5. GPU: sw_align(packed, window) → banded Gotoh
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
cimport numpy as np
import numpy as np

np.import_array()

# ── C-level padding + encode (no Python overhead) ─────────────

cdef unsigned char _ENC_C[256]
# Initialize C lookup table
for c, v in [(65,0),(67,1),(71,2),(84,3),(97,0),(99,1),(103,2),(116,3),(78,3),(110,3)]:
    _ENC_C[c] = v

cdef bytes pad_and_encode_stream(str fastq_path, int n_reads, int read_len):
    """Stream FASTQ → pad + 2-bit encode in ONE pass. No intermediate list.
    Eliminates: 6GB read(), split(b'\\n'), 24M-item list extraction.

    Each read is packed into its own byte-aligned block of rl_bytes =
    (read_len+3)>>2 bytes (padded with 'A'=00 if read_len % 4 != 0). This
    MUST match the per-read stride the GPU kernels use to index into the
    packed reads buffer (`reads + rid * ((read_len+3)>>2)`); packing the
    whole stream as one contiguous bitstream instead (ignoring per-read
    boundaries) desyncs GPU indexing from the second read onward whenever
    read_len isn't a multiple of 4 (e.g. 150bp, 250bp)."""
    cdef Py_ssize_t total = <Py_ssize_t>n_reads * <Py_ssize_t>read_len
    cdef int rl_bytes = (read_len + 3) >> 2
    cdef Py_ssize_t packed_total = <Py_ssize_t>n_reads * <Py_ssize_t>rl_bytes
    cdef unsigned char* buf = <unsigned char*>malloc(total)
    cdef unsigned char* packed = <unsigned char*>malloc(packed_total)
    cdef unsigned char b0, b1, b2, b3
    cdef int st = 0, rlen, reads_copied
    cdef Py_ssize_t line_num, r, base_in_read, rbase, obase, i
    cdef bytes seq

    if buf == NULL or packed == NULL:
        free(buf); free(packed)
        raise MemoryError("malloc failed for %d bases" % total)

    memset(buf, 65, total)

    # Stream file, extract sequence lines (every 2nd of 4)
    line_num = 0
    reads_copied = 0
    st = 0
    with open(fastq_path, 'rb') as f:
        for line_bytes in f:
            if line_num & 3 == 1:
                seq = line_bytes.rstrip(b'\r\n')
                rlen = len(seq)
                if rlen > read_len: rlen = read_len
                if rlen > 0 and reads_copied < n_reads:
                    memcpy(buf + st, <unsigned char*>seq, rlen)
                st += read_len
                reads_copied += 1
            line_num += 1

    # 2-bit encode PER READ: each read gets its own byte-aligned block of
    # rl_bytes bytes, matching the GPU kernels' `reads + rid * rl_bytes` stride.
    for r in range(n_reads):
        rbase = r * read_len
        obase = r * rl_bytes
        for base_in_read in range(0, read_len, 4):
            i = rbase + base_in_read
            b0 = _ENC_C[buf[i]]
            b1 = _ENC_C[buf[i+1]] if base_in_read+1 < read_len else 0
            b2 = _ENC_C[buf[i+2]] if base_in_read+2 < read_len else 0
            b3 = _ENC_C[buf[i+3]] if base_in_read+3 < read_len else 0
            packed[obase + (base_in_read >> 2)] = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3

    cdef bytes result = packed[:packed_total]
    free(buf)
    free(packed)
    return result

# Keep numpy table for reference encoding (one-time, small)
_ENCODE = np.zeros(256, dtype=np.uint8)
for c, v in [(65,0),(67,1),(71,2),(84,3),(97,0),(99,1),(103,2),(116,3),(78,3),(110,3)]:
    _ENCODE[c] = v

cdef bytes encode_2bit_py(bytes seq):
    """Pack reference DNA → 2-bit (one-time, small ref)."""
    cdef int n = len(seq)
    cdef np.ndarray[np.uint8_t, ndim=1] enc = _ENCODE[bytearray(seq)]
    cdef np.ndarray[np.uint8_t, ndim=1] packed = np.zeros((n + 3) >> 2, dtype=np.uint8)
    cdef int i
    for i in range(0, n, 4):
        packed[i>>2] = ((enc[i]   if i < n     else 0) << 6) | \
                       ((enc[i+1] if i+1 < n   else 0) << 4) | \
                       ((enc[i+2] if i+2 < n   else 0) << 2) | \
                       ((enc[i+3] if i+3 < n   else 0))
    return packed.tobytes()

# ── CUDA Runtime API ───────────────────────────────────────────
cdef extern from "cuda_runtime.h":
    ctypedef int cudaError_t
    int cudaMalloc(void** devPtr, size_t size)
    int cudaFree(void* devPtr)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaMemcpyHostToDevice
    int cudaMemcpyDeviceToHost
    int cudaGetLastError()
    int cudaSuccess
    int cudaErrorInvalidValue
    int cudaErrorMemoryAllocation
    const char* cudaGetErrorString(cudaError_t error)
    int cudaDeviceSynchronize()

cdef extern from "numpy/arrayobject.h":
    void* PyArray_DATA(np.ndarray arr)

# ── Kernels (2-bit packed signatures) ─────────────────────────
cdef extern from "kernels.h":
    int launch_build_index(
        const unsigned char* ref, int ref_len, int k, int w,
        unsigned long long* table_keys, int* table_vals,
        int table_size, int max_vals_per_key,
    )
    int launch_seed_reads(
        const unsigned char* reads, int n_reads, int read_len,
        const unsigned long long* table_keys, const int* table_vals,
        int table_size, int max_vals_per_key, int k, int w,
        int* out_rp, int* out_fp,
    )
    int launch_sw_align(
        const unsigned char* reads, const unsigned char* ref, int ref_len,
        const int* anchor_rp, const int* anchor_fp,
        int n_reads, int read_len,
        int band_width, int gap_open, int gap_extend,
        float* scores, int* read_start, int* read_end,
        int* ref_start, int* ref_end,
    )


# ── CUDA error helper ───────────────────────────────────────────
cdef void _check_cuda(int err, str msg) except *:
    if err != cudaSuccess:
        raise RuntimeError(f"CUDA error code {err}: {msg}")

cdef class HybAligner2:
    """Cython+CUDA aligner with 2-bit packed DNA."""

    cdef:
        unsigned char* d_ref
        int ref_len
        unsigned long long* d_table_keys
        int* d_table_vals
        int table_size
        int k, w, max_vals

    def __cinit__(self):
        self.d_ref = NULL
        self.d_table_keys = NULL
        self.d_table_vals = NULL
        self.k = 8
        self.w = 5
        self.max_vals = 8
        self.table_size = 0

    def __dealloc__(self):
        if self.d_ref:          cudaFree(self.d_ref)
        if self.d_table_keys:   cudaFree(self.d_table_keys)
        if self.d_table_vals:   cudaFree(self.d_table_vals)

    def load_reference(self, str fasta_path, int k=0, int w=0):
        """Load FASTA, 2-bit encode, upload to GPU, build hash table.

        Args:
            fasta_path: Path to reference FASTA file.
            k: k-mer size (default: 10, range 8-15).
            w: window size for minimizer (default: k//2 + 1).

        Raises:
            FileNotFoundError: FASTA file not found.
            ValueError: Invalid reference or parameters.
            RuntimeError: CUDA allocation or kernel failure.
        """
        cdef bytes ref_data, ref_packed
        cdef int rlen, tsize, err
        cdef int mv = self.max_vals

        # ── Input validation ──
        if k < 0: k = 0
        if k == 0: k = 10
        if k < 4 or k > 31:
            raise ValueError(f"k-mer size must be 4-31, got {k}")
        if w < 0: w = 0
        if w == 0: w = max(5, k // 2 + 1)
        if w < 2:
            raise ValueError(f"window size must be >= 2, got {w}")
        self.k = k
        self.w = w

        # Read FASTA
        with open(fasta_path, 'rb') as f:
            lines = f.read().split(b'\n')
        parts = [l for l in lines if not l.startswith(b'>')]
        ref_data = b''.join(parts)
        rlen = len(ref_data)
        if rlen < self.k + self.w:
            raise ValueError(f"Reference too short: {rlen}bp (need >= {self.k + self.w})")

        # 2-bit encode
        ref_packed = encode_2bit_py(ref_data)
        cdef int packed_len = len(ref_packed)

        # Upload packed reference to GPU
        if self.d_ref: cudaFree(self.d_ref)
        if self.d_table_keys: cudaFree(self.d_table_keys)
        if self.d_table_vals: cudaFree(self.d_table_vals)
        self.d_ref = NULL
        self.d_table_keys = NULL
        self.d_table_vals = NULL

        err = cudaMalloc(<void**>&self.d_ref, packed_len)
        _check_cuda(err, f"cudaMalloc ref ({packed_len} bytes)")
        err = cudaMemcpy(self.d_ref, <const unsigned char*>ref_packed, packed_len, cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy ref H2D")
        self.ref_len = rlen

        # Size hash table dynamically from reference length
        tsize = 1
        n_windows = max(1, rlen // self.w)
        while tsize < n_windows * 2:
            tsize *= 2
        if tsize < (1 << 18): tsize = 1 << 18
        if tsize > (1 << 24): tsize = 1 << 24

        # Allocate GPU hash table
        err = cudaMalloc(<void**>&self.d_table_keys, tsize * sizeof(unsigned long long))
        _check_cuda(err, f"cudaMalloc table_keys ({tsize * 8} bytes)")
        err = cudaMalloc(<void**>&self.d_table_vals, tsize * mv * sizeof(int))
        _check_cuda(err, f"cudaMalloc table_vals ({tsize * mv * 4} bytes)")

        # Init table on host, upload
        cdef np.ndarray ek = np.full(tsize, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        cdef np.ndarray ev = np.full(tsize * mv, -1, dtype=np.int32)
        err = cudaMemcpy(self.d_table_keys, PyArray_DATA(ek), tsize * 8, cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy table_keys init H2D")
        err = cudaMemcpy(self.d_table_vals, PyArray_DATA(ev), tsize * mv * 4, cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy table_vals init H2D")

        err = launch_build_index(
            self.d_ref, rlen, self.k, self.w,
            self.d_table_keys, self.d_table_vals, tsize, mv,
        )
        if err != 0:
            raise RuntimeError(f"build_index kernel failed (error {err})")
        self.table_size = tsize

    def align(self, str fastq_path, int band_width=50, int gap_open=5, int gap_extend=2):
        """Align FASTQ reads. CPU: encode+chain. GPU: seed+SW.

        Args:
            fastq_path: Path to FASTQ file.
            band_width: Band width for SW (capped at 80).
            gap_open: Gap open penalty.
            gap_extend: Gap extend penalty.

        Returns:
            dict with alignment results.

        Raises:
            FileNotFoundError: FASTQ file not found.
            ValueError: Empty input or invalid parameters.
            RuntimeError: GPU allocation or kernel failure.
        """
        cdef bytes padded_packed
        cdef int n_reads, read_len, i, n_seeded, err
        cdef unsigned char* d_reads
        cdef int* d_rp
        cdef int* d_fp
        cdef int* d_anchor_rp
        cdef int* d_anchor_fp
        cdef float* d_scores
        cdef int* d_rs
        cdef int* d_re
        cdef int* d_fs
        cdef int* d_fe
        cdef np.ndarray[np.int32_t, ndim=1] rp_arr, fp_arr
        cdef np.ndarray[np.int32_t, ndim=1] anchor_rp_arr, anchor_fp_arr
        cdef np.ndarray[np.float32_t, ndim=1] scores_arr
        cdef np.ndarray[np.int32_t, ndim=1] rs_arr, re_arr, fs_arr, fe_arr

        # ── Input validation ──
        if band_width <= 0: band_width = 20
        if gap_open < 0: gap_open = 5
        if gap_extend < 0: gap_extend = 2
        if self.table_size == 0:
            raise RuntimeError("No reference loaded. Call load_reference() first.")

        # ── Quick scan: count reads + find max read length ──
        n_reads = 0
        read_len = 0
        cdef int line_idx = 0, rl
        with open(fastq_path, 'rb') as f:
            for line_bytes in f:
                if line_idx & 3 == 1:  # sequence line
                    n_reads += 1
                    rl = len(line_bytes.rstrip(b'\r\n'))
                    if rl > read_len: read_len = rl
                line_idx += 1

        if n_reads == 0:
            raise ValueError(f"No reads found in {fastq_path}")
        if read_len < self.k:
            raise ValueError(f"Reads too short: {read_len}bp (need >= k={self.k})")

        # ── Stream encode (no intermediate list) ────────────
        padded_packed = pad_and_encode_stream(fastq_path, n_reads, read_len)
        # Must match pad_and_encode_stream's per-read byte-aligned stride.
        cdef int plen = n_reads * ((read_len + 3) >> 2)

        # ── Upload packed reads to GPU ─────────────────────
        err = cudaMalloc(<void**>&d_reads, plen)
        _check_cuda(err, f"cudaMalloc reads ({plen} bytes)")
        err = cudaMemcpy(d_reads, <const unsigned char*>padded_packed, plen, cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy reads H2D")

        # ── Phase 1: GPU seed reads (single best anchor) ──
        rp_arr = np.full(n_reads, -1, dtype=np.int32)
        fp_arr = np.full(n_reads, -1, dtype=np.int32)
        err = cudaMalloc(<void**>&d_rp, n_reads * sizeof(int))
        _check_cuda(err, f"cudaMalloc rp ({n_reads * 4} bytes)")
        err = cudaMalloc(<void**>&d_fp, n_reads * sizeof(int))
        _check_cuda(err, f"cudaMalloc fp ({n_reads * 4} bytes)")

        err = launch_seed_reads(
            d_reads, n_reads, read_len,
            self.d_table_keys, self.d_table_vals,
            self.table_size, self.max_vals, self.k, self.w,
            d_rp, d_fp,
        )
        if err != 0:
            cudaFree(d_reads); cudaFree(d_rp); cudaFree(d_fp)
            raise RuntimeError(f"seed_reads kernel failed (error {err})")

        err = cudaMemcpy(PyArray_DATA(rp_arr), d_rp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy rp D2H")
        err = cudaMemcpy(PyArray_DATA(fp_arr), d_fp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy fp D2H")
        cudaFree(d_rp); cudaFree(d_fp)

        # ── Phase 2: CPU anchor chaining (numpy-vectorized) ──
        anchor_rp_arr = np.where((rp_arr >= 0) & (fp_arr >= 0), rp_arr, -1).astype(np.int32)
        anchor_fp_arr = np.where((rp_arr >= 0) & (fp_arr >= 0), fp_arr, -1).astype(np.int32)
        n_seeded = int((anchor_rp_arr >= 0).sum())

        # Upload anchors to GPU
        err = cudaMalloc(<void**>&d_anchor_rp, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc anchor_rp")
        err = cudaMalloc(<void**>&d_anchor_fp, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc anchor_fp")
        err = cudaMemcpy(d_anchor_rp, PyArray_DATA(anchor_rp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy anchor_rp H2D")
        err = cudaMemcpy(d_anchor_fp, PyArray_DATA(anchor_fp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)
        _check_cuda(err, "cudaMemcpy anchor_fp H2D")

        # ── Phase 3: GPU windowed SW ───────────────────────
        scores_arr = np.zeros(n_reads, dtype=np.float32)
        rs_arr = np.zeros(n_reads, dtype=np.int32)
        re_arr = np.zeros(n_reads, dtype=np.int32)
        fs_arr = np.zeros(n_reads, dtype=np.int32)
        fe_arr = np.zeros(n_reads, dtype=np.int32)

        err = cudaMalloc(<void**>&d_scores, n_reads * sizeof(float))
        _check_cuda(err, "cudaMalloc scores")
        err = cudaMalloc(<void**>&d_rs, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc rs")
        err = cudaMalloc(<void**>&d_re, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc re")
        err = cudaMalloc(<void**>&d_fs, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc fs")
        err = cudaMalloc(<void**>&d_fe, n_reads * sizeof(int))
        _check_cuda(err, "cudaMalloc fe")

        err = launch_sw_align(
            d_reads, self.d_ref, self.ref_len,
            d_anchor_rp, d_anchor_fp,
            n_reads, read_len,
            band_width, gap_open, gap_extend,
            d_scores, d_rs, d_re, d_fs, d_fe,
        )
        if err != 0:
            cudaFree(d_reads)
            cudaFree(d_anchor_rp); cudaFree(d_anchor_fp)
            cudaFree(d_scores); cudaFree(d_rs); cudaFree(d_re); cudaFree(d_fs); cudaFree(d_fe)
            raise RuntimeError(f"sw_align kernel failed (error {err})")

        err = cudaMemcpy(PyArray_DATA(scores_arr), d_scores, n_reads * sizeof(float), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy scores D2H")
        err = cudaMemcpy(PyArray_DATA(rs_arr), d_rs, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy rs D2H")
        err = cudaMemcpy(PyArray_DATA(re_arr), d_re, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy re D2H")
        err = cudaMemcpy(PyArray_DATA(fs_arr), d_fs, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy fs D2H")
        err = cudaMemcpy(PyArray_DATA(fe_arr), d_fe, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        _check_cuda(err, "cudaMemcpy fe D2H")

        cudaFree(d_reads)
        cudaFree(d_anchor_rp); cudaFree(d_anchor_fp)
        cudaFree(d_scores); cudaFree(d_rs); cudaFree(d_re); cudaFree(d_fs); cudaFree(d_fe)

        # Results
        cdef float sm = 0.0
        cdef float smax = 0.0
        if n_reads > 0:
            smax = float(np.max(scores_arr))
            pos_vals = scores_arr[scores_arr > 0]
            if len(pos_vals) > 0:
                sm = float(np.mean(pos_vals))
        return {
            "n_reads": n_reads, "n_seeded": n_seeded,
            "scores": scores_arr,
            "read_start": rs_arr, "read_end": re_arr,
            "ref_start": fs_arr, "ref_end": fe_arr,
            "score_mean": sm, "score_max": smax,
        }
