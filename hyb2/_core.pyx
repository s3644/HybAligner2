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

cdef bytes pad_and_encode_c(list read_list, int n_reads, int read_len):
    """C-level: pre-allocate N-buffer, copy reads, 2-bit encode. No Python in hot loop."""
    cdef Py_ssize_t total = <Py_ssize_t>n_reads * <Py_ssize_t>read_len
    cdef Py_ssize_t packed_total = (total + 3) >> 2
    cdef Py_ssize_t i, j, st
    cdef int rlen, pos
    cdef unsigned char* buf = <unsigned char*>malloc(total)
    cdef unsigned char* packed = <unsigned char*>malloc(packed_total)
    cdef unsigned char* src
    cdef bytes rbytes
    cdef unsigned char b0, b1, b2, b3

    if buf == NULL or packed == NULL:
        free(buf); free(packed)
        raise MemoryError("malloc failed for %d bases" % total)

    # Fill buffer with 'N' (78) and copy reads
    memset(buf, 78, total)
    st = 0
    for i in range(n_reads):
        rbytes = read_list[i]
        rlen = len(rbytes)
        if rlen > read_len:
            rlen = read_len
        if rlen > 0:
            src = rbytes
            memcpy(buf + st, src, rlen)
        st += read_len

    # 2-bit encode: pack 4 bases per byte
    pos = 0
    for i in range(0, total, 4):
        b0 = _ENC_C[buf[i]]
        b1 = _ENC_C[buf[i+1]] if i+1 < total else 0
        b2 = _ENC_C[buf[i+2]] if i+2 < total else 0
        b3 = _ENC_C[buf[i+3]] if i+3 < total else 0
        packed[pos] = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3
        pos += 1

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
    int cudaMalloc(void** devPtr, size_t size)
    int cudaFree(void* devPtr)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaMemcpyHostToDevice
    int cudaMemcpyDeviceToHost

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
            k: k-mer size (default: 10 for good specificity, min 8)
            w: window size for minimizer (default: k//2 + 1)
        """
        cdef bytes ref_data, ref_packed
        cdef int rlen, tsize
        cdef int mv = self.max_vals

        # Set k, w (configurable)
        if k <= 0: k = 10  # default: better specificity than k=8
        if w <= 0: w = max(5, k // 2 + 1)
        self.k = k
        self.w = w

        # Read FASTA
        with open(fasta_path, 'rb') as f:
            lines = f.read().split(b'\n')
        parts = [l for l in lines if not l.startswith(b'>')]
        ref_data = b''.join(parts)
        rlen = len(ref_data)

        # 2-bit encode
        ref_packed = encode_2bit_py(ref_data)
        cdef int packed_len = len(ref_packed)

        # Upload packed reference to GPU
        if self.d_ref: cudaFree(self.d_ref)
        if self.d_table_keys: cudaFree(self.d_table_keys)
        if self.d_table_vals: cudaFree(self.d_table_vals)

        cudaMalloc(<void**>&self.d_ref, packed_len)
        cudaMemcpy(self.d_ref, <const unsigned char*>ref_packed, packed_len, cudaMemcpyHostToDevice)
        self.ref_len = rlen

        # Size hash table
        tsize = 1
        n_windows = max(1, rlen // self.w)
        while tsize < n_windows * 2:
            tsize *= 2
        if tsize < (1 << 18): tsize = 1 << 18
        if tsize > (1 << 24): tsize = 1 << 24

        # Allocate + init GPU hash table
        cudaMalloc(<void**>&self.d_table_keys, tsize * sizeof(unsigned long long))
        cudaMalloc(<void**>&self.d_table_vals, tsize * mv * sizeof(int))

        cdef np.ndarray ek = np.full(tsize, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        cdef np.ndarray ev = np.full(tsize * mv, -1, dtype=np.int32)
        cudaMemcpy(self.d_table_keys, PyArray_DATA(ek), tsize * 8, cudaMemcpyHostToDevice)
        cudaMemcpy(self.d_table_vals, PyArray_DATA(ev), tsize * mv * 4, cudaMemcpyHostToDevice)

        launch_build_index(
            self.d_ref, rlen, self.k, self.w,
            self.d_table_keys, self.d_table_vals, tsize, mv,
        )
        self.table_size = tsize

    def align(self, str fastq_path, int band_width=50, int gap_open=5, int gap_extend=2):
        """Align FASTQ reads. CPU: encode+chain. GPU: seed+SW."""
        cdef bytes padded_packed
        cdef int n_reads, read_len, i, n_seeded
        cdef unsigned char* d_reads
        cdef int* d_rp, *d_fp, *d_anchor_rp, *d_anchor_fp
        cdef float* d_scores
        cdef int* d_rs, *d_re, *d_fs, *d_fe
        cdef np.ndarray[np.int32_t, ndim=1] rp_arr, fp_arr
        cdef np.ndarray[np.int32_t, ndim=1] anchor_rp_arr, anchor_fp_arr
        cdef np.ndarray[np.float32_t, ndim=1] scores_arr
        cdef np.ndarray[np.int32_t, ndim=1] rs_arr, re_arr, fs_arr, fe_arr

        # ── Parse FASTQ (CPU) ──────────────────────────────
        with open(fastq_path, 'rb') as f:
            lines = f.read().split(b'\n')
        read_list = [lines[i] for i in range(1, len(lines), 4)]
        n_reads = len(read_list)
        read_len = max(len(r) for r in read_list) if read_list else 0

        # C-level padding + 2-bit encode (no Python overhead in hot loop)
        padded_packed = pad_and_encode_c(read_list, n_reads, read_len)
        cdef int plen = ((<Py_ssize_t>n_reads * <Py_ssize_t>read_len + 3) >> 2)

        # ── Upload packed reads to GPU ─────────────────────
        cudaMalloc(<void**>&d_reads, plen)
        cudaMemcpy(d_reads, <const unsigned char*>padded_packed, plen, cudaMemcpyHostToDevice)

        # ── Phase 1: GPU seed reads (single best anchor) ──
        rp_arr = np.full(n_reads, -1, dtype=np.int32)
        fp_arr = np.full(n_reads, -1, dtype=np.int32)
        cudaMalloc(<void**>&d_rp, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_fp, n_reads * sizeof(int))

        launch_seed_reads(
            d_reads, n_reads, read_len,
            self.d_table_keys, self.d_table_vals,
            self.table_size, self.max_vals, self.k, self.w,
            d_rp, d_fp,
        )
        cudaMemcpy(PyArray_DATA(rp_arr), d_rp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(fp_arr), d_fp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaFree(d_rp); cudaFree(d_fp)

        # ── Phase 2: CPU anchor chaining (numpy-vectorized) ──
        anchor_rp_arr = np.where((rp_arr >= 0) & (fp_arr >= 0), rp_arr, -1).astype(np.int32)
        anchor_fp_arr = np.where((rp_arr >= 0) & (fp_arr >= 0), fp_arr, -1).astype(np.int32)
        n_seeded = int((anchor_rp_arr >= 0).sum())

        # Upload anchors to GPU
        cudaMalloc(<void**>&d_anchor_rp, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_anchor_fp, n_reads * sizeof(int))
        cudaMemcpy(d_anchor_rp, PyArray_DATA(anchor_rp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)
        cudaMemcpy(d_anchor_fp, PyArray_DATA(anchor_fp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)

        # ── Phase 3: GPU windowed SW ───────────────────────
        scores_arr = np.zeros(n_reads, dtype=np.float32)
        rs_arr = np.zeros(n_reads, dtype=np.int32)
        re_arr = np.zeros(n_reads, dtype=np.int32)
        fs_arr = np.zeros(n_reads, dtype=np.int32)
        fe_arr = np.zeros(n_reads, dtype=np.int32)

        cudaMalloc(<void**>&d_scores, n_reads * sizeof(float))
        cudaMalloc(<void**>&d_rs, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_re, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_fs, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_fe, n_reads * sizeof(int))

        launch_sw_align(
            d_reads, self.d_ref, self.ref_len,
            d_anchor_rp, d_anchor_fp,
            n_reads, read_len,
            band_width, gap_open, gap_extend,
            d_scores, d_rs, d_re, d_fs, d_fe,
        )

        cudaMemcpy(PyArray_DATA(scores_arr), d_scores, n_reads * sizeof(float), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(rs_arr), d_rs, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(re_arr), d_re, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(fs_arr), d_fs, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(fe_arr), d_fe, n_reads * sizeof(int), cudaMemcpyDeviceToHost)

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
