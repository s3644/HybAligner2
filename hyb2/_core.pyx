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
from libc.string cimport memcpy
cimport numpy as np
import numpy as np

np.import_array()

# ── 2-bit DNA encoding (CPU, vectorized via numpy) ───────────
# A=0, C=1, G=2, T=3, N=0  (packed: 4 bases per byte, MSB first)
_ENCODE = np.zeros(256, dtype=np.uint8)
for c, v in [(65,0),(67,1),(71,2),(84,3),(97,0),(99,1),(103,2),(116,3)]:
    _ENCODE[c] = v

cdef bytes encode_2bit_py(bytes seq):
    """Pack ASCII DNA → 2-bit packed bytes (CPU-numpy, fast)."""
    cdef int n = len(seq)
    cdef np.ndarray[np.uint8_t, ndim=1] enc = _ENCODE[bytearray(seq)]
    # Pack 4 bases per byte: [b0 b1 b2 b3] → b0<<6 | b1<<4 | b2<<2 | b3
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

    def load_reference(self, str fasta_path):
        """Load FASTA, 2-bit encode, upload to GPU, build hash table."""
        cdef bytes ref_data, ref_packed
        cdef int rlen, tsize
        cdef int mv = self.max_vals

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
        cudaMalloc(<void**>&self.d_ref, packed_len)
        cudaMemcpy(self.d_ref, <const unsigned char*>ref_packed, packed_len, cudaMemcpyHostToDevice)
        self.ref_len = rlen  # store base count, not packed bytes

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

        # 2-bit encode all reads (CPU, padded to read_len)
        cdef bytes packed
        cdef int plen = (read_len + 3) >> 2  # packed bytes per read
        parts = []
        for r in read_list:
            padded = r.ljust(read_len, b'N')[:read_len]
            parts.append(encode_2bit_py(padded))
        padded_packed = b''.join(parts)

        # ── Upload packed reads to GPU ─────────────────────
        cudaMalloc(<void**>&d_reads, n_reads * plen)
        cudaMemcpy(d_reads, <const unsigned char*>padded_packed, n_reads * plen, cudaMemcpyHostToDevice)

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

        # ── Phase 2: CPU anchor chaining (single-anchor passthrough) ──
        anchor_rp_arr = np.full(n_reads, -1, dtype=np.int32)
        anchor_fp_arr = np.full(n_reads, -1, dtype=np.int32)
        n_seeded = 0
        for i in range(n_reads):
            if rp_arr[i] >= 0 and fp_arr[i] >= 0:
                anchor_rp_arr[i] = rp_arr[i]
                anchor_fp_arr[i] = fp_arr[i]
                n_seeded += 1

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

        pos = scores_arr > 0
        return {
            "n_reads": n_reads,
            "n_seeded": n_seeded,
            "scores": scores_arr,
            "read_start": rs_arr, "read_end": re_arr,
            "ref_start": fs_arr, "ref_end": fe_arr,
            "score_mean": float(np.mean(scores_arr[pos])) if pos.any() else 0.0,
            "score_max": float(np.max(scores_arr)) if n_reads else 0.0,
        }
