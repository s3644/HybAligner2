# cython: language_level=3, boundscheck=False, wraparound=False
"""HybAligner2 Cython Core — direct CUDA Runtime API + CPU anchor chaining.

Pipeline:
  1. GPU: build_index(ref)        → minimizer hash table
  2. GPU: seed_reads(reads)       → anchor positions
  3. CPU: chain_anchors()         → 1D DP over diagonals
  4. GPU: sw_align(windows)       → banded Gotoh, anchor-aware

Zero Python in hot path. Both CPU and GPU utilized.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport numpy as np
import numpy as np

np.import_array()

# ── CUDA Runtime API ───────────────────────────────────────────
cdef extern from "cuda_runtime.h":
    int cudaMalloc(void** devPtr, size_t size)
    int cudaFree(void* devPtr)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaMemcpyHostToDevice
    int cudaMemcpyDeviceToHost

cdef extern from "numpy/arrayobject.h":
    void* PyArray_DATA(np.ndarray arr)

# ── Our kernels (compiled from kernels.cu via kernels.h) ──────
cdef extern from "kernels.h":
    int launch_build_index(
        const char* ref, int ref_len, int k, int w,
        unsigned long long* table_keys, int* table_vals,
        int table_size, int max_vals_per_key,
    )
    int launch_seed_reads(
        const char* reads, int n_reads, int read_len,
        const unsigned long long* table_keys, const int* table_vals,
        int table_size, int max_vals_per_key, int k, int w,
        int* out_rp, int* out_fp,
    )
    int launch_sw_align(
        const char* reads, const char* ref, int ref_len,
        const int* anchor_rp, const int* anchor_fp,
        int n_reads, int read_len,
        int band_width, int gap_open, int gap_extend,
        float* scores, int* read_start, int* read_end,
        int* ref_start, int* ref_end,
    )

# ── Anchor chaining (minimap2-style 1D DP, runs on CPU) ────────
cdef void chain_anchors_cpu(
    int* rp_in, int* fp_in, int n, int max_gap, int bandwidth,
    int* rp_out, int* fp_out,
):
    """Simple 1D DP chaining: pick the best colinear chain.
    For Phase 1: returns the single best anchor per read.
    Input arrays may contain multiple anchors per read (future).
    """
    cdef int i, j, best_idx
    cdef int best_score, score, di, dj, dr, df
    cdef float gap_penalty

    if n <= 1:
        if n == 1:
            rp_out[0] = rp_in[0]
            fp_out[0] = fp_in[0]
        return

    # For single-anchor-per-read (current seed_reads behavior),
    # just pass through. Multi-anchor chaining will be Phase 2.
    best_idx = 0
    best_score = 0
    for i in range(n):
        if rp_in[i] >= 0 and fp_in[i] >= 0:
            # Score: prefer anchors near center of read (heuristic)
            score = 1000 - abs(fp_in[i] - rp_in[i])  # prefer consistent diagonal
            if score > best_score:
                best_score = score
                best_idx = i

    if best_score > 0:
        rp_out[0] = rp_in[best_idx]
        fp_out[0] = fp_in[best_idx]
    else:
        rp_out[0] = -1
        fp_out[0] = -1


cdef class HybAligner2:
    """Cython+CUDA aligner. GPU for heavy compute, CPU for chaining."""

    cdef:
        char* d_ref
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
        """Load FASTA, upload to GPU, build minimizer hash table."""
        cdef bytes ref_bytes, ref_data
        cdef int rlen, tsize
        cdef np.ndarray[np.uint64_t, ndim=1] empty_keys
        cdef np.ndarray[np.int32_t, ndim=1] empty_vals
        cdef int mv = self.max_vals

        # Read FASTA
        with open(fasta_path, 'rb') as f:
            lines = f.read().split(b'\n')
        parts = [l for l in lines if not l.startswith(b'>')]
        ref_data = b''.join(parts)
        rlen = len(ref_data)

        # Upload reference to GPU
        cudaMalloc(<void**>&self.d_ref, rlen)
        cudaMemcpy(self.d_ref, <const char*>ref_data, rlen, cudaMemcpyHostToDevice)
        self.ref_len = rlen

        # Size hash table: ~2× the number of windows
        tsize = 1
        n_windows = max(1, rlen // self.w)
        while tsize < n_windows * 2:
            tsize *= 2
        if tsize < (1 << 18): tsize = 1 << 18  # minimum 256K slots
        if tsize > (1 << 24): tsize = 1 << 24  # cap at 16M slots (~128MB keys)

        # Allocate GPU hash table
        cudaMalloc(<void**>&self.d_table_keys, tsize * sizeof(unsigned long long))
        cudaMalloc(<void**>&self.d_table_vals, tsize * mv * sizeof(int))

        # Initialize to empty
        cdef np.ndarray ek_arr = np.full(tsize, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        cdef np.ndarray ev_arr = np.full(tsize * mv, -1, dtype=np.int32)
        cudaMemcpy(self.d_table_keys, PyArray_DATA(ek_arr), tsize * 8, cudaMemcpyHostToDevice)
        cudaMemcpy(self.d_table_vals, PyArray_DATA(ev_arr), tsize * mv * 4, cudaMemcpyHostToDevice)

        launch_build_index(
            self.d_ref, rlen, self.k, self.w,
            self.d_table_keys, self.d_table_vals, tsize, mv,
        )
        self.table_size = tsize

    def align(self, str fastq_path, int band_width=50, int gap_open=5, int gap_extend=2):
        """Align FASTQ reads against loaded reference. CPU+GPU pipeline."""
        cdef bytes padded
        cdef int n_reads, read_len, i, n_seeded
        cdef char* d_reads
        cdef int* d_rp, *d_fp
        cdef int* d_anchor_rp, *d_anchor_fp
        cdef float* d_scores
        cdef int* d_rs, *d_re, *d_fs, *d_fe
        cdef np.ndarray[np.int32_t, ndim=1] rp_arr, fp_arr, anchor_rp_arr, anchor_fp_arr
        cdef np.ndarray[np.float32_t, ndim=1] scores_arr
        cdef np.ndarray[np.int32_t, ndim=1] rs_arr, re_arr, fs_arr, fe_arr

        # ── Parse FASTQ (CPU) ──────────────────────────────
        with open(fastq_path, 'rb') as f:
            lines = f.read().split(b'\n')
        read_list = [lines[i] for i in range(1, len(lines), 4)]
        n_reads = len(read_list)
        read_len = max(len(r) for r in read_list) if read_list else 0

        padded = b''.join(r.ljust(read_len, b'N')[:read_len] for r in read_list)

        # ── Upload reads to GPU ────────────────────────────
        cudaMalloc(<void**>&d_reads, n_reads * read_len)
        cudaMemcpy(d_reads, <const char*>padded, n_reads * read_len, cudaMemcpyHostToDevice)

        # ── Phase 1: GPU seed reads ────────────────────────
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

        # Download seeds to CPU
        cudaMemcpy(PyArray_DATA(rp_arr), d_rp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(PyArray_DATA(fp_arr), d_fp, n_reads * sizeof(int), cudaMemcpyDeviceToHost)
        cudaFree(d_rp); cudaFree(d_fp)

        # ── Phase 2: CPU anchor chaining ───────────────────
        anchor_rp_arr = np.full(n_reads, -1, dtype=np.int32)
        anchor_fp_arr = np.full(n_reads, -1, dtype=np.int32)

        for i in range(n_reads):
            if rp_arr[i] >= 0 and fp_arr[i] >= 0:
                anchor_rp_arr[i] = rp_arr[i]
                anchor_fp_arr[i] = fp_arr[i]

        n_seeded = 0
        for i in range(n_reads):
            if anchor_rp_arr[i] >= 0:
                n_seeded += 1

        # Upload anchors to GPU
        cudaMalloc(<void**>&d_anchor_rp, n_reads * sizeof(int))
        cudaMalloc(<void**>&d_anchor_fp, n_reads * sizeof(int))
        cudaMemcpy(d_anchor_rp, PyArray_DATA(anchor_rp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)
        cudaMemcpy(d_anchor_fp, PyArray_DATA(anchor_fp_arr), n_reads * sizeof(int), cudaMemcpyHostToDevice)

        # ── Phase 3: GPU windowed SW alignment ─────────────
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

        # ── Cleanup GPU ────────────────────────────────────
        cudaFree(d_reads)
        cudaFree(d_anchor_rp); cudaFree(d_anchor_fp)
        cudaFree(d_scores); cudaFree(d_rs); cudaFree(d_re); cudaFree(d_fs); cudaFree(d_fe)

        # ── Results ──
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
