# cython: language_level=3
"""HybAligner2 Cython Core — direct CUDA Runtime API bindings.

Zero Python in hot path. All GPU work dispatched via compiled C.

Memory layout:
  reads:  contiguous n_reads × read_len bytes (padded)
  ref:    contiguous ref_len bytes
  anchors: n_reads × 2 int32 (read_pos, ref_pos)
  scores:  n_reads float32
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

np.import_array()

# CUDA Runtime API (linked via setup.py)
cdef extern from "cuda_runtime.h":
    int cudaMalloc(void** devPtr, size_t size)
    int cudaFree(void* devPtr)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaMemcpyKind "cudaMemcpyKind"
    int cudaMemcpyHostToDevice "cudaMemcpyHostToDevice"
    int cudaMemcpyDeviceToHost "cudaMemcpyDeviceToHost"

# Our CUDA kernels (compiled from kernels.cu)
cdef extern from "kernels.cu":
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
        const char* reads, const char* ref,
        int n_reads, int read_len, int ref_len,
        int band_width, int gap_open, int gap_extend,
        float* scores, int* rs, int* re, int* fs, int* fe,
    )


cdef class HybAligner2:
    """Minimal Cython+CUDA aligner. One object per reference."""
    
    cdef:
        char* d_ref          # GPU: reference sequence
        int ref_len
        unsigned long long* d_table_keys  # GPU: hash table keys
        int* d_table_vals                 # GPU: hash table values
        int table_size
        int k, w

    def __cinit__(self):
        self.d_ref = NULL
        self.d_table_keys = NULL
        self.d_table_vals = NULL
        self.k = 8
        self.w = 5

    def __dealloc__(self):
        if self.d_ref: cudaFree(self.d_ref)
        if self.d_table_keys: cudaFree(self.d_table_keys)
        if self.d_table_vals: cudaFree(self.d_table_vals)

    def load_reference(self, str fasta_path):
        """Load FASTA, upload to GPU, build seed index."""
        cdef bytes ref_bytes, ref_padded
        cdef int rlen, tsize
        cdef int max_vals = 8

        # Read FASTA
        with open(fasta_path, 'rb') as f:
            lines = f.read().split(b'\n')
        parts = [l for l in lines if not l.startswith(b'>')]
        ref_bytes = b''.join(parts)
        rlen = len(ref_bytes)

        # Pad to multiple of 4 for alignment
        pad = (4 - (rlen % 4)) % 4
        ref_padded = ref_bytes + b'N' * pad
        rlen = len(ref_padded)

        # Upload reference to GPU
        cudaMalloc(<void**>&self.d_ref, rlen)
        cudaMemcpy(self.d_ref, ref_padded, rlen, cudaMemcpyHostToDevice)
        self.ref_len = rlen

        # Build seed index on GPU
        tsize = 1 << 20  # 1M slots
        cudaMalloc(<void**>&self.d_table_keys, tsize * 8)    # uint64
        cudaMalloc(<void**>&self.d_table_vals, tsize * max_vals * 4)  # int32
        # Initialize table to empty
        import numpy as np
        empty_keys = np.full(tsize, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        empty_vals = np.full(tsize * max_vals, -1, dtype=np.int32)
        cudaMemcpy(self.d_table_keys, empty_keys.ctypes.data, tsize * 8, cudaMemcpyHostToDevice)
        cudaMemcpy(self.d_table_vals, empty_vals.ctypes.data, tsize * max_vals * 4, cudaMemcpyHostToDevice)

        launch_build_index(
            self.d_ref, rlen, self.k, self.w,
            self.d_table_keys, self.d_table_vals,
            tsize, max_vals,
        )
        self.table_size = tsize

    def align(self, str fastq_path, int band_width=50, int gap_open=5, int gap_extend=2):
        """Align FASTQ reads against loaded reference. All GPU."""
        cdef bytes reads_padded
        cdef int n_reads, read_len
        cdef char* d_reads
        cdef int* d_rp, *d_fp, *d_rs, *d_re, *d_fs, *d_fe
        cdef float* d_scores
        cdef np.ndarray rp_arr, fp_arr, scores_arr, rs_arr, re_arr, fs_arr, fe_arr

        # Parse FASTQ
        with open(fastq_path, 'rb') as f:
            lines = f.read().split(b'\n')
        read_list = [lines[i] for i in range(1, len(lines), 4)]
        n_reads = len(read_list)
        read_len = max(len(r) for r in read_list) if read_list else 0

        # Pad reads
        padded = b''.join(r[:read_len].ljust(read_len, b'N') for r in read_list)

        # Upload reads to GPU
        cudaMalloc(<void**>&d_reads, n_reads * read_len)
        cudaMemcpy(d_reads, padded, n_reads * read_len, cudaMemcpyHostToDevice)

        # ── Phase 1: Seed reads on GPU ──
        rp_arr = np.full(n_reads, -1, dtype=np.int32)
        fp_arr = np.full(n_reads, -1, dtype=np.int32)
        cudaMalloc(<void**>&d_rp, n_reads * 4)
        cudaMalloc(<void**>&d_fp, n_reads * 4)
        cudaMemcpy(d_rp, rp_arr.ctypes.data, n_reads * 4, cudaMemcpyHostToDevice)
        cudaMemcpy(d_fp, fp_arr.ctypes.data, n_reads * 4, cudaMemcpyHostToDevice)

        launch_seed_reads(
            d_reads, n_reads, read_len,
            self.d_table_keys, self.d_table_vals,
            self.table_size, 8, self.k, self.w,
            d_rp, d_fp,
        )

        # Download anchors
        cudaMemcpy(rp_arr.ctypes.data, d_rp, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaMemcpy(fp_arr.ctypes.data, d_fp, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaFree(d_rp); cudaFree(d_fp)

        # ── Phase 2: Banded SW on GPU (uses full ref for simplicity) ──
        scores_arr = np.zeros(n_reads, dtype=np.float32)
        rs_arr = np.zeros(n_reads, dtype=np.int32)
        re_arr = np.zeros(n_reads, dtype=np.int32)
        fs_arr = np.zeros(n_reads, dtype=np.int32)
        fe_arr = np.zeros(n_reads, dtype=np.int32)

        cudaMalloc(<void**>&d_scores, n_reads * 4)
        cudaMalloc(<void**>&d_rs, n_reads * 4)
        cudaMalloc(<void**>&d_re, n_reads * 4)
        cudaMalloc(<void**>&d_fs, n_reads * 4)
        cudaMalloc(<void**>&d_fe, n_reads * 4)

        launch_sw_align(
            d_reads, self.d_ref,
            n_reads, read_len, self.ref_len,
            band_width, gap_open, gap_extend,
            d_scores, d_rs, d_re, d_fs, d_fe,
        )

        cudaMemcpy(scores_arr.ctypes.data, d_scores, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaMemcpy(rs_arr.ctypes.data, d_rs, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaMemcpy(re_arr.ctypes.data, d_re, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaMemcpy(fs_arr.ctypes.data, d_fs, n_reads * 4, cudaMemcpyDeviceToHost)
        cudaMemcpy(fe_arr.ctypes.data, d_fe, n_reads * 4, cudaMemcpyDeviceToHost)

        cudaFree(d_reads); cudaFree(d_scores)
        cudaFree(d_rs); cudaFree(d_re); cudaFree(d_fs); cudaFree(d_fe)

        pos = scores_arr > 0
        return {
            "n_reads": n_reads,
            "scores": scores_arr,
            "read_start": rs_arr, "read_end": re_arr,
            "ref_start": fs_arr, "ref_end": fe_arr,
            "score_mean": float(np.mean(scores_arr[pos])) if pos.any() else 0.0,
            "score_max": float(np.max(scores_arr)) if n_reads else 0.0,
        }
