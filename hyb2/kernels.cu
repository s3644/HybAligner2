/* kernels.cu — HybAligner2 GPU kernels (Phase 2: 2-bit packed DNA).
 *
 * Pipeline: build_index → seed_reads → (CPU chains) → sw_align
 * GPU: GB10 Blackwell sm_120, 228KB shared mem per SM, CUDA 13.x.
 * DNA encoding: 2 bits/base (A=00,C=01,G=10,T=11), 4 bases/byte.
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* Max band_width supported by fixed-size local arrays (161 = 2*80+1).
   Kernel silently caps at this to avoid stack corruption. */
#define MAX_BAND_WIDTH 80

typedef unsigned long long u64;

/* ── 2-bit DNA helpers ──────────────────────────────────────── */
__device__ __forceinline__ int get_base(const uint8_t* p, int pos) {
    int byte = pos >> 2;           /* pos / 4 */
    int shift = 6 - ((pos & 3) << 1); /* 6 - 2*(pos%4): bits 6,4,2,0 */
    return (p[byte] >> shift) & 3;
}

__constant__ int SCORE[5][5] = {
    { 2,-3,-1,-3,-1}, {-3, 2,-3,-1,-1}, {-1,-3, 2,-3,-1},
    {-3,-1,-3, 2,-1}, {-1,-1,-1,-1,-1},
};

/* ── Minimizer hashing (2-bit packed) ───────────────────────── */
__device__ u64 kmer_hash_2bit(const uint8_t* s, int p, int k) {
    u64 f = 0, r = 0;
    for (int i = 0; i < k; i++) {
        int b = get_base(s, p + i);
        f = (f << 2) | b;
        r = (r >> 2) | ((u64)(3 - b) << (2 * (k - 1)));
    }
    return f < r ? f : r;
}

__device__ u64 rehash(u64 k, int a) {
    k = (~k) + (a << 21); k ^= (k >> 24);
    k = (k + (k << 3)) + (k << 8); k ^= (k >> 14);
    k = (k + (k << 2)) + (k << 4); k ^= (k >> 28);
    k += (k << 31); return k;
}

/* ── Kernel 1: Build minimizer hash table (2-bit packed ref) ── */
__global__ void build_index(
    const uint8_t* ref, int rl, int k, int w,
    u64* tk, int* tv, int ts, int mv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nw = rl - k - w + 2;
    if (tid >= nw / w) return;
    int win = tid * w;
    u64 mh = ~0ULL; int mp = -1;
    for (int o = 0; o < w && win + o + k <= rl; o++) {
        u64 h = kmer_hash_2bit(ref, win + o, k);
        if (h < mh) { mh = h; mp = win + o; }
    }
    if (mp < 0) return;
    for (int a = 0; a < 32; a++) {
        u64 p = (a == 0) ? mh : rehash(mh, a);
        unsigned s = p % ts;
        u64 old = atomicCAS(&tk[s], 0xFFFFFFFFFFFFFFFFULL, mh);
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == mh) {
            int b = s * mv;
            for (int v = 0; v < mv; v++)
                if (atomicCAS(&tv[b + v], -1, mp) == -1) break;
            return;
        }
    }
}

/* ── Kernel 2: Seed reads (2-bit packed, single best anchor) ── */
__global__ void seed_reads(
    const uint8_t* reads, int nr, int rl,
    const u64* tk, const int* tv, int ts, int mv,
    int k, int w,
    int* orp, int* ofp)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= nr) return;  /* padding threads: out-of-bounds, must not write */
    const uint8_t* s = reads + rid * ((rl + 3) >> 2);
    int nw = rl - k - w + 2;
    orp[rid] = -1; ofp[rid] = -1;
    if (nw <= 0) return;
    for (int win = 0; win < nw; win += w) {
        u64 mh = ~0ULL; int mp = -1;
        for (int o = 0; o < w && win + o + k <= rl; o++) {
            u64 h = kmer_hash_2bit(s, win + o, k);
            if (h < mh) { mh = h; mp = win + o; }
        }
        if (mp < 0) continue;
        for (int a = 0; a < 32; a++) {
            u64 p = (a == 0) ? mh : rehash(mh, a);
            unsigned s2 = p % ts;
            if (tk[s2] == mh) { orp[rid] = mp; ofp[rid] = tv[s2 * mv]; return; }
            if (tk[s2] == 0xFFFFFFFFFFFFFFFFULL) break;
        }
    }
}

__global__ void sw_align_local(
    const uint8_t* __restrict__ reads,
    const uint8_t* __restrict__ ref, int ref_len,
    const int* __restrict__ anchor_rp, const int* __restrict__ anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* __restrict__ scores,
    int* __restrict__ read_start, int* __restrict__ read_end,
    int* __restrict__ ref_start, int* __restrict__ ref_end)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    /* Cap band_width to avoid overflow of the shared-memory allocation
       (must match the clamp the launcher used to size shmem). */
    if (band_width > MAX_BAND_WIDTH) band_width = MAX_BAND_WIDTH;
    int band = 2 * band_width + 1;  /* at most 161 */
    int T = blockDim.x;

    /* Per-thread scratch lives in real on-chip shared memory instead of
       fixed-size local arrays (which nvcc cannot keep in registers when
       indexed by a runtime variable — they silently spill to slow
       local/global memory). Layout is [plane][k][thread] so that all
       threads in a warp touch consecutive addresses for a given k
       (coalesced, bank-conflict-free), since every thread in the block
       iterates the same k range in lockstep. */
    extern __shared__ int smem[];
    int* prev_M  = smem + 0 * band * T;
    int* prev_Ix = smem + 1 * band * T;
    int* prev_Iy = smem + 2 * band * T;
    int* curr_M  = smem + 3 * band * T;
    int* curr_Ix = smem + 4 * band * T;
    int* curr_Iy = smem + 5 * band * T;
    int tid = threadIdx.x;
#define AT(arr, k) arr[(k) * T + tid]

    if (rid >= n_reads) return;
    int rl_bytes = (read_len + 3) >> 2;
    const uint8_t* read = reads + rid * rl_bytes;

    int ar = anchor_rp[rid], af = anchor_fp[rid];
    if (ar < 0 || af < 0) {
        scores[rid] = 0.0f;
        read_start[rid] = read_len; read_end[rid] = 0;
        ref_start[rid] = ref_len; ref_end[rid] = 0;
        return;
    }
    /* Center the band on the anchor diagonal diag = af - ar.
       The band covers ref positions [diag - bw, diag + bw] at each read position i,
       so overall ref window = [diag - bw, diag + read_len - 1 + bw]. */
    int diag = af - ar;
    int rws = diag - band_width;
    int rwe = diag + read_len + band_width;
    if (rws < 0) rws = 0;
    if (rwe > ref_len) rwe = ref_len;

    for (int k = 0; k < band; k++) AT(prev_M,k) = AT(prev_Ix,k) = AT(prev_Iy,k) = 0;
    int best = 0, best_i = 0, best_j = 0;

    for (int i = 0; i < read_len; i++) {
        int rc = get_base(read, i);
        int js = diag + i - band_width, je = diag + i + band_width;
        if (js < rws) js = rws;
        if (je >= rwe) je = rwe - 1;
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width;
            AT(curr_M,k) = AT(curr_Ix,k) = AT(curr_Iy,k) = 0;
        }
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width, s = SCORE[rc][get_base(ref,jj)];
            int d = AT(prev_M,k); if (AT(prev_Ix,k)>d) d=AT(prev_Ix,k); if (AT(prev_Iy,k)>d) d=AT(prev_Iy,k);
            AT(curr_M,k) = (d+s>0)?d+s:0;
            int ix = 0;
            if (k+1<band) { int fm=AT(prev_M,k+1)-gap_open-gap_extend, fi=AT(prev_Ix,k+1)-gap_extend; ix=(fm>fi)?fm:fi; }
            AT(curr_Ix,k) = (ix>0)?ix:0;
        }
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width, iy = 0;
            if (k>0) { int fm=AT(curr_M,k-1)-gap_open-gap_extend, fi=AT(curr_Iy,k-1)-gap_extend; iy=(fm>fi)?fm:fi; }
            AT(curr_Iy,k) = (iy>0)?iy:0;
        }
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width;
            if (AT(curr_M,k)>best) { best=AT(curr_M,k); best_i=i; best_j=jj; }
        }
        for (int k = 0; k < band; k++) {
            int t = AT(prev_M,k); AT(prev_M,k)=AT(curr_M,k); AT(curr_M,k)=t;
            t = AT(prev_Ix,k); AT(prev_Ix,k)=AT(curr_Ix,k); AT(curr_Ix,k)=t;
            t = AT(prev_Iy,k); AT(prev_Iy,k)=AT(curr_Iy,k); AT(curr_Iy,k)=t;
        }
    }
    scores[rid]=(float)best;
    if (best > 0) {
        /* Traceback-free boundaries: best (i,j) is the endpoint.
           Walk back along the best path by re-tracing along the
           diagonal to estimate the start; for simplicity use the
           alignment's diagonal span. */
        int ref_diag = best_j - best_i;
        read_start[rid] = 0;       /* full-read estimate */
        read_end[rid] = read_len;
        ref_start[rid] = ref_diag;
        ref_end[rid] = ref_diag + read_len;
    } else {
        read_start[rid] = read_len; read_end[rid] = 0;
        ref_start[rid] = ref_len; ref_end[rid] = 0;
    }
    if (ref_start[rid] < 0) ref_start[rid] = 0;
    if (ref_end[rid] > ref_len) ref_end[rid] = ref_len;
}
#undef AT

/* ── C-callable launchers ───────────────────────────────────── */
extern "C" {

int launch_build_index(
    const uint8_t* ref, int rl, int k, int w,
    u64* tk, int* tv, int ts, int mv)
{
    int nw = rl - k - w + 2;
    int blocks = (nw / w + 255) / 256;
    if (blocks <= 0) return -1;
    build_index<<<blocks, 256>>>(ref, rl, k, w, tk, tv, ts, mv);
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_seed_reads(
    const uint8_t* reads, int nr, int rl,
    const u64* tk, const int* tv, int ts, int mv,
    int k, int w, int* orp, int* ofp)
{
    int blocks = (nr + 255) / 256;
    if (blocks <= 0) return -1;
    seed_reads<<<blocks, 256>>>(reads, nr, rl, tk, tv, ts, mv, k, w, orp, ofp);
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_sw_align(
    const uint8_t* reads, const uint8_t* ref, int ref_len,
    const int* anchor_rp, const int* anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* scores, int* rs, int* re, int* fs, int* fe)
{
    int bw = band_width;
    if (bw > MAX_BAND_WIDTH) bw = MAX_BAND_WIDTH;
    if (bw < 0) bw = 0;
    int band = 2 * bw + 1;
    size_t bytes_per_thread = 6ULL * band * sizeof(int);

    /* Query and opt into the device's real max shared memory per block
       (default CUDA cap is 48KB; Blackwell/Hopper-class parts allow a
       much larger opt-in limit — query it, don't assume a number). */
    static int budget = 0;
    static bool attr_set = false;
    if (budget == 0) {
        cudaDeviceGetAttribute(&budget, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (budget <= 0) budget = 48 * 1024;
    }

    int threads = (int)(budget / bytes_per_thread);
    if (threads > 256) threads = 256;
    if (threads < 1) threads = 1;
    size_t shmem_bytes = threads * bytes_per_thread;

    if (!attr_set) {
        cudaFuncSetAttribute(sw_align_local, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
        attr_set = true;
    }

    int blocks = (n_reads + threads - 1) / threads;
    sw_align_local<<<blocks, threads, shmem_bytes>>>(
        reads, ref, ref_len, anchor_rp, anchor_fp,
        n_reads, read_len, band_width, gap_open, gap_extend,
        scores, rs, re, fs, fe);
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} /* extern "C" */
