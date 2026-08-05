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

/* ── Kernel 2: Multi-seed extraction for chaining ── */
/* Extracts up to MAX_ANCHORS anchors per read for downstream chaining */
#define MAX_ANCHORS 32

__global__ void seed_reads_multi(
    const uint8_t* reads, int nr, int rl,
    const u64* tk, const int* tv, int ts, int mv,
    int k, int w,
    int* out_rp, int* out_fp, int* out_counts)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= nr) { out_counts[rid] = 0; return; }
    const uint8_t* s = reads + rid * ((rl + 3) >> 2);
    int nw = rl - k - w + 2;
    int count = 0;
    if (nw <= 0) { out_counts[rid] = 0; return; }
    
    /* Collect all anchors (not just first) */
    for (int win = 0; win < nw && count < MAX_ANCHORS; win += w) {
        u64 mh = ~0ULL; int mp = -1;
        for (int o = 0; o < w && win + o + k <= rl; o++) {
            u64 h = kmer_hash_2bit(s, win + o, k);
            if (h < mh) { mh = h; mp = win + o; }
        }
        if (mp < 0) continue;
        
        /* Hash table lookup */
        for (int a = 0; a < 32 && count < MAX_ANCHORS; a++) {
            u64 p = (a == 0) ? mh : rehash(mh, a);
            unsigned s2 = p % ts;
            if (tk[s2] == mh) { 
                out_rp[rid * MAX_ANCHORS + count] = mp;
                out_fp[rid * MAX_ANCHORS + count] = tv[s2 * mv];
                count++;
                break;
            }
            if (tk[s2] == 0xFFFFFFFFFFFFFFFFULL) break;
        }
    }
    out_counts[rid] = count;
}

/* ── Kernel 3: Anchor chaining with 1D DP over diagonals ── */
/* Implements minimap2-style chaining: maximize score with colinear anchors */
__global__ void chain_anchors(
    const int* rp, const int* fp, const int* counts, int n_reads,
    int max_gap, int penalty,
    int* best_rp, int* best_fp)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_reads) { best_rp[rid] = -1; best_fp[rid] = -1; return; }
    
    int n = counts[rid];
    if (n == 0) { best_rp[rid] = -1; best_fp[rid] = -1; return; }
    if (n == 1) {
        best_rp[rid] = rp[rid * MAX_ANCHORS];
        best_fp[rid] = fp[rid * MAX_ANCHORS];
        return;
    }
    
    /* DP arrays in registers (small n <= MAX_ANCHORS) */
    int dp[MAX_ANCHORS];
    int prev[MAX_ANCHORS];
    for (int i = 0; i < n; i++) {
        dp[i] = 1;  /* Each anchor has base score 1 */
        prev[i] = -1;
    }
    
    /* O(n^2) DP: find best chain ending at each anchor */
    for (int i = 1; i < n; i++) {
        int ri = rp[rid * MAX_ANCHORS + i];
        int fi = fp[rid * MAX_ANCHORS + i];
        int diag_i = fi - ri;
        
        for (int j = 0; j < i; j++) {
            int rj = rp[rid * MAX_ANCHORS + j];
            int fj = fp[rid * MAX_ANCHORS + j];
            int diag_j = fj - rj;
            
            /* Check colinearity: same diagonal and proper order */
            int gap_r = ri - rj;
            int gap_f = fi - fj;
            int gap_diff = abs(gap_r - gap_f);
            
            if (gap_r > 0 && gap_f > 0 && gap_diff <= max_gap) {
                /* Same diagonal, forward direction */
                int score = dp[j] + 1;
                /* Penalize diagonal shift */
                if (diag_i != diag_j) score -= penalty;
                
                if (score > dp[i]) {
                    dp[i] = score;
                    prev[i] = j;
                }
            }
        }
    }
    
    /* Find best ending anchor */
    int best_idx = 0, best_score = dp[0];
    for (int i = 1; i < n; i++) {
        if (dp[i] > best_score) {
            best_score = dp[i];
            best_idx = i;
        }
    }
    
    /* Return the last anchor in the best chain */
    best_rp[rid] = rp[rid * MAX_ANCHORS + best_idx];
    best_fp[rid] = fp[rid * MAX_ANCHORS + best_idx];
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
    if (rid >= n_reads) return;
    int rl_bytes = (read_len + 3) >> 2;
    const uint8_t* read = reads + rid * rl_bytes;
    /* Cap band_width to avoid overflow of fixed-size local arrays */
    if (band_width > MAX_BAND_WIDTH) band_width = MAX_BAND_WIDTH;
    int band = 2 * band_width + 1;  /* at most 161 */

    /* Fixed-size local arrays (L1-cached, no shmem limit) */
    int prev_M[MAX_BAND_WIDTH * 2 + 1];
    int prev_Ix[MAX_BAND_WIDTH * 2 + 1];
    int prev_Iy[MAX_BAND_WIDTH * 2 + 1];
    int curr_M[MAX_BAND_WIDTH * 2 + 1];
    int curr_Ix[MAX_BAND_WIDTH * 2 + 1];
    int curr_Iy[MAX_BAND_WIDTH * 2 + 1];

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

    for (int k = 0; k < band; k++) prev_M[k] = prev_Ix[k] = prev_Iy[k] = 0;
    int best = 0, best_i = 0, best_j = 0;

    for (int i = 0; i < read_len; i++) {
        int rc = get_base(read, i);
        int js = diag + i - band_width, je = diag + i + band_width;
        if (js < rws) js = rws;
        if (je >= rwe) je = rwe - 1;
        for (int jj = js; jj <= je; jj++)
            curr_M[jj-(diag+i)+band_width] = curr_Ix[jj-(diag+i)+band_width] = curr_Iy[jj-(diag+i)+band_width] = 0;
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width, s = SCORE[rc][get_base(ref,jj)];
            int d = prev_M[k]; if (prev_Ix[k]>d) d=prev_Ix[k]; if (prev_Iy[k]>d) d=prev_Iy[k];
            curr_M[k] = (d+s>0)?d+s:0;
            int ix = 0;
            if (k+1<band) { int fm=prev_M[k+1]-gap_open-gap_extend, fi=prev_Ix[k+1]-gap_extend; ix=(fm>fi)?fm:fi; }
            curr_Ix[k] = (ix>0)?ix:0;
        }
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width, iy = 0;
            if (k>0) { int fm=curr_M[k-1]-gap_open-gap_extend, fi=curr_Iy[k-1]-gap_extend; iy=(fm>fi)?fm:fi; }
            curr_Iy[k] = (iy>0)?iy:0;
        }
        for (int jj = js; jj <= je; jj++) {
            int k = jj-(diag+i)+band_width;
            if (curr_M[k]>best) { best=curr_M[k]; best_i=i; best_j=jj; }
        }
        for (int k = 0; k < band; k++) {
            int t = prev_M[k]; prev_M[k]=curr_M[k]; curr_M[k]=t;
            t = prev_Ix[k]; prev_Ix[k]=curr_Ix[k]; curr_Ix[k]=t;
            t = prev_Iy[k]; prev_Iy[k]=curr_Iy[k]; curr_Iy[k]=t;
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

/* ── C-callable launchers with optional streams ─────────────── */
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

int launch_seed_reads_multi(
    const uint8_t* reads, int nr, int rl,
    const u64* tk, const int* tv, int ts, int mv,
    int k, int w, int* out_rp, int* out_fp, int* out_counts)
{
    int blocks = (nr + 255) / 256;
    if (blocks <= 0) return -1;
    seed_reads_multi<<<blocks, 256>>>(reads, nr, rl, tk, tv, ts, mv, k, w, out_rp, out_fp, out_counts);
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_chain_anchors(
    const int* rp, const int* fp, const int* counts, int n_reads,
    int max_gap, int penalty,
    int* best_rp, int* best_fp)
{
    int threads = 256;
    int blocks = (n_reads + threads - 1) / threads;
    chain_anchors<<<blocks, threads>>>(rp, fp, counts, n_reads, max_gap, penalty, best_rp, best_fp);
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
    int threads = 256;
    int blocks = (n_reads + threads - 1) / threads;
    sw_align_local<<<blocks, threads>>>(
        reads, ref, ref_len, anchor_rp, anchor_fp,
        n_reads, read_len, band_width, gap_open, gap_extend,
        scores, rs, re, fs, fe);
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Stream-aware launchers for async operation */
int launch_seed_reads_multi_async(
    const uint8_t* reads, int nr, int rl,
    const u64* tk, const int* tv, int ts, int mv,
    int k, int w, int* out_rp, int* out_fp, int* out_counts, void* stream)
{
    int blocks = (nr + 255) / 256;
    if (blocks <= 0) return -1;
    seed_reads_multi<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        reads, nr, rl, tk, tv, ts, mv, k, w, out_rp, out_fp, out_counts);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_chain_anchors_async(
    const int* rp, const int* fp, const int* counts, int n_reads,
    int max_gap, int penalty,
    int* best_rp, int* best_fp, void* stream)
{
    int threads = 256;
    int blocks = (n_reads + threads - 1) / threads;
    chain_anchors<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        rp, fp, counts, n_reads, max_gap, penalty, best_rp, best_fp);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_sw_align_async(
    const uint8_t* reads, const uint8_t* ref, int ref_len,
    const int* anchor_rp, const int* anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* scores, int* rs, int* re, int* fs, int* fe,
    void* stream)
{
    int threads = 256;
    int blocks = (n_reads + threads - 1) / threads;
    sw_align_local<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        reads, ref, ref_len, anchor_rp, anchor_fp,
        n_reads, read_len, band_width, gap_open, gap_extend,
        scores, rs, re, fs, fe);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} /* extern "C" */
