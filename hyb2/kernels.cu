/* kernels.cu — HybAligner2 GPU kernels.
 *
 * Pipeline: build_index → seed_reads → (CPU chains anchors) → sw_align
 * GPU: GB10 Blackwell sm_120, 228KB shared mem per SM, CUDA 13.x.
 * Recurrence: banded Smith-Waterman-Gotoh (affine gap), double-buffered.
 */

#include <cuda_runtime.h>

typedef unsigned long long u64;

/* ── DNA helpers ────────────────────────────────────────────── */
__device__ __forceinline__ int base2idx(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return 4;
    }
}

__constant__ int SCORE[5][5] = {
    { 2,-3,-1,-3,-1}, {-3, 2,-3,-1,-1}, {-1,-3, 2,-3,-1},
    {-3,-1,-3, 2,-1}, {-1,-1,-1,-1,-1},
};

/* ── Minimizer hashing ──────────────────────────────────────── */
__device__ u64 kmer_hash(const char* s, int p, int k) {
    u64 f=0,r=0;
    for(int i=0;i<k;i++){unsigned char b=base2idx(s[p+i]);f=(f<<2)|b;r=(r>>2)|((u64)(3-b)<<(2*(k-1)));}
    return f<r?f:r;
}
__device__ u64 rehash(u64 k,int a){k=(~k)+(a<<21);k^=(k>>24);k=(k+(k<<3))+(k<<8);k^=(k>>14);k=(k+(k<<2))+(k<<4);k^=(k>>28);k+=(k<<31);return k;}

/* ── Kernel 1: Build minimizer hash table from reference ────── */
__global__ void build_index(const char* ref,int rl,int k,int w,u64* tk,int* tv,int ts,int mv){
    int tid=blockIdx.x*blockDim.x+threadIdx.x,nw=rl-k-w+2;
    if(tid>=nw/w)return;
    int win=tid*w;u64 mh=~0ULL;int mp=-1;
    for(int o=0;o<w&&win+o+k<=rl;o++){u64 h=kmer_hash(ref,win+o,k);if(h<mh){mh=h;mp=win+o;}}
    if(mp<0)return;
    for(int a=0;a<32;a++){u64 p=(a==0)?mh:rehash(mh,a);unsigned s=p%ts;
        u64 old=atomicCAS(&tk[s],0xFFFFFFFFFFFFFFFFULL,mh);
        if(old==0xFFFFFFFFFFFFFFFFULL||old==mh){int b=s*mv;for(int v=0;v<mv;v++){if(atomicCAS(&tv[b+v],-1,mp)==-1)break;}return;}
    }
}

/* ── Kernel 2: Seed reads against hash table ────────────────── */
__global__ void seed_reads(const char* reads,int nr,int rl,const u64* tk,const int* tv,int ts,int mv,int k,int w,int* orp,int* ofp){
    int rid=blockIdx.x*blockDim.x+threadIdx.x;
    if(rid>=nr){orp[rid]=-1;ofp[rid]=-1;return;}
    const char* s=reads+rid*rl;int nw=rl-k-w+2;orp[rid]=-1;ofp[rid]=-1;
    if(nw<=0)return;
    for(int win=0;win<nw;win+=w){u64 mh=~0ULL;int mp=-1;
        for(int o=0;o<w&&win+o+k<=rl;o++){u64 h=kmer_hash(s,win+o,k);if(h<mh){mh=h;mp=win+o;}}
        if(mp<0)continue;
        for(int a=0;a<32;a++){u64 p=(a==0)?mh:rehash(mh,a);unsigned s2=p%ts;
            if(tk[s2]==mh){orp[rid]=mp;ofp[rid]=tv[s2*mv];return;}
            if(tk[s2]==0xFFFFFFFFFFFFFFFFULL)break;
        }
    }
}

/* ── Kernel 3: Banded Smith-Waterman-Gotoh with anchor windowing
 * Double-buffered: prev_M/Ix/Iy + curr_M/Ix/Iy = 6 × band ints/thread.
 * Threads/block auto-capped to fit GB10 228KB shared memory. ─── */
__global__ void sw_align(
    const char* __restrict__ reads, const char* __restrict__ ref, int ref_len,
    const int* __restrict__ anchor_rp, const int* __restrict__ anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* __restrict__ scores,
    int* __restrict__ read_start, int* __restrict__ read_end,
    int* __restrict__ ref_start, int* __restrict__ ref_end)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_reads) return;
    const char* read = reads + rid * read_len;
    int band = 2 * band_width + 1;

    extern __shared__ int sh[];
    int* my = sh + threadIdx.x * (6 * band);
    int* prev_M  = my;
    int* prev_Ix = prev_M  + band;
    int* prev_Iy = prev_Ix + band;
    int* curr_M  = prev_Iy + band;
    int* curr_Ix = curr_M  + band;
    int* curr_Iy = curr_Ix + band;

    /* Reference window from anchor */
    int ar = anchor_rp[rid], af = anchor_fp[rid];
    int rws;
    if (ar >= 0 && af >= 0) {
        rws = af - ar - band_width;
        if (rws < 0) rws = 0;
        if (rws >= ref_len) rws = ref_len - 1;
    } else { rws = 0; }
    int rwe = rws + read_len + band;
    if (rwe > ref_len) rwe = ref_len;

    for (int k = 0; k < band; k++) prev_M[k] = prev_Ix[k] = prev_Iy[k] = 0;
    int best = 0, best_i = 0, best_j = 0;

    for (int i = 0; i < read_len; i++) {
        int rc = base2idx(read[i]);
        int js = rws + i - band_width, je = rws + i + band_width;
        if (js < rws) js = rws;
        if (je >= rwe) je = rwe - 1;

        for (int jj = js; jj <= je; jj++) {
            int k = jj - (rws + i) + band_width;
            curr_M[k] = curr_Ix[k] = curr_Iy[k] = 0;
        }
        /* Pass 1: M and Ix */
        for (int jj = js; jj <= je; jj++) {
            int k = jj - (rws + i) + band_width;
            int s = SCORE[rc][base2idx(ref[jj])];
            int diag = prev_M[k];
            if (prev_Ix[k] > diag) diag = prev_Ix[k];
            if (prev_Iy[k] > diag) diag = prev_Iy[k];
            curr_M[k] = (diag + s > 0) ? diag + s : 0;
            int ix = 0;
            if (k + 1 < band) {
                int fm = prev_M[k+1] - gap_open - gap_extend;
                int fi = prev_Ix[k+1] - gap_extend;
                ix = (fm > fi) ? fm : fi;
            }
            curr_Ix[k] = (ix > 0) ? ix : 0;
        }
        /* Pass 2: Iy */
        for (int jj = js; jj <= je; jj++) {
            int k = jj - (rws + i) + band_width;
            int iy = 0;
            if (k > 0) {
                int fm = curr_M[k-1] - gap_open - gap_extend;
                int fi = curr_Iy[k-1] - gap_extend;
                iy = (fm > fi) ? fm : fi;
            }
            curr_Iy[k] = (iy > 0) ? iy : 0;
        }
        /* Track best */
        for (int jj = js; jj <= je; jj++) {
            int k = jj - (rws + i) + band_width;
            if (curr_M[k] > best) { best = curr_M[k]; best_i = i; best_j = jj; }
        }
        /* Swap prev ↔ curr */
        int* tmp;
        tmp = prev_M;  prev_M  = curr_M;  curr_M  = tmp;
        tmp = prev_Ix; prev_Ix = curr_Ix; curr_Ix = tmp;
        tmp = prev_Iy; prev_Iy = curr_Iy; curr_Iy = tmp;
    }

    scores[rid] = (float)best;
    read_start[rid] = (best > 0) ? 0 : read_len;
    read_end[rid]   = (best > 0) ? read_len : 0;
    ref_start[rid]  = (best > 0) ? (best_j - best_i) : ref_len;
    ref_end[rid]    = (best > 0) ? (best_j - best_i + read_len) : 0;
    if (ref_start[rid] < 0) ref_start[rid] = 0;
    if (ref_end[rid] > ref_len) ref_end[rid] = ref_len;
}

/* ── C-callable launchers ───────────────────────────────────── */
extern "C" {

int launch_build_index(const char* ref,int rl,int k,int w,u64* tk,int* tv,int ts,int mv){
    int nw=rl-k-w+2,blocks=(nw/w+255)/256;
    if(blocks<=0)return-1;
    build_index<<<blocks,256>>>(ref,rl,k,w,tk,tv,ts,mv);
    cudaDeviceSynchronize();
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int launch_seed_reads(const char* reads,int nr,int rl,const u64* tk,const int* tv,int ts,int mv,int k,int w,int* orp,int* ofp){
    int blocks=(nr+255)/256;
    if(blocks<=0)return-1;
    seed_reads<<<blocks,256>>>(reads,nr,rl,tk,tv,ts,mv,k,w,orp,ofp);
    cudaDeviceSynchronize();
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int launch_sw_align(const char* reads,const char* ref,int ref_len,const int* anchor_rp,const int* anchor_fp,int n_reads,int read_len,int band_width,int gap_open,int gap_extend,float* scores,int* rs,int* re,int* fs,int* fe){
    int band=2*band_width+1;
    int shmem_per_thread=6*band*(int)sizeof(int);
    /* Cap: use at most 48KB dynamic shmem (safe for all sm_120 configs).
     * If this isn't enough threads, reduce band_width at call site. */
    int max_shmem = 48 * 1024;
    int max_threads = max_shmem / shmem_per_thread;
    if (max_threads < 1) max_threads = 1;
    int threads = max_threads;
    if (threads > 256) threads = 256;
    if (threads < 1)   threads = 1;
    int shmem = threads * shmem_per_thread;
    cudaFuncSetAttribute(sw_align,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem);
    int blocks=(n_reads+threads-1)/threads;
    sw_align<<<blocks,threads,shmem>>>(reads,ref,ref_len,anchor_rp,anchor_fp,n_reads,read_len,band_width,gap_open,gap_extend,scores,rs,re,fs,fe);
    cudaDeviceSynchronize();
    return cudaGetLastError()==cudaSuccess?0:-1;
}

} /* extern "C" */
