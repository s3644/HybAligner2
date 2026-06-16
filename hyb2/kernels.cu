#include <cuda_runtime.h>

typedef unsigned long long u64;

__device__ unsigned char base2bit(char c) {
    return (c=='A'||c=='a')?0:(c=='C'||c=='c')?1:(c=='G'||c=='g')?2:3;
}
__device__ u64 kmer_hash(const char* s, int p, int k) {
    u64 f=0,r=0;
    for(int i=0;i<k;i++){unsigned char b=base2bit(s[p+i]);f=(f<<2)|b;r=(r>>2)|((u64)(3-b)<<(2*(k-1)));}
    return f<r?f:r;
}
__device__ u64 rehash(u64 k,int a){k=(~k)+(a<<21);k^=(k>>24);k=(k+(k<<3))+(k<<8);k^=(k>>14);k=(k+(k<<2))+(k<<4);k^=(k>>28);k+=(k<<31);return k;}

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
__device__ int sw_score(char a,char b){return(a==b)?2:-3;}
__global__ void sw_align(const char* reads,const char* ref,int nr,int rl,int rfl,int bw,int go,int ge,float* scores,int* rs,int* re,int* fs,int* fe){
    int rid=blockIdx.x*blockDim.x+threadIdx.x;if(rid>=nr)return;
    const char* rseq=reads+rid*rl;int band=2*bw+1;
    extern __shared__ int sh[];int* M=sh+threadIdx.x*band*3;int* Ix=M+band;int* Iy=Ix+band;
    float ms=0;int mi=0,mj=0;
    for(int i=0;i<rl;i++){for(int k2=0;k2<band;k2++){int j=i+k2-bw;if(j<0||j>=rfl){M[k2]=Ix[k2]=Iy[k2]=0;continue;}
        int diag=(k2>0)?M[k2]:0,up=(j>0&&k2<band-1)?Ix[k2+1]:0,left=(j>0&&k2>0)?Iy[k2-1]:0;
        int match=diag+sw_score(rseq[i],ref[j]);M[k2]=(match>0)?match:0;
        int gx=((k2<band-1)?Ix[k2+1]-ge:-999),ox=((k2<band-1)?M[k2+1]-go-ge:-999);Ix[k2]=(gx>ox)?gx:ox;
        int gy=((k2>0)?Iy[k2-1]-ge:-999),oy=((k2>0)?M[k2-1]-go-ge:-999);Iy[k2]=(gy>oy)?gy:oy;
        if(M[k2]>ms){ms=(float)M[k2];mi=i;mj=j;}
    }}
    scores[rid]=ms;rs[rid]=0;re[rid]=rl;fs[rid]=mj-mi;if(fs[rid]<0)fs[rid]=0;fe[rid]=fs[rid]+rl;if(fe[rid]>rfl)fe[rid]=rfl;
}
extern "C" {
int launch_build_index(const char* ref,int rl,int k,int w,u64* tk,int* tv,int ts,int mv){
    int nw=rl-k-w+2,blocks=(nw/w+255)/256;build_index<<<blocks,256>>>(ref,rl,k,w,tk,tv,ts,mv);cudaDeviceSynchronize();return cudaGetLastError()==cudaSuccess?0:-1;
}
int launch_seed_reads(const char* reads,int nr,int rl,const u64* tk,const int* tv,int ts,int mv,int k,int w,int* orp,int* ofp){
    int blocks=(nr+255)/256;seed_reads<<<blocks,256>>>(reads,nr,rl,tk,tv,ts,mv,k,w,orp,ofp);cudaDeviceSynchronize();return cudaGetLastError()==cudaSuccess?0:-1;
}
int launch_sw_align(const char* reads,const char* ref,int nr,int rl,int rfl,int bw,int go,int ge,float* scores,int* rs,int* re,int* fs,int* fe){
    int band=2*bw+1,shmem=256*band*3*sizeof(int);
    cudaFuncSetAttribute(sw_align,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem);
    int blocks=(nr+255)/256;sw_align<<<blocks,256,shmem>>>(reads,ref,nr,rl,rfl,bw,go,ge,scores,rs,re,fs,fe);cudaDeviceSynchronize();return cudaGetLastError()==cudaSuccess?0:-1;
}
}
