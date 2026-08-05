/* kernels.h — C-compatible declarations for HybAligner2 (2-bit packed). */
#ifndef HYB2_KERNELS_H
#define HYB2_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int launch_build_index(
    const uint8_t* ref, int ref_len, int k, int w,
    unsigned long long* table_keys, int* table_vals,
    int table_size, int max_vals_per_key);

int launch_seed_reads_multi(
    const uint8_t* reads, int n_reads, int read_len,
    const unsigned long long* table_keys, const int* table_vals,
    int table_size, int max_vals_per_key, int k, int w,
    int* out_rp, int* out_fp, int* out_counts);

int launch_chain_anchors(
    const int* rp, const int* fp, const int* counts, int n_reads,
    int max_gap, int penalty,
    int* best_rp, int* best_fp);

int launch_sw_align(
    const uint8_t* reads, const uint8_t* ref, int ref_len,
    const int* anchor_rp, const int* anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* scores, int* read_start, int* read_end,
    int* ref_start, int* ref_end);

/* Stream-aware async launchers */
int launch_seed_reads_multi_async(
    const uint8_t* reads, int n_reads, int read_len,
    const unsigned long long* table_keys, const int* table_vals,
    int table_size, int max_vals_per_key, int k, int w,
    int* out_rp, int* out_fp, int* out_counts, void* stream);

int launch_chain_anchors_async(
    const int* rp, const int* fp, const int* counts, int n_reads,
    int max_gap, int penalty,
    int* best_rp, int* best_fp, void* stream);

int launch_sw_align_async(
    const uint8_t* reads, const uint8_t* ref, int ref_len,
    const int* anchor_rp, const int* anchor_fp,
    int n_reads, int read_len,
    int band_width, int gap_open, int gap_extend,
    float* scores, int* read_start, int* read_end,
    int* ref_start, int* ref_end,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif /* HYB2_KERNELS_H */
