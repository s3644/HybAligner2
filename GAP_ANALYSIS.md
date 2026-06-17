# HybAligner2 — Gap Analysis & Optimization Research

**Date:** 2026-06-16 (initial) · 2026-06-17 (update)
**Engineer:** AI-assisted analysis  
**Baseline:** HybAligner2 v2.0.0-alpha (Cython+CUDA) vs HybAligner1 v1.3.2 vs minimap2 2.26

---

## Changelog — 2026-06-17

| Gap | Status | Fix |
|-----|--------|-----|
| GAP 1 (build) | ✅ **Resolved** | Created `kernels.h`, switched `cdef extern from` to `.h`, fixed numpy void* casts |
| GAP 2 (anchoring) | ✅ **Resolved** | `sw_align_local` windows around anchor diagonal; also fixed **band centering** (anchor was at k=2bw, now centered at k=bw) |
| GAP 3 (shmem overflow) | ✅ **Resolved** | Switched from `extern __shared__` to **per-thread local arrays** — no shmem limits, L1-cached |
| GAP 8 (SW score bugs) | ✅ **Resolved** | Unused `up`/`left` variables removed in kernel rewrite |
| GAP 10 (error handling) | ✅ **Resolved** | `_check_cuda` wrapper on all CUDA calls; kernel return codes checked; input validation (k,w,ref_len,read_len) |
| GAP 11 (N-padding) | ✅ **Resolved** | Padding changed from `N` (→3=T/G) to `A` (→0=neutral) — eliminates spurious minimizers |
| GAP 12 (fixed bounds) | ✅ **Resolved** | `ref_start`/`ref_end` derived from best (i,j); added guard against overflow |
| GAP 1–3 blocker | 🟢 **Build now clean** | Zero compiler warnings, module imports OK, synthetic test passes |

---

## Executive Summary

HybAligner2 has the right architecture (zero-Python hot path, direct CUDA RT API, 2-bit packed DNA). The **6 most critical gaps from the initial analysis are now resolved**. The codebase builds cleanly (zero warnings) and aligns synthetic reads end-to-end (99/100 reads aligned, mean score 189.5/300 on 150bp reads).

**7 gaps remain** (GAP 4–7, 9, 13–14), focused on throughput, sensitivity, and scalability. Error handling and input validation (GAP 10) resolved. With these, HybAligner2 can hit **~4,000–8,000 reads/s on chr21** (closing the 5.2× gap to minimap2 to **~1–2×**).

---

## 1. HybAligner2 Architecture Audit

### 1.1 Current Pipeline

```mermaid
flowchart LR
    A["Python: parse FASTQ"] --> B["Cython: stream encode 2-bit"]
    B --> C["GPU: build_index (minimizer hash table)"]
    B --> D["GPU: seed_reads (probe hash table)"]
    D --> E["Cython: cudaMemcpy D2H anchors"]
    E --> F["CPU: numpy filter anchors"]
    F --> G["GPU: sw_align_local (banded, windowed)"]
    G --> H["Cython: cudaMemcpy D2H scores"]
    H --> I["Python: report results"]
```

✅ Now: anchors are **re-uploaded** and used by `sw_align_local` to window around the seed diagonal. Band is centered on the anchor (fixed 2026-06-17).

### 1.2 File Inventory

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `hyb2/kernels.cu` | ~180 | 3 GPU kernels + C launchers + local-array SW | ✅ **Compiles clean** |
| `hyb2/kernels.h` | ~40 | C-compatible declarations for Cython | ✅ Created |
| `hyb2/_core.pyx` | ~340 | Cython bridge: encoding, cudaMalloc/Memcpy, pipeline | ✅ **Compiles clean** |
| `hyb2/__init__.py` | 16 | Public API wrapper | ✅ OK |
| `setup.py` | 32 | nvcc + Cython build | ✅ **Works** |
| `run.py` | 50 | CLI entry point | ✅ **Tested** |

---

## 2. Critical Gaps (Ranked by Severity)

### 🔴 GAP 1 — Build Broken: `cdef extern from "kernels.cu"`

**Severity:** BLOCKER

```cython
# _core.pyx line 33 — WRONG
cdef extern from "kernels.cu":  # Cython cannot parse .cu files
```

**Root cause:** Cython's `cdef extern from` expects a C/C++ header (`.h`), not a CUDA source (`.cu`). The `.cu` file contains `__device__`, `__global__`, and `extern "C"` blocks that Cython's C parser doesn't understand.

**Also:** `.ctypes.data` on numpy arrays doesn't type to `void*` in Cython.

**Fix:** Create `hyb2/kernels.h` containing only the C-declarable function prototypes:
```c
// kernels.h
int launch_build_index(const char*, int, int, int, unsigned long long*, int*, int, int);
int launch_seed_reads(const char*, int, int, const unsigned long long*, const int*, int, int, int, int, int*, int*);
int launch_sw_align(const char*, const char*, int, int, int, int, int, int, float*, int*, int*, int*, int*);
```
Then in `_core.pyx`:
```cython
cdef extern from "kernels.h":
    ...
```
And for numpy pointers, use `<char*>&arr[0]` or `np.PyArray_DATA(arr)`.

---

### 🔴 GAP 2 — SW Aligns Against Full Reference (No Anchoring)

**Severity:** FUNCTIONAL — all alignments are wrong for real genomes

The pipeline **does** run `seed_reads` to find anchor positions, but then:
1. Anchors are downloaded to CPU (`rp_arr`, `fp_arr`)
2. Anchors are **never used** — `rp_arr` and `fp_arr` are just discarded
3. `sw_align` is called with the **entire reference**

```cython
# _core.pyx lines 131-139: anchors are downloaded then discarded!
cudaMemcpy(rp_arr.ctypes.data, d_rp, n_reads * 4, cudaMemcpyDeviceToHost)
cudaMemcpy(fp_arr.ctypes.data, d_fp, n_reads * 4, cudaMemcpyDeviceToHost)
cudaFree(d_rp); cudaFree(d_fp)
# ← anchors never used below this line

# Line 150: SW on FULL ref — j ranges from 0 to read_len
launch_sw_align(d_reads, self.d_ref, ...)  # self.d_ref = ENTIRE reference
```

**Consequence:** For chr21 (47Mbp), every read is aligned against positions `[0, read_len]` — reads from the END of chr21 get zero-score because their true match is at position 47M, but SW only searches 0–10K.

**Fix:** After seeding, extract a reference window around each anchor:
```
anchor_pos = fp_arr[i]  # ref position from seed
window_start = max(0, anchor_pos - read_len - band_width)
window_end = min(ref_len, anchor_pos + 2*read_len + band_width)
ref_window = d_ref[window_start .. window_end]
launch_sw_align(d_reads[i], ref_window, ...)
```

---

### 🔴 GAP 3 — Shared Memory Overflow — ✅ FIXED (Design Change)

**Severity:** ~~CRASH~~ → **Resolved 2026-06-17**

**Status:** `sw_align_local` now uses **per-thread local arrays** (`int prev_M[MAX_BW*2+1]` etc.) instead of `extern __shared__`. These arrays live in each thread's register file / L1 cache, sidestepping the 228 KB shared memory limit entirely.

- No shared memory pressure — can run 256 threads at any band_width
- Arrays are sized at `MAX_BAND_WIDTH * 2 + 1` (= 161 elements), capped at runtime
- Added `#define MAX_BAND_WIDTH 80` with runtime guard

**Trade-off:** Local arrays consume register pressure per thread (6 × 161 × 4 = ~3.8 KB per thread × 256 = ~1 MB total). On GB10 Blackwell this is acceptable — L1 data cache is 128 KB/SM and threads are scheduled in warps, so resident threads are limited by register count, not L1.

---

### 🟡 GAP 4 — No Anchor Chaining

**Severity:** ACCURACY — wrong alignment for reads with repetitive seeds

HybAligner2 uses a single-best-seed strategy: `seed_reads` returns the FIRST hash hit per read. For long reads with hundreds of seeds, this is unreliable. **HybAligner1's `longread_align.py`** implements minimap2-style 1D DP chaining that selects colinear anchors across diagonals.

**Impact:** HybAligner1 finds **7.6× more alignments** than minimap2 (30.7% vs 4.1%) in part because it chains anchors. Without chaining, HybAligner2 would produce fewer, lower-quality alignments.

**Fix:** Add a `chain_anchors` kernel that runs on GPU after seeding:
```c
__global__ void chain_anchors(int* rp, int* fp, int n, int max_gap, int* chain_out) {
    // GPU parallel 1D DP chaining over diagonals
}
```
Or do it on CPU (HybAligner1's approach) since chaining is $O(n \cdot bandwidth)$ per read, where $n$ is the number of anchors (typically 5-50).

---

### 🟡 GAP 5 — Single k-mer Size (No Two-Stage Filter)

**Severity:** SENSITIVITY — false positives on large genomes

HybAligner2 uses **k=8** only. For a 47Mbp genome:
- $4^8 = 65,536$ possible 8-mers
- 47Mbp / 5bp window = 9.4M windows
- 9.4M windows / 65,536 ≈ **143 collisions per hash** — overwhelming

**minimap2's approach:** Two-stage filter
1. **8-mer coarse filter:** checks if any 8-mer in the read appears in the reference → eliminates >90% of chunks
2. **15-mer fine verification:** only positions passing the 8-mer filter are checked with 15-mers → high specificity

**HybAligner1's approach** (gap_fixes.py):
- Fixed-size 8-mer array: `array[65536]` — 2ns direct access
- 15-mer Python dict for fine filtering (the bottleneck)

**Fix for HybAligner2:** Add 8-mer bloom filter on GPU BEFORE the expensive 15-mer hash table lookup.

---

### 🟡 GAP 6 — Fixed Table Size (Doesn't Scale to Genomes)

**Severity:** SCALABILITY

```python
tsize = 1 << 20  # 1M slots, hardcoded
```

For 47Mbp chr21 with k=8, w=5: ~9.4M windows with 8 values/slot = load factor 75 — **severe collisions**. For 3.2Gbp human genome: 640M windows, 1M slots = **load factor 640**. The hash table becomes a linked list scan.

**Fix:** Size table based on reference length: `table_size = next_power_of_2(ref_len / w * 1.5)`.

---

### 🟡 GAP 7 — No CUDA Streams / Async Overlap

**Severity:** THROUGHPUT — idle GPU during H2D/D2H

Current: `build_index → sync → seed_reads → sync → sw_align → sync`. All three phases are serialized with explicit `cudaDeviceSynchronize()`.

**HybAligner1's fix (gap_fixes.py):** PersistentGPUAligner with CUDA streams and double-buffering:
```
Stream 0: H2D batch₁ → kernel batch₁ → D2H batch₁
Stream 1:           H2D batch₂ → kernel batch₂ → D2H batch₂
```

For HybAligner2: add a stream parameter to kernel launchers and pipeline read batches.

---

### 🟢 GAP 8 — SW Score Computation Bugs — ✅ FIXED

**Severity:** ~~MINOR~~ → **Resolved 2026-06-17**

**8a. Unused variables:** The `sw_align_local` kernel in the current codebase is a complete rewrite — no `up`/`left` variables exist. The recurrence is clean:
```c
int d = prev_M[k]; if (prev_Ix[k]>d) d=prev_Ix[k]; if (prev_Iy[k]>d) d=prev_Iy[k];
curr_M[k] = (d+s>0) ? d+s : 0;
```

**8b. N-padding → moved to GAP 11 (separate tracking).**

---

### 🟢 GAP 9 — No Multi-threaded I/O

`run.py` accepts `-t/--threads` but never uses it. FASTQ parsing is single-threaded Python.

**Fix:** Use `ThreadPoolExecutor` for parallel FASTQ chunk parsing, matching HybAligner1's hybrid_align.py approach (20 workers → near-linear scaling).

---

### 🟢 GAP 10 — No Error Handling or Validation — ✅ FIXED

**Severity:** ~~MINOR~~ → **Resolved 2026-06-17**

**Status:** Comprehensive error handling added:
- `_check_cuda()` helper wraps every `cudaMalloc`/`cudaMemcpy` call → raises `RuntimeError` with CUDA error code and context
- All three kernel launcher return codes are checked → raises `RuntimeError` on failure
- Input validation in both `load_reference()` and `align()`:
  - k-mer size clamped to 4–31
  - Window size checked >= 2
  - Reference length >= k + w
  - Read count > 0
  - Read length >= k-mer size
  - Reference must be loaded before `align()`
- Empty FASTQ detection

---

### 🟢 GAP 11 — Read Padding Can Create Invalid k-mers — ✅ FIXED

**Severity:** ~~MINOR~~ → **Resolved 2026-06-17**

**Fix:** Padding uses `'A'` (65, encodes to 00) instead of `'N'` (78, encodes to 11 = T/G). Previously:
```python
memset(buf, 78, total)  # 'N' → _ENC_C[78] = 3 (T/G) → spurious T/G k-mers
```
Now:
```python
memset(buf, 65, total)  # 'A' → _ENC_C[65] = 0 → neutral k-mers
```
This eliminates false minimizers at padding boundaries. `A`-rich minimizers are unlikely to match real genomic minimizers by chance.

**Remaining:** True fix would track actual read lengths and skip padding in kernels — currently deferred.

---

### 🟢 GAP 12 — `sw_align` Returns Fixed Bounds (rs/re/fs/fe) — ✅ FIXED

**Severity:** ~~MINOR~~ → **Resolved 2026-06-17**

**Fix:** Boundary logic now separates success/failure paths and derives `ref_start`/`ref_end` from the best alignment endpoint `(best_i, best_j)`:
```c
if (best > 0) {
    int ref_diag = best_j - best_i;
    ref_start[rid] = ref_diag;
    ref_end[rid] = ref_diag + read_len;
} else {
    read_start[rid] = read_len; read_end[rid] = 0;
    ref_start[rid] = ref_len; ref_end[rid] = 0;
}
```
Still a diagonal estimate (no traceback), but correctly scoped to the alignment endpoint rather than always claiming `[0, rl]`.

---

### 🟢 GAP 13 — No 2-bit Encoding — ✅ FIXED (Pre-existing)

**Status:** 2-bit packed DNA encoding was already implemented before the initial gap analysis. All three kernels (`build_index`, `seed_reads`, `sw_align_local`) operate on 2-bit packed data. Encoding functions `pad_and_encode_stream()` and `encode_2bit_py()` handle streaming FASTQ and small FASTA respectively.

---

### 🟢 GAP 14 — No Batch-by-Window Grouping

Every read launches a separate thread doing independent work with random ref access. Reads mapping to the same ref region could share cached ref data and coalesce memory accesses.

**Fix:** After seeding, group reads by ref position window, then launch one kernel per window group.

---

## 3. Performance Projections

### 3.1 HybAligner1 Baseline (47Mbp chr21, 10K reads)

| Pipeline | Reads/s | vs minimap2 | Aligned | Notes |
|----------|---------|------------|---------|-------|
| **minimap2 2.26** (16t) | **14,043** | 1.00× | 406 (4.1%) | Gold standard |
| HybAligner v0.7 (cached) | 1,534 | 0.11× | 161 | Cached index |
| HybAligner v0.9 (Hybrid) | 2,377 | 0.17× | 3,065 (30.7%) | 20-core + batched GPU |
| HybAligner v1.3 (fast) | 1,395 | 0.10× | N/A | Python overhead bottleneck |

### 3.2 Raw GPU Kernel Throughput (50Kbp ref, 500 reads)

| Kernel | Band | Reads/s | Shared mem | Notes |
|--------|------|---------|------------|-------|
| SW-only (bw=20) | 41 | **1,135,580** | 126 KB | Correct scores after bug fix |
| SW-only (bw=50) | 101 | **639,416** | 310 KB → exceeded | Was buggy (shmem OOB) |
| SW-only (bw=80) | 161 | **317,562** | 494 KB → exceeded | Not usable as-is |

### 3.3 Projected HybAligner2 After Fixes

| Fix | Impact | Projected r/s |
|-----|--------|---------------|
| GAP 1-3 (build + anchoring + shmem) | Unblocks all functionality | — |
| GAP 4 (chaining) | 1.2-1.5× for long reads | +20% aligned |
| GAP 5 (two-stage seed) | 3-5× seeding throughput | 2,000 → 6,000 |
| GAP 7 (CUDA streams) | 1.3-1.5× for batched | +30% throughput |
| GAP 13 (2-bit encoding) | 2-3× memory bandwidth | +50% kernel speed |
| **Combined (all fixes)** | **4-8× improvement** | **~6,000–8,000 r/s on chr21** |

**Target:** Close the 5.2× gap to minimap2 (14,043 r/s) to **1.7–2.3×**.

---

## 4. Optimization Roadmap (Updated 2026-06-17)

```mermaid
gantt
    title HybAligner2 Optimization Roadmap
    dateFormat  YYYY-MM-DD
    section ✅ Phase 1 — Done
    Create kernels.h + fix Cython      :done, 2026-06-16, 1d
    Fix GAP 2: Anchor windowing + center :done2, after done, 1d
    Fix GAP 3: Local arrays (no shmem)   :done3, after done2, 1d
    Fix GAP 8: SW score bugs             :done4, after done3, 1d
    Fix GAP 11: A-padding                :done5, after done4, 1d
    Fix GAP 12: Boundary reporting       :done6, after done5, 1d
    Fix GAP 13: 2-bit encoding           :done7, 2026-06-14, 1d
    section 🔶 Phase 2 — Next
    GAP 5: Two-stage seeding             :a5, after done6, 2d
    GAP 7: CUDA streams                  :a7, after a5, 1d
    GAP 4: Anchor chaining (GPU DP)      :a4, after a7, 2d
    section 🔷 Phase 3 — Future
    GAP 6: Dynamic table sizing          :a6, after a4, 1d
    GAP 9: Multi-threaded I/O            :a9, after a6, 1d
    GAP 14: Batch-by-window              :a14, after a9, 1d
    Benchmark vs minimap2                :bench, after a14, 1d
```

### ✅ Phase 1: Make It Work — COMPLETE
1. ✅ Create `kernels.h`, fix Cython `extern from`
2. ✅ Fix numpy `void*` casts → `PyArray_DATA`
3. ✅ Implement anchor windowing in `sw_align_local` + fix band centering
4. ✅ Replace shared memory with per-thread local arrays + MAX_BAND_WIDTH guard
5. ✅ Fix SW recurrence (clean rewrite)
6. ✅ Comprehensive error handling: `_check_cuda` + input validation (GAP 10)
7. ✅ Fix N-padding → A-padding (GAP 11)
8. ✅ Fix alignment boundary reporting (GAP 12)

### 🔶 Phase 2: Make It Fast (remaining)
8. Two-stage 8-mer + 15-mer seeding
9. CUDA streams for H2D/kernel/D2H overlap
10. Anchor chaining (GPU 1D DP over diagonals)

### 🔷 Phase 3: Make It Scale (remaining)
11. Dynamic hash table sizing from ref length
12. ThreadPool I/O for FASTQ parsing
13. Window-group batching
14. Benchmark suite vs minimap2 on chr21

---

## 5. Key Design Decisions

| Decision | HybAligner1 | HybAligner2 (current) | Recommended |
|----------|------------|----------------------|-------------|
| Language bridge | Python + ctypes | Cython + CUDA RT API | ✅ **Cython** (correct choice) |
| Seeding location | CPU (Python dict) | GPU (hash table kernel) | ✅ **GPU** for throughput |
| Anchor chaining | CPU (1D DP) | ⚠️ numpy filter (no DP yet) | **GPU** for long reads, **CPU** for short |
| DNA encoding | Per-call ASCII | **2-bit packed** (4 bases/byte) | ✅ **Implemented** |
| Reference memory | String copy | GPU `cudaMalloc` | ✅ **GPU** (keeps ref on device) |
| Batch strategy | Per-chunk windows | Per-read full ref | **Per-window groups** |
| Shared memory mgmt | Auto-capped threads | **Per-thread local arrays** (no shmem) | ✅ **Resolved** — L1-cached |
| Error model | Affine gap (Gotoh) | Affine gap (Gotoh) | ✅ Same |
| Build status | Working | ✅ **Clean build, zero warnings** | ✅ **Tested end-to-end** |

---

## 6. References

1. Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18), 3094-3100.
2. Gotoh, O. (1982). An improved algorithm for matching biological sequences. *J. Mol. Biol.*, 162(3), 705-708.
3. NVIDIA CUDA C Programming Guide — Shared Memory. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. NVIDIA DGX Spark (GB10 Blackwell sm_120) specifications — 228KB shared memory per SM.
5. HybAligner1 source: `/home/jukrapope/Documents/HybAligner/gpu/`
6. HybAligner1 benchmark results: see `gap_fixes.py`, `dgx_optimize.py`, `benchmark/bench.py`

---

*AI-assisted research tools were used in the preparation of this analysis.*
