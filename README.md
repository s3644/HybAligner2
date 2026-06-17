# HybAligner2

**GPU-accelerated DNA sequence aligner** — Cython + CUDA, 2-bit packed encoding, banded Smith-Waterman-Gotoh on NVIDIA GB10 Blackwell.

Targets **~4,000–8,000 reads/s on chr21** (within 1–2× of minimap2).

---

## Pipeline

```
FASTQ → [CPU] 2-bit encode → [GPU] seed (minimizer hash) → [CPU] anchor chain → [GPU] banded SW → results
```

| Stage | Location | What |
|-------|----------|------|
| Parse + 2-bit encode | CPU (Cython) | Stream FASTQ, pad to uniform length, pack 4 bases/byte |
| Build index | GPU | Minimizer hash table from reference (2-bit packed) |
| Seed reads | GPU | Probe hash table → one anchor per read |
| Chain anchors | CPU (numpy) | Filter colinear anchors — placeholder for full 1D DP |
| Banded SW | GPU | Gotoh affine-gap alignment around anchor window |

### DNA Encoding

2 bits per base: `A=00, C=01, G=10, T=11` — 4 bases per byte, 4× memory reduction vs ASCII. Padding uses `A` (00) to avoid spurious minimizers from `N`→`T/G` encoding.

---

## Build

### Prerequisites

- CUDA 13.0+ (`nvcc` on `$PATH`)
- Cython ≥ 3.0
- NumPy
- GCC / g++

### Compile

```bash
python setup.py build_ext --inplace
```

This runs:
1. `nvcc -arch=sm_120` → `hyb2/kernels.o`
2. Cython → C++ → `hyb2/_core.cpython-*.so`

### Quick check

```bash
python -c "from hyb2 import HybAligner2; print('OK')"
```

---

## Usage

```bash
python run.py reads.fastq ref.fa -b 20 -k 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `-b, --band` | 20 | Band width (±bases). Capped at 80 for fixed-size arrays. |
| `-k, --kmer` | 10 | K-mer size (8–15). Higher = more specific. |
| `-w, --window` | auto | Minimizer window (`k//2 + 1`). |
| `--go` | 5 | Gap open penalty. |
| `--ge` | 2 | Gap extend penalty. |
| `-q, --quiet` | — | Suppress verbose output. |

### Example

```bash
# 100 reads × 150bp vs 5 Kbp reference
python run.py /tmp/test_reads.fastq /tmp/test_ref.fa -b 20 -k 10

# Output:
#   Reads: 100  Seeded: 99  Aligned: 99
#   Time: 5ms  (21438 reads/s)
#   Score: mean=189.5  max=295
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   run.py (CLI)                  │
├─────────────────────────────────────────────────┤
│               HybAligner2 (Cython)              │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │ 2-bit    │ │ cudaMalloc│ │ anchor chain    │  │
│  │ encode   │ │ cudaMemcpy│ │ (numpy where)   │  │
│  └──────────┘ └──────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────┤
│              GPU Kernels (CUDA)                  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │build_idx │ │seed_reads│ │  sw_align_local │  │
│  │(minimizer│ │(hash     │ │  (banded Gotoh) │  │
│  │ hash tbl)│ │ probe)   │ │                  │  │
│  └──────────┘ └──────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language bridge | **Cython** + CUDA RT API | Zero Python overhead in hot path, no ctypes |
| DNA storage | **2-bit packed** on GPU | 4× bandwidth vs ASCII, fits in L1$ |
| SW buffers | **Local arrays** (per-thread) | Sidesteps shared memory overflow at large bands |
| Hash table | **Open-addressing** with rehash | Simple, no linked-list traversal on GPU |
| Anchor chaining | **CPU numpy** (simple filter) | Sufficient for short reads; GPU kernel planned |

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `hyb2/kernels.cu` | ~180 | 3 GPU kernels + `extern "C"` launchers |
| `hyb2/kernels.h` | 40 | C-compatible declarations for Cython |
| `hyb2/_core.pyx` | ~340 | Cython bridge: encoding, memory mgmt, orchestration |
| `hyb2/__init__.py` | 16 | Public API |
| `setup.py` | 32 | nvcc + Cython build |
| `run.py` | 50 | CLI entry point |
| `GAP_ANALYSIS.md` | — | Gap analysis & optimization research |

---

## Status

**2026-06-17** — Build clean (zero warnings). Basic alignment working.

### Resolved Gaps

| Gap | Issue | Fix |
|-----|-------|-----|
| GAP 1 | Cython couldn't parse `.cu` | Created `kernels.h`, switched `extern from` to `.h` |
| GAP 2 | SW ran on full ref (no anchoring) | `sw_align_local` windows around anchor diagonal |
| GAP 3 | Shared memory overflow | Switched to **per-thread local arrays** (L1-cached) |
| GAP 8 | Unused SW variables | Cleaned up in kernel rewrite |
| GAP 10 | No error handling | `_check_cuda` on all CUDA calls + input validation |
| GAP 11 | `N`-padding created false minimizers | Switched to `A`-padding (encodes to 00) |
| GAP 12 | Fixed alignment bounds reporting | Refined ref_start/ref_end from best (i,j) |

### Remaining Work

| Gap | Priority | Description |
|-----|----------|-------------|
| GAP 4 | Medium | Anchor chaining (1D DP over diagonals) |
| GAP 5 | Medium | Two-stage 8-mer + 15-mer seeding |
| GAP 6 | Low | Dynamic hash table sizing |
| GAP 7 | Medium | CUDA streams for async H2D/kernel/D2H |
| GAP 9 | Low | Multi-threaded FASTQ I/O |
| GAP 14 | Low | Batch-by-window grouping |

---

## Benchmark Notes

- **Test system:** NVIDIA GB10 Blackwell (sm_120), 228 KB shared mem/SM
- **Baseline throughput (5 Kbp ref, 100 reads × 150 bp):** ~21K reads/s
- **chr21 projection (47 Mbp):** ~4,000–8,000 reads/s after GAP 4–7

See [`GAP_ANALYSIS.md`](./GAP_ANALYSIS.md) for detailed performance analysis.

---

## License

MIT — see LICENSE file.
