#!/usr/bin/env python3
"""HybAligner2 CLI — CPU+GPU hybrid aligner."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hyb2 import HybAligner2

def main():
    import argparse
    p = argparse.ArgumentParser(description="HybAligner2 — CPU+GPU aligner")
    p.add_argument("fastq", help="Input FASTQ")
    p.add_argument("ref", help="Reference FASTA")
    p.add_argument("-b", "--band", type=int, default=20, help="Band width (auto-capped by GPU shmem)")
    p.add_argument("-k", "--kmer", type=int, default=10, help="K-mer size (8-15, higher=more specific)")
    p.add_argument("-w", "--window", type=int, default=0, help="Minimizer window (auto: k//2+1)")
    p.add_argument("--go", type=int, default=5, help="Gap open penalty")
    p.add_argument("--ge", type=int, default=2, help="Gap extend penalty")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"HybAligner2 v2.2 — CPU+GPU hybrid")
        print(f"  FASTQ: {args.fastq}")
        print(f"  Ref:   {args.ref}")
        print(f"  Band:  ±{args.band}  K-mer: {args.kmer}  W: {args.window or args.kmer//2+1}  Gap: {args.go}/{args.ge}")

    aln = HybAligner2()

    t0 = time.perf_counter()
    aln.load_reference(args.ref, k=args.kmer, w=args.window)
    load_ms = (time.perf_counter() - t0) * 1000
    if not args.quiet:
        print(f"  Index: {load_ms:.0f}ms")

    t0 = time.perf_counter()
    result = aln.align(args.fastq, band_width=args.band, gap_open=args.go, gap_extend=args.ge)
    align_ms = (time.perf_counter() - t0) * 1000

    n_aligned = int((result["scores"] > 0).sum())
    throughput = result["n_reads"] / (align_ms / 1000) if align_ms > 0 else 0

    print(f"  Reads: {result['n_reads']}  Seeded: {result['n_seeded']}  Aligned: {n_aligned}")
    print(f"  Time: {align_ms:.0f}ms  ({throughput:.0f} reads/s)")
    print(f"  Score: mean={result['score_mean']:.1f}  max={result['score_max']:.0f}")

if __name__ == "__main__":
    main()
