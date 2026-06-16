#!/usr/bin/env python3
"""HybAligner2 CLI — one command, maximum speed."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hyb2 import HybAligner2

def main():
    import argparse
    p = argparse.ArgumentParser(description="HybAligner2 — Cython+CUDA aligner")
    p.add_argument("fastq", help="Input FASTQ")
    p.add_argument("ref", help="Reference FASTA")
    p.add_argument("-b", "--band", type=int, default=50, help="Band width")
    p.add_argument("-t", "--threads", type=int, default=1, help="CPU threads (for I/O only)")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"HybAligner2 v2.0.0")
        print(f"  FASTQ: {args.fastq}")
        print(f"  Ref:   {args.ref}")
        print(f"  Band:  ±{args.band}")

    aln = HybAligner2()
    t0 = time.perf_counter()
    aln.load_reference(args.ref)
    result = aln.align(args.fastq, band_width=args.band)
    elapsed = (time.perf_counter() - t0) * 1000

    n_aligned = sum(1 for s in result["scores"] if s > 0)
    throughput = result["n_reads"] / (elapsed / 1000)
    
    print(f"  Reads: {result['n_reads']}, Aligned: {n_aligned}")
    print(f"  Time: {elapsed:.0f}ms, {throughput:.0f} reads/s")
    print(f"  Score: mean={result['score_mean']:.1f}, max={result['score_max']:.0f}")

if __name__ == "__main__":
    main()
