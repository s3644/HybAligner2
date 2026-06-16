"""HybAligner2 — Minimal CUDA + Cython aligner. Zero Python in hot path.

Architecture:
  GPU kernels (kernels.cu):  seed → chain → SW — all on GPU
  Cython bridge (_core.pyx): direct CUDA Runtime API, no ctypes
  Python API (__init__.py):  parse FASTQ, dispatch, collect results

Build: python setup.py build_ext --inplace
Run:   python run.py reads.fastq ref.fa
"""

from ._core import HybAligner2

__version__ = "2.0.0"
__all__ = ["HybAligner2"]
