"""Build HybAligner2: Cython + CUDA extension."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

# Compile CUDA kernels to object file
cuda_src = "hyb2/kernels.cu"
cuda_obj = "hyb2/kernels.o"

subprocess.run([
    "nvcc", "-c", cuda_src, "-o", cuda_obj,
    "-O3", "-arch=sm_120",  # GB10 Blackwell
    "-Xcompiler", "-fPIC",
], check=True)

ext = Extension(
    "hyb2._core",
    sources=["hyb2/_core.pyx"],
    extra_objects=[cuda_obj],
    include_dirs=[np.get_include(), "/usr/local/cuda/include"],
    library_dirs=["/usr/local/cuda/lib64"],
    libraries=["cudart"],
    extra_compile_args=["-O3"],
    language="c++",
)

setup(
    name="hyb2",
    version="2.0.0",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
    packages=["hyb2"],
    package_data={"hyb2": ["kernels.cu"]},
)
