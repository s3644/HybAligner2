"""Build HybAligner2: Cython + CUDA extension."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os, subprocess

# Compile CUDA kernels → object file
cuda_src = "hyb2/kernels.cu"
cuda_obj = "hyb2/kernels.o"

subprocess.run([
    "nvcc", "-c", cuda_src, "-o", cuda_obj,
    "-O3", "-arch=sm_120",
    "-Xcompiler", "-fPIC",
], check=True)

ext = Extension(
    "hyb2._core",
    sources=["hyb2/_core.pyx"],
    extra_objects=[cuda_obj, "/usr/local/cuda/lib64/libcudart_static.a"],
    include_dirs=[np.get_include(), "/usr/local/cuda/include", "."],
    libraries=["cuda", "dl", "pthread", "rt"],
    extra_compile_args=["-O3", "-Wno-unused-variable"],
    extra_link_args=["-Wl,-rpath,/usr/local/cuda/lib64"],
    language="c++",
)

setup(
    name="hyb2",
    version="2.0.1",
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
    ),
)
