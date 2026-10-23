"""Isolated native-CUDA routing/FP8 packing extension for the Sage3 fork.

The original Sage3 FP4 attention and quantization extensions remain untouched.
This first-stage extension prepares the per-V-tile branch and FP8 operand that
will be consumed by the mixed PV path in the copied Sage3 CUTLASS mainloop.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="sage3-native-mixed-router",
    ext_modules=[
        CUDAExtension(
            name="sage3_mixed_router_cuda",
            sources=["v_tile_router.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-gencode=arch=compute_120a,code=sm_120a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
