"""Build E117 sparse V router alongside the unchanged dense baseline."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sage3-native-mixed-router-sparse",
    ext_modules=[CUDAExtension(
        name="sage3_mixed_router_sparse_cuda",
        sources=["v_tile_router_sparse.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo",
                     "-gencode=arch=compute_120a,code=sm_120a"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
