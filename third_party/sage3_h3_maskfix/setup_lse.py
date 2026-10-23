"""Build an isolated corrected Sage3 FP4 core that additionally writes LSE."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
cutlass = Path(os.environ["E114_CUTLASS_ROOT"])
if not (cutlass / "include/cute/tensor.hpp").is_file():
    raise RuntimeError(f"Invalid CUTLASS source: {cutlass}")

setup(
    name="sage3-h3-maskfix-lse",
    ext_modules=[CUDAExtension(
        name="fp4attn_lse_cuda",
        sources=[str(root / "sageattn3/blackwell/api.cu")],
        include_dirs=[str(root / "sageattn3"), str(cutlass / "include"),
                      str(cutlass / "tools/util/include")],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                     "--expt-extended-lambda", "--use_fast_math", "-lineinfo",
                     "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG",
                     "-DQBLKSIZE=128", "-DKBLKSIZE=128", "-DCTA256",
                     "-DDQINRMEM", "-DEXECMODE=0", "-DSAGE3_EXPORT_LSE",
                     "-gencode=arch=compute_120a,code=sm_120a"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
