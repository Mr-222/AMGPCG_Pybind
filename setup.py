from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="amgpcg_pybind",
    version="0.0.0",
    packages=["amgpcg_pybind"],
    ext_modules=[
        CUDAExtension(
            name="amgpcg_cuda",
            sources=[
                "amgpcg_torch.cu",
                "solver/amgpcg.cu",
                "solver/trim_poisson.cu",
                "common/data_io.cu",
                "common/mem.cc",
                "common/timer.cu",
                "common/util.cu",
            ],
            include_dirs=["common", "solver"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
