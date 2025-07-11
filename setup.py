from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="amgpcg_pybind",
    version="0.0.0",
    ext_modules=[
        CUDAExtension(
            name="amgpcg_pybind",
            sources=[
                "amgpcg_torch.cpp",
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
