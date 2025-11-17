from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# предположительно должно работать
setup(
    name="xnor_matmult",
    ext_modules=[
        CppExtension(
            name="xnor_matmult",
            sources=["xnor_matmult.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-march=native"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
