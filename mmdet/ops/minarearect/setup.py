from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='minarearect',
    ext_modules=[
        CUDAExtension('minarearect', [
            'src/minarearect_cuda.cpp',
            'src/minarearect_kernel.cu',
        ]),
        CUDAExtension(
            'minarearect',
            ['src/minarearect_cuda.cpp', 'src/minarearect_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension})

