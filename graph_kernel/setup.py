from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graphop',
    ext_modules=[
        CUDAExtension('graphop', [
            'graphop.cpp',
            'graphop_kernel.cu'],
    #         extra_compile_args={'cxx': ['-g'],
    #                             'nvcc': ['-O2', '-arch=compute_60', '-code=sm_60']}),
    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
