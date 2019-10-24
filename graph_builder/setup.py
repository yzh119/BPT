import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(name='graph builder',
      ext_modules=[
          Extension('graphbuilder',
                    sources=['builder.pyx'],
                    extra_compile_args=['-O3'],
                    language='c++')
      ],
      cmdclass = {'build_ext': build_ext},
      include_dirs=[np.get_include()],
)
