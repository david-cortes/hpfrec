from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

if platform.system() == "Windows":
	ext_mod = Extension("hpfrec.cython_loops",
                             sources=["hpfrec/cython_loops.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['/openmp', '/O2'])
else:
	ext_mod = Extension("hpfrec.cython_loops",
                             sources=["hpfrec/cython_loops.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-fopenmp', '-O3'],
                             extra_link_args=['-fopenmp'])
### Comment: -Ofast gives worse speed than -O2 or -O3

setup(
  name = 'hpfrec',
  packages = ['hpfrec'],
  install_requires=[
   'pandas>=0.21',
   'numpy',
   'scipy',
   'cython'
],
  version = '0.2.2.2',
  description = 'Hierarchical Poisson matrix factorization for recommender systems',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/hpfrec',
  download_url = 'https://github.com/david-cortes/hpfrec/archive/0.2.2.2.tar.gz',
  keywords = ['poisson', 'probabilistic', 'non-negative', 'factorization', 'variational inference', 'collaborative filtering'],
  classifiers = [],

  cmdclass = {'build_ext': build_ext},
  ext_modules = [ext_mod,
    ]
)
