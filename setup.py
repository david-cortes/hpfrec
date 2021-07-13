try:
	import setuptools
	from setuptools import setup
	from setuptools import Extension
except:
	from distutils.core import setup
	from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys, os

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
	def build_extensions(self):
		c = self.compiler.compiler_type
		if c == 'msvc': # visual studio
			for e in self.extensions:
				e.extra_compile_args = ['/openmp', '/O2']
		else: # gcc and clang
			for e in self.extensions:
				e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c99']
				e.extra_link_args = ['-fopenmp']
				### Comment: -Ofast gives worse speed than -O2 or -O3

		## Note: apple will by default alias 'gcc' to 'clang', and will ship its own "special"
		## 'clang' which has no OMP support and nowadays will purposefully fail to compile when passed
		## '-fopenmp' flags. If you are using mac, and have an OMP-capable compiler,
		## comment out the code below, or set 'use_omp' to 'True'.
		if not use_omp:
			for e in self.extensions:
				e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
				e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']

		build_ext.build_extensions(self)


use_omp = (("enable-omp" in sys.argv)
		   or ("-enable-omp" in sys.argv)
		   or ("--enable-omp" in sys.argv))
if use_omp:
	sys.argv = [a for a in sys.argv if a not in ("enable-omp", "-enable-omp", "--enable-omp")]
if os.environ.get('ENABLE_OMP') is not None:
	use_omp = True
if sys.platform[:3] != "dar":
	use_omp = True

### Shorthand for apple computer:
### uncomment line below
# use_omp = True

setup(
	name = 'hpfrec',
	packages = ['hpfrec'],
	install_requires=[
	 'pandas>=0.24',
	 'numpy>=1.18',
	 'scipy',
	 'cython'
],
	version = '0.2.4',
	description = 'Hierarchical Poisson matrix factorization for recommender systems',
	author = 'David Cortes',
	author_email = 'david.cortes.rivera@gmail.com',
	url = 'https://github.com/david-cortes/hpfrec',
	keywords = ['poisson', 'probabilistic', 'non-negative', 'factorization', 'variational inference', 'collaborative filtering'],
	classifiers = [],

	cmdclass = {'build_ext': build_ext_subclass},
	ext_modules = [ Extension("hpfrec.cython_loops_float", sources=["hpfrec/cython_float.pyx"], include_dirs=[numpy.get_include()]),
					Extension("hpfrec.cython_loops_double", sources=["hpfrec/cython_double.pyx"], include_dirs=[numpy.get_include()])]
)

if not use_omp:
	import warnings
	apple_msg  = "\n\n\nMacOS detected. Package will be built without multi-threading capabilities, "
	apple_msg += "due to Apple's lack of OpenMP support in default clang installs. In order to enable it, "
	apple_msg += "install the package directly from GitHub: https://www.github.com/david-cortes/hpfrec\n"
	apple_msg += "Using 'python setup.py install enable-omp'. "
	apple_msg += "You'll also need an OpenMP-capable compiler.\n\n\n"
	warnings.warn(apple_msg)
