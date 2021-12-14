try:
	import setuptools
	from setuptools import setup
	from setuptools import Extension
except:
	from distutils.core import setup
	from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys, os, subprocess, warnings, re

found_omp = True
def set_omp_false():
	global found_omp
	found_omp = False

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
	def build_extensions(self):
		if self.compiler.compiler_type == 'msvc':
			for e in self.extensions:
				e.extra_compile_args += ['/openmp', '/O2', '/fp:fast']
		else:
			self.add_march_native()
			self.add_openmp_linkage()
			self.add_no_math_errno()

			for e in self.extensions:
				# e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c99']
				# e.extra_link_args = ['-fopenmp']
				### Comment: -Ofast gives worse speed than -O2 or -O3
				e.extra_compile_args += ['-O2', '-std=c99']

		build_ext.build_extensions(self)

	def add_march_native(self):
		arg_march_native = "-march=native"
		arg_mcpu_native = "-mcpu=native"
		if self.test_supports_compile_arg(arg_march_native):
			for e in self.extensions:
				e.extra_compile_args.append(arg_march_native)
		elif self.test_supports_compile_arg(arg_mcpu_native):
			for e in self.extensions:
				e.extra_compile_args.append(arg_mcpu_native)

	def add_no_math_errno(self):
		arg_fnme = "-fno-math-errno"
		if self.test_supports_compile_arg(arg_fnme):
			for e in self.extensions:
				e.extra_compile_args.append(arg_fnme)
				e.extra_link_args.append(arg_fnme)

	def add_openmp_linkage(self):
		arg_omp1 = "-fopenmp"
		arg_omp2 = "-qopenmp"
		arg_omp3 = "-xopenmp"
		args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
		if self.test_supports_compile_arg(arg_omp1):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp1)
				e.extra_link_args.append(arg_omp1)
		elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp):
			for e in self.extensions:
				e.extra_compile_args += ["-Xclang", "-fopenmp"]
				e.extra_link_args += ["-lomp"]
		elif self.test_supports_compile_arg(arg_omp2):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp2)
				e.extra_link_args.append(arg_omp2)
		elif self.test_supports_compile_arg(arg_omp3):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp3)
				e.extra_link_args.append(arg_omp3)
		else:
			set_omp_false()
			for e in self.extensions:
				e.sources = [re.sub(r"^(.*)return1\.pyx$", r"\1return0.pyx", s) for s in e.sources]

	def test_supports_compile_arg(self, comm):
		is_supported = False
		try:
			if not hasattr(self.compiler, "compiler"):
				return False
			if not isinstance(comm, list):
				comm = [comm]
			print("--- Checking compiler support for option '%s'" % " ".join(comm))
			fname = "hpfrec_compiler_testing.c"
			with open(fname, "w") as ftest:
				ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
			try:
				cmd = [self.compiler.compiler[0]]
			except:
				cmd = list(self.compiler.compiler)
			val_good = subprocess.call(cmd + [fname])
			try:
				val = subprocess.call(cmd + comm + [fname])
				is_supported = (val == val_good)
			except:
				is_supported = False
		except:
			pass
		try:
			os.remove(fname)
		except:
			pass
		return is_supported


setup(
	name = 'hpfrec',
	packages = ['hpfrec'],
	install_requires=[
	 'pandas>=0.24',
	 'numpy>=1.18',
	 'scipy',
	 'cython'
],
	version = '0.2.5-2',
	description = 'Hierarchical Poisson matrix factorization for recommender systems',
	author = 'David Cortes',
	author_email = 'david.cortes.rivera@gmail.com',
	url = 'https://github.com/david-cortes/hpfrec',
	keywords = ['poisson', 'probabilistic', 'non-negative', 'factorization', 'variational inference', 'collaborative filtering'],
	classifiers = [],

	cmdclass = {'build_ext': build_ext_subclass},
	ext_modules = [ Extension(
						"hpfrec.cython_loops_float",
						sources=["hpfrec/cython_float.pyx"],
						include_dirs=[numpy.get_include()]
					),
					Extension(
						"hpfrec.cython_loops_double",
						sources=["hpfrec/cython_double.pyx"],
						include_dirs=[numpy.get_include()]
					),
					Extension(
						"hpfrec._check_openmp",
						sources=["hpfrec/return1.pyx"],
						include_dirs=[numpy.get_include()]
					) ]
)

if not found_omp:
	omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
	omp_msg += " To enable multi-threading, first install OpenMP"
	if (sys.platform[:3] == "dar"):
		omp_msg += " - for macOS: 'brew install libomp'\n"
	else:
		omp_msg += " modules for your compiler. "
	
	omp_msg += "Then reinstall this package from scratch: 'pip install --force-reinstall hpfrec'.\n"
	warnings.warn(omp_msg)
