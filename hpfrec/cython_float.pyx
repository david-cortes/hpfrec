import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport sdot
from scipy.special.cython_special cimport psi, gamma
import ctypes

## TODO: use libc.math once Cython 0.30 is released
# from libc.math cimport log, exp
cdef extern from "<math.h>":
	double log(double x) nogil
	float logf(float) nogil
	double exp(double x) nogil
	float expf(float) nogil
	const float HUGE_VALF
	const double HUGE_VAL
	const long double HUGE_VALL

c_real_t = ctypes.c_float
ctypedef float real_t

ctypedef real_t (*blas_dot)(int*, real_t*, int*, real_t*, int*) nogil
cdef blas_dot tdot = sdot

ctypedef real_t (*real_t_fun)(real_t) nogil
cdef real_t_fun exp_t = expf
cdef real_t_fun log_t = logf

cdef real_t HUGE_VAL_T = HUGE_VALF

include "cython_loops.pxi"
