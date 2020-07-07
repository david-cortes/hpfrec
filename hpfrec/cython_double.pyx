from scipy.linalg.cython_blas cimport ddot
import ctypes

## TODO: use libc.math once Cython 0.30 is released
# from libc.math cimport log, exp
cdef extern from "<math.h>":
	double log(double x) nogil
	double exp(double x) nogil
	const double HUGE_VAL
	const long double HUGE_VALL

c_real_t = ctypes.c_double
ctypedef double real_t

ctypedef real_t (*blas_dot)(int*, real_t*, int*, real_t*, int*) nogil
cdef blas_dot tdot = ddot

ctypedef real_t (*real_t_fun)(real_t) nogil
cdef real_t_fun exp_t = exp
cdef real_t_fun log_t = log

cdef real_t HUGE_VAL_T = HUGE_VAL

include "cython_loops.pxi"
