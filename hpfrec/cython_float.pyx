#cython: language_level=3
import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport sdot
from scipy.special.cython_special cimport psi, gamma
import ctypes

from libc.math cimport log, exp, logf, expf, HUGE_VALF, HUGE_VAL, HUGE_VALL

c_real_t = ctypes.c_float
ctypedef float real_t

ctypedef real_t (*blas_dot)(int*, real_t*, int*, real_t*, int*) nogil
cdef blas_dot tdot = sdot

ctypedef real_t (*real_t_fun)(real_t) nogil
cdef real_t_fun exp_t = expf
cdef real_t_fun log_t = logf

cdef real_t HUGE_VAL_T = HUGE_VALF

include "cython_loops.pxi"
