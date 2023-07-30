from scipy.linalg.cython_blas cimport ddot, ddot as tdot
import ctypes

from libc.math cimport log, exp, HUGE_VAL, HUGE_VALL
from libc.math cimport log as log_t, exp as exp_t, HUGE_VAL as HUGE_VAL_T

c_real_t = ctypes.c_double
ctypedef double real_t

include "cython_loops.pxi"
