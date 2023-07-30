import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport sdot as tdot
from scipy.special.cython_special cimport psi, gamma
import ctypes

from libc.math cimport log, exp, logf as log_t, expf as exp_t, HUGE_VALF as HUGE_VAL_T, HUGE_VAL, HUGE_VALL

c_real_t = ctypes.c_float
ctypedef float real_t

include "cython_loops.pxi"
