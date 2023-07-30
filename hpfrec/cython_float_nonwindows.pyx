#cython: language_level=3, legacy_implicit_noexcept=True
import ctypes
from libc.math cimport HUGE_VAL, HUGE_VALL, HUGE_VALL as LD_HUGE_VAL

## Note: As of the end of 2018, MSVC is still stuck with OpenMP 2.0 (released 2002), which does not support
## parallel for loops with unsigend iterators. If you are using a different compiler, this part can be safely removed
## See also: https://github.com/cython/cython/issues/3136
ctypedef size_t ind_type
ctypedef long double long_double_type
obj_ind_type = ctypes.c_size_t
obj_long_double_type = ctypes.c_longdouble

include "cython_float.pxi"
