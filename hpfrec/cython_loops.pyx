import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport log, exp
from scipy.special.cython_special cimport psi, gamma
from scipy.linalg.cython_blas cimport sdot
import time, os
import ctypes

### Helper functions
####################
def cast_float(n):
	return <float> n

def cast_int(n):
	return <int> n

def cast_np_int(a):
	return a.astype(int)

## Main function
################
def fit_hpf(float a, float a_prime, float b_prime,
			float c, float c_prime, float d_prime,
			np.ndarray[float, ndim=1] Y,
			np.ndarray[int, ndim=1] ix_u,
			np.ndarray[int, ndim=1] ix_i,
			np.ndarray[float, ndim=2] Theta,
			np.ndarray[float, ndim=2] Beta,
			int maxiter, str stop_crit, int check_every, float stop_thr,
			str save_folder, int random_seed, int verbose,
			int nthreads, int par_sh, int has_valset,
			np.ndarray[float, ndim=1] Yval,
			np.ndarray[int, ndim=1] ix_u_val,
			np.ndarray[int, ndim=1] ix_i_val,
			int full_llk):
	## useful information
	cdef int nU = <int> Theta.shape[0]
	cdef int nI = <int> Beta.shape[0]
	cdef int nY = <int> Y.shape[0]
	cdef int k = <int> Theta.shape[1]

	cdef int nYv
	if has_valset>0:
		nYv = <int> Yval.shape[0]

	if verbose>0:
		print "Initializing parameters..."

	### Comment: I'm not entirely sure how to initialize the variables according to the prior, and the
	### initialization here differs from the implementation of the paper's author.

	## initializing parameters
	if random_seed > 0:
		np.random.seed(random_seed)
	Theta[:,:] = np.random.gamma(a, 1/b_prime, size=(nU, k)).astype('float32')
	Beta[:,:] = np.random.gamma(c, 1/d_prime, size=(nI, k)).astype('float32')

	### Comment: the code above seems to give worse likelihood in the first iterations, but better
	### local optima in the end, compared to initializing them like this:
		# cdef np.ndarray[double, ndim=2] ksi = np.random.gamma(a_prime, b_prime/a_prime, size=(nU,1))
		# Theta[:,:] = np.random.gamma(a, 1/ksi, size=(nU, k)).astype('float32')
		# cdef np.ndarray[double, ndim=2] eta = np.random.gamma(c_prime, d_prime/c_prime, size=(nI,1))
		# Beta[:,:] = np.random.gamma(c, 1/eta, size=(nI, k)).astype('float32')

	cdef float k_shp = a_prime + k*a
	cdef float t_shp = c_prime + k*c
	cdef np.ndarray[float, ndim=2] k_rte = b_prime + Theta.sum(axis=1, keepdims=True)
	cdef np.ndarray[float, ndim=2] t_rte = d_prime + Beta.sum(axis=1, keepdims=True)

	cdef np.ndarray[float, ndim=2] Gamma_rte = np.random.gamma(a_prime, b_prime/a_prime, size=(nU, 1)).astype('float32') + \
												Beta.sum(axis=0, keepdims=True)
	cdef np.ndarray[float, ndim=2] Lambda_rte = np.random.gamma(c_prime, d_prime/c_prime, size=(nI, 1)).astype('float32') + \
												Theta.sum(axis=0, keepdims=True)

	cdef np.ndarray[float, ndim=2] Gamma_shp = Gamma_rte * Theta * np.random.uniform(low=.85, high=1.15, size=(nU, k)).astype('float32')
	cdef np.ndarray[float, ndim=2] Lambda_shp = Lambda_rte * Beta * np.random.uniform(low=.85, high=1.15, size=(nI, k)).astype('float32')
	np.nan_to_num(Gamma_shp, copy=False)
	np.nan_to_num(Lambda_shp, copy=False)
	np.nan_to_num(Gamma_rte, copy=False)
	np.nan_to_num(Lambda_rte, copy=False)

	cdef np.ndarray[float, ndim=2] phi = np.empty((nY, k), dtype='float32')

	cdef float add_k_rte = a_prime/b_prime
	cdef float add_t_rte = c_prime/d_prime
	cdef np.ndarray[long double, ndim=1] errs = np.zeros(2, dtype=ctypes.c_longdouble)

	cdef long double last_crit = - (10**37)
	cdef np.ndarray[float, ndim=2] Theta_prev
	if stop_crit == 'diff-norm':
		Theta_prev = Theta.copy()

	cdef int one = 1
	if verbose>0:
		print "Initializing optimization procedure..."
	cdef double st_time = time.time()

	### Main loop
	cdef int i
	for i in range(maxiter):
		update_phi(&Gamma_shp[0,0], &Gamma_rte[0,0], &Lambda_shp[0,0], &Lambda_rte[0,0],
						  &phi[0,0], &Y[0], k,
						  &ix_u[0], &ix_i[0], nY, nthreads)

		Gamma_rte = k_shp/k_rte + Beta.sum(axis=0, keepdims=True)

		### Comment: don't put this part before the update for Gamma rate
		Gamma_shp = np.zeros((Gamma_shp.shape[0], Gamma_shp.shape[1]), dtype='float32')
		Lambda_shp = np.zeros((Lambda_shp.shape[0], Lambda_shp.shape[1]), dtype='float32')
		if par_sh>0:
			update_G_n_L_sh_par(&Gamma_shp[0,0], &Lambda_shp[0,0],
							  &phi[0,0], k,
							  &ix_u[0], &ix_i[0], nY, nthreads)
		else:
			update_G_n_L_sh(&Gamma_shp[0,0], &Lambda_shp[0,0],
							  &phi[0,0], k,
							  &ix_u[0], &ix_i[0], nY)
		Gamma_shp += a
		Lambda_shp += c
		Theta[:,:] = Gamma_shp/Gamma_rte

		### Comment: these operations are pretty fast in numpy, so I preferred not to parallelize them.
		### Moreover, compiler optimizations do a very poor job at parallelizing sums by columns.
		Lambda_rte = t_shp/t_rte + Theta.sum(axis=0, keepdims=True)
		Beta[:,:] = Lambda_shp/Lambda_rte

		k_rte = add_k_rte + Theta.sum(axis=1, keepdims=True)
		t_rte = add_t_rte + Beta.sum(axis=1, keepdims=True)

		## assessing convergence
		if check_every>0:
			if ((i+1) % check_every) == 0:

				if stop_crit == 'diff-norm':
					last_crit = np.linalg.norm(Theta - Theta_prev)
					if verbose:
						print_norm_diff(i+1, check_every, <float> last_crit)
					if last_crit < stop_thr:
						break
					Theta_prev = Theta.copy()

				else:

					if has_valset:
						llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Yval[0],
									  &ix_u_val[0], &ix_i_val[0], nYv, k,
									  &errs[0], nthreads, verbose, full_llk)
						errs[0] -= Theta[ix_u_val].sum(axis=0).dot(Beta[ix_i_val].sum(axis=0))
						errs[1] = np.sqrt(errs[1]/nYv)
					else:
						llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Y[0],
									  &ix_u[0], &ix_i[0], nY, k,
									  &errs[0], nthreads, verbose, full_llk)
						errs[0] -= Theta.sum(axis=0).dot(Beta.sum(axis=0))
						errs[1] = np.sqrt(errs[1]/nY)

					if verbose>0:
						print_llk_iter(i+1, <long long> errs[0], <double> errs[1], has_valset)

					if stop_crit != 'maxiter':
						if (i+1) == check_every:
							last_crit = errs[0]
						else:
							if (1 - errs[0]/last_crit) <= stop_thr:
								break
							last_crit = errs[0]

	if (stop_crit == 'diff-norm') or (stop_crit == 'maxiter'):
		if verbose>0:
			if has_valset:
				llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Yval[0],
							  &ix_u_val[0], &ix_i_val[0], nYv, k,
							  &errs[0], nthreads, verbose, full_llk)
				errs[0] -= Theta[ix_u_val].sum(axis=0).dot(Beta[ix_i_val].sum(axis=0))
				errs[1] = np.sqrt(errs[1]/nYv)
			else:
				llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Y[0],
							  &ix_u[0], &ix_i[0], nY, k,
							  &errs[0], nthreads, verbose, full_llk)
				errs[0] -= Theta.sum(axis=0).dot(Beta.sum(axis=0))
				errs[1] = np.sqrt(errs[1]/nY)

	cdef double end_tm = (time.time()-st_time)/60
	if verbose:
		print_final_msg(i+1, <long long> errs[0], <double> errs[1], end_tm)

	if save_folder != "":
		if verbose:
			print "Saving final parameters to .csv files..."
		np.savetxt(os.path.join(save_folder, "Theta.csv"), Theta, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "Beta.csv"), Beta, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "Gamma_shp.csv"), Gamma_shp, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "Gamma_rte.csv"), Gamma_rte, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "Lambda_shp.csv"), Lambda_shp, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "Lambda_rte.csv"), Lambda_rte, fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "kappa_rte.csv"), k_rte.reshape(-1,1), fmt="%.10f", delimiter=',')
		np.savetxt(os.path.join(save_folder, "tau_rte.csv"), t_rte.reshape(-1,1), fmt="%.10f", delimiter=',')

	return None

### External llk function
#########################
def calc_llk(np.ndarray[float, ndim=1] Y, np.ndarray[int, ndim=1] ix_u, np.ndarray[int, ndim=1] ix_i,
			 np.ndarray[float, ndim=2] Theta, np.ndarray[float, ndim=2] Beta, int k, int nthreads, int full_llk):
	cdef np.ndarray[long double, ndim=1] o = np.zeros(1, dtype='float128')
	llk_plus_rmse(&Theta[0,0], &Beta[0,0],
			 &Y[0], &ix_u[0], &ix_i[0],
			 <int> Y.shape[0], k,
			 &o[0], nthreads, 0, full_llk)
	o[0] -= Theta[ix_u].sum(axis=0).dot(Beta[ix_i].sum(axis=0))
	return o[0]

### Internal C functions
########################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_phi(float* G_sh, float* G_rt, float* L_sh, float* L_rt,
					 float* phi, float* Y, int k,
					 int* ix_u, int* ix_i, int nY, int nthreads) nogil:
	cdef int uid, iid
	cdef int uid_st, iid_st, phi_st
	cdef float sumphi
	cdef int i, j
	for i in prange(nY, schedule='static', num_threads=nthreads):
		uid = ix_u[i]
		iid = ix_i[i]
		sumphi = 0
		uid_st = k*uid
		iid_st = k*iid
		phi_st = i*k
		for j in range(k):
			phi[phi_st + j] = exp(  psi(G_sh[uid_st + j]) - log(G_rt[uid_st + j]) +psi(L_sh[iid_st + j]) - log(L_rt[iid_st + j])  )
			sumphi += phi[phi_st + j]
		for j in range(k):
			phi[phi_st + j] *= Y[i]/sumphi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh_par(float* G_sh, float* L_sh,
						  float* phi, int k,
						  int* ix_u, int* ix_i, int nY, int nthreads) nogil:
	cdef int i, j
	for i in prange(nY, schedule='static', num_threads=nthreads):
		for j in range(k):
			G_sh[ix_u[i]*k + j] += phi[i*k + j]
			L_sh[ix_i[i]*k + j] += phi[i*k + j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh(float* G_sh, float* L_sh,
						  float* phi, int k,
						  int* ix_u, int* ix_i, int nY) nogil:
	cdef int i, j
	for i in range(nY):
		for j in range(k):
			G_sh[ix_u[i]*k + j] += phi[i*k + j]
			L_sh[ix_i[i]*k + j] += phi[i*k + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void llk_plus_rmse(float* T, float* B, float* Y,
						int* ix_u, int* ix_i, int nY, int k,
						long double* out, int nthreads, int add_mse, int full_llk) nogil:
	cdef int i
	cdef int one = 1
	cdef float yhat
	cdef long double out1 = 0
	cdef long double out2 =  0
	if add_mse:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &T[ix_u[i] * k], &one, &B[ix_i[i] * k], &one)
				out1 += Y[i]*log(yhat) - log(gamma(Y[i] + 1))
				out2 += (Y[i] - yhat)**2
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &T[ix_u[i] * k], &one, &B[ix_i[i] * k], &one)
				out1 += Y[i]*log(yhat)
				out2 += (Y[i] - yhat)**2
		out[0] = out1
		out[1] = out2
	else:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log(sdot(&k, &T[ix_u[i] * k], &one, &B[ix_i[i] * k], &one)) - log(gamma(Y[i] + 1))
			out[0] = out1
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log(sdot(&k, &T[ix_u[i] * k], &one, &B[ix_i[i] * k], &one))
			out[0] = out1
	### Comment: adding += directly to *out triggers compiler optimizations that produce
	### different (and wrong) results across different runs.

### Printing output
###################
def print_norm_diff(int it, int check_every, float normdiff):
	print "Iteration %d | Norm(Theta_{%d} - Theta_{%d}): %.5f" % (it, it, it-check_every, normdiff)

def print_llk_iter(int it, long long llk, double rmse, int has_valset):
	cdef str dataset_type
	if has_valset:
		dataset_type = "val"
	else:
		dataset_type = "train"
	msg = "Iteration %d | " + dataset_type + " llk: %d | " + dataset_type + " rmse: %.4f"
	print msg % (it, llk, rmse)

def print_final_msg( int it, long long llk, double rmse, double end_tm):
	print "\n\nOptimization finished"
	print "Final log-likelihood: %d" % llk
	print "Final RMSE: %.4f" % rmse
	print "Minutes taken (optimization part): %.1f" % end_tm
	print ""
