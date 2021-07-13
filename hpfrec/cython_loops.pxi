import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from scipy.special.cython_special cimport psi, gamma, loggamma
import time, os
import ctypes

## Note: As of the end of 2018, MSVC is still stuck with OpenMP 2.0 (released 2002), which does not support
## parallel for loops with unsigend iterators. If you are using a different compiler, this part can be safely removed
## See also: https://github.com/cython/cython/issues/3136
IF UNAME_SYSNAME == "Windows":
	obj_ind_type = ctypes.c_longlong
	ctypedef long long ind_type
	ctypedef double long_double_type
	obj_long_double_type = ctypes.c_double
	LD_HUGE_VAL = HUGE_VAL
ELSE:
	obj_ind_type = ctypes.c_size_t
	ctypedef size_t ind_type
	ctypedef long double long_double_type
	obj_long_double_type = ctypes.c_longdouble
	LD_HUGE_VAL = HUGE_VALL


### Helper functions
####################
def cast_real_t(n):
	return <real_t> n

def cast_int(n):
	return <int> n

def cast_ind_type(n):
	return <ind_type> n

### Procedures reusable by package ctpfrec
##########################################
def get_csc_data(ix_u, ix_i, Y, nU, nI):
	from scipy.sparse import coo_matrix, csc_matrix
	X = coo_matrix((Y, (ix_u, ix_i)), shape=(nU, nI))
	X = csc_matrix(X)
	return X.indptr.astype(obj_ind_type), X.indices.astype(obj_ind_type), X.data.astype(c_real_t)

def get_unique_items_batch(np.ndarray[ind_type, ndim=1] users_this_batch,
						   np.ndarray[ind_type, ndim=1] st_ix_u,
						   np.ndarray[ind_type, ndim=1] ix_i, int nthreads,
						   return_ix=False):
	cdef ind_type nusers_batch = users_this_batch.shape[0]
	cdef np.ndarray[ind_type, ndim=1] st_pos = np.empty(nusers_batch + 1, dtype=users_this_batch.dtype)
	st_pos[0] = 0
	get_i_batch_pass1(&st_ix_u[0], &users_this_batch[0], &st_pos[0], nusers_batch)
	cdef np.ndarray[ind_type, ndim=1] arr_items_batch = np.empty(st_pos[-1], dtype=ix_i.dtype)
	get_i_batch_pass2(&st_ix_u[0], &st_pos[0], &arr_items_batch[0], &ix_i[0], &users_this_batch[0],
					nusers_batch, nthreads)
	items = np.unique(arr_items_batch)
	if return_ix:
		return items, st_pos
	else:
		return items

def save_parameters(verbose, save_folder, file_names, obj_list):
	if verbose:
		print("Saving final parameters to .csv files...")

	for i in range(len(file_names)):
		np.savetxt(os.path.join(save_folder, file_names[i]), obj_list[i], fmt="%.10f", delimiter=',')

def assess_convergence(int i, check_every, stop_crit, last_crit, stop_thr,
					   np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Theta_prev,
					   np.ndarray[real_t, ndim=2] Beta, ind_type nY,
					   np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u, np.ndarray[ind_type, ndim=1] ix_i, ind_type nYv,
					   np.ndarray[real_t, ndim=1] Yval, np.ndarray[ind_type, ndim=1] ix_u_val, np.ndarray[ind_type, ndim=1] ix_i_val,
					   np.ndarray[long_double_type, ndim=1] errs, ind_type k, int nthreads, int verbose, int full_llk, has_valset):

	if stop_crit == 'diff-norm':
		last_crit = np.linalg.norm(Theta - Theta_prev)
		if verbose:
			print_norm_diff(i+1, check_every, <real_t> last_crit)
		if last_crit < stop_thr:
			return True, last_crit
		Theta_prev[:,:] = Theta.copy()

	else:

		if has_valset:
			llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Yval[0],
						  &ix_u_val[0], &ix_i_val[0], nYv, k,
						  &errs[0], nthreads, verbose, full_llk)
			errs[0] -= sum_prediction(&Theta[0,0], &Beta[0,0], &ix_u_val[0], &ix_i_val[0], nYv, <int> k, nthreads)
			errs[1] = np.sqrt(errs[1]/nYv)
		else:
			llk_plus_rmse(&Theta[0,0], &Beta[0,0], &Y[0],
						  &ix_u[0], &ix_i[0], nY, k,
						  &errs[0], nthreads, verbose, full_llk)
			errs[0] -= Theta.sum(axis=0).dot(Beta.sum(axis=0))
			errs[1] = np.sqrt(errs[1]/nY)

		if verbose:
			print_llk_iter(<int> (i+1), <long long> errs[0], <double> errs[1], has_valset)

		if stop_crit != 'maxiter':
			if (i+1) == check_every:
				last_crit = errs[0]
			else:
				if (1. - errs[0]/last_crit) <= stop_thr:
					return True, last_crit
				last_crit = errs[0]
	
	return False, last_crit

def eval_after_term(stop_crit, int verbose, int nthreads, int full_llk, ind_type k, ind_type nY, ind_type nYv, has_valset,
					np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Beta,
					np.ndarray[long_double_type, ndim=1] errs,
					np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u, np.ndarray[ind_type, ndim=1] ix_i,
					np.ndarray[real_t, ndim=1] Yval, np.ndarray[ind_type, ndim=1] ix_u_val, np.ndarray[ind_type, ndim=1] ix_i_val):
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
			return errs[0]

### Random initializer for parameters
#####################################
def initialize_parameters(Theta, Beta, random_seed,
						  a, a_prime, b_prime, c, c_prime, d_prime):
	### Comment: this is the initialization that was used in the original HPF code.
	### It doesn't exactly follow the paper's instructions about 'initializing according to prior',
	### but it gives better results than other initializations.

	nU = Theta.shape[0]
	nI = Beta.shape[0]
	k = Theta.shape[1]
	
	rng = np.random.Generator(np.random.MT19937(seed = random_seed if random_seed > 0 else None))
	
	k_rte = np.empty((nU,1), dtype=c_real_t)
	t_rte = np.empty((nI,1), dtype=c_real_t)
	k_rte[:,:] = b_prime
	t_rte[:,:] = d_prime

	Gamma_rte = a_prime + 0.01 * rng.random(size=(nU, k), dtype=c_real_t)
	Lambda_rte = c_prime + 0.01 * rng.random(size=(nI, k), dtype=c_real_t)

	Gamma_shp = a_prime + 0.01 * rng.random(size=(nU, k), dtype=c_real_t)
	Lambda_shp = c_prime + 0.01 * rng.random(size=(nI, k), dtype=c_real_t)

	Theta[:,:] = Gamma_shp / Gamma_rte
	Beta[:,:] = Lambda_shp / Lambda_rte

	return Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte

### Main function
#################
def fit_hpf(real_t a, real_t a_prime, real_t b_prime,
			real_t c, real_t c_prime, real_t d_prime,
			np.ndarray[real_t, ndim=1] Y,
			np.ndarray[ind_type, ndim=1] ix_u,
			np.ndarray[ind_type, ndim=1] ix_i,
			np.ndarray[real_t, ndim=2] Theta,
			np.ndarray[real_t, ndim=2] Beta,
			int maxiter, str stop_crit, int check_every, real_t stop_thr,
			users_per_batch, items_per_batch, step_size, int sum_exp_trick,
			np.ndarray[ind_type, ndim=1] st_ix_u,
			str save_folder, int random_seed, int verbose,
			int nthreads, int par_sh, int has_valset,
			np.ndarray[real_t, ndim=1] Yval,
			np.ndarray[ind_type, ndim=1] ix_u_val,
			np.ndarray[ind_type, ndim=1] ix_i_val,
			int full_llk, int keep_all_objs, int alloc_full_phi):
	## useful information
	cdef ind_type nU = Theta.shape[0]
	cdef ind_type nI = Beta.shape[0]
	cdef ind_type nY = Y.shape[0]
	cdef ind_type k = Theta.shape[1]

	cdef ind_type nYv
	if has_valset>0:
		nYv = Yval.shape[0]

	cdef real_t k_shp = a_prime + k*a
	cdef real_t t_shp = c_prime + k*c

	if verbose>0:
		print("Initializing parameters...")

	cdef np.ndarray[real_t, ndim=2] Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte
	Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte = initialize_parameters(
		Theta, Beta, random_seed, a, a_prime, b_prime, c, c_prime, d_prime)

	cdef np.ndarray[real_t, ndim=2] phi
	if ((users_per_batch == 0) and (items_per_batch == 0)) or alloc_full_phi:
		if verbose>0:
			print("Allocating Phi matrix...")
		phi = np.empty((nY, k), dtype = c_real_t)

	cdef np.ndarray[ind_type, ndim=1] users_numeration, items_numeration, users_this_batch, items_this_batch, st_ix_id_batch
	cdef ind_type nUbatch, nIbatch
	cdef real_t multiplier_batch, step_prev
	cdef np.ndarray[ind_type, ndim=1] st_ix_i_copy, ix_u_copy
	cdef np.ndarray[real_t, ndim=1] Ycopy
	full_updates = True
	if items_per_batch>0:
		if verbose:
			print("Creating item indices for stochastic optimization...")
		items_numeration = np.arange(nI, dtype=obj_ind_type)
		nbatches_i = int(np.ceil(float(nI) / float(items_per_batch)))
		st_ix_i_copy, ix_u_copy, Ycopy = get_csc_data(ix_u, ix_i, Y, nU, nI)
		full_updates = False
	if users_per_batch != 0:
		users_numeration = np.arange(nU, dtype=obj_ind_type)
		nbatches_u = int(np.ceil(float(nU) / float(users_per_batch)))
		full_updates = False

	rng = np.random.default_rng(seed = random_seed if random_seed > 0 else None)

	cdef real_t add_k_rte = a_prime/b_prime
	cdef real_t add_t_rte = c_prime/d_prime
	cdef np.ndarray[long_double_type, ndim=1] errs = np.zeros(2, dtype=obj_long_double_type)

	cdef long_double_type last_crit = - LD_HUGE_VAL
	cdef np.ndarray[real_t, ndim=2] Theta_prev
	if stop_crit == 'diff-norm':
		Theta_prev = Theta.copy()
	else:
		Theta_prev = np.empty((0,0), dtype = c_real_t)

	cdef int one = 1
	if verbose>0:
		print("Initializing optimization procedure...")
	cdef double st_time = time.time()

	### Main loop
	cdef int i
	for i in range(maxiter):

		## Full-batch updates
		if full_updates:

			update_phi(&Gamma_shp[0,0], &Gamma_rte[0,0], &Lambda_shp[0,0], &Lambda_rte[0,0],
							  &phi[0,0], &Y[0], k, sum_exp_trick,
							  &ix_u[0], &ix_i[0], nY, nthreads)

			Gamma_rte = k_shp/k_rte + Beta.sum(axis=0, keepdims=True)

			### Comment: don't put this part before the update for Gamma rate
			Gamma_shp[:,:] = a
			Lambda_shp[:,:] = c
			if par_sh:
				## this produces inconsistent results across runs, so there's a non-parallel version too
				update_G_n_L_sh_par(&Gamma_shp[0,0], &Lambda_shp[0,0],
								  &phi[0,0], k,
								  &ix_u[0], &ix_i[0], nY, nthreads)
			else:
				update_G_n_L_sh(&Gamma_shp[0,0], &Lambda_shp[0,0],
								  &phi[0,0], k,
								  &ix_u[0], &ix_i[0], nY)

			Theta[:,:] = Gamma_shp/Gamma_rte

			### Comment: these operations are pretty fast in numpy, so I preferred not to parallelize them.
			### Moreover, compiler optimizations from .pyx files do a very poor job at parallelizing sums by columns.
			Lambda_rte = t_shp/t_rte + Theta.sum(axis=0, keepdims=True)
			Beta[:,:] = Lambda_shp/Lambda_rte

			k_rte = add_k_rte + Theta.sum(axis=1, keepdims=True)
			t_rte = add_t_rte + Beta.sum(axis=1, keepdims=True)

		## Mini-batch epoch (stochastic variational inference)
		else:
			step_size_batch = <real_t> step_size(i)
			step_prev = 1 - step_size_batch
			if (users_per_batch>0) and (items_per_batch>0):
				if ((i+1) % 2) == 0:
					user_epoch = True
				else:
					user_epoch = False
			elif (users_per_batch>0) and (items_per_batch==0):
				user_epoch = True
			else:
				user_epoch = False

			if user_epoch:
				## users epoch
				rng.shuffle(users_numeration)
				for bt in range(nbatches_u):
					st_batch = bt * users_per_batch
					end_batch = min(nU, (bt + 1) * users_per_batch)
					users_this_batch = users_numeration[st_batch : end_batch]
					multiplier_batch = float(nU) / float(end_batch - st_batch)
					nUbatch = users_this_batch.shape[0]

					if alloc_full_phi:
						items_this_batch = get_unique_items_batch(users_this_batch, st_ix_u, ix_i, nthreads, False)
					else:
						items_this_batch, st_ix_id_batch = get_unique_items_batch(users_this_batch, st_ix_u, ix_i, nthreads, True)
						phi = np.empty((st_ix_id_batch[-1], k), dtype = c_real_t)

					if alloc_full_phi:
						update_phi_csr(&Gamma_shp[0,0], &Gamma_rte[0,0], &Lambda_shp[0,0], &Lambda_rte[0,0],
								   &phi[0,0], &Y[0], &ix_i[0], &st_ix_u[0], &users_this_batch[0],
								   k, nUbatch, nthreads)
					else:
						update_phi_csr_small(&Gamma_shp[0,0], &Gamma_rte[0,0], &Lambda_shp[0,0], &Lambda_rte[0,0],
								   &phi[0,0], &Y[0], &ix_i[0], &st_ix_u[0], &users_this_batch[0], &st_ix_id_batch[0],
								   k, nUbatch, nthreads)
					
					Gamma_rte = k_shp/k_rte + Beta.sum(axis=0, keepdims=True)
					
					Lambda_shp_prev = Lambda_shp[items_this_batch,:].copy()

					Gamma_shp[users_this_batch,:] = a
					Lambda_shp[items_this_batch,:] = c

					if alloc_full_phi:
						update_G_n_L_sh_csr(&Gamma_shp[0,0], &Lambda_shp[0,0], &phi[0,0],
										k, nUbatch, nthreads,
										&ix_i[0], &st_ix_u[0], &users_this_batch[0])
					else:
						update_G_n_L_sh_csr_small(&Gamma_shp[0,0], &Lambda_shp[0,0], &st_ix_id_batch[0], &phi[0,0],
										k, nUbatch, nthreads,
										&ix_i[0], &st_ix_u[0], &users_this_batch[0])

					Lambda_shp[items_this_batch,:] = step_size_batch * multiplier_batch * Lambda_shp[items_this_batch,:] + step_prev * Lambda_shp_prev

					Theta[:,:] = Gamma_shp / Gamma_rte

					Lambda_rte[items_this_batch,:] = step_size_batch * (t_shp/t_rte[items_this_batch] + Theta.sum(axis=0, keepdims=False)) + step_prev * Lambda_rte[items_this_batch,:]

					Beta[:,:] = Lambda_shp / Lambda_rte

					k_rte[users_this_batch] = step_size_batch * (add_k_rte + Theta[users_this_batch].sum(axis=1, keepdims=True)) + step_prev * k_rte[users_this_batch]
					t_rte[items_this_batch] = step_size_batch * (add_t_rte + Beta[items_this_batch].sum(axis=1, keepdims=True)) + step_prev * t_rte[items_this_batch]

			else:
				## items epoch
				rng.shuffle(items_numeration)
				for bt in range(nbatches_i):
					st_batch = bt * items_per_batch
					end_batch = min(nI, (bt + 1) * items_per_batch)
					items_this_batch = items_numeration[st_batch : end_batch]
					multiplier_batch = float(nI) / float(end_batch - st_batch)
					nIbatch = items_this_batch.shape[0]

					if alloc_full_phi:
						users_this_batch = get_unique_items_batch(items_this_batch, st_ix_i_copy, ix_u_copy, nthreads, False)
					else:
						users_this_batch, st_ix_id_batch = get_unique_items_batch(items_this_batch, st_ix_i_copy, ix_u_copy, nthreads, True)
						phi = np.empty((st_ix_id_batch[-1], k), dtype = c_real_t)

					if alloc_full_phi:
						update_phi_csr(&Lambda_shp[0,0], &Lambda_rte[0,0], &Gamma_shp[0,0], &Gamma_rte[0,0],
								   &phi[0,0], &Ycopy[0], &ix_u_copy[0], &st_ix_i_copy[0], &items_this_batch[0],
								   k, nIbatch, nthreads)
					else:
						update_phi_csr_small(&Lambda_shp[0,0], &Lambda_rte[0,0], &Gamma_shp[0,0], &Gamma_rte[0,0],
								   &phi[0,0], &Ycopy[0], &ix_u_copy[0], &st_ix_i_copy[0], &items_this_batch[0], &st_ix_id_batch[0],
								   k, nIbatch, nthreads)
					
					Lambda_rte = t_shp/t_rte + Theta.sum(axis=0, keepdims=True)
					
					Gamma_shp_prev = Gamma_shp[users_this_batch,:].copy()

					Gamma_shp[users_this_batch,:] = a
					Lambda_shp[items_this_batch,:] = c

					if alloc_full_phi:
						update_G_n_L_sh_csr(&Lambda_shp[0,0], &Gamma_shp[0,0], &phi[0,0],
										k, nIbatch, nthreads,
										&ix_u_copy[0], &st_ix_i_copy[0], &items_this_batch[0])
					else:
						update_G_n_L_sh_csr_small(&Lambda_shp[0,0], &Gamma_shp[0,0], &st_ix_id_batch[0], &phi[0,0],
										k, nIbatch, nthreads,
										&ix_u_copy[0], &st_ix_i_copy[0], &items_this_batch[0])

					Gamma_shp[users_this_batch,:] = step_size_batch * multiplier_batch * Gamma_shp[users_this_batch,:] + step_prev * Gamma_shp_prev

					Beta[:,:] = Lambda_shp / Lambda_rte

					Gamma_rte[users_this_batch,:] = step_size_batch * (k_shp/k_rte[users_this_batch] + Beta.sum(axis=0, keepdims=False)) + step_prev * Gamma_rte[users_this_batch,:]

					Theta[:,:] = Gamma_shp / Gamma_rte

					k_rte[users_this_batch] = step_size_batch * (add_k_rte + Theta[users_this_batch].sum(axis=1, keepdims=True)) + step_prev * k_rte[users_this_batch]
					t_rte[items_this_batch] = step_size_batch * (add_t_rte + Beta[items_this_batch].sum(axis=1, keepdims=True)) + step_prev * t_rte[items_this_batch]


		## assessing convergence
		if check_every>0:
			if ((i+1) % check_every) == 0:

				has_converged, last_crit = assess_convergence(
					i, check_every, stop_crit, last_crit, stop_thr,
					Theta, Theta_prev,
					Beta, nY,
					Y, ix_u, ix_i, nYv,
					Yval, ix_u_val, ix_i_val,
					errs, k, nthreads, verbose, full_llk, has_valset
					)

				if has_converged:
					break

	## last metrics once it finishes optimizing
	last_llk = eval_after_term(
		stop_crit, verbose, nthreads, full_llk, k, nY, nYv, has_valset,
		Theta, Beta, errs,
		Y, ix_u, ix_i,
		Yval, ix_u_val, ix_i_val
		)

	cdef double end_tm = (time.time()-st_time)/60
	if verbose:
		print_final_msg(i+1, <long long> errs[0], <double> errs[1], end_tm)

	if save_folder != "":
		save_parameters(verbose, save_folder,
						["Theta", "Beta", "Gamma_shp", "Gamma_rte", "Lambda_shp", "Lambda_rte", "kappa_rte", "tau_rte"],
						[Theta, Beta, Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte])

	## returning objects as needed
	if keep_all_objs:
		temp = (Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte)
	else:
		temp = None
	return i, temp, last_llk


### Functions for updates without a complete refit
##################################################
def partial_fit(np.ndarray[real_t, ndim=1] Y_batch,
				np.ndarray[ind_type, ndim=1] ix_u_batch, np.ndarray[ind_type, ndim=1] ix_i_batch,
				np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Beta,
				np.ndarray[real_t, ndim=2] Gamma_shp, np.ndarray[real_t, ndim=2] Gamma_rte,
				np.ndarray[real_t, ndim=2] Lambda_shp, np.ndarray[real_t, ndim=2] Lambda_rte,
				np.ndarray[real_t, ndim=2] k_rte, np.ndarray[real_t, ndim=2] t_rte,
				real_t add_k_rte, real_t add_t_rte, real_t a, real_t c,
				real_t k_shp, real_t t_shp, ind_type k,
				users_this_batch, items_this_batch, par_sh,
				real_t step_size_batch, real_t multiplier_batch,
				int nthreads, user_batch
				):
	cdef ind_type nYbatch = Y_batch.shape[0]
	cdef np.ndarray[real_t, ndim=2] phi = np.empty((nYbatch, k), dtype = c_real_t)
	cdef real_t step_prev = 1 - step_size_batch
	update_phi(&Gamma_shp[0,0], &Gamma_rte[0,0], &Lambda_shp[0,0], &Lambda_rte[0,0],
				  &phi[0,0], &Y_batch[0], k, 1,
				  &ix_u_batch[0], &ix_i_batch[0], nYbatch, nthreads)
	
	if user_batch:
		Gamma_rte[:,:] = k_shp/k_rte + Beta.sum(axis=0, keepdims=True)
		Lambda_shp_prev = Lambda_shp[items_this_batch,:].copy()
	else:
		Lambda_rte[:,:] = t_shp/t_rte + Theta.sum(axis=0, keepdims=True)
		Gamma_shp_prev = Gamma_shp[users_this_batch,:].copy()

	Gamma_shp[users_this_batch,:] = a
	Lambda_shp[items_this_batch,:] = c

	if par_sh>0:
		update_G_n_L_sh_par(&Gamma_shp[0,0], &Lambda_shp[0,0],
						  &phi[0,0], k,
						  &ix_u_batch[0], &ix_i_batch[0],  nYbatch, nthreads)
	else:
		update_G_n_L_sh(&Gamma_shp[0,0], &Lambda_shp[0,0],
						  &phi[0,0], k,
						  &ix_u_batch[0], &ix_i_batch[0],  nYbatch)

	if user_batch:
		Lambda_shp[items_this_batch,:] = step_size_batch * multiplier_batch * Lambda_shp[items_this_batch,:] + step_prev * Lambda_shp_prev
		Theta[:,:] = Gamma_shp / Gamma_rte
		Lambda_rte[items_this_batch,:] = step_size_batch * (t_shp/t_rte[items_this_batch] + Theta.sum(axis=0, keepdims=False)) + step_prev * Lambda_rte[items_this_batch,:]
		Beta[:,:] = Lambda_shp / Lambda_rte
	else:
		Gamma_shp[users_this_batch,:] = step_size_batch * multiplier_batch * Gamma_shp[users_this_batch,:] + step_prev * Gamma_shp_prev
		Beta[:,:] = Lambda_shp / Lambda_rte
		Gamma_rte[users_this_batch,:] = step_size_batch * (k_shp/k_rte[users_this_batch] + Beta.sum(axis=0, keepdims=False)) + step_prev * Gamma_rte[users_this_batch,:]
		Theta[:,:] = Gamma_shp / Gamma_rte

	k_rte[:,:] = step_size_batch * (add_k_rte + Theta.sum(axis=1, keepdims=True)) + step_prev * k_rte
	t_rte[:,:] = step_size_batch * (add_t_rte + Beta.sum(axis=1, keepdims=True)) + step_prev * t_rte


def calc_user_factors(real_t a, real_t a_prime, real_t b_prime,
					  real_t c, real_t c_prime, real_t d_prime,
					  np.ndarray[real_t, ndim=1] Y,
					  np.ndarray[ind_type, ndim=1] ix_i,
					  np.ndarray[real_t, ndim=1] Theta,
					  np.ndarray[real_t, ndim=2] Beta,
					  np.ndarray[real_t, ndim=2] Lambda_shp,
					  np.ndarray[real_t, ndim=2] Lambda_rte,
					  ind_type nY, ind_type k, int maxiter, int nthreads, int random_seed,
					  real_t stop_thr, int return_all):
	
	## initializing parameters
	cdef real_t k_shp = a_prime + k*a
	cdef real_t t_shp = c_prime + k*c
	rng = np.random.default_rng(seed = random_seed if random_seed > 0 else None)
	Theta[:] = rng.gamma(a, 1/b_prime, size = k).astype(c_real_t)
	cdef real_t k_rte = b_prime + Theta.sum()
	cdef np.ndarray[real_t, ndim=1] Gamma_rte = rng.gamma(a_prime, b_prime/a_prime, size=1).astype(c_real_t) + \
											Beta.sum(axis=0)
	cdef np.ndarray[real_t, ndim=1] Gamma_shp = Gamma_rte * Theta * rng.uniform(low=.85, high=1.15, size=k).astype(c_real_t)
	np.nan_to_num(Gamma_shp, copy=False)
	np.nan_to_num(Gamma_rte, copy=False)
	cdef np.ndarray[real_t, ndim=2] phi = np.empty((nY, k), dtype = c_real_t)
	cdef real_t add_k_rte = a_prime/b_prime
	cdef np.ndarray[real_t, ndim=1] Theta_prev = Theta.copy()
	cdef np.ndarray[ind_type, ndim=1] u_repeated = np.zeros(nY, dtype=obj_ind_type)

	## running the loop
	for i in range(maxiter):
		update_phi(&Gamma_shp[0], &Gamma_rte[0], &Lambda_shp[0,0], &Lambda_rte[0,0],
				   &phi[0,0], &Y[0], k, 1, &u_repeated[0], &ix_i[0], nY, nthreads)
		Gamma_rte = (k_shp/k_rte + Beta.sum(axis=0, keepdims=True)).reshape(-1)
		Gamma_shp = a + phi.sum(axis=0)
		Theta[:] = Gamma_shp / Gamma_rte
		k_rte = add_k_rte + Theta.sum()

		## checking for early stop
		if np.linalg.norm(Theta - Theta_prev) < stop_thr:
			break
		Theta_prev = Theta.copy()

	if return_all:
		return Gamma_shp, Gamma_rte, phi/Y.reshape((-1, 1))
	else:
		return None


### External llk function
#########################
def calc_llk(np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u, np.ndarray[ind_type, ndim=1] ix_i,
			 np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Beta, ind_type k, int nthreads, int full_llk):
	cdef np.ndarray[long_double_type, ndim=1] o = np.zeros(1, dtype=obj_long_double_type)
	llk_plus_rmse(&Theta[0,0], &Beta[0,0],
			 &Y[0], &ix_u[0], &ix_i[0],
			 <ind_type> Y.shape[0], k,
			 &o[0], nthreads, 0, full_llk)
	cdef int kint = k
	o[0] -= sum_prediction(&Theta[0,0], &Beta[0,0], &ix_u[0], &ix_i[0], <ind_type> Y.shape[0], kint, nthreads)
	return o[0]

### External prediction function
################################
def predict_arr(np.ndarray[real_t, ndim=2] M1, np.ndarray[real_t, ndim=2] M2, np.ndarray[ind_type, ndim=1] ix_u, np.ndarray[ind_type, ndim=1] ix_i, int nthreads):
	cdef ind_type n = ix_u.shape[0]
	cdef int k = M1.shape[1]
	cdef np.ndarray[real_t, ndim=1] out = np.zeros(n, dtype = c_real_t)
	predict_multiple(&out[0], &M1[0,0], &M2[0,0], &ix_u[0], &ix_i[0], n, k, nthreads)
	return out

### Internal C functions
########################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_phi(real_t* G_sh, real_t* G_rt, real_t* L_sh, real_t* L_rt,
					 real_t* phi, real_t* Y, ind_type k, int sum_exp_trick,
					 ind_type* ix_u, ind_type* ix_i, ind_type nY, int nthreads) nogil:
	cdef ind_type uid, iid
	cdef ind_type uid_st, iid_st, phi_st
	cdef real_t sumphi, maxval
	cdef ind_type i, j

	if sum_exp_trick:
		for i in prange(nY, schedule='static', num_threads=nthreads):
			uid = ix_u[i]
			iid = ix_i[i]
			sumphi = 0
			maxval = - HUGE_VAL_T
			uid_st = k*uid
			iid_st = k*iid
			phi_st = i*k
			for j in range(k):
				phi[phi_st + j] = psi(G_sh[uid_st + j]) - log(G_rt[uid_st + j]) + psi(L_sh[iid_st + j]) - log(L_rt[iid_st + j])
				if phi[phi_st + j] > maxval:
					maxval = phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] = exp_t(phi[phi_st + j] - maxval)
				sumphi += phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] *= Y[i]/sumphi

	else:
		for i in prange(nY, schedule='static', num_threads=nthreads):
			uid = ix_u[i]
			iid = ix_i[i]
			sumphi = 0
			uid_st = k*uid
			iid_st = k*iid
			phi_st = i*k
			for j in range(k):
				phi[phi_st + j] = exp(  psi(G_sh[uid_st + j]) - log(G_rt[uid_st + j]) + psi(L_sh[iid_st + j]) - log(L_rt[iid_st + j])  )
				sumphi += phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] *= Y[i]/sumphi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh_par(real_t* G_sh, real_t* L_sh,
						  real_t* phi, ind_type k,
						  ind_type* ix_u, ind_type* ix_i, ind_type nY, int nthreads) nogil:
	cdef ind_type i, j
	for i in prange(nY, schedule='static', num_threads=nthreads):
		for j in range(k):
			G_sh[ix_u[i]*k + j] += phi[i*k + j]
			L_sh[ix_i[i]*k + j] += phi[i*k + j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh(real_t* G_sh, real_t* L_sh,
						  real_t* phi, ind_type k,
						  ind_type* ix_u, ind_type* ix_i, ind_type nY) nogil:
	cdef ind_type i, j
	for i in range(nY):
		for j in range(k):
			G_sh[ix_u[i]*k + j] += phi[i*k + j]
			L_sh[ix_i[i]*k + j] += phi[i*k + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void llk_plus_rmse(real_t* T, real_t* B, real_t* Y,
						ind_type* ix_u, ind_type* ix_i, ind_type nY, ind_type kszt,
						long_double_type* out, int nthreads, int add_mse, int full_llk) nogil:
	cdef ind_type i
	cdef int one = 1
	cdef real_t yhat
	cdef long_double_type out1 = 0
	cdef long_double_type out2 =  0
	cdef int k = <int> kszt
	if add_mse:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
				out1 += Y[i]*log(yhat) - loggamma(Y[i] + 1.)
				out2 += (Y[i] - yhat)**2
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
				out1 += Y[i]*log_t(yhat)
				out2 += (Y[i] - yhat)**2
		out[0] = out1
		out[1] = out2
	else:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log(tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)) - loggamma(Y[i] + 1.)
			out[0] = out1
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log_t(tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one))
			out[0] = out1
	### Comment: adding += directly to *out triggers compiler optimizations that produce
	### different (and wrong) results across different runs.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_phi_csr(real_t* G_sh, real_t* G_rt, real_t* L_sh, real_t* L_rt,
						 real_t* phi, real_t* Y, ind_type* ix_i, ind_type* st_ix_u, ind_type* u_arr,
						 ind_type k, ind_type nU, int nthreads) nogil:
	cdef ind_type u, i, j
	cdef ind_type uid, n_uid
	cdef ind_type st_G, st_L, phi_st, y_ix
	cdef real_t sumrow, maxval
	for u in prange(nU, schedule='dynamic', num_threads=nthreads):
		uid = u_arr[u]
		n_uid = st_ix_u[uid + 1] - st_ix_u[uid]
		st_G = k * uid
		for i in range(n_uid):
			y_ix = i + st_ix_u[uid]
			st_L = k * ix_i[y_ix]
			phi_st = y_ix * k
			sumrow = 0
			maxval = - HUGE_VAL_T
			for j in range(k):
				phi[phi_st + j] = psi(G_sh[st_G + j]) - log(G_rt[st_G + j]) + psi(L_sh[st_L + j]) - log(L_rt[st_L + j])
				if phi[phi_st + j] > maxval:
					maxval = phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] = exp_t(phi[phi_st + j] - maxval)
				sumrow += phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] *= Y[y_ix] / sumrow

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_phi_csr_small(real_t* G_sh, real_t* G_rt, real_t* L_sh, real_t* L_rt,
						 	   real_t* phi, real_t* Y, ind_type* ix_i, ind_type* st_ix_u, ind_type* u_arr,
						 	   ind_type* st_phi_u, ind_type k, ind_type nU, int nthreads) nogil:
	cdef ind_type u, i, j
	cdef ind_type uid, n_uid
	cdef ind_type st_G, st_L, phi_st, y_ix
	cdef real_t sumrow, maxval
	for u in prange(nU, schedule='dynamic', num_threads=nthreads):
		uid = u_arr[u]
		n_uid = st_ix_u[uid + 1] - st_ix_u[uid]
		st_G = k * uid
		for i in range(n_uid):
			y_ix = i + st_ix_u[uid]
			st_L = k * ix_i[y_ix]
			phi_st = (st_phi_u[u] + i) * k
			sumrow = 0
			maxval = - HUGE_VAL_T
			for j in range(k):
				phi[phi_st + j] = psi(G_sh[st_G + j]) - log(G_rt[st_G + j]) + psi(L_sh[st_L + j]) - log(L_rt[st_L + j])
				if phi[phi_st + j] > maxval:
					maxval = phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] = exp_t(phi[phi_st + j] - maxval)
				sumrow += phi[phi_st + j]
			for j in range(k):
				phi[phi_st + j] *= Y[y_ix] / sumrow

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh_csr(real_t* G_sh, real_t* L_sh,
							  real_t* phi, ind_type k, ind_type nU, int nthreads,
							  ind_type* ix_i, ind_type* st_ix_u, ind_type* u_arr) nogil:
	cdef ind_type u, i, j
	cdef ind_type uid, n_uid
	cdef ind_type st_ix_G, st_ix_L, st_ix_phi
	for u in prange(nU, schedule='dynamic', num_threads=nthreads):
		uid = u_arr[u]
		n_uid = st_ix_u[uid + 1] - st_ix_u[uid]
		st_ix_G = uid * k
		for i in range(n_uid):
			st_ix_L = ix_i[i + st_ix_u[uid]] * k
			st_ix_phi = (i + st_ix_u[uid]) * k
			for j in range(k):
				G_sh[st_ix_G + j] += phi[st_ix_phi + j]
				L_sh[st_ix_L + j] += phi[st_ix_phi + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh_csr_small(real_t* G_sh, real_t* L_sh, ind_type* st_phi_u,
							  		real_t* phi, ind_type k, ind_type nU, int nthreads,
							  		ind_type* ix_i, ind_type* st_ix_u, ind_type* u_arr) nogil:
	cdef ind_type u, i, j
	cdef ind_type uid, n_uid
	cdef ind_type st_ix_G, st_ix_L, st_ix_phi
	for u in prange(nU, schedule='dynamic', num_threads=nthreads):
		uid = u_arr[u]
		n_uid = st_ix_u[uid + 1] - st_ix_u[uid]
		st_ix_G = uid * k
		for i in range(n_uid):
			st_ix_phi = (st_phi_u[u] + i) * k
			st_ix_L = ix_i[i + st_ix_u[uid]] * k
			for j in range(k):
				G_sh[st_ix_G + j] += phi[st_ix_phi + j]
				L_sh[st_ix_L + j] += phi[st_ix_phi + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void get_i_batch_pass1(ind_type* st_ix_u, ind_type* u_arr, ind_type* out, ind_type nU) nogil:
	cdef ind_type st_out = 0
	cdef ind_type u, n_uid, i
	for u in range(nU):
		n_uid = st_ix_u[u_arr[u] + 1] - st_ix_u[u_arr[u]]
		st_out += n_uid
		out[u + 1] = st_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void get_i_batch_pass2(ind_type* st_ix_u, ind_type* st_ix_out, ind_type* out, ind_type* ix_i, ind_type* u_arr,
						  ind_type nU, int nthreads) nogil:
	cdef ind_type i, u
	cdef ind_type uid, n_uid, st_out
	for u in prange(nU, schedule='dynamic', num_threads=nthreads):
		uid = u_arr[u]
		n_uid = st_ix_u[uid + 1] - st_ix_u[uid]
		st_out = st_ix_out[u]
		for i in range(n_uid):
			out[st_out + i] = ix_i[st_ix_u[uid] + i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void predict_multiple(real_t* out, real_t* M1, real_t* M2, ind_type* ix_u, ind_type* ix_i, ind_type n, int k, int nthreads) nogil:
	
	cdef int one = 1
	cdef ind_type kszt = k
	cdef ind_type i
	for i in prange(n, schedule='static', num_threads=nthreads):
		out[i] = tdot(&k, &M1[ix_u[i] * kszt], &one, &M2[ix_i[i] * kszt], &one)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef long_double_type sum_prediction(real_t* M1, real_t* M2, ind_type* ix_u, ind_type* ix_i, ind_type n, int k, int nthreads) nogil:
	
	cdef long_double_type out = 0
	cdef int one = 1
	cdef ind_type kszt = k
	cdef ind_type i
	for i in prange(n, schedule='static', num_threads=nthreads):
		out += tdot(&k, &M1[ix_u[i] * kszt], &one, &M2[ix_i[i] * kszt], &one)
	return out


### Printing output
###################
def print_norm_diff(int it, int check_every, real_t normdiff):
	print("Iteration %d | Norm(Theta_{%d} - Theta_{%d}): %.5f" % (it, it, it-check_every, normdiff))

def print_llk_iter(int it, long long llk, double rmse, int has_valset):
	cdef str dataset_type
	if has_valset:
		dataset_type = "val"
	else:
		dataset_type = "train"
	msg = "Iteration %d | " + dataset_type + " llk: %d | " + dataset_type + " rmse: %.4f"
	print(msg % (it, llk, rmse))

def print_final_msg( int it, long long llk, double rmse, double end_tm):
	print("\n\nOptimization finished")
	print("Final log-likelihood: %d" % llk)
	print("Final RMSE: %.4f" % rmse)
	print("Minutes taken (optimization part): %.1f" % end_tm)
	print("")
