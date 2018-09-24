import pandas as pd, numpy as np
import multiprocessing, os, warnings
import hpfrec.cython_loops as cython_loops
import ctypes, types, inspect
from scipy.sparse import coo_matrix, csr_matrix
pd.options.mode.chained_assignment = None

class HPF:
	"""
	Hierarchical Poisson Factorization

	Model for recommending items based on probabilistic Poisson factorization
	on sparse count data (e.g. number of times a user played different songs),
	using mean-field variational inference with coordinate-ascent.
	Can also use stochastic variational inference (using mini batches of data).

	Can use different stopping criteria for the opimization procedure:

	1) Run for a fixed number of iterations (stop_crit='maxiter').
	2) Calculate the Poisson log-likelihood every N iterations (stop_crit='train-llk' and check_every)
	   and stop once {1 - curr/prev} is below a certain threshold (stop_thr)
	3) Calculate the Poisson log-likelihood in a user-provided validation set (stop_crit='val-llk', val_set and check_every)
	   and stop once {1 - curr/prev} is below a certain threshold. For this criterion, you might want to lower the
	   default threshold (see Note).
	4) Check the the difference in the user-factor matrix after every N iterations (stop_crit='diff-norm', check_every)
	   and stop once the *l2-norm* of this difference is below a certain threshold (stop_thr).
	   Note that this is **not a percent** difference as it is for log-likelihood criteria, so you should put a larger
	   value than the default here.
	   This is a much faster criterion to calculate and is recommended for larger datasets.
	
	If passing reindex=True, it will internally reindex all user and item IDs. Your data will not require
	reindexing if the IDs for users and items in counts_df meet the following criteria:

	1) Are all integers.
	2) Start at zero.
	3) Don't have any enumeration gaps, i.e. if there is a user '4', user '3' must also be there.

	If you only want to obtain the fitted parameters and use your own API later for recommendations,
	you can pass produce_dicts=False and pass a folder where to save them in csv format (they are also
	available as numpy arrays in this object's Theta and Beta attributes). Otherwise, the model
	will create Python dictionaries with entries for each user and item, which can take quite a bit of
	RAM memory. These can speed up predictions later through this package's API.

	Passing verbose=True will also print RMSE (root mean squared error) at each iteration.
	For slighly better speed pass verbose=False once you know what a good threshold should be
	for your data.

	Note
	----
	DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
	need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

	Note
	----
	If 'check_every' is not None and stop_crit is not 'diff-norm', it will, every N iterations,
	calculate the log-likelihood of the data. By default, this is NOT the full likelihood, (not including a constant
	that depends on the data but not on the parameters and which is quite slow to compute). The reason why
	it's calculated by default like this is because otherwise it can result it overflow (number is too big for the data
	type), but be aware that if not adding this constant, the number can turn positive
	and will mess with the stopping criterion for likelihood.

	Note
	----
	If you pass a validation set, it will calculate the Poisson log-likelihood **of the non-zero observations
	only**, rather than the complete likelihood that includes also the combinations of users and items
	not present in the data (assumed to be zero), thus it's more likely that you might see positive numbers here.
	
	Note
	----
	Compared to ALS, iterations from this algorithm are a lot faster to compute, so don't be scared about passing
	large numbers for maxiter.

	Note
	----
	In some unlucky cases, the parameters will become NA in the first iteration, in which case you should see
	weird values for log-likelihood and RMSE. If this happens, try again with a different random seed.

	Note
	----
	Fitting in mini-batches is more prone to numerical instability and compared to full-batch
	variational inference, it is more likely that all your parameters will turn to NaNs (which
	means the optimization procedure failed).

	Parameters
	----------
	k : int
		Number of latent factors to use.
	a : float
		Shape parameter for the user-factor matrix.
	a_prime : float
		Shape parameter and dividend of the rate parameter for the user activity vector.
	b_prime : float
		Divisor of the rate parameter for the user activity vector.
	c : float
		Shape parameter for the item-factor matrix.
	c_prime : float
		Shape parameter and dividend of the rate parameter for the item popularity vector.
	d_prime : float
		Divisor o the rate parameter for the item popularity vector.
	ncores : int
		Number of cores to use to parallelize computations.
		If set to -1, will use the maximum available on the computer.
	stop_crit : str, one of 'maxiter', 'train-llk', 'val-llk', 'diff-norm'
		Stopping criterion for the optimization procedure.
	check_every : None or int
		Calculate log-likelihood every N iterations.
	stop_thr : float
		Threshold for proportion increase in log-likelihood or l2-norm for difference between matrices.
	users_per_batch : None or int
		Number of users to take for each batch update in stochastic variational inference. If passing None both here
		and for 'items_per_batch', will perform full-batch variational inference, which leads to better results but on
		larger datasets takes longer to converge.
		If passing a number for both 'users_per_batch' and 'items_per_batch', it will alternate between epochs in which
		it samples by user and epochs in which it samples by item - this leads to faster convergence and is recommended,
		but using only one type leads to lower memory requirements and might have a use case if memory is limited.
	items_per_batch : None or int
		Number of items to take for each batch update in stochastic variational inference. If passing None both here
		and for 'users_per_batch', will perform full-batch variational inference, which leads to better results but on
		larger datasets takes longer to converge.
		If passing a number for both 'users_per_batch' and 'items_per_batch', it will alternate between epochs in which
		it samples by user and epochs in which it samples by item - this leads to faster convergence and is recommended,
		but using only one type leads to lower memory requirements and might have a use case if memory is limited.
	step_size : function(int) -> float in (0, 1)
		Function that takes the iteration/epoch number as input (starting at zero) and produces the step size
		for the global parameters as output (only used when fitting with stochastic variational inference).
		The step size must be a number between zero and one, and should be decresing with bigger iteration numbers.
		Ignored when passing users_per_batch=None.
	maxiter : int or None
		Maximum number of iterations for which to run the optimization procedure. This corresponds to epochs when
		fitting in batches of users. Recommended to use a lower number when passing a batch size.
	reindex : bool
		Whether to reindex data internally.
	verbose : bool
		Whether to print convergence messages.
	random_seed : int or None
		Random seed to use when starting the parameters.
	allow_inconsistent_math : bool
		Whether to allow inconsistent floating-point math (producing slightly different results on each run)
		which would allow parallelization of the updates for the shape parameters of Lambda and Gamma.
		Ignored (forced to True) in stochastic optimization mode.
	full_llk : bool
		Whether to calculate the full Poisson log-likehood, including terms that don't depend on the model parameters
		(thus are constant for a given dataset).
	alloc_full_phi : bool
		Whether to allocate the full Phi matrix (size n_samples * k) when using stochastic optimization. Doing so
		will make it a bit faster, but it will use more memory.
		Ignored when passing both 'users_per_batch=None' and 'items_per_batch=None'.
	keep_data : bool
		Whether to keep information about which user was associated with each item
		in the training set, so as to exclude those items later when making Top-N
		recommendations.
	save_folder : str or None
		Folder where to save all model parameters as csv files.
	produce_dicts : bool
		Whether to produce Python dictionaries for users and items, which
		are used to speed-up the prediction API of this package. You can still predict without
		them, but it might take some additional miliseconds (or more depending on the
		number of users and items).
	keep_all_objs : bool
		Whether to keep intermediate objects/variables in the object that are not necessary for
		predictions - these are: Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte
		(when passing True here, the model object will have these extra attributes too).
		Without these objects, it's not possible to call functions that alter the model parameters
		given new information after it's already fit.
	sum_exp_trick : bool
		Whether to use the sum-exp trick when scaling the multinomial parameters - that is, calculating them as
		exp(val - maxval)/sum_{val}(exp(val - maxval)) in order to avoid numerical overflow if there are
		too large numbers. For this kind of model, it is unlikely that it will be required, and it adds a
		small overhead, but if you notice NaNs in the results or in the likelihood, you might give this option a try.
	
	Attributes
	----------
	Theta : array (nusers, k)
		User-factor matrix.
	Beta : array (nitems, k)
		Item-factor matrix.
	user_mapping_ : array (nusers,)
		ID of the user (as passed to .fit) corresponding to each row of Theta.
	item_mapping_ : array (nitems,)
		ID of the item (as passed to .fit) corresponding to each row of Beta.
	user_dict_ : dict (nusers)
		Dictionary with the mapping between user IDs (as passed to .fit) and rows of Theta.
	item_dict_ : dict (nitems)
		Dictionary with the mapping between item IDs (as passed to .fit) and rows of Beta.
	is_fitted : bool
		Whether the model has been fit to some data.
	niter : int
		Number of iterations for which the fitting procedure was run.

	References
	----------
	[1] Scalable Recommendation with Hierarchical Poisson Factorization (Gopalan, P., Hofman, J.M. and Blei, D.M., 2015)
	[2] Stochastic variational inference (Hoffman, M.D., Blei, D.M., Wang, C. and Paisley, J., 2013)
	"""
	def __init__(self, k=30, a=0.3, a_prime=0.3, b_prime=1.0,
				 c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
				 stop_crit='train-llk', check_every=10, stop_thr=1e-3,
				 users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
				 maxiter=100, reindex=True, verbose=True,
				 random_seed = None, allow_inconsistent_math=False, full_llk=False,
				 alloc_full_phi=False, keep_data=True, save_folder=None,
				 produce_dicts=True, keep_all_objs=True, sum_exp_trick=False):

		## checking input
		assert isinstance(k, int)
		if isinstance(a, int):
			a = float(a)
		if isinstance(a_prime, int):
			a_prime = float(a_prime)
		if isinstance(b_prime, int):
			b_prime = float(b_prime)
		if isinstance(c, int):
			c = float(c)
		if isinstance(c_prime, int):
			c_prime = float(c_prime)
		if isinstance(d_prime, int):
			d_prime = float(d_prime)
			
		assert isinstance(a, float)
		assert isinstance(a_prime, float)
		assert isinstance(b_prime, float)
		assert isinstance(c, float)
		assert isinstance(c_prime, float)
		assert isinstance(d_prime, float)
		
		assert a>0
		assert a_prime>0
		assert b_prime>0
		assert c>0
		assert c_prime>0
		assert d_prime>0
		assert k>0
		
		if ncores == -1:
			ncores = multiprocessing.cpu_count()
			if ncores is None:
				ncores = 1 
		assert ncores>0
		assert isinstance(ncores, int)

		if random_seed is not None:
			assert isinstance(random_seed, int)

		assert stop_crit in ['maxiter', 'train-llk', 'val-llk', 'diff-norm']

		if maxiter is not None:
			assert maxiter>0
			assert isinstance(maxiter, int)
		else:
			maxiter = 10**10
			if stop_crit!='maxiter':
				raise ValueError("If 'stop_crit' is set to 'maxiter', must provide a maximum number of iterations.")
			
		if check_every is not None:
			assert isinstance(check_every, int)
			assert check_every>0
			assert check_every<=maxiter
		else:
			if stop_crit != 'maxiter':
				raise ValueError("If 'stop_crit' is not 'maxiter', must input after how many iterations to calculate it.")
			check_every = 0

		if isinstance(stop_thr, int):
			stop_thr = float(stop_thr)
		if stop_thr is not None:
			assert stop_thr>0
			assert isinstance(stop_thr, float)
			
		if save_folder is not None:
			save_folder = os.path.expanduser(save_folder)
			assert os.path.exists(save_folder)
			
		verbose = bool(verbose)
		if (stop_crit == 'maxiter') and (not verbose):
			check_every = 0

		if not isinstance(step_size, types.FunctionType):
			raise ValueError("'step_size' must be a function.")
		if len(inspect.getfullargspec(step_size).args) < 1:
			raise ValueError("'step_size' must be able to take the iteration number as input.")
		assert (step_size(0) >= 0) and (step_size(0) <= 1)
		assert (step_size(1) >= 0) and (step_size(1) <= 1)
		if users_per_batch is not None:
			if isinstance(users_per_batch, float):
				users_per_batch = int(users_per_batch)
			assert isinstance(users_per_batch, int)
			assert users_per_batch > 0
		else:
			users_per_batch = 0

		if items_per_batch is not None:
			if isinstance(items_per_batch, float):
				items_per_batch = int(items_per_batch)
			assert isinstance(items_per_batch, int)
			assert items_per_batch > 0
		else:
			items_per_batch = 0
		
		## storing these parameters
		self.k = k
		self.a = a
		self.a_prime = a_prime
		self.b_prime = b_prime
		self.c = c
		self.c_prime = c_prime
		self.d_prime = d_prime

		self.ncores = ncores
		self.allow_inconsistent_math = bool(allow_inconsistent_math)
		self.random_seed = random_seed
		self.stop_crit = stop_crit
		self.reindex = bool(reindex)
		self.keep_data = bool(keep_data)
		self.maxiter = maxiter
		self.check_every = check_every
		self.stop_thr = stop_thr
		self.save_folder = save_folder
		self.verbose = verbose
		self.produce_dicts = bool(produce_dicts)
		self.full_llk = bool(full_llk)
		self.alloc_full_phi = bool(alloc_full_phi)
		self.keep_all_objs = bool(keep_all_objs)
		self.sum_exp_trick = bool(sum_exp_trick)
		self.step_size = step_size
		self.users_per_batch = users_per_batch
		self.items_per_batch = items_per_batch

		if not self.reindex:
			self.produce_dicts = False
		
		## initializing other attributes
		self.Theta = None
		self.Beta = None
		self.user_mapping_ = None
		self.item_mapping_ = None
		self.user_dict_ = None
		self.item_dict_ = None
		self.is_fitted = False
		self.niter = None
	
	def fit(self, counts_df, val_set=None):
		"""
		Fit Hierarchical Poisson Model to sparse count data

		Fits a hierarchical Poisson model to count data using mean-field approximation with either
		full-batch coordinate-ascent or mini-batch stochastic coordinate-ascent.
		
		Note
		----
		DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
		need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

		Note
		----
		Forcibly terminating the procedure should still keep the last calculated shape and rate
		parameter values, but is not recommended. If you need to make predictions on a forced-terminated
		object, set the attribute 'is_fitted' to 'True'.

		Note
		----
		Fitting in mini-batches is more prone to numerical instability and compared to full-batch
		variational inference, it is more likely that all your parameters will turn to NaNs (which
		means the optimization procedure failed).

		Parameters
		----------
		counts_df : pandas data frame (nobs, 3)
			Input data with one row per non-zero observation, consisting of triplets ('UserId', 'ItemId', 'Count').
			Must containin columns 'UserId', 'ItemId', and 'Count'.
			Combinations of users and items not present are implicitly assumed to be zero by the model.
		val_set : pandas data frame (nobs, 3)
			Validation set on which to monitor log-likelihood. Same format as counts_df.

		Returns
		-------
		self : obj
			Copy of this object
		"""

		## a basic check
		if self.stop_crit == 'val-llk':
			if val_set is None:
				raise ValueError("If 'stop_crit' is set to 'val-llk', must provide a validation set.")

		## running each sub-process
		if self.verbose:
			self._print_st_msg()
		self._process_data(counts_df)
		if self.verbose:
			self._print_data_info()
		if (val_set is not None) and (self.stop_crit!='diff-norm') and (self.stop_crit!='train-llk'):
			self._process_valset(val_set)
		else:
			self.val_set = None
			
		self._cast_before_fit()
		self.niter = self._fit()
		
		## after terminating optimization
		if self.keep_data:
			if self.users_per_batch == 0:
				self._store_metadata()
			else:
				self._st_ix_user = self._st_ix_user[:-1]
		if self.produce_dicts and self.reindex:
			self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
			self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}
		self.is_fitted = True
		del self.input_df
		del self.val_set
		
		return self
	
	def _process_data(self, input_df):
		if isinstance(input_df, np.ndarray):
			assert len(input_df.shape) > 1
			assert input_df.shape[1] >= 3
			input_df = input_df.values[:,:3]
			input_df.columns = ['UserId', 'ItemId', "Count"]
			
		if input_df.__class__.__name__ == 'DataFrame':
			assert input_df.shape[0] > 0
			assert 'UserId' in input_df.columns.values
			assert 'ItemId' in input_df.columns.values
			assert 'Count' in input_df.columns.values
			self.input_df = input_df[['UserId', 'ItemId', 'Count']]
		else:
			raise ValueError("'input_df' must be a pandas data frame or a numpy array")

		if self.stop_crit in ['maxiter', 'diff-norm']:
			thr = 0
		else:
			thr = 0.9
		obs_zero = input_df.Count.values <= thr
		if obs_zero.sum() > 0:
			msg = "'counts_df' contains observations with a count value less than 1, these will be ignored."
			msg += " Any user or item associated exclusively with zero-value observations will be excluded."
			msg += " If using 'reindex=False', make sure that your data still meets the necessary criteria."
			warnings.warn(msg)
			input_df = input_df.loc[~obs_zero]
			
		if self.reindex:
			self.input_df['UserId'], self.user_mapping_ = pd.factorize(self.input_df.UserId)
			self.input_df['ItemId'], self.item_mapping_ = pd.factorize(self.input_df.ItemId)
			self.nusers = self.user_mapping_.shape[0]
			self.nitems = self.item_mapping_.shape[0]
			self.user_mapping_ = np.array(self.user_mapping_).reshape(-1)
			self.item_mapping_ = np.array(self.item_mapping_).reshape(-1)
			if (self.save_folder is not None) and self.reindex:
				if self.verbose:
					print("\nSaving user and item mappings...\n")
				pd.Series(self.user_mapping_).to_csv(os.path.join(self.save_folder, 'users.csv'), index=False)
				pd.Series(self.item_mapping_).to_csv(os.path.join(self.save_folder, 'items.csv'), index=False)
		else:
			self.nusers = self.input_df.UserId.max() + 1
			self.nitems = self.input_df.ItemId.max() + 1

		if self.save_folder is not None:
			with open(os.path.join(self.save_folder, "hyperparameters.txt"), "w") as pf:
				pf.write("a: %.3f\n" % self.a)
				pf.write("a_prime: %.3f\n" % self.a_prime)
				pf.write("b_prime: %.3f\n" % self.b_prime)
				pf.write("c: %.3f\n" % self.c)
				pf.write("c_prime: %.3f\n" % self.c_prime)
				pf.write("d_prime: %.3f\n" % self.d_prime)
				pf.write("k: %d\n" % self.k)
				if self.random_seed is not None:
					pf.write("random seed: %d\n" % self.random_seed)
				else:
					pf.write("random seed: None\n")
		
		self.input_df['Count'] = self.input_df.Count.astype('float32')
		self.input_df['UserId'] = self.input_df.UserId.astype(ctypes.c_int)
		self.input_df['ItemId'] = self.input_df.ItemId.astype(ctypes.c_int)

		if self.users_per_batch != 0:
			if self.nusers < self.users_per_batch:
				warnings.warn("Batch size passed is larger than number of users. Will set it to nusers/10.")
				self.users_per_batch = ctypes.c_int(np.ceil(self.nusers/10))
			self.input_df.sort_values('UserId', inplace=True)
			self._store_metadata(for_partial_fit=True)

		return None
		
	def _process_valset(self, val_set, valset=True):
		if isinstance(val_set, np.ndarray):
			assert len(val_set.shape) > 1
			assert val_set.shape[1] >= 3
			val_set = val_set.values[:,:3]
			val_set.columns = ['UserId', 'ItemId', "Count"]
			
		if val_set.__class__.__name__ == 'DataFrame':
			assert val_set.shape[0] > 0
			assert 'UserId' in val_set.columns.values
			assert 'ItemId' in val_set.columns.values
			assert 'Count' in val_set.columns.values
			self.val_set = val_set[['UserId', 'ItemId', 'Count']]
		else:
			raise ValueError("'val_set' must be a pandas data frame or a numpy array")
			
		if self.stop_crit == 'val-llk':
			thr = 0
		else:
			thr = 0.9
		obs_zero = self.val_set.Count.values <= thr
		if obs_zero.sum() > 0:
			msg = "'val_set' contains observations with a count value less than 1, these will be ignored."
			warnings.warn(msg)
			self.val_set = self.val_set.loc[~obs_zero]

		if self.reindex:
			self.val_set['UserId'] = pd.Categorical(self.val_set.UserId, self.user_mapping_).codes
			self.val_set['ItemId'] = pd.Categorical(self.val_set.ItemId, self.item_mapping_).codes
			self.val_set = self.val_set.loc[(self.val_set.UserId != (-1)) & (self.val_set.ItemId != (-1))]
			if self.val_set.shape[0] == 0:
				if valset:
					warnings.warn("Validation set has no combinations of users and items"+
								  " in common with training set. If 'stop_crit' was set"+
								  " to 'val-llk', will now be switched to 'train-llk'.")
					if self.stop_crit == 'val-llk':
						self.stop_crit = 'train-llk'
					self.val_set = None
				else:
					raise ValueError("'input_df' has no combinations of users and items"+
									 "in common with the training set.")
			else:
				self.val_set.reset_index(drop=True, inplace=True)
				self.val_set['Count'] = self.val_set.Count.astype('float32')
				self.val_set['UserId'] = self.val_set.UserId.astype(ctypes.c_int)
				self.val_set['ItemId'] = self.val_set.ItemId.astype(ctypes.c_int)
		return None
			
	def _store_metadata(self, for_partial_fit=False):
		if self.verbose and for_partial_fit:
			print("Creating user indices for stochastic optimization...")
		X = coo_matrix((self.input_df.Count.values, (self.input_df.UserId.values, self.input_df.ItemId.values)))
		X = csr_matrix(X)
		self._n_seen_by_user = X.indptr[1:] - X.indptr[:-1]
		if for_partial_fit:
			self._st_ix_user = X.indptr.astype(ctypes.c_int)
			self.input_df.sort_values('UserId', inplace=True)
		else:
			self._st_ix_user = X.indptr[:-1]
		self.seen = X.indices
		return None

	def _cast_before_fit(self):
		## setting all parameters and data to the right type
		self.Theta = np.empty((self.nusers, self.k), dtype='float32')
		self.Beta = np.empty((self.nitems, self.k), dtype='float32')
		self.k = cython_loops.cast_int(self.k)
		self.nusers = cython_loops.cast_int(self.nusers)
		self.nitems = cython_loops.cast_int(self.nitems)
		self.ncores = cython_loops.cast_int(self.ncores)
		self.maxiter = cython_loops.cast_int(self.maxiter)
		self.verbose = cython_loops.cast_int(self.verbose)
		if self.random_seed is None:
			self.random_seed = 0
		self.random_seed = cython_loops.cast_int(self.random_seed)
		self.check_every = cython_loops.cast_int(self.check_every)

		self.stop_thr = cython_loops.cast_float(self.stop_thr)
		self.a = cython_loops.cast_float(self.a)
		self.a_prime = cython_loops.cast_float(self.a_prime)
		self.b_prime = cython_loops.cast_float(self.b_prime)
		self.c = cython_loops.cast_float(self.c)
		self.c_prime = cython_loops.cast_float(self.c_prime)
		self.d_prime = cython_loops.cast_float(self.d_prime)

		if self.save_folder is None:
			self.save_folder = ""
	
	def _fit(self):

		if self.val_set is None:
			use_valset = cython_loops.cast_int(0)
			self.val_set = pd.DataFrame(np.empty((0,3)), columns=['UserId','ItemId','Count'])
			self.val_set['UserId'] = self.val_set.UserId.astype(ctypes.c_int)
			self.val_set['ItemId'] = self.val_set.ItemId.astype(ctypes.c_int)
			self.val_set['Count'] = self.val_set.Count.values.astype('float32')
		else:
			use_valset = cython_loops.cast_int(1)

		if self.users_per_batch == 0:
			self._st_ix_user = np.arange(1).astype(ctypes.c_int)

		self.niter, temp = cython_loops.fit_hpf(
			self.a, self.a_prime, self.b_prime,
			self.c, self.c_prime, self.d_prime,
			self.input_df.Count.values, self.input_df.UserId.values, self.input_df.ItemId.values,
			self.Theta, self.Beta,
			self.maxiter, self.stop_crit, self.check_every, self.stop_thr,
			self.users_per_batch, self.items_per_batch,
			self.step_size, cython_loops.cast_int(self.sum_exp_trick),
			self._st_ix_user.astype(ctypes.c_int),
			self.save_folder, self.random_seed, self.verbose,
			self.ncores, cython_loops.cast_int(self.allow_inconsistent_math),
			use_valset,
			self.val_set.Count.values, self.val_set.UserId.values, self.val_set.ItemId.values,
			cython_loops.cast_int(self.full_llk), cython_loops.cast_int(self.keep_all_objs),
			cython_loops.cast_int(self.alloc_full_phi)
			)

		if self.users_per_batch == 0:
			del self._st_ix_user

		if self.keep_all_objs:
			self.Gamma_shp = temp[0]
			self.Gamma_rte = temp[1]
			self.Lambda_shp = temp[2]
			self.Lambda_rte = temp[3]
			self.k_rte = temp[4]
			self.t_rte = temp[5]

	def _process_data_single(self, counts_df):
		assert self.is_fitted
		assert self.keep_all_objs
		if isinstance(counts_df, np.ndarray):
			assert len(counts_df.shape) > 1
			assert counts_df.shape[1] >= 2
			counts_df = counts_df.values[:,:2]
			counts_df.columns = ['ItemId', "Count"]
			
		if counts_df.__class__.__name__ == 'DataFrame':
			assert counts_df.shape[0] > 0
			assert 'ItemId' in counts_df.columns.values
			assert 'Count' in counts_df.columns.values
			counts_df = counts_df[['ItemId', 'Count']]
		else:
			raise ValueError("'counts_df' must be a pandas data frame or a numpy array")
			
		if self.reindex:
			if self.produce_dicts:
				try:
					counts_df['ItemId'] = counts_df.ItemId.map(lambda x: self.item_dict_[x])
				except:
					raise ValueError("Can only make calculations for items that were in the training set.")
			else:
				counts_df['ItemId'] = pd.Categorical(counts_df.ItemId.values, self.item_mapping_).codes
				if (counts_df.ItemId == -1).sum() > 0:
					raise ValueError("Can only make calculations for items that were in the training set.")

		counts_df['ItemId'] = counts_df.ItemId.values.astype(ctypes.c_int)
		counts_df['Count'] = counts_df.ItemId.values.astype(ctypes.c_float)
		return counts_df

	def partial_fit(self, counts_df, batch_type='users', step_size=None,
					nusers=None, nitems=None, users_in_batch=None, items_in_batch=None,
					new_users=False, new_items=False, random_seed=None):
		"""
		Updates the model with batches of data from a subset of users or items

		Note
		----
		You must pass either the **full set of user-item interactions** that are non-zero for some
		subset of users, or the **full set of item-user intersactions** that are non-zero for some
		subset of items.
		Otherwise, if passing a random sample of triplets, the model will not converge to reasonable results.

		Note
		----
		All user and items IDs must be integers starting at one, without gaps in the numeration.

		Note
		----
		For better results, fit the model with full-batch iterations (using the 'fit' method).
		Adding new users and/or items without refitting the model might result in worsened results
		for existing users/items. For adding users without altering the parameters for items or for
		other users, see the method 'add_user'.

		Note
		----
		Fitting in mini-batches is more prone to numerical instability and compared to full-batch
		variational inference, it is more likely that all your parameters will turn to NaNs (which
		means the optimization procedure failed).

		Parameters
		----------
		counts_df : data frame (n_samples, 3)
			Data frame with the user-item interactions for some subset of users. Must have columns
			'UserId', 'ItemId', 'Count'.
		batch_type : str, one of 'users' or 'items'
			Whether 'counts_df' contains a sample of users with all their item counts ('users'), or a
			sample of items with all their user counts ('items').
		step_size : None or float in (0, 1)
			Step size with which to update the global variables in the model. Must be a number between
			zero and one. If passing None, will determine it according to the step size function with which
			the model was initialized and the number of iterations or calls to partial fit that have been
			performed. If no valid function was passed at the initialization, it will use 1/sqrt(i+1).
		nusers : int
			Total number of users (not just in this batch!). Only required if calling partial_fit for the
			first time on a model object that hasn't been fit.
		nitems : int
			Total number of items (not just in this batch!). Only required if calling partial_fit for the
			first time on a model object that hasn't been fit.
		users_in_batch : None or array (n_users_sample,)
			Users that are present int counts_df. If passing None, will determine the unique elements in
			counts_df.UserId, but passing them if you already have them will skip this step.
		items_in_batch : None or array (n_items_sample,)
			Items that are present int counts_df. If passing None, will determine the unique elements in
			counts_df.ItemId, but passing them if you already have them will skip this step.
		new_users : bool
			Whether the data contains new users with numeration greater than the number of users with which
			the model was initially fit. **For better results refit the model including all users/items instead
			of adding them afterwards**.
		new_items : bool
			Whether the data contains new items with numeration greater than the number of items with which
			the model was initially fit. **For better results refit the model including all users/items instead
			of adding them afterwards**.
		random_seed : int
			Random seed to be used for the initialization of new user/item parameters. Ignored when
			new_users=False and new_items=False.

		Returns
		-------
		self : obj
			Copy of this object.
		"""

		if self.reindex:
			raise ValueError("'partial_fit' can only be called when using reindex=False.")
		if not self.keep_all_objs:
			raise ValueError("'partial_fit' can only be called when using keep_all_objs=True.")
		if self.keep_data:
			try:
				self.seen
				msg = "When using 'partial_fit', the list of items seen by each user is not updated "
				msg += "with the data passed here."
				warnings.warn(msg)
			except:
				msg = "When fitting the model through 'partial_fit' without calling 'fit' beforehand, "
				msg += "'keep_data' will be forced to False."
				warnings.warn(msg)
				self.keep_data = False

		assert batch_type in ['users', 'items']
		if batch_type == 'users':
			user_batch = True
		else:
			user_batch = False

		if nusers is None:
			try:
				nusers = self.nusers
			except:
				raise ValueError("Must specify total number of users when calling 'partial_fit' for the first time.")
		if nitems is None:
			try:
				nitems = self.nitems
			except:
				raise ValueError("Must specify total number of items when calling 'partial_fit' for the first time.")

		try:
			if self.nusers is None:
				self.nusers = nusers
		except:
			self.nusers = nusers
		try:
			if self.nitems is None:
				self.nitems = nitems
		except:
			self.nitems = nitems

		if step_size is None:
			try:
				self.step_size(0)
				try:
					step_size = self.step_size(self.niter)
				except:
					self.niter = 0
					step_size = 1.0
			except:
				try:
					step_size = 1 / np.sqrt(self.niter + 2)
				except:
					self.niter = 0
					step_size = 1.0
		assert step_size >= 0
		assert step_size <= 1

		if random_seed is not None:
			if isinstance(random_seed, float):
				random_seed = int(random_seed)
			assert isinstance(random_seed, int)

		if counts_df.__class__.__name__ == "ndarray":
			counts_df = pd.DataFrame(counts_df)
			counts_df.columns[:3] = ['UserId', 'ItemId', 'Count']

		assert counts_df.__class__.__name__ == "DataFrame"
		assert 'UserId' in counts_df.columns.values
		assert 'ItemId' in counts_df.columns.values
		assert 'Count' in counts_df.columns.values
		assert counts_df.shape[0] > 0

		Y_batch = counts_df.Count.values.astype('float32')
		ix_u_batch = counts_df.UserId.values.astype(ctypes.c_int)
		ix_i_batch = counts_df.ItemId.values.astype(ctypes.c_int)

		if users_in_batch is None:
			users_in_batch = np.unique(ix_u_batch)
		else:
			users_in_batch = np.array(users_in_batch).astype(ctypes.c_int)
		if items_in_batch is None:
			items_in_batch = np.unique(ix_i_batch)
		else:
			items_in_batch = np.array(items_in_batch).astype(ctypes.c_int)

		if (self.Theta is None) or (self.Beta is None):
			self._cast_before_fit()
			self.Gamma_shp, self.Gamma_rte, self.Lambda_shp, self.Lambda_rte, \
			self.k_rte, self.t_rte = cython_loops.initialize_parameters(
				self.Theta, self.Beta, self.random_seed, self.a, self.a_prime,
				self.b_prime, self.c, self.c_prime, self.d_prime)
			self.Theta = self.Gamma_shp / self.Gamma_rte
			self.Beta = self.Lambda_shp / self.Lambda_rte

		if new_users:
			if not self.keep_all_objs:
				raise ValueError("Can only add users without refitting when using keep_all_objs=True")
			nusers_now = ix_u_batch.max() + 1
			nusers_add = self.nusers - nusers_now
			if nusers_add < 1:
				raise ValueError("There are no new users in the data passed to 'partial_fit'.")
			self._initialize_extra_users(nusers_add, random_seed)
			self.nusers += nusers_add

		if new_items:
			if not self.keep_all_objs:
				raise ValueError("Can only add items without refitting when using keep_all_objs=True")
			nitems_now = ix_i_batch.max() + 1
			nitems_add = self.nitems - nitems_now
			if nitems_add < 1:
				raise ValueError("There are no new items in the data passed to 'partial_fit'.")
			self._initialize_extra_items(nitems_add, random_seed)
			self.nitems += nitems_add

		k_shp = cython_loops.cast_float(self.a_prime + self.k * self.a)
		t_shp = cython_loops.cast_float(self.c_prime + self.k * self.c)
		add_k_rte = cython_loops.cast_float(self.a_prime / self.b_prime)
		add_t_rte = cython_loops.cast_float(self.c_prime / self.d_prime)
		multiplier_batch = float(nusers) / users_in_batch.shape[0]

		cython_loops.partial_fit(
					Y_batch,
					ix_u_batch, ix_i_batch,
					self.Theta, self.Beta,
					self.Gamma_shp, self.Gamma_rte,
					self.Lambda_shp, self.Lambda_rte,
					self.k_rte, self.t_rte,
					add_k_rte, add_t_rte, self.a, self.c,
					k_shp, t_shp, cython_loops.cast_int(self.k),
					users_in_batch, items_in_batch,
					cython_loops.cast_int(self.allow_inconsistent_math),
					cython_loops.cast_float(step_size), cython_loops.cast_float(multiplier_batch),
					self.ncores, user_batch
				)

		self.niter += 1
		self.is_fitted = True
		return self

	def _initialize_extra_users(self, n, seed):
		if seed is not None:
			np.random.seed(seed)

		new_Theta = np.random.gamma(self.a, 1/self.b_prime, size=(n, self.k)).astype('float32')
		self.Theta = np.r_[self.Theta, new_Theta]
		self.k_rte = np.r_[self.k_rte, b_prime + new_Theta.sum(axis=1, keepdims=True)]
		new_Gamma_rte = np.random.gamma(self.a_prime, self.b_prime/self.a_prime, size=(n, 1)).astype('float32') \
							+ self.Beta.sum(axis=0, keepdims=True)
		self.Gamma_rte = np.r_[self.Gamma_rte, new_Gamma_rte]
		self.Gamma_shp = np.r_[self.Gamma_shp, new_Gamma_rte * new_Theta * \
								np.random.uniform(low=.85, high=1.15, size=(n, self.k)).astype('float32')]

	def _initialize_extra_items(self, n, seed):
		if seed is not None:
			np.random.seed(seed)

		new_Beta = np.random.gamma(self.c, 1/self.d_prime, size=(n, self.k)).astype('float32')
		self.Beta = np.r_[self.Beta, new_Beta]
		self.t_rte = np.r_[self.t_rte, self.d_prime + new_Beta.sum(axis=1, keepdims=True)]
		new_Lambda_rte = np.random.gamma(self.c_prime, self.d_prime/self.c_prime, size=(n, 1)).astype('float32') \
							+ self.Theta.sum(axis=0, keepdims=True)
		self.Lambda_rte = np.r_[self.Lambda_rte, new_Lambda_rte]
		self.Lambda_shp = np.r_[self.Lambda_shp, new_Lambda_rte * new_Beta * \
									 np.random.uniform(low=.85, high=1.15, size=(n, self.k)).astype('float32')]

	def _check_input_predict_factors(self, ncores, random_seed, stop_thr, maxiter):
		if ncores == -1:
			ncores = multiprocessing.cpu_count()
			if ncores is None:
				ncores = 1 
		assert ncores>0
		assert isinstance(ncores, int)

		assert isinstance(random_seed, int)
		assert random_seed > 0

		if isinstance(stop_thr, int):
			stop_thr = float(stop_thr)
		assert stop_thr>0
		assert isinstance(stop_thr, float)

		if isinstance(maxiter, float):
			maxiter = int(maxiter)
		assert isinstance(maxiter, int)
		assert maxiter > 0

		return ncores, random_seed, stop_thr, maxiter

	def predict_factors(self, counts_df, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3, return_all=False):
		"""
		Gets latent factors for a user given her item counts

		This is similar to obtaining topics for a document in LDA.

		Note
		----
		This function will NOT modify any of the item parameters.

		Note
		----
		This function only works with one user at a time.

		Note
		----
		This function is prone to producing all NaNs values.

		Parameters
		----------
		counts_df : DataFrame or array (nsamples, 2)
			Data Frame with columns 'ItemId' and 'Count', indicating the non-zero item counts for a
			user for whom it's desired to obtain latent factors.
		maxiter : int
			Maximum number of iterations to run.
		ncores : int
			Number of threads/cores to use. With data for only one user, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
			If passing -1, it will determine the maximum number of cores in the system and use that.
		random_seed : int
			Random seed used to initialize parameters.
		stop_thr : float
			If the l2-norm of the difference between values of Theta_{u} between interations is less
			than this, it will stop. Smaller values of 'k' should require smaller thresholds.
		return_all : bool
			Whether to return also the intermediate calculations (Gamma_shp, Gamma_rte). When
			passing True here, the output will be a tuple containing (Theta, Gamma_shp, Gamma_rte, Phi)

		Returns
		-------
		latent_factors : array (k,)
			Calculated latent factors for the user, given the input data
		"""

		ncores, random_seed, stop_thr, maxiter = self._check_input_predict_factors(ncores, random_seed, stop_thr, maxiter)

		## processing the data
		counts_df = self._process_data_single(counts_df)

		## calculating the latent factors
		Theta = np.empty(self.k, dtype='float32')
		temp = cython_loops.calc_user_factors(
								 self.a, self.a_prime, self.b_prime,
								 self.c, self.c_prime, self.d_prime,
								 counts_df.Count.values,
								 counts_df.ItemId.values,
								 Theta, self.Beta,
								 self.Lambda_shp,
								 self.Lambda_rte,
								 cython_loops.cast_int(counts_df.shape[0]), cython_loops.cast_int(self.k),
								 cython_loops.cast_int(int(maxiter)), cython_loops.cast_int(ncores),
								 cython_loops.cast_int(int(random_seed)), cython_loops.cast_float(stop_thr),
								 cython_loops.cast_int(bool(return_all))
								 )

		if np.isnan(Theta).sum() > 0:
			raise ValueError("NaNs encountered in the result. Failed to produce latent factors.")

		if return_all:
			return (Theta, temp[0], temp[1], temp[2])
		else:
			return Theta

	def add_user(self, user_id, counts_df, update_existing=False, maxiter=10, ncores=1,
				 random_seed=1, stop_thr=1e-3, update_all_params=None):
		"""
		Add a new user to the model or update parameters for a user according to new data
		
		Note
		----
		This function will NOT modify any of the item parameters.

		Note
		----
		This function only works with one user at a time. For updating many users at the same time,
		use 'partial_fit' instead.

		Note
		----
		For betters results, refit the model again from scratch.

		Note
		----
		This function is prone to producing all NaNs values.

		Parameters
		----------
		user_id : obj
			Id to give to be user (when adding a new one) or Id of the existing user whose parameters are to be
			updated according to the data in 'counts_df'. **Make sure that the data type is the same that was passed
			in the training data, so if you have integer IDs, don't pass a string as ID**.
		counts_df : data frame or array (nsamples, 2)
			Data Frame with columns 'ItemId' and 'Count'. If passing a numpy array, will take the first two columns
			in that order. Data containing user/item interactions **from one user only** for which to add or update
			parameters. Note that you need to pass *all* the user-item interactions for this user when making an update,
			not just the new ones.
		update_existing : bool
			Whether this should be an update of the parameters for an existing user (when passing True), or
			an addition of a new user that was not in the model before (when passing False).
		maxiter : int
			Maximum number of iterations to run.
		ncores : int
			Number of threads/cores to use. With data for only one user, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
		random_seed : int
			Random seed used to initialize parameters.
		stop_thr : float
			If the l2-norm of the difference between values of Theta_{u} between interations is less
			than this, it will stop. Smaller values of 'k' should require smaller thresholds.
		update_all_params : bool
			Whether to update also the item parameters in each iteration. If passing True, will update them
			with a step size given determined by the number of iterations already taken and the step_size function
			given as input in the model constructor call.

		Returns
		-------
		True : bool
			Will return True if the process finishes successfully.
		"""

		ncores, random_seed, stop_thr, maxiter = self._check_input_predict_factors(ncores, random_seed, stop_thr, maxiter)

		if update_existing:
			## checking that the user already exists
			if self.produce_dicts and self.reindex:
				user_id = self.user_dict_[user_id]
			else:
				if self.reindex:
					user_id = pd.Categorical(np.array([user_id]), self.user_mapping_).codes[0]
					if user_id == -1:
						raise ValueError("User was not present in the training data.")

		## processing the data
		counts_df = self._process_data_single(counts_df)
		
		if update_all_params:
			counts_df['UserId'] = user_id
			counts_df['UserId'] = counts_df.UserId.astype(ctypes.c_int)
			self.partial_fit(counts_df, new_users=(not update_existing))
			Theta_prev = self.Theta[-1].copy()
			for i in range(maxiter - 1):
				self.partial_fit(counts_df)
				new_Theta = self.Theta[-1]
				if np.linalg.norm(new_Theta - Theta_prev) <= stop_thr:
					break
				else:
					Theta_prev = self.Theta[-1].copy()
		else:
			## calculating the latent factors
			Theta = np.empty(self.k, dtype='float32')
			temp = cython_loops.calc_user_factors(
								 self.a, self.a_prime, self.b_prime,
								 self.c, self.c_prime, self.d_prime,
								 counts_df.Count.values,
								 counts_df.ItemId.values,
								 Theta, self.Beta,
								 self.Lambda_shp,
								 self.Lambda_rte,
								 cython_loops.cast_int(counts_df.shape[0]), cython_loops.cast_int(self.k),
								 cython_loops.cast_int(maxiter), cython_loops.cast_int(ncores),
								 cython_loops.cast_int(random_seed), cython_loops.cast_int(stop_thr),
								 cython_loops.cast_int(self.keep_all_objs)
								 )

			if np.isnan(Theta).sum() > 0:
				raise ValueError("NaNs encountered in the result. Failed to produce latent factors.")

			## adding the data to the model
			if update_existing:
				self.Theta[user_id] = Theta
				if self.keep_all_objs:
					self.Gamma_shp[user_id] = temp[0]
					self.Gamma_rte[user_id] = temp[1]
					self.k_rte[user_id] = self.a_prime/self.b_prime + \
										(temp[0].reshape((1,-1))/temp[1].reshape((1,-1))).sum(axis=1, keepdims=True)
			else:
				if self.reindex:
					new_id = self.user_mapping_.shape[0]
					self.user_mapping_ = np.r_[self.user_mapping_, user_id]
					if self.produce_dicts:
						self.user_dict_[user_id] = new_id
				self.Theta = np.r_[self.Theta, Theta.reshape((1, self.k))]
				if self.keep_all_objs:
					self.Gamma_shp = np.r_[self.Gamma_shp, temp[0].reshape((1, self.k))]
					self.Gamma_rte = np.r_[self.Gamma_rte, temp[1].reshape((1, self.k))]
					self.k_rte = np.r_[self.k_rte, self.a_prime/self.b_prime + \
									(temp[0].reshape((1,-1))/temp[1].reshape((1,-1))).sum(axis=1, keepdims=True)]
				self.nusers += 1

		## updating the list of seen items for this user
		if self.keep_data:
			if update_existing:
				n_seen_by_user_before = self._n_seen_by_user[user_id]
				self._n_seen_by_user[user_id] = counts_df.shape[0]
				self.seen = np.r_[self.seen[:user_id], counts_df.ItemId.values, self.seen[(user_id + 1):]]
				self._st_ix_user[(user_id + 1):] += self._n_seen_by_user[user_id] - n_seen_by_user_before
			else:
				self._n_seen_by_user = np.r_[self._n_seen_by_user, counts_df.shape[0]]
				self._st_ix_user = np.r_[self._st_ix_user, self.seen.shape[0]]
				self.seen = np.r_[self.seen, counts_df.ItemId.values]

		return True
	
	def predict(self, user, item):
		"""
		Predict count for combinations of users and items
		
		Note
		----
		You can either pass an individual user and item, or arrays representing
		tuples (UserId, ItemId) with the combinatinons of users and items for which
		to predict (one row per prediction).

		Parameters
		----------
		user : array-like (npred,) or obj
			User(s) for which to predict each item.
		item: array-like (npred,) or obj
			Item(s) for which to predict for each user.
		"""
		assert self.is_fitted
		if isinstance(user, list) or isinstance(user, tuple):
			user = np.array(user)
		if isinstance(item, list) or isinstance(item, tuple):
			item = np.array(item)
		if user.__class__.__name__=='Series':
			user = user.values
		if item.__class__.__name__=='Series':
			item = item.values
			
		if isinstance(user, np.ndarray):
			if len(user.shape) > 1:
				user = user.reshape(-1)
			assert user.shape[0] > 0
			if self.reindex:
				if user.shape[0] > 1:
					user = pd.Categorical(user, self.user_mapping_).codes
				else:
					if self.user_dict_ is not None:
						try:
							user = self.user_dict_[user]
						except:
							user = -1
					else:
						user = pd.Categorical(user, self.user_mapping_).codes[0]
		else:
			if self.reindex:
				if self.user_dict_ is not None:
					try:
						user = self.user_dict_[user]
					except:
						user = -1
				else:
					user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
			user = np.array([user])
			
		if isinstance(item, np.ndarray):
			if len(item.shape) > 1:
				item = item.reshape(-1)
			assert item.shape[0] > 0
			if self.reindex:
				if item.shape[0] > 1:
					item = pd.Categorical(item, self.item_mapping_).codes
				else:
					if self.item_dict_ is not None:
						try:
							item = self.item_dict_[item]
						except:
							item = -1
					else:
						item = pd.Categorical(item, self.item_mapping_).codes[0]
		else:
			if self.reindex:
				if self.item_dict_ is not None:
					try:
						item = self.item_dict_[item]
					except:
						item = -1
				else:
					item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
			item = np.array([item])

		assert user.shape[0] == item.shape[0]
		
		if user.shape[0] == 1:
			if (user[0] == -1) or (item[0] == -1):
				return np.nan
			else:
				return self.Theta[user].dot(self.Beta[item].T).reshape(-1)[0]
		else:
			nan_entries = (user == -1) | (item == -1)
			if nan_entries.sum() == 0:
				return (self.Theta[user] * self.Beta[item]).sum(axis=1)
			else:
				non_na_user = user[~nan_entries]
				non_na_item = item[~nan_entries]
				out = np.empty(user.shape[0], dtype=self.Theta.dtype)
				out[~nan_entries] = (self.Theta[non_na_user] * self.Beta[non_na_item]).sum(axis=1)
				out[nan_entries] = np.nan
				return out
		
	
	def topN(self, user, n=10, exclude_seen=True, items_pool=None):
		"""
		Recommend Top-N items for a user

		Outputs the Top-N items according to score predicted by the model.
		Can exclude the items for the user that were associated to her in the
		training set, and can also recommend from only a subset of user-provided items.

		Parameters
		----------
		user : obj
			User for which to recommend.
		n : int
			Number of top items to recommend.
		exclude_seen: bool
			Whether to exclude items that were associated to the user in the training set.
		items_pool: None or array
			Items to consider for recommending to the user.
		
		Returns
		-------
		rec : array (n,)
			Top-N recommended items.
		"""
		if isinstance(n, float):
			n = int(n)
		assert isinstance(n ,int)
		if self.reindex:
			if self.produce_dicts:
				try:
					user = self.user_dict_[user]
				except:
					raise ValueError("Can only predict for users who were in the training set.")
			else:
				user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
				if user == -1:
					raise ValueError("Can only predict for users who were in the training set.")
		if exclude_seen and not self.keep_data:
			raise Exception("Can only exclude seen items when passing 'keep_data=True' to .fit")
			
		if items_pool is None:
			allpreds = - (self.Theta[user].dot(self.Beta.T))
			if exclude_seen:
				n_ext = np.min([n + self._n_seen_by_user[user], self.Beta.shape[0]])
				rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
				seen = self.seen[self._st_ix_user[user] : self._st_ix_user[user] + self._n_seen_by_user[user]]
				rec = np.setdiff1d(rec, seen)
				rec = rec[np.argsort(allpreds[rec])[:n]]
				if self.reindex:
					return self.item_mapping_[rec]
				else:
					return rec

			else:
				n = np.min([n, self.Beta.shape[0]])
				rec = np.argpartition(allpreds, n-1)[:n]
				rec = rec[np.argsort(allpreds[rec])]
				if self.reindex:
					return self.item_mapping_[rec]
				else:
					return rec

		else:
			if isinstance(items_pool, list) or isinstance(items_pool, tuple):
				items_pool = np.array(items_pool)
			if items_pool.__class__.__name__=='Series':
				items_pool = items_pool.values
			if isinstance(items_pool, np.ndarray):  
				if len(items_pool.shape) > 1:
					items_pool = items_pool.reshape(-1)
				if self.reindex:
					items_pool_reind = pd.Categorical(items_pool, self.item_mapping_).codes
					nan_ix = (items_pool_reind == -1)
					if nan_ix.sum() > 0:
						items_pool_reind = items_pool_reind[~nan_ix]
						msg = "There were " + ("%d" % int(nan_ix.sum())) + " entries from 'item_pool'"
						msg += "that were not in the training data and will be exluded."
						warnings.warn(msg)
					del nan_ix
					if items_pool_reind.shape[0] == 0:
						raise ValueError("No items to recommend.")
					elif items_pool_reind.shape[0] == 1:
						raise ValueError("Only 1 item to recommend.")
					else:
						pass
			else:
				raise ValueError("'items_pool' must be an array.")

			if self.reindex:
				allpreds = - self.Theta[user].dot(self.Beta[items_pool_reind].T)
			else:
				allpreds = - self.Theta[user].dot(self.Beta[items_pool].T)
			n = np.min([n, items_pool.shape[0]])
			if exclude_seen:
				n_ext = np.min([n + self._n_seen_by_user[user], items_pool.shape[0]])
				rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
				seen = self.seen[self._st_ix_user[user] : self._st_ix_user[user] + self._n_seen_by_user[user]]
				if self.reindex:
					rec = np.setdiff1d(items_pool_reind[rec], seen)
					allpreds = - self.Theta[user].dot(self.Beta[rec].T)
					return self.item_mapping_[rec[np.argsort(allpreds)[:n]]]
				else:
					rec = np.setdiff1d(items_pool[rec], seen)
					allpreds = - self.Theta[user].dot(self.Beta[rec].T)
					return rec[np.argsort(allpreds)[:n]]
			else:
				rec = np.argpartition(allpreds, n-1)[:n]
				return items_pool[rec[np.argsort(allpreds[rec])]]

	
	def eval_llk(self, input_df, full_llk=False):
		"""
		Evaluate Poisson log-likelihood (plus constant) for a given dataset
		
		Note
		----
		This Poisson log-likelihood is calculated only for the combinations of users and items
		provided here, so it's not a complete likelihood, and it might sometimes turn out to
		be a positive number because of this.
		Will filter out the input data by taking only combinations of users
		and items that were present in the training set.

		Parameters
		----------
		input_df : pandas data frame (nobs, 3)
			Input data on which to calculate log-likelihood, consisting of IDs and counts.
			Must contain one row per non-zero observaion, with columns 'UserId', 'ItemId', 'Count'.
			If a numpy array is provided, will assume the first 3 columns
			contain that info.
		full_llk : bool
			Whether to calculate terms of the likelihood that depend on the data but not on the
			parameters. Ommitting them is faster, but it's more likely to result in positive values.

		Returns
		-------
		llk : dict
			Dictionary containing the calculated log-likelihood and the number of
			observations that were used to calculate it.
		"""
		assert self.is_fitted
		self._process_valset(input_df, valset=False)
		self.ncores = cython_loops.cast_int(self.ncores)
		out = {'llk': cython_loops.calc_llk(self.val_set.Count.values,
											self.val_set.UserId.values,
											self.val_set.ItemId.values,
											self.Theta,
											self.Beta,
											self.k,
											self.ncores,
											cython_loops.cast_int(bool(full_llk))),
			   'nobs':self.val_set.shape[0]}
		del self.val_set
		return out

	def _print_st_msg(self):
		print("**********************************")
		print("Hierarchical Poisson Factorization")
		print("**********************************")
		print("")

	def _print_data_info(self):
		print("Number of users: %d" % self.nusers)
		print("Number of items: %d" % self.nitems)
		print("Latent factors to use: %d" % self.k)
		print("")
