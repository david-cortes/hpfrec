import pandas as pd, numpy as np
import multiprocessing, os, warnings
import hpfrec.cython_loops as cython_loops
import ctypes
pd.options.mode.chained_assignment = None

class HPF:
    """
    Hierarchical Poisson Factorization

    Model for recommending items based on probabilistic Poisson factorization
    on sparse count data (e.g. number of times a user played different songs),
    using variational inference with coordinate-ascent.

    Can use different stopping criteria for the opimization procedure:

    1) Run for a fixed number of iterations (stop_crit='maxiter').
    2) Calculate the log-likelihood every N iterations (stop_crit='train-llk' and check_every)
       and stop once {1 - curr/prev} is below a certain threshold (stop_thr)
    3) Calculate the log-likelihood in a user-provided validation set (stop_crit='val-llk', val_set and check_every)
       and stop once {1 - curr/prev} is below a certain threshold. For this criterion, you might want to lower the
       default threshold (see Note).
    4) Check the the difference in the user-factor matrix after every N iterations (stop_crit='diff-norm', check_every)
       and stop once the *l2-norm* of this difference is below a certain threshold (stop_thr).
       Note that this is *not a percent* difference as it is for log-likelihood criteria, so you should put a larger
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
    RAM memory. These are required for making predictions later through this package's API.

    Passing verbose=True will also print RMSE (root mean squared error) at each iteration.
    For slighly better speed pass verbose=False once you know what a good threshold should be
    for your data.

    Note
    ----
    If 'check_every' is not None and stop_crit is not 'diff-norm', it will, every N iterations,
    calculate the log-likelihood of the data. By default, this is the full likelihood, including a constant
    that depends on the data but not on the parameters and which is quite slow to compute. The reason why
    it's calculated by default like this is because, if not adding this constant, the number can turn positive
    and will mess with the stopping criterion for likelihood. You can nevertheless choose to turn this constant off
    if you are confident that your likelihood values will not get positive.

    Note
    ----
    If you pass a validation set, it will calculate the log-likelihood *of the non-zero observations
    only*, rather than the complete likelihood that includes also the combinations of users and items
    not present in the data (assumed to be zero), thus it's more likely that you might see positive numbers here.
    Compared to ALS, iterations from this algorithm are a lot faster to compute, so don't be scared about passing
    large numbers for maxiter.

    Note
    ----
    In some unlucky cases, the parameters will become NA in the first iteration, in which case you should see
    weird values for log-likelihood and RMSE. If this happens, try again with a different random seed.

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
    maxiter : int
        Maximum number of iterations for which to run the optimization procedure.
    reindex : bool
        Whether to reindex data internally.
    random_seed : int or None
        Random seed to use when starting the parameters.
    allow_inconsistent_math : bool
        Whether to allow inconsistent floating-point math (producing slightly different results on each run)
        which would allow parallelization of the updates for the shape parameters of Lambda and Gamma.
    verbose : bool
        Whether to print convergence messages.
    keep_data : bool
        Whether to keep information about which user was associated with each item
        in the training set, so as to exclude those items later when making Top-N
        recommendations.
    save_folder : str or None
        Folder where to save all model parameters as csv files.
    produce_dicts : bool
        Whether to produce Python dictionaries for users and items, which
        are used by the prediction API of this package.
    
    Attributes
    ----------
    Theta : array (nusers, k)
        User-factor matrix.
    Beta : array (nitems, k)
        Item-factor matrix.
    user_mapping_ : array (nusers,)
        ID of the user (as passed to .fit) of each row of Theta.
    item_mapping_ : array (nitems,)
        ID of the item (as passed to .fit) of each row of Beta.
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
    [1] Scalable Recommendation with Hierarchical Poisson Factorization (P. Gopalan, 2015)
    """
    def __init__(self, k=30, a=0.3, a_prime=0.3, b_prime=1.0,
                 c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
                 stop_crit='train-llk', check_every=10, stop_thr=1e-3,
                 maxiter=100, reindex=True, random_seed = None,
                 allow_inconsistent_math=False, verbose=True, full_llk=True,
                 keep_data=True, save_folder=None, produce_dicts=True):

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

        Fits a hierarchical Poisson model to count data using mean-field approximation with coordinate-ascent.
        
        Note
        ----
        Forcibly terminating the procedure should still keep the last calculated Theta and Beta in the
        object attributes, but is not recommended.

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
            
        self._fit()
        
        ## after terminating optimization
        if not self.produce_dicts:
            return True

        if self.keep_data:
            self._store_metadata()
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
            
        if self.reindex:
            self.val_set['UserId'] = pd.Categorical.from_codes(self.val_set.UserId, self.user_mapping_).\
                                        get_values()
            self.val_set['ItemId'] = pd.Categorical.from_codes(self.val_set.ItemId, self.user_mapping_).\
                                        get_values()
            self.val_set = self.val_set.loc[(~self.val_set.UserId.isnull()) & (~self.val_set.ItemId.isnull())]
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
            
    def _store_metadata(self):
        self.seen = self.input_df[['UserId', 'ItemId']].copy()
        self.seen.sort_values(['UserId', 'ItemId'], inplace=True)
        self.seen.reset_index(drop = True, inplace = True)
        self._n_seen_by_user = self.seen.groupby('UserId')['ItemId'].agg(lambda x: len(x)).values
        self._st_ix_user = np.cumsum(self._n_seen_by_user)
        self._st_ix_user = np.r_[[0], self._st_ix_user[:self._st_ix_user.shape[0]-1]]
        self.seen = self.seen.ItemId.values
        return None
    
    def _fit(self):
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

        if self.val_set is None:
            use_valset = cython_loops.cast_int(0)
            self.val_set = pd.DataFrame(np.empty((0,3)), columns=['UserId','ItemId','Count'])
            self.val_set['UserId'] = self.val_set.UserId.astype(ctypes.c_int)
            self.val_set['ItemId'] = self.val_set.ItemId.astype(ctypes.c_int)
            self.val_set['Count'] = self.val_set.Count.values.astype('float32')
        else:
            use_valset = cython_loops.cast_int(1)

        self.niter = cython_loops.fit_hpf(
            self.a, self.a_prime, self.b_prime,
            self.c, self.c_prime, self.d_prime,
            self.input_df.Count.values, self.input_df.UserId.values, self.input_df.ItemId.values,
            self.Theta, self.Beta,
            self.maxiter, self.stop_crit, self.check_every, self.stop_thr,
            self.save_folder, self.random_seed, self.verbose,
            self.ncores, cython_loops.cast_int(self.allow_inconsistent_math),
            use_valset,
            self.val_set.Count.values, self.val_set.UserId.values, self.val_set.ItemId.values,
            cython_loops.cast_int(self.full_llk)
            )
    
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
            if self.reindex:
                try:
                    user = pd.Series(user).map(lambda x: self.user_dict_[x])
                except:
                    raise ValueError("Can only predict for users who were in the training set.")
        else:
            if self.reindex:
                try:
                    user = self.user_dict_[user]
                except:
                    raise ValueError("Can only predict for users who were in the training set.")
            user = np.array([user])
            
        if isinstance(item, np.ndarray):  
            if len(item.shape) > 1:
                item = item.reshape(-1)
            if self.reindex:
                try:
                    item = pd.Series(item).map(lambda x: self.item_dict_[x])
                except:
                    raise ValueError("Can only predict for items that were in the training set.")
        else:
            if self.reindex:
                try:
                    item = self.item_dict_[item]
                except:
                    raise ValueError("Can only predict for items that were in the training set.")
            item = np.array([item])

        assert user.shape[0] == item.shape[0]
        
        if user.shape[0] == 1:
            return self.Theta[user].dot(self.Beta[item].T)
        else:
            return (self.Theta[user] * self.Beta[item]).sum(axis=1)
        
    
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
            try:
                user = self.user_dict_[user]
            except:
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
                    try:
                        items_pool_reind = pd.Series(items_pool).map(lambda x: self.item_dict_[x])
                    except:
                        raise ValueError("Can only predict for items that were in the training set.")
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

    
    def eval_llk(self, input_df, full_llk=True):
        """
        Evaluate log-likelihood (plus constant) for a given dataset
        
        Note
        ----
        This log-likelihood is calculated only for the combinations of users and items
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
