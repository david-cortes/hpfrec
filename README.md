# Hierarchical Poisson Factorization

This is a Python package for hierarchical Poisson factorization, a form of probabilistic matrix factorization used for recommender systems with implicit count data, based on the paper _Scalable Recommendation with Hierarchical Poisson Factorization (P. Gopalan, 2015)_.

Although the package was created with recommender systems in mind, it can also be used for other domains, e.g. as a faster alternative to LDA (Latent Ditichlet Allocation), where users become documents and items become words.

Supports parallelization, full-batch variational inference, mini-batch stochastic variational inference (alternating between epochs sampling batches of users and epochs sampling batches of items), and different stopping criteria for the coordinate-ascent procedure. The main computations are written in fast Cython code.

As a point of reference, fitting the model through full-batch updates to the MillionSong TasteProfile dataset (48M records from 1M users on 380K items) took around 45 minutes on a server from Google Cloud with Skylake CPU when using 24 cores.

For a similar package using also item/user side information see [ctpfrec](https://github.com/david-cortes/ctpfrec).

For a non-Bayesian version which can produce sparse factors see [poismf](https://github.com/david-cortes/poismf).

## Model description

The model consists in producing a non-negative low-rank matrix factorization of counts data (such as number of times each user played each song in some internet service) `Y ~= UV'`, produced by a generative model as follows:
```
ksi_u ~ Gamma(a_prime, a_prime/b_prime)
Theta_uk ~ Gamma(a, ksi_u)

eta_i ~ Gamma(c_prime, c_prime/d_prime)
Beta_ik ~ Gamma(c, eta_i)

Y_ui ~ Poisson(Theta_u' Beta_i)
```
The parameters are fit using mean-field approximation (a form of Bayesian variational inference) with coordinate ascent (updating each parameter separately until convergence).

## Installation

Package is available on PyPI, can be installed with:

```
pip install hpfrec
```

As it contains Cython code, it requires a C compiler. In Windows, this usually means it requires a Visual Studio Build Tools installation (with MSVC140 component for `conda`) (or MinGW + GCC), and if using Anaconda, might also require configuring it to use said Visual Studio instead of MinGW, otherwise the installation from `pip` might fail. For more details see this guide:
[Cython Extensions On Windows](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows)

On Python 2.7 on Windows, it might additionally require installing extra Visual Basic modules (untested).

On Linux, the `pip` install should work out-of-the-box, as long as the system has `gcc`.

**Note for macOS users:** on macOS, the Python version of this package will compile **without** multi-threading capabilities. This is due to default apple's redistribution of clang not providing OpenMP modules, and aliasing it to gcc which causes confusions in build scripts. If you have a non-apple version of clang with the OpenMP modules, or if you have gcc installed, you can compile this package with multi-threading enabled by setting up an environment variable `ENABLE_OMP=1`:
```
export ENABLE_OMP=1
pip install hpfrec
```
(Alternatively, can also pass argument `enable-omp` to the setup.py file: `python setup.py install enable-omp`)

**Note2:** the setup script uses a PEP517 environment, which means it will create an isolated virtual environment, install its build dependencies there, compile, and then copy to the actual environment. This can causes issues - for example, if one has NumPy<1.20 and the build environment installs NumPy>=1.20, there will be a binary incompatibility which will make the package fail to import. To avoid PEP517, install with:
```
pip install --no-use-pep517 hpfrec
```

## Sample usage

```python
import pandas as pd, numpy as np
from hpfrec import HPF

## Generating sample counts data
nusers = 10**2
nitems = 10**2
nobs   = 10**4

np.random.seed(1)
counts_df = pd.DataFrame({
	'UserId' : np.random.randint(nusers, size=nobs),
	'ItemId' : np.random.randint(nitems, size=nobs),
	'Count' :  (np.random.gamma(1,1, size=nobs) + 1).astype('int32')
	})
counts_df = counts_df.loc[~counts_df[['UserId', 'ItemId']].duplicated()].reset_index(drop=True)

## Initializing the model object
recommender = HPF()

## For stochastic variational inference, need to select batch size (number of users)
recommender = HPF(users_per_batch = 20)

## Full function call
recommender = HPF(
	k=30, a=0.3, a_prime=0.3, b_prime=1.0,
	c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
	stop_crit='train-llk', check_every=10, stop_thr=1e-3,
	users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
	maxiter=100, use_float=True, reindex=True, verbose=True,
	random_seed=None, allow_inconsistent_math=False, full_llk=False,
	alloc_full_phi=False, keep_data=True, save_folder=None,
	produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
)

## Fitting the model to the data
recommender.fit(counts_df)

## Fitting the model while monitoring a validation set
recommender = HPF(stop_crit='val-llk')
recommender.fit(counts_df, val_set=counts_df.sample(10**2))
## Note: a real validation should NEVER be a subset of the training set

## Fitting the model to data in batches passed by the user
recommender = HPF(reindex=False, keep_data=False)
users_batch1 = np.unique(np.random.randint(10**2, size=20))
users_batch2 = np.unique(np.random.randint(10**2, size=20))
users_batch3 = np.unique(np.random.randint(10**2, size=20))
recommender.partial_fit(counts_df.loc[counts_df.UserId.isin(users_batch1)], nusers=10**2, nitems=10**2)
recommender.partial_fit(counts_df.loc[counts_df.UserId.isin(users_batch2)])
recommender.partial_fit(counts_df.loc[counts_df.UserId.isin(users_batch3)])

## Making predictions
# recommender.topN(user=10, n=10, exclude_seen=True) ## not available when using 'partial_fit'
recommender.topN(user=10, n=10, exclude_seen=False, items_pool=np.array([1,2,3,4]))
recommender.predict(user=10, item=11)
recommender.predict(user=[10,10,10], item=[1,2,3])
recommender.predict(user=[10,11,12], item=[4,5,6])

## Evaluating Poisson likelihood
recommender.eval_llk(counts_df, full_llk=True)

## Determining latent factors for a new user, given her item interactions
nobs_new = 20
np.random.seed(2)
counts_df_new = pd.DataFrame({
	'ItemId' : np.random.choice(np.arange(nitems), size=nobs_new, replace=False),
	'Count' : np.random.gamma(1,1, size=nobs_new).astype('int32')
	})
counts_df_new = counts_df_new.loc[counts_df_new.Count > 0].reset_index(drop=True)
recommender.predict_factors(counts_df_new)

## Adding a user without refitting the whole model
recommender.add_user(user_id=nusers+1, counts_df=counts_df_new)

## Updating data for an existing user without refitting the whole model
chosen_user = counts_df.UserId.values[10]
recommender.add_user(user_id=chosen_user, counts_df=counts_df_new, update_existing=True)
```

If passing `reindex=True`, all user and item IDs that you pass to `.fit` will be reindexed internally (they need to be hashable types like `str`, `int` or `tuple`), and you  can use these same IDs to make predictions later. The IDs returned by `predict` and `topN` are these IDs passed to `.fit` too.

For a more detailed example, see the IPython notebook [recommending songs with EchoNest MillionSong dataset](http://nbviewer.jupyter.org/github/david-cortes/hpfrec/blob/master/example/hpfrec_echonest.ipynb) illustrating its usage with the EchoNest TasteProfile dataset.

## Documentation

Documentation is available at readthedocs: [http://hpfrec.readthedocs.io](http://hpfrec.readthedocs.io/en/latest/)

It is also internally documented through docstrings (e.g. you can try `help(hpfrec.HPF))`, `help(hpfrec.HPF.fit)`, etc.

## Serializing (pickling) the model

Don't use `pickle` to save an `HPF` object, as it will fail due to problems with lambda functions. Rather, use `dill` instead, which has the same syntax as pickle:

```python
import dill
from hpfrec import HPF

h = HPF()
dill.dump(h, open("HPF_obj.dill", "wb"))
h = dill.load(open("HPF_obj.dill", "rb"))
```

## Speeding up optimization procedure

For faster fitting and predictions, use SciPy and NumPy libraries compiled against MKL or OpenBLAS. These come by default with MKL in Anaconda installations.

The constructor for HPF allows some parameters to make it run faster (if you know what you're doing): these are `allow_inconsistent_math=True`, `full_llk=False`, `stop_crit='diff-norm'`, `reindex=False`, `verbose=False`. See the documentation for more details.

Using stochastic variational inference, which fits the data in smaller batches containing all the user-item interactions only for subsets of users, might converge in fewer iterations (epochs), but the results tend be slightly worse.

## Troubleshooting

* Package uses only one CPU core: make sure that your C compiler supports OpenMP (both Visual Studio and GCC do in default installations, but with MinGW you might need additional modules).
* Error with `vcvarsall.bat`: see installation instructions (you need to configure your Python installation to use Visual Studio and set the correct paths to libraries). If you are using Python 2, try installing under a Python 3 environment instead and the problem might disappear.
* Parameters turn to NaN: you might have run into an unlucky parmeter initialization. Try using a different random seed, or changing the number of latent factors (`k`). If passing `reindex=False`, try changing to `reindex=True`.

The package has only been tested under Python 3.6.

## References
* [1] Gopalan, Prem, Jake M. Hofman, and David M. Blei. "Scalable Recommendation with Hierarchical Poisson Factorization." UAI. 2015.
* [2] Gopalan, Prem, Jake M. Hofman, and David M. Blei. "Scalable recommendation with poisson factorization." arXiv preprint arXiv:1311.1704 (2013).
* [3] Hoffman, Matthew D., et al. "Stochastic variational inference." The Journal of Machine Learning Research 14.1 (2013): 1303-1347.
