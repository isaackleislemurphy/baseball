"""
Hyperparamter tuning using GPs as a emulator model to approximate the loss landscape of an estimator.
"""

from tqdm import trange
from typing import Any, Callable, Dict, Literal, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Kernel


def _support(x: float, p_xh: Any) -> float:
    """
    Clamp a value within the support of a given distribution.

    Parameters
    ----------
    x : float
        The input value to clamp.
    p_xh : Any
        A scipy.stats distribution object (or similar) that implements a
        `.support()` method returning a tuple (min, max).

    Returns
    -------
    float
        The value `x` clamped between the lower and upper bounds of `p_xh`.
    """
    return min(max(x, p_xh.support()[0]), p_xh.support()[1])


class GPHPTuner:
    """
    Hyperparameter tuner using Gaussian Process emulator model.

    This class performs hyperparameter optimization by maintaining a Gaussian Process (GP)
    emulator model. It approximates the relationship between hyperparameters (`xh`) and
    cross-validation losses (`yl`).

    Note that it supports both continuous and discrete parameters, though that's probably
    overstating the discrete compatibilites..all that's going on there is that it optimizes
    these discrete parameters in the real space, and then rounds them off at the end.

    TODOS:
        [ ] `.predict_proba()` support
        [ ] custom KFold support

    Attributes
    ----------
    approx : sklearn.gaussian_process.GaussianProcessRegressor
        The internal Gaussian Process emulator model used to approximate the loss landscape.
    dim : int
        The total number of hyperparameters (continuous + discrete) being optimized.
    dim_continuous : int
        The number of continuous hyperparameters.
    dim_discrete : int
        The number of discrete hyperparameters.
    estimator : Any
        The **uninstantiated** sklearn-compatible estimator provided during initialization.
    kfold : sklearn.model_selection.KFold
        The cross-validation splitting strategy initialized with the specified `cv` count.
    loss_fn : Callable
        The scoring function (wrapped via `sklearn.metrics.make_scorer`) used to evaluate
        model performance.
    params_continuous : Dict[str, Any]
        The dictionary mapping continuous hyperparameter names to their distribution objects.
    params_discrete : Dict[str, Any]
        The dictionary mapping discrete hyperparameter names to their distribution objects.
    param_names_continuous : Tuple[str, ...]
        A sorted tuple of the continuous hyperparameter names, used to ensure consistent
        indexing in the internal arrays.
    param_names_discrete : Tuple[str, ...]
        A sorted tuple of the discrete hyperparameter names, used to ensure consistent
        indexing in the internal arrays.
    xh_continuous : np.ndarray
        History of continuous hyperparameter values tried so far.
        Shape: `(n_trials, n_continuous_params)`.
    xh_discrete : np.ndarray
        History of discrete hyperparameter values tried so far.
        Shape: `(n_trials, n_discrete_params)`.
    yl : np.ndarray
        History of observed cross-validated loss values corresponding to the parameters in
        `xh_continuous` and `xh_discrete`. Shape: `(n_trials,)`.
    _mu_xh : np.ndarray
        Internal cache of the mean values of the prior distributions. Used to scale raw
        hyperparameters into the approximation space (Z-scoring).
    _sigma_xh : np.ndarray
        Internal cache of the standard deviation values of the prior distributions. Used
        to scale raw hyperparameters into the approximation space.
    """

    def __init__(
        self,
        estimator: Any,
        params_continuous: Dict[str, Any],
        params_discrete: Dict[str, Any],
        kernel: Kernel = RationalQuadratic() + WhiteKernel(noise_level=1e-4),
        cv: int = 20,
        loss_fn: Callable = mean_absolute_error,
    ) -> None:
        """
        Parameters
        ----------
        estimator : Any
            An unfitted sklearn-compatible estimator that implements `fit` and `predict`.
        params_continuous : Dict[str, Any]
            A dictionary mapping continuous hyperparameter names to scipy.stats distribution objects.
            The distributions get used to make the draws for the random search; good chance these are
            uniform but to each their own.
        params_discrete : Dict[str, Any]
            A dictionary mapping discrete hyperparameter names to scipy.stats distribution objects.
            The distributions get used to make the draws for the random search; good chance these are
            uniform but to each their own.
        kernel : sklearn.gaussian_process.kernels.Kernel, optional
            The kernel to use for the Gaussian Process Regressor.
            Default is RationalQuadratic() + WhiteKernel(noise_level=1e-4).
        cv : int, default=20
            The number of splits for Cross-Validation.
        loss_fn : Callable, default=sklearn.metrics.mean_absolute_error
            The loss function to minimize. Must be compatible with sklearn's `make_scorer`.
        """

        self.estimator = estimator
        self.kfold = KFold(n_splits=cv, shuffle=True, random_state=None)
        self.loss_fn = make_scorer(loss_fn)

        # everything related to discrete params goes in here
        self.params_discrete = params_discrete  # discrete hyparameter names
        self.param_names_discrete = tuple(sorted(params_discrete.keys()))
        self.xh_discrete = []  # discrete hyperparameter values

        # everything related to continuous params goes in here
        self.params_continuous = params_continuous
        self.param_names_continuous = tuple(sorted(params_continuous.keys()))
        self.xh_continuous = []

        # store dimensions
        self.dim_continuous = len(params_continuous)
        self.dim_discrete = len(params_discrete)
        self.dim = len(params_continuous) + len(params_discrete)

        # cache the h-scaling statistics
        self._calculate_xh_scaling_statistics()

        # instantiate a GP
        self.approx = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
        )

        self.yl = []

    def _propose_random_discrete_param_config(self, random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Generate a random configuration for the discrete parameters.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        Dict[str, float]
            A dictionary of discrete parameter names and randomly sampled values, i.e.
            ```
            {discrete_hyperparam: value}
            ```
        Note that hyperparameters are merely rounded, not casted to integers, since they'll need to
        remain floats when subsequently and imminently plugged into the GP emulator
        """
        return {param: round(dist.rvs(random_state=random_state)) for param, dist in self.params_discrete.items()}

    def _propose_random_continuous_param_config(self, random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Generate a random configuration for the continuous parameters.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        Dict
            A dictionary of continuous parameter names and randomly sampled values, i.e.
            ```
            {continuous_hyperparam: value}
            ```
        """
        return {param: dist.rvs(random_state=random_state) for param, dist in self.params_continuous.items()}

    def _propose_random_param_config(self, random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Generate a random configuration for all parameters (discrete and continuous).

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        Dict
            A dictionary containing one full set of hyperparameters.
        """
        return {
            **self._propose_random_discrete_param_config(random_state=random_state),
            **self._propose_random_continuous_param_config(random_state=random_state),
        }

    def sample_random_point(self, **kwargs: Dict) -> np.ndarray:
        """
        Sample a random point from the hyperparameter prior distributions.

        Parameters
        ----------
        **kwargs : Dict
            Keyword arguments passed to the internal proposal methods
            (e.g., `random_state`).

        Returns
        -------
        np.ndarray
            A 1D array representing the random point. Continuous parameters
            come first, followed by discrete parameters.
        """
        params = self._propose_random_param_config(**kwargs)
        random_pt = [params[item] for item in self.param_names_continuous + self.param_names_discrete]
        return np.array(random_pt).astype(float)

    def _calculate_xh_scaling_statistics(self) -> np.ndarray:
        """
        Calculate and cache the mean and standard deviation of the prior distributions.

        These statistics (`_mu_xh` and `_sigma_xh`) are used to scale the
        raw hyperparameters (`h`) into the approximation space inputs (`xh`).
        """
        # mean
        self._mu_xh = np.array(
            [self.params_continuous[item].mean() for item in self.param_names_continuous]
            + [self.params_discrete[item].mean() for item in self.param_names_discrete]
        )[
            None, :
        ]  # (1, P)

        # standard deviation
        self._sigma_xh = np.array(
            [self.params_continuous[item].std() for item in self.param_names_continuous]
            + [self.params_discrete[item].std() for item in self.param_names_discrete]
        )[
            None, :
        ]  # (1, P)

    def _cv_score(self, x: np.ndarray, y: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the Cross-Validation score for a set of hyperparameters.

        Parameters
        ----------
        x : np.ndarray
            The training features (inputs).
        y : np.ndarray
            The training targets (responses).
        params : Dict
            The hyperparameter configuration to evaluate.

        Returns
        -------
        np.ndarray
            Array of scores from cross-validation.
        """
        return cross_val_score(self.estimator(**params), x, y, cv=self.kfold, scoring=self.loss_fn, n_jobs=-1)

    def _scale_xh(self, h: np.ndarray) -> np.ndarray:
        """
        Scale raw hyperparameters (`h`) to the approximation space (`xh`). Plain old z-scoring before you
        stuff it into a GP.

        Parameters
        ----------
        h : np.ndarray
            Raw hyperparameter values.

        Returns
        -------
        np.ndarray
            Standardized hyperparameters (Z-scores based on priors).
        """
        return (h - self._mu_xh) / self._sigma_xh

    def _unscale_xh(self, h: np.ndarray) -> np.ndarray:
        """
        Unscale approximation space inputs (`xh`) back to raw hyperparameters.

        Parameters
        ----------
        h : np.ndarray
            Scaled hyperparameter values (approximation space inputs).

        Returns
        -------
        np.ndarray
            Raw hyperparameter values in their original scale.
        """
        return h * self._sigma_xh + self._mu_xh

    def _get_xh(self) -> np.ndarray:
        """
        Retrieve and scale the history of hyperparameters.

        Returns
        -------
        np.ndarray
            The inputs for the approximation space (`xh`).
            These are the concatenated continuous (left) and discrete histories (right), scaled.
        """
        h = np.hstack([self.xh_continuous, self.xh_discrete])
        return self._scale_xh(h)

    def _get_yl(self) -> np.ndarray:
        """
        Retrieve the history of observed CV losses.

        Returns
        -------
        np.ndarray
            The history of observed CV losses. Used as a target for the emulator.
        """
        y = np.array(self.yl)
        return y

    def _fit_gp(self) -> None:
        """
        Fit the Gaussian Process surrogate model using current `xh` and `yl`.
        """
        xh, yh = self._get_xh(), self._get_yl()
        self.approx.fit(xh, yh)

    def _predict_gp(self, xh: np.ndarray) -> float:
        """
        Predict the loss for given scaled inputs using the emulator.

        Parameters
        ----------
        xh : np.ndarray
            Inputs in the approximation space (scaled hyperparameters).

        Returns
        -------
        float
            Predicted CV loss value.
        """
        xh = np.array(xh).astype(float)
        if len(xh.shape) == 1:
            xh = xh[None, :]
        yl_hat = self.approx.predict(xh, return_std=False)
        return yl_hat

    def _run_random_search(self, x: np.ndarray, y: np.ndarray, n_iter: int = 50) -> None:
        """
        Execute a random search (burn-in) phase to initialize the history. Gotta give the
        emulator something to train on.

        Parameters
        ----------
        x : np.ndarray
            The training features.
        y : np.ndarray
            The training targets.
        n_iter : int, default=50
            Number of random configurations to evaluate.
        """
        # do a random search of `n_iter` points
        for _ in trange(n_iter):
            # randomly propose some parameters
            params_iter = self._propose_random_param_config()
            # CV-scoring under those parameters
            yl_iter = self._cv_score(x, y, params_iter)
            # update the inputs, for eventual use in the GP
            self.xh_continuous += [[params_iter[item] for item in self.param_names_continuous]]
            self.xh_discrete += [[params_iter[item] for item in self.param_names_discrete]]
            self.yl += [np.mean(yl_iter)]
        # stack up the arrays
        self.xh_continuous = np.vstack(self.xh_continuous)
        self.xh_discrete = np.vstack(self.xh_discrete)
        self.yl = np.array(self.yl)

    def _random_search_approximation(self, n_draws: int = 10_000) -> np.ndarray:
        """
        Propose a new point by:
            (i) randomly sampling a bunch of inputs
            (ii) plugging them all into the approximator
            (iii) taking that point which has the lowest approximated CV loss.

        This strategy corresponds to the 'stochastic' option in `minimization_strategy`.

        Note/thoughts: you probably want to optimize over the GP's surface, but leaving this here
        as a sanity checking mechanism. Also, maybe there's some epsilon-greedy approach to mix
        this  in with LBFGS GP minimizations in `_minimize_approximation`, to every once in
        awhile give yourself an extra chance to hop out of a local minima??? Restarts should
        have you covered though.

        Parameters
        ----------
        n_draws : int, default=10_000
            Number of points to sample from the approximation space.

        Returns
        -------
        np.ndarray
            The hyperparameter configuration (unscaled) that minimizes the
            predicted loss on the GP surface among the sampled points.
        """
        # propose a bunch of hyperparam candidates (on the z-scale)
        param_draws = norm().rvs((250_000, self.dim))  # param here
        # use the GP to approximate their loss
        approx_yl = self._predict_gp(param_draws)
        # take the best point, and bump it back to the scale of the original parameter space
        new_params = self._unscale_xh(param_draws[np.argmin(approx_yl)])

        return new_params

    def _minimize_approximation(self, n_gp_restarts: int) -> np.ndarray:
        """
        Propose a new point by minimizing the GP approximation surface via L-BFGS-B.

        Parameters
        ----------
        n_gp_restarts : int
            Number of times to restart the optimizer with different random seeds
            to avoid local minima in the approximation surface.

        Returns
        -------
        np.ndarray
            The hyperparameter configuration (unscaled) that minimizes the
            GP surface.
        """
        best_loss, new_params = np.inf, None
        for _ in range(n_gp_restarts):
            opt_attempt = minimize(self._predict_gp, norm().rvs(self.dim), method="L-BFGS-B")
            if opt_attempt.fun < best_loss:
                best_loss = opt_attempt.fun
                new_params = self._unscale_xh(opt_attempt.x)
        return new_params

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_random_burn_in: int = 50,
        n_gp_iter: int = 25,
        n_gp_restarts: int = 10,
        minimization_strategy: Literal["lbfgs", "stochastic"] = "lbfgs",
    ) -> None:
        """
        Run the Gaussian Process Hyperparameter Tuning loop.

        This process consists of two phases:
        1. A random burn-in phase to gather initial statistics.
        2. An iterative phase where the GP is fitted to existing history,
           and new points are proposed by minimizing the GP surface.

        Parameters
        ----------
        x : np.ndarray
            The training features. Note that these are the _actual_ model inputs.
        y : np.ndarray
            The training targets. Note that these are the _actual_ model responses.
        n_random_burn_in : int, default=50
            Number of random points to evaluate before starting GP optimization.
        n_gp_iter : int, default=25
            Number of Bayesian optimization iterations (GP fit + proposal + evaluation).
        n_gp_restarts : int, default=10
            Number of restarts for the internal optimizer when finding the minimum
            of the GP surface (only used if strategy is "lbfgs").
        minimization_strategy : Literal["lbfgs", "stochastic"], default="lbfgs"
            Strategy to minimize the GP acquisition function:
            - "lbfgs": Uses scipy.optimize.minimize (L-BFGS-B).
            - "stochastic": Samples many points and picks the minimum.
        """
        # randomly search `n_random_burn_in` points; you'll start your GP approximation training on this.
        self._run_random_search(x, y, n_iter=n_random_burn_in)

        # fit the initial approx gp
        self._fit_gp()

        for _ in trange(n_gp_iter):
            # -----------------------------------------------------------------------------------
            # 1.) Propose a new hyperparamter configuration, either through optimization
            # or just searching a zillion points on the approximation function that we wouldn't
            # otherwise have time to cross-validate
            # -----------------------------------------------------------------------------------

            if minimization_strategy == "lbfgs":
                new_params = self._minimize_approximation(n_gp_restarts=n_gp_restarts)
            elif minimization_strategy == "stochastic":
                new_params = self._random_search_approximation()
            else:
                raise ValueError("`minimization_strategy` must be one of 'lbfgs', 'stochastic'")

            # -----------------------------------------------------------------------------------
            # 2.) Ensure hyperparams are on their provided support (and rounded if discrete)
            # run through continuous
            # -----------------------------------------------------------------------------------
            for idx in range(self.dim_continuous):
                new_params[:, idx] = _support(
                    new_params[:, idx], self.params_continuous[self.param_names_continuous[idx]]
                )
            # run through discrete
            offset = self.dim_continuous
            for idx in range(self.dim_discrete):
                # make sure everything is supported
                new_params[:, offset + idx] = _support(
                    new_params[:, offset + idx], self.params_discrete[self.param_names_discrete[idx]]
                )
                # round off discrete params
                new_params[:, offset + idx] = np.round(new_params[:, offset + idx])

            # -----------------------------------------------------------------------------------
            # 3.) Do a round of cross-validation with the hyperparameters
            # pull into Sklearn-friendly dict
            # -----------------------------------------------------------------------------------
            params_iter = {
                **{param: new_params[:, i].item() for i, param in enumerate(self.param_names_continuous)},
                **{
                    param: int(new_params[:, self.dim_continuous + i].item())
                    for i, param in enumerate(self.param_names_discrete)
                },
            }
            # score the model with the new, proposed params
            yl_iter = self._cv_score(x, y, params_iter)

            # -----------------------------------------------------------------------------------
            # 4.) Update inputs/outputs for GP approximation
            # -----------------------------------------------------------------------------------
            self.yl = np.append(self.yl, yl_iter.mean())
            self.xh_continuous = np.vstack([self.xh_continuous, new_params[:, : self.dim_continuous]])
            self.xh_discrete = np.vstack([self.xh_discrete, new_params[:, -self.dim_discrete :]])

            # -----------------------------------------------------------------------------------
            # 5.) Retrain the GP approximator
            # -----------------------------------------------------------------------------------
            self._fit_gp()

    def fit_summary(self) -> pd.DataFrame:
        """ """
        param_summary = pd.DataFrame(
            self._unscale_xh(self._get_xh()), columns=self.param_names_continuous + self.param_names_discrete
        )
        param_summary.iloc[:, -self.dim_discrete :] = param_summary.iloc[:, -self.dim_discrete :].astype(int)
        param_summary[self.loss_fn._score_func.__name__] = self.yl
        return param_summary
