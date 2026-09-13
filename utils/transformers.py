""" """

from typing import Any, Dict

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class GroupedStandardScaler(BaseEstimator, TransformerMixin):
    """
    Applies StandardScaler independently to subsets of the data based on group labels.

    This transformer behaves like `sklearn.preprocessing.StandardScaler`, but instead
    of standardizing the entire dataset globally, it fits a separate scaler for each
    unique group provided in the `groups` array.

    Parameters
    ----------
    **scaler_kwargs : dict
        Keyword arguments passed directly to the underlying `sklearn.preprocessing.StandardScaler`
        (e.g., `with_mean=True`, `with_std=False`).

    Attributes
    ----------
    scalers_ : dict
        A dictionary mapping each unique group label to its fitted `StandardScaler` instance.
    """

    def __init__(self, **scaler_kwargs: Any) -> None:
        self.scaler_kwargs: Dict[str, Any] = scaler_kwargs
        self.scalers_: Dict[Any, StandardScaler] = {}

    def fit(self, X: np.ndarray, groups: np.ndarray) -> "GroupedStandardScaler":
        """
        Fits a separate StandardScaler for each unique group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-group mean and standard deviation.
        groups : array-like of shape (n_samples,)
            An array of group labels corresponding to each sample in `X`.

        Returns
        -------
        self : GroupedStandardScaler
            The fitted estimator.
        """
        X_arr = np.asarray(X)
        groups_arr = np.asarray(groups)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.scalers_ = {}
        unique_groups = np.unique(groups_arr)

        for group in unique_groups:
            mask = groups_arr == group
            scaler = StandardScaler(**self.scaler_kwargs)
            scaler.fit(X_arr[mask])
            self.scalers_[group] = scaler

        return self

    def transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted scaler for each corresponding group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be standardized.
        groups : array-like of shape (n_samples,)
            An array of group labels corresponding to each sample in `X`.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The standardized data.

        Raises
        ------
        ValueError
            If a group label in `groups` was not encountered during the `fit` step.
        """
        X_arr = np.asarray(X)
        groups_arr = np.asarray(groups)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_transformed = np.empty_like(X_arr, dtype=float)

        for group in np.unique(groups_arr):
            mask = groups_arr == group
            if group not in self.scalers_:
                raise ValueError(f"Group '{group}' was not seen during fit().")

            X_transformed[mask] = self.scalers_[group].transform(X_arr[mask])

        return X_transformed

    def fit_transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Fits the scalers to the groups and transforms the data in a single step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be standardized.
        groups : array-like of shape (n_samples,)
            An array of group labels corresponding to each sample in `X`.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The standardized data.
        """
        return self.fit(X, groups).transform(X, groups)


class RankGaussianTransformer:
    """
    Transforms data to follow a standard normal distribution using rank-based
    transformation.

    This transformer maps the input distribution to a uniform distribution
    (via ranks/quantiles) and then maps those quantiles to a standard normal
    distribution using the inverse cumulative distribution function (probit).

    Parameters
    ----------
    clip : bool, default=True
        If True, quantiles are clipped slightly inward from 0.0 and 1.0 to
        prevent the inverse normal calculation from producing infinity.

    Attributes
    ----------
    dims : tuple or None
        The shape of the input data seen during fitting.
    is_fit : bool
        Whether the transformer has been fitted.
    quantile_transformer : sklearn.preprocessing.QuantileTransformer
        The underlying transformer used to convert data to uniform quantiles.
    """

    def __init__(self, clip: bool = True) -> None:
        self.clip = clip
        self.dims = None
        self.is_fit = False
        self.quantile_transformer = QuantileTransformer()

    def _resize(self, x: np.ndarray) -> np.ndarray:
        """
        Reshapes the input array to a 2D format expected by the transformer.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            The reshaped array with shape (n_samples, 1).

        Raises
        ------
        ValueError
            If the input array has more than 2 dimensions.
        """
        if len(x.shape) == 1:
            x = x[:, None]
        elif len(x.shape) > 2:
            raise ValueError("Must provide (n, ) or (n, 1) array to `InverseNormalTransformer`")
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Fit the QuantileTransformer to the data.

        Parameters
        ----------
        x : np.ndarray
            The input data to fit, shape (n_samples, ) or (n_samples, 1).
        """
        self.dims = x.shape
        self.quantile_transformer.fit(self._resize(x))
        self.is_fit = True

    def _clip_quantiles(self, q: np.ndarray) -> np.ndarray:
        """
        Clips quantile values to a finite range to ensure numerical stability.

        This prevents the inverse survival function (norm.isf) from returning
        infinity when given 0.0 or 1.0.

        Parameters
        ----------
        q : np.ndarray
            Array of quantile values (between 0 and 1).

        Returns
        -------
        np.ndarray
            The clipped quantiles.
        """
        q_nonzero = q[(q > 0.0) & (q < 1.0)]
        upr = (1.0 + q_nonzero.max()) / 2.0
        lwr = q_nonzero.min() / 2.0
        q_clipped = np.minimum(np.maximum(q, lwr), upr)
        return q_clipped

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data to a standard normal distribution.

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data in the original shape.
        """
        assert self.is_fit, "`.fit()` method has not yet been called"
        quantiles = self.quantile_transformer.transform(self._resize(x))
        if self.clip:
            quantiles = self._clip_quantiles(quantiles)

        return -norm.isf(quantiles).reshape(self.dims)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        x : np.ndarray
            Input data to fit and transform.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Map data from the standard normal distribution back to the original distribution.

        Parameters
        ----------
        z : np.ndarray
            The gaussian-distributed data.

        Returns
        -------
        np.ndarray
            The data mapped back to the original feature space.
        """
        return self.quantile_transformer.inverse_transform(self._resize(norm.cdf(z))).reshape(self.dims)


class MultiColumnTransformer:
    """
    Applies a specific transformer class to each column of a 2D array independently.

    This wrapper iterates over the columns of the input array, creating and
    fitting a separate instance of the provided transformer class for each column.

    Parameters
    ----------
    transformer : type
        A class (not an instance) that implements standard fit/transform/inverse_transform
        methods (e.g., RankGaussianTransformer).

    Attributes
    ----------
    transformer_class : type
        The class constructor for the transformers.
    transformers : tuple
        A tuple containing the fitted transformer instances for each column.
    n_columns : int or None
        The number of columns observed during fit.
    """

    def __init__(self, transformer: Any) -> None:
        self.transformer_class = transformer
        self.transformers = []
        self.n_columns = None

    def fit(self, x: np.ndarray) -> None:
        """
        Fit a separate transformer for each column in the input.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).
        """
        assert len(x.shape) == 2, "`x` must have shape `(n, d)`"
        self.n_columns = x.shape[-1]
        self.transformers = tuple([self.transformer_class() for _ in range(self.n_columns)])
        for x, trf in zip(x.T, self.transformers):
            trf.fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform each column using its corresponding fitted transformer.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array of the same shape.
        """
        return np.vstack([trf.transform(x) for (x, trf) in zip(x.T, self.transformers)]).T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fit and transform the data in one step.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transform each column using its corresponding fitted transformer.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Inverse transformed array of the same shape.
        """
        return np.vstack([trf.inverse_transform(x) for (x, trf) in zip(x.T, self.transformers)]).T
