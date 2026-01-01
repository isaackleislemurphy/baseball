"""
Models for Expected Pitch Movement (xMovement).

This module implements a GP regression approach to establish baseline movement
expectations (horizontal and vertical break) based on a pitcher's
basic, physical release characteristics (arm angle and velocity).
"""

import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from baseball.data.savant.pitch.etl import load_pitch_data
from baseball.projects.pitch_quality.data.etl import partition_pitch_data
from baseball.utils.general import load_pickled_object, write_pickled_object

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XMVMT_OBJECT_PATH = os.path.join(SCRIPT_DIR.replace("submodels", "objects"), "xmvmt_models.pkl")

# Inputs: basic physical characteristics of the release
EXPECTED_MVMT_INPUTS = {
    "FF": ["arm_angle", "release_speed"],
    "SI": ["arm_angle", "release_speed"],
    "BB": ["arm_angle", "release_speed", "cos_spin_axis_pitcher_neutral", "sin_spin_axis_pitcher_neutral"],
    "OS": ["arm_angle", "release_speed"],
}
# have any/all xMvmt feature, for use in pandas column slicing and munging
EXPECTED_MVMT_INPUTS_UNION = sorted(list(set().union(*EXPECTED_MVMT_INPUTS.values())))
# EXPECTED_MVMT_INPUTS = ["arm_angle", "release_speed"]

# Outputs: basic movement metrics to model. No SSW for now.
EXPECTED_MVMT_OUTPUTS = ["pfx_z", "pfx_x_pitcher_neutral"]

# Minimum sample size to include a pitcher-season in the training set
MIN_EXPECTED_MVMT_PITCHES = 25


class ExpectedMovement:
    """
    A model suite for predicting expected pitch movement based on release characteristics.

    This class trains separate Gaussian Process Regressors for specific pitch types
    (for now, FF and SI) to predict horizontal and vertical movement
    based on arm angle and release speed.

    Attributes
    ----------
    inputs : list[str]
        The feature columns used for prediction (e.g., ['arm_angle', 'release_speed']).
    outputs : list[str]
        The target columns to predict (e.g., ['pfx_x_pitcher_neutral', 'pfx_z']).
    pitch_types : tuple[str]
        The specific pitch types to model (e.g., ("FF", "SI")).
    scalers : dict
        A dictionary mapping (pitch_type, output_metric) tuples to fitted StandardScaler instances.
    fits : dict
        A dictionary mapping (pitch_type, output_metric) tuples to fitted GaussianProcessRegressor instances.
    """

    def __init__(
        self,
        pitch_group_types: tuple[str] = ("BB", "SI", "FF", "OS"),
        kernel: Kernel = (
            RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e1))
        ),
        random_state: int = 2026,
    ) -> None:
        """
        Initialize the ExpectedMovement model suite.

        Parameters
        ----------
        pitch_group_types : tuple[str], default ("BB", "SI", "FF", "OS")
            Pitch groups to model. Each group gets its own set of Gaussian Process
            regressors for each movement output. Groups should align with keys in
            `EXPECTED_MVMT_INPUTS`.
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel used for all Gaussian Process regressors. Passed through directly
            to sklearn's `GaussianProcessRegressor`.
        random_state : int, default 2026
            Random seed for optimizer restarts and reproducibility.
        """
        self.inputs = EXPECTED_MVMT_INPUTS
        self.outputs = EXPECTED_MVMT_OUTPUTS
        self.pitch_group_types = pitch_group_types

        # Initialize scalers for input features
        self.scalers = {
            (pitch_type, output): StandardScaler()
            for pitch_type, output in itertools.product(self.pitch_group_types, self.outputs)
        }

        # Initialize GPR models
        # NOTE: Updated to use the `kernel` argument passed to __init__ rather than hardcoding.
        self.fits = {
            (pitch_type, output): GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=random_state,
                n_restarts_optimizer=10,
            )
            for pitch_type, output in itertools.product(self.pitch_group_types, self.outputs)
        }

    def _assign_xmvmt_pitch_group(self, pitch_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign pitch-level rows to xMovement pitch groups.

        Fastballs (FF, SI) are kept distinct, while all other pitches fall back to
        their broader `pitch_group` classification. This determines which submodel
        is used during training and prediction.

        Parameters
        ----------
        pitch_data_df : pd.DataFrame
            Pitch-level or aggregated pitch data containing `pitch_type` and
            `pitch_group` columns.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with an added `xmvmt_pitch_group` column.
        """
        pitch_data_df["xmvmt_pitch_group"] = np.where(
            pitch_data_df["pitch_type"].isin(("FF", "SI")).values,
            pitch_data_df["pitch_type"].values,
            pitch_data_df["pitch_group"].values,
        )
        return pitch_data_df

    def _aggregate_pitch_data(
        self, pitch_data_df: pd.DataFrame, min_pitches: int = MIN_EXPECTED_MVMT_PITCHES
    ) -> pd.DataFrame:
        """
        Aggregate pitch-level data to pitcher-season-pitch-group averages.

        Training is performed on aggregated pitcher-season data rather than individual
        pitches.

        Notably, by averaging down to player-season, I'm cutting corners. But this
        _significantly_ reduces training time and  necessary compute; additionally, not wholly
        unreasonable since batter expectations  of movement relative to speed / slot might reflect
        an aggregation, rather than a pitch-by-pitch level memory. For instance, a batter
        might associate Hoby Milner's general SI shape with his general slot.

        Parameters
        ----------
        pitch_data_df : pd.DataFrame
            Raw Statcast pitch-level data.
        min_pitches : int, default 25
            Minimum number of pitches required for a pitcher-season-pitch-group
            to be included in the training set.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with one row per pitcher, season, pitch group,
            and pitch type.
        """

        # aggregations
        _agg_fns = {item: np.mean for item in EXPECTED_MVMT_INPUTS_UNION + EXPECTED_MVMT_OUTPUTS}
        _agg_fns.update({"n_pitches": np.sum})

        # average over movement, slot, and velo
        pitch_type_avg_df = (
            pitch_data_df.dropna(subset=EXPECTED_MVMT_INPUTS_UNION + EXPECTED_MVMT_OUTPUTS)
            .assign(n_pitches=1)
            .groupby(["pitcher", "season", "pitch_group", "pitch_type"], as_index=False)
            .agg(_agg_fns)
        )
        pitch_type_avg_df = pitch_type_avg_df.query(f"n_pitches >= {min_pitches}")
        return pitch_type_avg_df

    def fit(self, pitch_data_df: pd.DataFrame) -> None:
        """
        Fit models for each pitch group and movement output.

        The method aggregates pitch-level data to pitcher-season averages, scales
        inputs using StandardScalers, and fits a separate GP for each
        (pitch_group, movement_output) pair.

        Parameters
        ----------
        pitch_data_df : pd.DataFrame
            Raw pitch-level Statcast data containing required input features
            and movement outputs.
        """
        pitch_type_avg_df = self._aggregate_pitch_data(pitch_data_df)
        pitch_type_avg_df = self._assign_xmvmt_pitch_group(pitch_type_avg_df)

        # iterate through pitch types and mappings
        for pitch_type in self.pitch_group_types:
            for output in self.outputs:
                print(f"...fitting {self.inputs[pitch_type]} --> {output} for {pitch_type}s...")

                # filter to relevant pitch type
                pitch_type_avg_df_ = pitch_type_avg_df.loc[pitch_type_avg_df["xmvmt_pitch_group"] == pitch_type]

                if pitch_type_avg_df_.empty:
                    print(f"   Warning: No data found for {pitch_type}. Skipping.")
                    continue

                # matrix time
                X = pitch_type_avg_df_[self.inputs[pitch_type]].values.astype(float)
                y = pitch_type_avg_df_[output].values.astype(float)

                # scale inputs
                X = self.scalers[(pitch_type, output)].fit_transform(X)

                # fit model
                self.fits[(pitch_type, output)].fit(X, y)

    def _predict(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate expected movement predictions for a batch of pitches.

        This is an internal method intended for chunked prediction. It applies the
        appropriate submodel based on pitch group, scales inputs using pre-fit
        scalers, and returns expected horizontal and vertical movement estimates.

        This method will happily attempt to predict on arbitrarily large DataFrames,
        which can be very slow for Gaussian Processes; use the public `predict()`
        wrapper for chunking and memory control.

        Parameters
        ----------
        pred_df : pd.DataFrame
            Pitch-level data to predict on. Must contain all required input
            features for the relevant pitch group and a `pitch_type` column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with expected movement columns:
            `x_pfx_z` and `x_pfx_x_pitcher_neutral`.

        Notes
        -----
        Missing input features are filled with 0.0 *after scaling*, which corresponds
        to mean imputation in standardized space. This primarily handles occasional
        tracking gaps (notably early Hawkeye seasons).
        """
        # begin with placeholders and nans
        y_hat = np.nan * np.zeros((pred_df.shape[0], len(self.outputs)))

        # add in xmvmt pitch group
        pred_df = self._assign_xmvmt_pitch_group(pred_df)

        for pitch_type in self.pitch_group_types:
            for col, output in enumerate(self.outputs):

                # extract the xmovement inputs, and scale them
                x_hat = self.scalers[(pitch_type, output)].transform(pred_df[self.inputs[pitch_type]].values)

                # fill with 0's (mostly a 2020 Hawkeye thing)
                x_hat = np.where(np.isnan(x_hat), 0.0, x_hat)

                # indices for the relevant pitch type
                pitch_type_idx = np.where(pred_df["pitch_type"].values == pitch_type)[0]

                # predict for rows where it's the right pitch type
                y_hat[pitch_type_idx, col] = self.fits[(pitch_type, output)].predict(x_hat[pitch_type_idx])

        y_hat = pd.DataFrame(y_hat, columns=["x_" + item for item in self.outputs], index=pred_df.index)
        return pd.concat([pred_df, y_hat], axis=1)

    def predict(self, pred_df: pd.DataFrame, chunk_size: int = 5_000, use_parallel: bool = False) -> pd.DataFrame:
        """
        Public-facing prediction method with chunking and optional parallelism.

        Wraps the internal `._predict()` method to safely handle large prediction
        jobs by processing the data in chunks. Parallel execution is supported
        via joblib for very large datasets.

        Parameters
        ----------
        pred_df : pd.DataFrame
            Pitch-level data to predict on.
        chunk_size : int, default 5_000
            Number of rows per prediction chunk.
        use_parallel : bool, default False
            If True, performs chunked prediction in parallel using joblib.

        Returns
        -------
        pd.DataFrame
            Pitch-level DataFrame augmented with expected movement estimates.
        """
        if use_parallel:
            _predict_df = lambda i: self._predict(pred_df.iloc[i : i + chunk_size])
            return Parallel(n_jobs=-1, verbose=2)(
                delayed(_predict_df)(i) for i in range(0, pred_df.shape[0], chunk_size)
            )
        else:
            return pd.concat(
                [self._predict(pred_df.iloc[i : i + chunk_size]) for i in trange(0, pred_df.shape[0], chunk_size)],
                axis=0,
            ).reset_index(drop=True)


def train_expected_movement_models() -> ExpectedMovement:
    """
    Train the Expected Movement (xMovement) models on historical training data.

    This function performs the following steps:
    1. Loads cached pitch data.
    2. Filters the data to include only games occurring on or before `TRAIN_TEST_CUTOFF_DATE`.
    3. Restricts the data to the specific subset of pitchers designated for training
       (via `load_training_partitions`) to prevent data leakage.
    4. Instantiates and fits the `ExpectedMovement` Gaussian Process models.

    Returns
    -------
    ExpectedMovement
        The fitted ExpectedMovement model suite, ready for prediction or serialization.
    """

    # load pitch data
    pitch_data_df_train = load_pitch_data(load_from_cache=True)

    # trim it down to only training data
    pitch_data_df_train, _, _ = partition_pitch_data(pitch_data_df_train)

    # instantiate xmovement models
    xmvmt_models = ExpectedMovement()
    print("xMovement models instantiated.")

    # fit 'em
    xmvmt_models.fit(pitch_data_df_train)
    print("xMovement models fit.")

    return xmvmt_models


def cache_expected_movement_models() -> None:
    """
    Train and serialize the Expected Movement models to disk.

    This acts as the main entry point for the training pipeline. It triggers the
    training process via `train_expected_movement_models` and pickles the resulting
    object to the path defined in `XMVMT_OBJECT_PATH`.

    Returns
    -------
    None
    """
    xmvmt_models = train_expected_movement_models()
    write_pickled_object(xmvmt_models, XMVMT_OBJECT_PATH)
    print(f"Expected movement model saved along: {XMVMT_OBJECT_PATH}")


def load_expected_movement_models() -> ExpectedMovement:
    """
    Loads a trained expected movement model suite.

    Returns
    -------
    ExpectedMovement
        An instantiated and fitted `ExpectedMovement` object.
    """
    return load_pickled_object(XMVMT_OBJECT_PATH)


if __name__ == "__main__":
    # load_pitch_data(save_to_cache=True, load_from_cache=False)
    cache_expected_movement_models()
