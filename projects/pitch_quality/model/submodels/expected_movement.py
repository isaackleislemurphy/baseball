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
EXPECTED_MVMT_INPUTS = ["arm_angle", "release_speed"]

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
        pitch_types: tuple[str] = ("SI", "FF"),
        kernel: Kernel = (
            RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e1))
        ),
        random_state: int = 2026,
    ) -> None:
        """
        Initialize the ExpectedMovement model structure.

        Parameters
        ----------
        pitch_types : tuple[str], default ("FF", "SI")
            The Statcast pitch type codes to include in the model.
        kernel : sklearn.gaussian_process.kernels.Kernel, optional
            The kernel to use for the Gaussian Process. Defaults to RationalQuadratic + WhiteKernel.
        random_state : int, default 2025
            Seed for reproducibility.
        """
        self.inputs = EXPECTED_MVMT_INPUTS
        self.outputs = EXPECTED_MVMT_OUTPUTS
        self.pitch_types = pitch_types

        # Initialize scalers for input features
        self.scalers = {
            (pitch_type, output): StandardScaler()
            for pitch_type, output in itertools.product(self.pitch_types, self.outputs)
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
            for pitch_type, output in itertools.product(self.pitch_types, self.outputs)
        }

    def _aggregate_pitch_data(
        self, pitch_data_df: pd.DataFrame, min_pitches: int = MIN_EXPECTED_MVMT_PITCHES
    ) -> pd.DataFrame:
        """
        Aggregate raw pitch-level data into pitcher-season-pitch_type averages.

        This reduces noise by training on the average characteristics of a pitcher's
        season rather than individual pitches.

        Parameters
        ----------
        pitch_data_df : pd.DataFrame
            Raw Statcast pitch data.
        min_pitches : int, default 25
            The minimum number of pitches of a specific type a pitcher must throw
            in a season to be included in the training set.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with one row per pitcher per season per pitch type.
        """
        # aggregations
        _agg_fns = {item: np.mean for item in EXPECTED_MVMT_INPUTS + EXPECTED_MVMT_OUTPUTS}
        _agg_fns.update({"n_pitches": np.sum})

        # average over movement, slot, and velo
        pitch_type_avg_df = (
            pitch_data_df.dropna(subset=EXPECTED_MVMT_INPUTS + EXPECTED_MVMT_OUTPUTS)
            .assign(n_pitches=1)
            .groupby(["pitcher", "season", "pitch_type"], as_index=False)
            .agg(_agg_fns)
        )
        pitch_type_avg_df = pitch_type_avg_df.query(f"n_pitches >= {min_pitches}")

        return pitch_type_avg_df

    def fit(self, pitch_data_df: pd.DataFrame) -> None:
        """
        Train the Gaussian Process models on the provided pitch data.

        The data is first aggregated by pitcher-season to create stable averages,
        normalized using StandardScalers, and then fitted to the GPR models.

        Parameters
        ----------
        pitch_data_df : pd.DataFrame
            Raw Statcast pitch data containing input features and target metrics.
        """
        pitch_type_avg_df = self._aggregate_pitch_data(pitch_data_df)

        # iterate through pitch types and mappings
        for pitch_type in self.pitch_types:
            for output in self.outputs:
                print(f"...fitting {self.inputs} --> {output} for {pitch_type}s...")

                # filter to relevant pitch type
                pitch_type_avg_df_ = pitch_type_avg_df.loc[pitch_type_avg_df["pitch_type"] == pitch_type]

                if pitch_type_avg_df_.empty:
                    print(f"   Warning: No data found for {pitch_type}. Skipping.")
                    continue

                # matrix time
                X = pitch_type_avg_df_[self.inputs].values.astype(float)
                y = pitch_type_avg_df_[output].values.astype(float)

                # scale inputs
                X = self.scalers[(pitch_type, output)].fit_transform(X)

                # fit model
                self.fits[(pitch_type, output)].fit(X, y)

    def _predict(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate expected movement predictions for the provided pitch data.

        This method iterates through the configured pitch types, scales the input features,
        and queries the trained Gaussian Process models to calculate expected movement. Note that
        it'll predict on any size `pred_df` you give it, so it can be really slow. That's why it's
        an internal method, and the external-facing `.predict()` automatically chunks things for memory
        management.

        Parameters
        ----------
        pred_df : pd.DataFrame
            The input DataFrame containing the pitch data to predict on.
            Must contain the feature columns specified in `self.inputs` (e.g., 'arm_angle', 'release_speed')
            and the 'pitch_type' column to route predictions to the correct submodel.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted "expected" movement values.
            - Columns are prefixed with "x_" (e.g., 'x_pfx_z').
            - The index matches the index of the input `pred_df`.
            - Rows corresponding to pitch types not included in `self.pitch_types` will contain NaNs.

        Notes
        -----
        - **Missing Data Handling:** If input features contain NaNs (e.g., missing arm angle),
          they are filled with 0.0 after scaling. Since the data is standardized, this imputes
          the missing value with the population mean, handling occasional gaps in tracking data
          (mainly 2020 Hawkeye issues).
        """
        # begin with placeholders and nans
        y_hat = np.nan * np.zeros((pred_df.shape[0], len(self.outputs)))

        for pitch_type in self.pitch_types:
            for col, output in enumerate(self.outputs):

                # extract the xmovement inputs, and scale them
                x_hat = self.scalers[(pitch_type, output)].transform(pred_df[self.inputs].values)

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
        Wrapper around `._predict()` for public-facing `predict()` method. Idea here is to do things in chunks,
        since matrix ops for a zillion unchunked rows w/ a GP will get huge (no idea why sklearn hasn't built
        this in)

        Parameters
        ----------
        pred_df : pd.DataFrame
            The input DataFrame containing the pitch data to predict on.
            Must contain the feature columns specified in `self.inputs` (e.g., 'arm_angle', 'release_speed')
            and the 'pitch_type' column to route predictions to the correct submodel.
        chunk_size : int, default=5_000
            Chunk size for GP prediction, for memory management. Default is 5K.
        use_parallel : bool, default=False
            If True, allows you to do the prediction in parallel w/ joblib. Can save time for huge prediction
            tasks.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted "expected" movement values.
            - Columns are prefixed with "x_" (e.g., 'x_pfx_z').
            - The index matches the index of the input `pred_df`.
            - Rows corresponding to pitch types not included in `self.pitch_types` will contain NaNs.
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
    cache_expected_movement_models()
