"""
Season- and count-based categorical log-offset model.

This module fits a multinomial logistic regression using count state
and season indicators to produce baseline log-probabilities for
categorical pitch outcomes. The fitted model is cached to disk and
used as a fixed offset in downstream pitch-quality models.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baseball.data.savant.pitch.constants import COUNTS
from baseball.projects.pitch_quality.data.etl import load_pitch_quality_model_data
from baseball.utils.general import load_pickled_object, write_pickled_object

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORICAL_LOG_OFFSET_OBJECT_PATH = os.path.join(
    SCRIPT_DIR.replace("submodels", "objects"), "categorical_log_offsets.pkl"
)


class CategoricalLogOffsets:
    def __init__(self) -> None:
        """
        Initialize the categorical log-offset model.

        Attributes are populated during fitting.
        """
        self.seasons = None
        self.baseline_features = COUNTS
        self.seasons = None

    def _add_season_indicator(self, pitch_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one-hot season indicator columns to pitch-level data.

        Seasons outside the fitted range are clipped to the minimum or
        maximum observed season.

        Parameters
        ----------
        pitch_data_df : pandas.DataFrame
            Pitch-level dataset containing a `season` column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one-hot encoded season indicator columns added.
        """
        # add indicator for seasons
        season_arr = np.minimum(np.maximum(pitch_data_df["season"].values, self.seasons.min()), self.seasons.max())
        for season in self.seasons:
            pitch_data_df[str(season)] = (season_arr == season).astype(int)
        return pitch_data_df

    def fit(self, **kwargs: dict) -> None:
        """
        Fit the categorical log-offset model.

        Loads pitch-level data, constructs baseline features consisting
        of count state and season indicators, and fits a multinomial
        logistic regression.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to
            `load_pitch_quality_model_data`.
        """
        # pull in the pitch data
        pitch_data_df = load_pitch_quality_model_data(**kwargs)

        # extract the unique seasons
        self.seasons = pitch_data_df["season"].unique()  # save me

        # add one-hot season indicators
        pitch_data_df = self._add_season_indicator(pitch_data_df)

        # these are the "baseline" features for the offset model
        self.baseline_features = COUNTS + [str(season) for season in self.seasons]  # save me

        # fit the offset model
        X, y = pitch_data_df[self.baseline_features].values.astype(float), pitch_data_df[
            "categorical_response_idx"
        ].values.astype(int)
        self.clf = LogisticRegression()  # save me
        self.clf.fit(X, y)

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        """
        Predict log-probabilities for categorical outcomes.

        Parameters
        ----------
        pred_df : pandas.DataFrame
            Pitch-level dataset.

        Returns
        -------
        numpy.ndarray
            Log-probabilities for each categorical outcome.
        """
        # don't mess with the original df
        pred_df = pred_df.copy()

        # add one-hot season indicators
        pred_df = self._add_season_indicator(pred_df)

        # exponentiate these guys and you have probabilities
        return self.clf.predict_log_proba(pred_df[self.baseline_features].values.astype(float))


def cache_categorical_log_offsets() -> None:
    """
    Fit and persist the categorical log-offset model to disk.
    """
    offsets = CategoricalLogOffsets()
    offsets.fit()
    write_pickled_object(offsets, CATEGORICAL_LOG_OFFSET_OBJECT_PATH)


def load_categorical_log_offsets() -> CategoricalLogOffsets:
    """
    Load a cached categorical log-offset model from disk.

    Returns
    -------
    CategoricalLogOffsets
        Restored offset model.
    """
    return load_pickled_object(CATEGORICAL_LOG_OFFSET_OBJECT_PATH)


if __name__ == "__main__":
    cache_categorical_log_offsets()
