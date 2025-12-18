"""
Training/Test Partition Management.

This module handles the creation and retrieval of pitcher-level train/test splits.
It ensures that:
1.  **No Data Leakage:** Pitchers are split by ID, preventing a pitcher's 2021 season
        from being in the training set while their 2022 season is in the test set.
2.  **Stratification:** The split coarsely preserves the distribution of arm slot
        ensuring that both sets have representative samples of various slots.
3.  **Reproducibility:** The split is cached to a CSV file to ensure consistent evaluation across different runs.
        That way, you can just load in the CSV and retrain whenever/wherever
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from baseball.data.savant.pitch.etl import load_pitch_data
from baseball.projects.pitch_quality.data.constants import (
    NUM_ARM_ANGLE_STRATIFICATION_BINS,
    PARTITION_CSV_PATH,
    TRAINING_PARTITION_SEED,
)


def cache_training_partitions() -> None:
    """
    Generate and persist a stratified train/test split of pitchers.

    This function loads historical pitch data, calculates the average arm angle for every
    unique pitcher, and stratifies them into training and testing sets. The result
    is saved to a CSV to ensure all future model runs use the exact same split.

    Logic
    -----
    1. **Aggregation:** Groups data by `pitcher` ID, not pitcher-season, to avoid leakage. See
        the note below.
    2. **Stratification:** Uses `KBinsDiscretizer` to bin pitchers by `arm_angle` (quantiles),
       ensuring the test set isn't accidentally biased towards a specific arm slot.
    3. **Persistence:** Saves the resulting mapping to `csvs/pitcher-id-partitions.csv`.

    Returns
    -------
    None
    """
    # pull in all pitches thrown here
    pitch_data_df = load_pitch_data(date_min="2020-01-01", date_max="2025-12-31")

    # aggregate pitchers by arm slots across seasons, as opposed to pitcher-seasons. Idea here is we
    # don't want David Hale 2020 in the training set and David Hale 2021 in the test set; rather, we
    # just want all his seasons either in training / testing
    pitchers = pitch_data_df.dropna(subset=["arm_angle"]).groupby(["pitcher"], as_index=False)["arm_angle"].mean()

    # stratify partitions by arm slot
    pitchers["strat_bin"] = (
        KBinsDiscretizer(
            n_bins=NUM_ARM_ANGLE_STRATIFICATION_BINS,
            random_state=TRAINING_PARTITION_SEED,
            strategy="quantile",
            encode="ordinal",
        )
        .fit_transform(pitchers[["arm_angle"]].values)
        .astype(int)
    )

    # partition the pitchers
    pitchers_train, pitchers_test = train_test_split(
        pitchers["pitcher"].values,
        test_size=0.25,
        random_state=TRAINING_PARTITION_SEED,
        shuffle=True,
        stratify=pitchers["strat_bin"].values,
    )

    # put everything together
    partition_df = pd.concat(
        [
            pd.DataFrame(dict(pitcher=pitchers_train, train=1, test=0)),
            pd.DataFrame(dict(pitcher=pitchers_test, train=0, test=1)),
        ],
        axis=0,
    ).reset_index(drop=True)

    # save CSV
    partition_df.to_csv(PARTITION_CSV_PATH, index=False)


def load_training_partitions() -> pd.DataFrame:
    """
    Load the cached pitcher partition map, generating it if necessary.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the partition flags.
        Columns: ['pitcher', 'train', 'test']
        - 'pitcher': The unique pitcher ID.
        - 'train': 1 if the pitcher is in the training set, 0 otherwise.
        - 'test': 1 if the pitcher is in the test set, 0 otherwise.
    """
    try:
        return pd.read_csv(PARTITION_CSV_PATH)
    except FileNotFoundError:
        print(f"Could not find cached training partitions...remaking with random seed = {TRAINING_PARTITION_SEED}")
        cache_training_partitions()
        return pd.read_csv(PARTITION_CSV_PATH)


if __name__ == "__main__":
    cache_training_partitions()
