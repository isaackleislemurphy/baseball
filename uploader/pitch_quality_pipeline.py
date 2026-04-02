import argparse
import os

import pandas as pd

from baseball.data.savant.pitch.load import load_pitch_data
from baseball.projects.pitch_quality.model.submodels.expected_movement.constants import (
    XMVMT_DUCK_DB_PARQUET_PATH,
    XMVMT_INPUTS_UNION,
    XMVMT_OUTPUTS,
)
from baseball.projects.pitch_quality.model.submodels.expected_movement.model import (
    ExpectedMovement,
    load_expected_movement_models,
)


def make_expected_movement_predictions(date_min: str, date_max: str) -> pd.DataFrame:
    """
    Generate expected movement predictions for pitches within a date range.

    After you've run `cache_expected_movement_models()`, use this function to make xMvmt
    predictions on new pitches (e.g., last night's).

    Parameters
    ----------
    date_min : str
        Start date for pitch data in format 'YYYY-MM-DD'.
    date_max : str
        End date for pitch data in format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing pitch data with expected movement predictions.

    Notes
    -----
    This function requires that expected movement models have been previously cached
    using `cache_expected_movement_models()`.
    """
    pitch_data_df = load_pitch_data(date_min=date_min, date_max=date_max)
    print("Pitch data loaded.")
    xmvmt = load_expected_movement_models()
    print("xMvmt models loaded")
    return xmvmt.predict(pitch_data_df)


def make_and_save_expected_movement_predictions(date_min: str, date_max: str) -> None:
    """
    Generate and save expected movement predictions to parquet file.

    This function creates expected movement predictions for a date range and saves
    the results as a parquet file in the configured output directory.

    Parameters
    ----------
    date_min : str
        Start date for pitch data in format 'YYYY-MM-DD'.
    date_max : str
        End date for pitch data in format 'YYYY-MM-DD'.

    Returns
    -------
    None

    Notes
    -----
    The output parquet file is saved with the naming convention:
    `xmvmt_{date_min}-{date_max}.parquet` in the directory specified by
    XMVMT_DUCK_DB_PARQUET_PATH.

    The saved DataFrame includes the following columns:
    - game_pk, at_bat_number, pitch_number, xmvmt_pitch_group
    - All columns from XMVMT_INPUTS_UNION
    - All columns from XMVMT_OUTPUTS
    """

    # make the xmvmt predictions
    xmvmt_preds = make_expected_movement_predictions(date_min=date_min, date_max=date_max)

    # slice down columns
    xmvmt_preds = xmvmt_preds[
        ["game_pk", "at_bat_number", "pitch_number", "xmvmt_pitch_group"] + XMVMT_INPUTS_UNION + XMVMT_OUTPUTS
    ]

    print("xMvmt predictions made")
    xmvmt_preds.to_parquet(os.path.join(XMVMT_DUCK_DB_PARQUET_PATH, f"xmvmt_{date_min}-{date_max}.parquet"))
    print("xMvmt saved to:", XMVMT_DUCK_DB_PARQUET_PATH)


def parse_args() -> argparse.Namespace:
    """
    Sets up argparser to intake:
    --date_min, minimum date ("YYYY-MM-DD") over which to make xmvmt preds.
    --date_max, maximum date ("YYYY-MM-DD") over which to make xmvmt preds.
    """

    parser = argparse.ArgumentParser(
        description="Generate and save expected movement predictions for baseball pitch data."
    )
    parser.add_argument(
        "--date_min", type=str, required=True, help="Start date for pitch data in format YYYY-MM-DD (e.g., 2025-01-01)"
    )
    parser.add_argument(
        "--date_max", type=str, required=True, help="End date for pitch data in format YYYY-MM-DD (e.g., 2025-12-31)"
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """Main fn"""
    args = parse_args()
    date_min, date_max = args.date_min, args.date_max
    make_and_save_expected_movement_predictions(date_min=date_min, date_max=date_max)


if __name__ == "__main__":
    main()
