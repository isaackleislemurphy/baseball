"""Extract fns to pull pitch tracking data from Savant"""

from datetime import datetime

import pandas as pd
import pybaseball as pb

import baseball.data.savant.pitch.constants as DATA_CONSTANTS
from baseball.data.savant.pitch.utils import get_season_savant_duckdb_filepath

TODAY = datetime.now().date()


def pull_raw_savant_pitch_data(
    date_max: str = str(TODAY), date_min: str = DATA_CONSTANTS.MIN_STATCAST_DATE
) -> pd.DataFrame:
    """
    Pull raw Statcast pitch-level data from Baseball Savant for a date range.

    This is a thin wrapper around `pybaseball.statcast()` that standardizes
    column names immediately after retrieval. No feature engineering or
    filtering is done here beyond renaming columns.

    Parameters
    ----------
    date_max : str, default=str(TODAY)
        Inclusive upper bound on game date (YYYY-MM-DD). Defaults to today.
    date_min : str, default=DATA_CONSTANTS.MIN_STATCAST_DATE
        Inclusive lower bound on game date (YYYY-MM-DD). Defaults to the
        earliest Statcast date supported by the project.

    Returns
    -------
    pd.DataFrame
        Raw pitch-level Statcast data with columns renamed according to
        `DATA_CONSTANTS.STATCAST_RENAMINGS`.

    Notes
    -----
    - This function intentionally stays close to the raw Savant schema.
    - Downstream functions are responsible for feature engineering,
      filtering, and storage.
    """
    data_df = pb.statcast(start_dt=date_min, end_dt=date_max).rename(columns=DATA_CONSTANTS.STATCAST_RENAMINGS)
    return data_df


def engineer_misc_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer lightweight, miscellaneous, non-model-specific pitch features.

    Adds a grab-bag of simple indicators and derived columns that are broadly
    useful across modeling tasks (classification, regression, hierarchical
    models). These features are cheap to compute and stable across seasons.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw Statcast pitch data with standardized column names.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with additional engineered columns.

    Engineered Features
    -------------------
    - is_bunt_attempt : int
        Indicator for bunt attempts, parsed from the pitch description.
    - pitch_group : str
        Coarse pitch grouping (e.g., FF, SI, BB) using project-level mappings.
    - is_oppo_hand : int
        Indicator for opposite-handed batter–pitcher matchups.
    - residual_speed : float
        Effective velocity minus release velocity.
    - pitch_outcome_category : str
        Coarse categorical pitch outcome derived from Statcast descriptions.

    Notes
    -----
    - This function does *not* drop rows or enforce completeness.
    - It is safe to apply before caching to DuckDB.
    """
    data_df = data_df.copy()

    # add in column for bunt attempts
    data_df["is_bunt_attempt"] = ["bunt" in item.lower() for item in data_df.description]
    data_df["is_bunt_attempt"] = data_df["is_bunt_attempt"].astype(int)

    # assign pitch groups
    data_df["pitch_group"] = data_df["pitch_type"].replace(DATA_CONSTANTS.PITCH_GROUP_MAPPINGS)

    # make an indicator for an opposite-hand platoon config
    data_df["is_oppo_hand"] = (data_df.bats != data_df.throws).astype(int)

    # offset effective velo
    data_df["residual_speed"] = data_df.effective_speed - data_df.release_speed

    # engineer the categorical response map
    data_df["pitch_outcome_category"] = data_df["description"].replace(DATA_CONSTANTS.PITCH_OUTCOME_CATEGORY_MAPPINGS)

    return data_df


def convert_feet_to_inches(data_df: pd.DataFrame, feet_cols: list[str] = ["pfx_x", "pfx_z"]) -> pd.DataFrame:
    """
    Convert selected movement columns from feet to inches. Just movement for now

    Parameters
    ----------
    data_df : pd.DataFrame
        Pitch-level Statcast data.
    feet_cols : list[str], default ["pfx_x", "pfx_z"]
        Columns measured in feet to convert to inches.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with converted units.
    """
    # copy dataframe, lest I drown in warnings
    data_df = data_df.copy()

    # feet --> inches
    data_df[feet_cols] *= 12

    return data_df


def _extract_raw_pitch_data(date_min: str, date_max: str) -> pd.DataFrame:
    """
    End-to-end extraction of raw pitch data for a date range.

    Convenience wrapper that pulls raw Savant data and applies the minimal
    feature engineering and unit conversions needed before storage.

    Parameters
    ----------
    date_min : str
        Inclusive lower bound on game date (YYYY-MM-DD).
    date_max : str
        Inclusive upper bound on game date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Cleaned, lightly engineered pitch-level dataset suitable for
        caching to parquet / DuckDB.

    Notes
    -----
    - This is the last step before data is persisted.
    - No modeling assumptions are baked in here.
    """

    return (
        pull_raw_savant_pitch_data(date_max=date_max, date_min=date_min)
        .pipe(engineer_misc_features)
        .pipe(convert_feet_to_inches)
    )


def extract_raw_pitch_data_byseason(season: int) -> pd.DataFrame:
    """
    Extract raw Statcast pitch data for a single MLB season.

    Parameters
    ----------
    season : int
        MLB season year (e.g., 2023).

    Returns
    -------
    pd.DataFrame
        Pitch-level Statcast data for the specified season, with basic
        feature engineering applied.

    Notes
    -----
    - Uses calendar-year bounds (Jan 1 – Dec 31).
    - Postseason games are included if present in Savant.
    """
    return _extract_raw_pitch_data(date_min=f"{season}-01-01", date_max=f"{season}-12-31")


def cache_raw_pitch_data_byseason(season: int) -> None:
    """
    Extract and persist raw pitch data for a season to DuckDB-compatible parquet.

    This function materializes season-level pitch data to disk, serving as the
    ingestion step for downstream DuckDB-based workflows.

    Parameters
    ----------
    season : int
        MLB season year to extract and cache.

    Returns
    -------
    None

    Side Effects
    ------------
    - Writes a parquet file to the season-specific DuckDB filepath.
    - Overwrites existing files if present. TODO: throw a warning here.

    Notes
    -----
    - Intended to be run once per season.
    - Downstream models should never hit Savant directly.
    """
    data_raw = extract_raw_pitch_data_byseason(season=season)
    data_raw.to_parquet(
        get_season_savant_duckdb_filepath(season),
        index=False,
    )
    print(f"Pitch data for {season} cached")


def init_duck_db_parquets() -> None:
    """
    Initialize DuckDB parquet files for all supported Statcast seasons.

    Iterates through the configured season range and caches raw pitch data
    for each season individually.

    Returns
    -------
    None

    Notes
    -----
    - Designed for one-time setup or full refreshes.
    - This can take a while and will hit the Savant API repeatedly.
        Make sure your wifi connection is solid.
    """
    for season in range(2017, 2026):
        cache_raw_pitch_data_byseason(season)


if __name__ == "__main__":
    init_duck_db_parquets()
