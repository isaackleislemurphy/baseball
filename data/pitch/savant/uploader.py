"""Extract fns to pull pitch tracking data from Savant"""

from datetime import datetime
from typing import Iterable

import pandas as pd
import pybaseball as pb

import baseball.data.savant.pitch.constants as DATA_CONSTANTS
from baseball.duckdb.database import make_write_path
from baseball.utils.general import read_yaml
from baseball.utils.logging import get_logger

TODAY = datetime.now().date()
LOGGER = get_logger()

# these are strictly Hawkeye features, so they'll be
# nulled out pre-2020. If the mass nulls for 2017-2019
# aren't explicitly casted to floats, then DuckDB may
# initialize the columns as ints, which'll bork further
# analysis. So these need to be casted explicitly to floats,
# particularly for parquets where the entire season will be
# null in these fields.
HAWKEYE_FLOAT_OVERRIDES = [
    "arm_angle",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
]


def scrape_savant_pitch_data(
    date_max: str = str(TODAY), date_min: str = DATA_CONSTANTS.MIN_STATCAST_DATE
) -> pd.DataFrame:
    """
    Pull raw pitch-level Statcast data from Baseball Savant across a date range.

    Thin wrapper around `pybaseball.statcast()` that just renames columns to
    project conventions right after the pull. No feature engineering, no
    filtering here — that's someone else's job downstream.

    Parameters
    ----------
    date_max : str, default=str(TODAY)
        Inclusive upper bound on game date (YYYY-MM-DD). Defaults to today.
    date_min : str, default=DATA_CONSTANTS.MIN_STATCAST_DATE
        Inclusive lower bound on game date (YYYY-MM-DD). Defaults to the
        earliest Statcast date the project supports.

    Returns
    -------
    pd.DataFrame
        Raw pitch-level Statcast data, columns renamed per
        `DATA_CONSTANTS.STATCAST_RENAMINGS`.

    Notes
    -----
    - Stays deliberately close to the raw Savant schema.
    - Feature engineering, filtering, and storage all happen downstream.
    """
    data_df = pb.statcast(start_dt=date_min, end_dt=date_max).rename(columns=DATA_CONSTANTS.STATCAST_RENAMINGS)
    return data_df


def engineer_misc_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tack on a grab-bag of lightweight, non-model-specific pitch features.

    Adds cheap, broadly-useful indicators and derived columns that come in
    handy across modeling tasks (classification, regression, hierarchical
    stuff). Nothing here is expensive to compute or season-dependent.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw Statcast pitch data with standardized column names.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the engineered columns tacked on.

    Engineered Features
    -------------------
    - is_bunt_attempt : int
        Indicator for bunt attempts, parsed out of the pitch description.
    - pitch_group : str
        Coarse pitch grouping (e.g., FF, SI, BB) via project-level mappings.
    - is_oppo_hand : int
        Indicator for opposite-handed batter–pitcher matchups.
    - residual_speed : float
        Effective velocity minus release velocity.
    - pitch_outcome_category : str
        Coarse categorical pitch outcome pulled from Statcast descriptions.

    Notes
    -----
    - Does *not* drop rows or enforce completeness.
    - Safe to run before caching to DuckDB.
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
    Convert selected columns from feet to inches. Just movement for now.

    Parameters
    ----------
    data_df : pd.DataFrame
        Pitch-level Statcast data.
    feet_cols : list[str], default ["pfx_x", "pfx_z"]
        Columns measured in feet that should get bumped to inches.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the units converted.
    """
    # copy dataframe, lest I drown in warnings
    data_df = data_df.copy()

    # feet --> inches
    data_df[feet_cols] *= 12

    return data_df


def _scrape_and_transform_pitch_data(date_min: str, date_max: str) -> pd.DataFrame:
    """
    Pull and lightly clean pitch data for a date range, end to end.

    Convenience wrapper that grabs the raw Savant data and runs the minimal
    feature engineering + unit conversions we want in place before anything
    hits disk.

    Parameters
    ----------
    date_min : str
        Inclusive lower bound on game date (YYYY-MM-DD).
    date_max : str
        Inclusive upper bound on game date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Cleaned, lightly-engineered pitch data, ready to cache to
        parquet / DuckDB.

    Notes
    -----
    - Last stop before the data gets persisted.
    - No modeling assumptions baked in here.
    """

    return (
        scrape_savant_pitch_data(date_max=date_max, date_min=date_min)
        .pipe(engineer_misc_features)
        .pipe(convert_feet_to_inches)
    )


def scrape_and_transform_pitch_data_byseason(season: int) -> pd.DataFrame:
    """
    Pull and clean pitch data for a single MLB season.

    Parameters
    ----------
    season : int
        MLB season year (e.g., 2023).

    Returns
    -------
    pd.DataFrame
        Pitch-level Statcast data for that season, with the basic feature
        engineering applied.

    Notes
    -----
    - Uses calendar-year bounds (Jan 1 – Dec 31).
    - Postseason games come along if Savant has them.
    """
    return _scrape_and_transform_pitch_data(date_min=f"{season}-01-01", date_max=f"{season}-12-31")


def upload_savant_pitch_data_byseason(season: int) -> None:
    """
    Pull a season of pitch data and stash it to DuckDB-friendly parquet.

    Materializes season-level pitch data to disk — this is the ingestion step
    that everything downstream in DuckDB leans on.

    Parameters
    ----------
    season : int
        MLB season year to pull and cache.

    Returns
    -------
    None

    Side Effects
    ------------
    - Writes a parquet file to the season-specific DuckDB filepath.
    - Overwrites whatever's already there. TODO: throw a warning here.

    Notes
    -----
    - Meant to run once per season.
    - Downstream models should never be hitting Savant directly.
    """
    # pull in the data
    LOGGER.info(f"Ingesting & transforming raw Savant pitch data for season = {season}")
    data_raw = scrape_and_transform_pitch_data_byseason(season=season)
    LOGGER.info(f"Raw Savant pitch data for season = {season} successfully ingested")

    # for pre-HE seasons, ensure mass nulls don't accidentally get tagged as ints and
    # mislead DuckDB down the line
    data_raw[HAWKEYE_FLOAT_OVERRIDES] = data_raw[HAWKEYE_FLOAT_OVERRIDES].astype(float)
    LOGGER.info("Hawkeye columns casted")

    # sort out table config + parquet storage
    table_config = read_yaml("duckdb/table_config/pitch__savant.yaml")

    # filename to store parquet
    parquet_filename = make_write_path(table_config, season=season)
    LOGGER.info("Pitch db table prepared. ")

    # save to parquet
    data_raw.to_parquet(parquet_filename, index=False)
    LOGGER.info(f"Savant pitch data uploaded to: {parquet_filename}")


def upload_savant_pitch_data(seasons: Iterable = range(2017, 2026)) -> None:
    """
    Stand up the DuckDB parquet files for every supported Statcast season.

    Walks the configured season range and caches raw pitch data one season
    at a time.

    Parameters
    ----------
    seasons : Iterable, default=range(2017, 2026)
        Seasons to pull and cache.

    Returns
    -------
    None

    Notes
    -----
    - Built for one-time setup or a full refresh.
    - This takes a while and hammers the Savant API repeatedly — make sure
      your wifi's solid.
    """
    for season in seasons:
        upload_savant_pitch_data_byseason(season)


if __name__ == "__main__":
    upload_savant_pitch_data()
