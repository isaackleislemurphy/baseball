"""Extract fns to pull pitch tracking data from Savant"""

import itertools
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pybaseball as pb

import baseball.data.savant.pitch.constants as DATA_CONSTANTS


def retrieve_statcast_pitch_data(
    date_max: str, date_min: Optional[str] = DATA_CONSTANTS.MIN_STATCAST_DATE
) -> pd.DataFrame:
    """
    Retrieve and process Statcast pitch data for a specified date range.

    Parameters:
    -----------
    date_max : str
            The end date (latest date) for the Statcast data retrieval.

    date_min : Optional[str]
            The start date (earliest date) for the Statcast data retrieval. Defaults to DATA_CONSTANTS.MIN_STATCAST_DATE.

    Returns:
    --------
    pd.DataFrame
            A Pandas DataFrame containing the Statcast pitch data with renamed columns.

    Notes:
    ------
    This function is a wrapper around `pybaseball`'s `statcast()` function and performs two main tasks:
    1. Retrieves Statcast pitch data for the specified date range.
    2. Renames the columns of the retrieved data based on predefined column renamings in DATA_CONSTANTS.STATCAST_RENAMINGS.

    The renamed columns help make the data more user-friendly and aligned with your preferences.

    Examples:
    ---------
    To retrieve Statcast pitch data for a specific date range:

    >>> start_date = "2023-01-01"
    >>> end_date = "2023-01-31"
    >>> statcast_data = retrieve_statcast_pitch_data(date_max=end_date, date_min=start_date)

    The `statcast_data` DataFrame will contain the Statcast pitch data with renamed columns.
    """
    data_df = pb.statcast(start_dt=date_min, end_dt=date_max).rename(columns=DATA_CONSTANTS.STATCAST_RENAMINGS)
    return data_df


def test_data_integrity(data_df: pd.DataFrame) -> None:
    """
    Perform data integrity checks on a FULLY PROCESSED Statcast pitch data DataFrame.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data to be checked for integrity.

    Raises:
    -------
    AssertionError
            If any of the data integrity checks fail, an AssertionError is raised with a corresponding error message.

    Checks:
    -------
    1. Ensures there are no missing values in the 'bats' column.
    2. Ensures there are no missing values in the 'throws' column.
    3. Ensures there are no missing values in the 'pitch_group' column.
    4. Validates that the 'pitch_type' column contains only valid pitch types defined in DATA_CONSTANTS.PITCH_TYPES.
    5. Validates that the 'throws' column contains only 'L' (Left) or 'R' (Right) values.
    6. Checks that there are no bunt attempts in the dataset (requires 'is_bunt_attempt' column).
    7. Validates that the 'description' column contains only valid categorical response values defined in DATA_CONSTANTS.PITCH_OUTCOME_CATEGORY_MAPPINGS.

    Notes:
    ------
    This function should be run at the END of the processing pipeline (e.g., inside `load_pitch_data`),
    as it checks for features like `is_bunt_attempt` and `pitch_group` that are created during feature engineering.
    Do not run this immediately after `retrieve_statcast_pitch_data`.

    Examples:
    ---------
    To test the integrity of a fully processed DataFrame:

    >>> loaded_data = load_pitch_data()
    >>> test_data_integrity(loaded_data)
    """
    assert data_df.bats.isna().sum() == 0, "Missing values for 'bats' column"
    assert data_df.throws.isna().sum() == 0, "Missing values for 'throws' column"
    assert data_df.pitch_group.isna().sum() == 0, "Missing values for 'pitch_group' column"
    assert (
        set(data_df.pitch_type.tolist()).difference(DATA_CONSTANTS.PITCH_TYPES) == set(),
        f"'pitch_type' column has values outside of {DATA_CONSTANTS.PITCH_TYPES}",
    )
    assert (
        set(data_df.throws.unique()).difference(["L", "R"]) == set()
    ), "'throws' column has values besides 'L' and 'R'"
    assert data_df.is_bunt_attempt.sum() == 0, f"You have {data_df.is_bunt_attempt.sum()} bunt attempts in your dataset"
    assert (
        set(data_df.description.tolist()).difference(DATA_CONSTANTS.PITCH_OUTCOME_CATEGORY_MAPPINGS.keys()) == set(),
        f"'description' column has invalid values: {set(data_df.description).difference(DATA_CONSTANTS.PITCH_OUTCOME_CATEGORY_MAPPINGS.keys())}",
    )


def filter_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and preprocess a Statcast pitch data DataFrame to ensure data quality and readability.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data to be filtered and processed.

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with filtered and sorted data for analysis.

    Filters and Preprocessing:
    --------------------------
    1. Filters the DataFrame to ensure only pitch types defined in DATA_CONSTANTS.PITCH_TYPES are included.
    2. Filters out non-competitive games (spring training) by checking 'game_type' against DATA_CONSTANTS.GAME_TYPES.
    3. Filters out eephus pitches and "fastballs" from position players pitching based on 'release_speed'.
    4. Filters out pitches with description "intent_ball" or "unknown_strike"
    5. Sorts the DataFrame for readability using the following columns: ['game_date', 'pitcher', 'at_bat_number', 'pitch_number'].
    6. Resets the DataFrame index for consistency.

    Notes:
    ------
    This function is used to filter and preprocess Statcast pitch data, ensuring data quality and readability for subsequent analysis and visualization.

    Examples:
    ---------
    To filter and preprocess a Statcast pitch data DataFrame:

    >>> statcast_data = retrieve_statcast_pitch_data(date_max="2023-01-31")
    >>> filtered_data = filter_data(statcast_data)

    The 'filtered_data' DataFrame will contain the filtered and sorted data for further analysis.
    """
    return (
        data_df
        # filter to ensure a "real" pitch offering
        .query(f"pitch_type in {DATA_CONSTANTS.PITCH_TYPES}")
        # filter to non-spring training games, i.e. competitive games
        .query(f"game_type in {DATA_CONSTANTS.GAME_TYPES}")
        # filter out eephus pitches and "fastballs" from position players pitching
        .query(f"release_speed >= {DATA_CONSTANTS.MIN_VELO}")
        # weird pitch types
        .query(f"description not in ('intent_ball', 'unknown_strike')")
        # sort for readability
        .sort_values(["game_date", "pitcher", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    )


def engineer_strike_zone_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional strike zone-related features in a Statcast pitch data DataFrame.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data to be augmented with strike zone features.

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with the added strike zone-related features.

    Features Engineered:
    --------------------
    1. 'sz_mid': The midpoint of the strike zone, calculated as the average of 'sz_top' and 'sz_bot'.
    2. 'sz_vertical_dist': The vertical distance of the strike zone, computed as the difference between 'sz_top' and 'sz_bot'.
    3. 'plate_z_mid_zone': Centered 'plate_z' on the middle of the strike zone by subtracting 'sz_mid'.
    4. 'plate_z_mid_zone_sc': Scaled 'plate_z_mid_zone' relative to the size of the strike zone.

    Notes:
    ------
    This function is used to engineer additional strike zone-related features in a Statcast pitch data DataFrame. These features can be useful for analyzing and visualizing pitch locations within the strike zone.

    The 'sz_vertical_dist' is checked to ensure that no entry is less than or equal to 0 to maintain data integrity.

    Examples:
    ---------
    To engineer strike zone features for a Statcast pitch data DataFrame:

    >>> statcast_data = retrieve_statcast_pitch_data(date_max="2023-01-31")
    >>> enriched_data = engineer_strike_zone_features(statcast_data)

    The 'enriched_data' DataFrame will contain the original data along with the newly engineered strike zone features.
    """
    data_df = data_df.copy()
    # determine midpoint of strike zone
    data_df["sz_mid"] = (data_df.sz_top + data_df.sz_bot) / 2
    # determine vertical strike zone dist
    data_df["sz_vertical_dist"] = data_df.sz_top - data_df.sz_bot
    assert data_df["sz_vertical_dist"].min() > 0, "'sz_vertical_dist' has an entry <= 0"
    # center plate_z on the middle of the zone
    data_df["plate_z_mid_zone"] = data_df.plate_z - data_df.sz_mid
    # scale `plate_z_mid_zone` relative to zone size
    data_df["plate_z_mid_zone_sc"] = data_df.plate_z_mid_zone / (data_df.sz_vertical_dist / 2)

    return data_df


def engineer_x_neutral_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer batter-neutral features related to horizontal pitch location in a Statcast pitch data DataFrame.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data to be augmented with batter-neutral features.

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with added batter-neutral features for horizontal pitch location.

    Features Engineered:
    --------------------
    1. 'pfx_x_neutral': Horizontal movement relative to the batter (positive away, negative towards).
    2. 'plate_x_neutral': Horizontal pitch location relative to the batter (positive away, negative towards).
    3. 'release_pos_x_neutral': Release position relative to the batter (positive away, negative towards).

    Notes:
    ------
    This function is used to engineer batter-neutral features that provide a standardized perspective on horizontal pitch location, considering the batter's side.

    For right-handed batters, the values remain unchanged, while for left-handed batters, the values are negated to ensure consistency in analysis.

    Examples:
    ---------
    To engineer batter-neutral features for horizontal pitch location in a Statcast pitch data DataFrame:

    >>> statcast_data = retrieve_statcast_pitch_data(date_max="2023-01-31")
    >>> enriched_data = engineer_x_neutral_features(statcast_data)

    The 'enriched_data' DataFrame will contain the original data along with the newly engineered batter-neutral features.
    """
    data_df = data_df.copy()
    # away from batter = positive value; towards/closer to batter = negative value
    for col in ["pfx_x", "plate_x", "release_pos_x"]:
        data_df[col + "_neutral"] = data_df[[col, "bats"]].apply(
            lambda df: df[col] if df["bats"] == "R" else -df[col], axis=1
        )
    return data_df


def parse_missingness(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the Statcast DataFrame, particularly for Expected Weighted On-Base Average (xWOBA).

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data.

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with missing values handled.

    Operations:
    -----------
    1. Drops rows where any features defined in `DATA_CONSTANTS.INPUT_VARS` are missing.
    2. Prints summary statistics about the rows dropped and xWOBA missingness.
    3. Imputes missing `xwoba` values with 0.0 (typically implying a non-contact or irrelevant outcome for xWOBA calculation).
    """

    data_df, shape_in = data_df.copy(), data_df.shape[0]
    data_df.dropna(subset=DATA_CONSTANTS.INPUT_VARS, inplace=True)
    print("- # Rows dropped due to missing features:", shape_in - data_df.shape[0])
    ### fill xwoba with 0
    print(
        "- % Rows missing xWOBA:",
        np.round(100 * data_df.query("type == 'X'").xwoba.isna().mean(), 2),
    )
    print("- # Rows missing xWOBA:", data_df.query("type == 'X'").xwoba.isna().sum())
    print(
        "- BB Types with missing xWOBA: ",
        data_df.query("type == 'X'")
        .loc[data_df.query("type == 'X'").xwoba.isna()]
        .groupby(["bb_type"], as_index=False)["pitch_number"]
        .count(),
    )
    data_df["xwoba"] = data_df["xwoba"].fillna(0.0)
    return data_df.reset_index(drop=True)


def engineer_bunt_indicator(data_df: pd.DataFrame, remove_bunts: Optional[bool] = True) -> pd.DataFrame:
    """
    Engineer a bunt attempt indicator in a Statcast pitch data DataFrame and optionally remove bunt data.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data to be augmented with a bunt attempt indicator.

    remove_bunts : Optional[bool], optional
            A flag to control whether to remove rows with bunt attempts (default: True).

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with the added 'is_bunt_attempt' column indicating bunt attempts.

    Features Engineered:
    --------------------
    1. 'is_bunt_attempt': A binary indicator (0 or 1) that flags rows with bunt attempts in the 'description' column.

    Notes:
    ------
    This function is used to engineer a bunt attempt indicator in a Statcast pitch data DataFrame. It adds a binary column 'is_bunt_attempt' that is 1 for rows with bunt attempts and 0 otherwise.

    If 'remove_bunts' is set to True, the function will also remove rows with bunt attempts, resulting in a filtered DataFrame.

    Examples:
    ---------
    To engineer a bunt attempt indicator in a Statcast pitch data DataFrame:

    >>> statcast_data = retrieve_statcast_pitch_data(date_max="2023-01-31")
    >>> enriched_data = engineer_bunt_indicator(statcast_data)

    The 'enriched_data' DataFrame will contain the original data along with the newly added 'is_bunt_attempt' column.

    To engineer the indicator and retain rows with bunt attempts:

    >>> enriched_data_with_bunts = engineer_bunt_indicator(statcast_data, remove_bunts=False)

    The 'enriched_data_with_bunts' DataFrame will include rows with bunt attempts, and 'is_bunt_attempt' will indicate bunt attempts.
    """
    data_df = data_df.copy()
    data_df["is_bunt_attempt"] = ["bunt" in item.lower() for item in data_df.description]
    data_df["is_bunt_attempt"] = data_df["is_bunt_attempt"].astype(int)
    if remove_bunts:
        data_df = data_df.query("is_bunt_attempt == 0").reset_index(drop=True)
    return data_df


def engineer_count_indicators(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode ball-strike counts and engineer count-based state flags.

    Parameters:
    -----------
    data_df : pd.DataFrame
            A Pandas DataFrame containing Statcast pitch data with 'balls' and 'strikes' columns.

    Returns:
    --------
    pd.DataFrame
            A new Pandas DataFrame with boolean/binary columns for specific counts and count states.

    Features Engineered:
    --------------------
    1. Specific count columns (e.g., '0_0', '3_2') via one-hot encoding.
    2. 'even_count': 1 if remaining balls equal remaining strikes, else 0.
    3. 'pitcher_ahead': 1 if the pitcher has fewer pitches remaining to a strikeout than the batter has to a walk.
    4. 'batter_ahead': 1 if the batter has fewer pitches remaining to a walk than the pitcher has to a strikeout.
    5. '2K': 1 if there are 2 strikes on the batter.
    """

    data_df = data_df.copy()
    # iterate through counts and encode
    for balls, strikes in itertools.product(range(4), range(3)):
        data_df[f"{balls}_{strikes}"] = (
            (data_df["balls"].values == balls) * (data_df["strikes"].values == strikes)
        ).astype(int)
    # also flag coarser count states
    data_df["even_count"] = ((4 - data_df["balls"]) == (3 - data_df["strikes"])).astype(int)
    data_df["pitcher_ahead"] = ((4 - data_df["balls"]) > (3 - data_df["strikes"])).astype(int)
    data_df["batter_ahead"] = ((4 - data_df["balls"]) < (3 - data_df["strikes"])).astype(int)
    data_df["2K"] = (data_df["strikes"] == 2).astype(int)
    return data_df


def load_pitch_data(date_min: str = "2020-01-01", date_max: str = str(datetime.today().date())) -> pd.DataFrame:
    """
    Load, preprocess, and engineer features for a Statcast pitch data DataFrame.

    Parameters:
    -----------
    date_min : str, default="2020-01-01"
            The earliest date to include in the dataset.
    date_max : str, default=today
            The latest date to include in the dataset.

    Returns:
    --------
    pd.DataFrame
            A Pandas DataFrame containing the preprocessed and enriched pitch data.

    Data Loading and Preprocessing:
    ------------------------------
    1. Loads Statcast pitch data within the specified date range using 'retrieve_statcast_pitch_data'.
    2. Applies a data cleaning pipeline using 'filter_data', 'engineer_strike_zone_features', 'engineer_x_neutral_features', and 'engineer_bunt_indicator'.
    3. Parses missingness in the data using 'parse_missingness', dropping inputs with missing features.

    Feature Engineering:
    --------------------
    4. Assigns 'pitch_group' based on 'pitch_type' using predefined mappings.
    5. Assigns 'pitch_group_idx' based on 'pitch_group' using predefined indices.
    6. Creates an 'is_oppo_hand' indicator for opposite-hand platoon configurations.
    7. Calculates 'residual_speed' (effective_speed - release_speed).
    8. Maps 'description' to 'pitch_outcome_category'.
    9. Creates a 'tracking_mask' indicating 'bip' (balls in play) in the categorical response.

    Data Integrity Check:
    ---------------------
    10. Calls 'test_data_integrity' to ensure data integrity and quality at the end of the pipeline.

    Examples:
    ---------
    To load and preprocess Statcast pitch data:

    >>> loaded_data = load_pitch_data()

    The 'loaded_data' DataFrame will contain the cleaned, enriched, and engineered pitch data ready for analysis.
    """
    data_df = retrieve_statcast_pitch_data(date_max=date_max, date_min=date_min)  # FIXME

    # apply data cleaning pipeline
    data_df = (
        data_df.pipe(filter_data)
        .pipe(engineer_strike_zone_features)
        .pipe(engineer_x_neutral_features)
        .pipe(engineer_bunt_indicator, remove_bunts=True)
        .pipe(engineer_count_indicators)
        .pipe(parse_missingness)
    )
    # assign pitch groups
    data_df["pitch_group"] = data_df["pitch_type"].replace(DATA_CONSTANTS.PITCH_GROUP_MAPPINGS)
    # assign indices for slicing the pitch groups
    data_df["pitch_group_idx"] = data_df["pitch_group"].replace(DATA_CONSTANTS.PITCH_GROUP_INDICES)
    # make an indicator for an opposite-hand platoon config
    data_df["is_oppo_hand"] = (data_df.bats != data_df.throws).astype(int)
    # offset effective velo
    data_df["residual_speed"] = data_df.effective_speed - data_df.release_speed
    # engineer the categorical response map
    data_df["pitch_outcome_category"] = data_df["description"].replace(DATA_CONSTANTS.PITCH_OUTCOME_CATEGORY_MAPPINGS)
    # mask out xwoba for non BIPs
    data_df["tracking_mask"] = data_df.pitch_outcome_category.values == "bip"
    # make sure data is in good shape
    test_data_integrity(data_df)

    return data_df
