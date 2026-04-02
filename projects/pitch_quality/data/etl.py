""" """

import pandas as pd

from baseball.constants import SHOHEI_OHTANI
from baseball.data.savant.pitch.load import make_sql_load_pitch_data
from baseball.duckdb.tables import TABLES
from baseball.projects.pitch_quality.data.cache_training_partitions import load_training_partitions
from baseball.projects.pitch_quality.data.constants import CATEGORICAL_RESPONSE_INDICES
from baseball.projects.pitch_quality.model.constants import TRAIN_TEST_CUTOFF_DATE
from baseball.utils.duckdb import query


def assign_probable_playing_totals(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify players as 'likely batter' or 'likely pitcher' based on seasonal usage.

    This function aggregates Plate Appearances (PA) and Batters Faced (BF) for every player
    in a given season. It creates binary flags to filter out position players pitching
    or pitchers batting during analysis. Kind of hacky, but Chadwick Github is spotty
    particularly with recent call ups so just doing it this way.

    Parameters
    ----------
    pitch_df : pd.DataFrame
        Standard pitch-level DataFrame containing 'game_pk', 'at_bat_number', 'pitch_number',
        'batter', 'pitcher', and 'season'.

    Returns
    -------
    pd.DataFrame
        A DataFrame keyed by ['season', 'player'] with the following columns:
        - 'pas': Total plate appearances as a batter.
        - 'bfs': Total batters faced as a pitcher.
        - 'is_likely_pitcher': 1.0 if BF >= PA (or if player is Shohei Ohtani).
        - 'is_likely_batter': 1.0 if PA >= BF (or if player is Shohei Ohtani).

    Notes
    -----
    Includes a hardcoded exception for Shohei Ohtani, who is flagged as both a likely
    pitcher and a likely batter regardless of the specific PA/BF ratio for that season.
    """
    # roll up pitches --> PAs
    pa_df = (
        pitch_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .tail(1)
    )
    # tally batter PAs
    batter_pas = (
        pa_df.rename(columns={"batter": "player"})
        .assign(pas=1)
        .groupby(["season", "player"], as_index=False)["pas"]
        .sum()
    )
    # tally pitcher BFs
    pitcher_bfs = (
        pa_df.rename(columns={"pitcher": "player"})
        .assign(bfs=1)
        .groupby(["season", "player"], as_index=False)["bfs"]
        .sum()
    )

    # combine the two
    playing_totals = batter_pas.merge(pitcher_bfs, on=["season", "player"], how="outer").fillna(0)

    # identify likely pitchers or batters
    playing_totals["is_likely_pitcher"] = (
        (playing_totals["bfs"] >= playing_totals["pas"]) | (playing_totals["player"] == SHOHEI_OHTANI)
    ).values.astype(float)
    playing_totals["is_likely_batter"] = (
        (playing_totals["bfs"] <= playing_totals["pas"]) | (playing_totals["player"] == SHOHEI_OHTANI)
    ).values.astype(float)

    return playing_totals


def load_pitch_quality_model_data(date_min: str = "2020-01-01", date_max: str = "2025-12-01") -> pd.DataFrame:
    """
    Load pitch data and filter for valid pitcher-vs-batter matchups.

    This function orchestrates the data pipeline for pitch quality modeling by:
    1. Loading raw pitch data for the specified date range.
    2. Encoding pitch outcomes into categorical indices.
    3. Filtering out "novelty" matchups (e.g., position players pitching) to ensure
       the model trains only on professional-standard pitching and hitting interactions.

    Parameters
    ----------
    date_min : str, default "2025-01-01"
        The start date for data retrieval (YYYY-MM-DD).
    date_max : str, default "2025-08-01"
        The end date for data retrieval (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing pitch-level data where both the pitcher and
        batter are deemed "likely" occupants of their respective roles.
    """
    # load the pitch-level data
    sql = f"""
    WITH pitch AS (
        {make_sql_load_pitch_data(date_min=date_min, date_max=date_max)}
    ),
    fa AS (
        SELECT * FROM '{TABLES.model_outputs.smoothed_fastball_shapes_player_season_game}'
    )
    SELECT
        p.*,
        fa.game_idx,
        --- FF diffs ---
        fa.release_speed_FF - p.release_speed AS release_speed_FF_delta,
        fa.release_pos_z_FF - p.release_pos_z AS release_pos_z_FF_delta,
        fa.pfx_z_FF - p.pfx_z AS pfx_z_FF_delta,
        fa.arm_angle_FF - p.arm_angle AS arm_angle_FF_delta,
        IF(p.bats = 'L', -1, 1) * (fa.pfx_x_FF - p.pfx_x) AS pfx_x_batter_neutral_FF_delta,

        --- SI diffs ---
        fa.release_speed_SI - p.release_speed AS release_speed_SI_delta,
        fa.release_pos_z_SI - p.release_pos_z AS release_pos_z_SI_delta,
        fa.pfx_z_SI - p.pfx_z AS pfx_z_SI_delta,
        fa.arm_angle_SI - p.arm_angle AS arm_angle_SI_delta,
        IF(p.bats = 'L', -1, 1) * (fa.pfx_x_SI - p.pfx_x) AS pfx_x_batter_neutral_SI_delta,

    FROM pitch p
    LEFT JOIN fa USING(pitcher, season, game_pk, game_date)
    """
    pitch_data_df = query(sql)

    # make the categorical response index
    pitch_data_df["categorical_response_idx"] = pitch_data_df["pitch_outcome_category"].replace(
        CATEGORICAL_RESPONSE_INDICES
    )

    # identify likely batters and hitters
    playing_totals = assign_probable_playing_totals(pitch_data_df)

    # identify real batters and pitchers
    valid_batters = playing_totals.query("is_likely_batter == 1").rename(columns={"player": "batter"})[
        ["season", "batter"]
    ]
    valid_pitchers = playing_totals.query("is_likely_pitcher == 1").rename(columns={"player": "pitcher"})[
        ["season", "pitcher"]
    ]

    # filter out pitchers batting, and batters pitching
    pitch_data_df = pitch_data_df.merge(valid_batters, on=["season", "batter"]).merge(
        valid_pitchers, on=["season", "pitcher"]
    )

    return pitch_data_df


def partition_pitch_data(pitch_data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split pitch data into Training, Test-Identity, and Future-Temporal subsets.

    This function joins the input data with the pitcher partition map to separate
    rows based on both time (pre/post cutoff) and pitcher identity (train/test split).

    Parameters
    ----------
    pitch_data_df : pd.DataFrame
        The raw pitch data to be split. Must contain 'game_date' and 'pitcher'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames:
        1. **Training Set:** Training pitchers (training_player == 1) in games on or before `TRAIN_TEST_CUTOFF_DATE`.
        2. **Test Identity Set:** All rows for pitchers designated as test subjects (training_player == 0),
            regardless of date.
        3. **Future Set:** All rows for games occurring after `TRAIN_TEST_CUTOFF_DATE`, regardless of pitcher identity.

    Notes
    -----
    - Uses a left join to attach partition flags; pitchers missing from the partition file
      are defaulted to `training_player = 0` (Test Identity).
    - **Overlap Warning:** The subsets are not mutually exclusive. A pitch thrown by a
      test-set pitcher after the cutoff date will appear in both the **Test Identity Set** and the **Future Set**.
    """
    # load the cached partitions
    partition_df = load_training_partitions()[["pitcher", "training_player"]]

    # join them onto `pitch_data_df`, but make sure you didn't alter the rows
    dim_in = pitch_data_df.shape[0]
    pitch_data_df_ = pitch_data_df.merge(partition_df, on="pitcher", how="left").fillna(value={"training_player": 0})
    dim_out = pitch_data_df_.shape[0]
    assert dim_in == dim_out

    return (
        pitch_data_df_.query(f"game_date <= '{TRAIN_TEST_CUTOFF_DATE}' & training_player == 1").reset_index(drop=True),
        pitch_data_df_.query("training_player == 0").reset_index(drop=True),
        pitch_data_df_.query(f"game_date > '{TRAIN_TEST_CUTOFF_DATE}'").reset_index(drop=True),
    )
