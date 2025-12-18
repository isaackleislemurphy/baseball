""" """

import pandas as pd

from baseball.data.savant.pitch.etl import load_pitch_data
from baseball.projects.pitch_quality.data.constants import CATEGORICAL_RESPONSE_INDICES
from baseball.constants import SHOHEI_OHTANI
from baseball.projects.pitch_quality.data.cache_training_partitions import load_training_partitions
from baseball.projects.pitch_quality.model.constants import TRAIN_TEST_CUTOFF_DATE


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
    pitch_data_df = load_pitch_data(date_min=date_min, date_max=date_max)

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


def filter_to_training_data(pitch_data_df: pd.DataFrame) -> pd.DataFrame:
    """ """

    # load pitch data, and trim it under `TRAIN_TEST_CUTOFF_DATE`
    pitch_data_df_train = pitch_data_df.query(f"game_date <= '{TRAIN_TEST_CUTOFF_DATE}'")

    # restrict to training pitchers
    partition_df = load_training_partitions().query("train == 1")
    pitch_data_df_train = pitch_data_df_train.merge(partition_df[["pitcher"]], on="pitcher")
    return pitch_data_df_train
