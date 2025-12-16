""" """

from datetime import datetime
import numpy as np
import pandas as pd

import pybaseball as pb

from baseball.data.chadwick.ids import load_raw_chadwick_people_csvs
from baseball.data.savant.pitch.etl import load_pitch_data
from baseball.projects.pitch_value.data.constants import CATEGORICAL_RESPONSE_INDICES
from baseball.constants import SHOHEI_OHTANI


def assign_probable_playing_totals(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns probable player roles (batter or pitcher) based on plate appearances (PAs) and batters faced (BFs)
    in a given season. If a player has more PA than TBF in a given season, they're labeled as a likely batter.
    Conversely, if they have more TBF than PA in a given season, they're labeled as a likely pitcher.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'season': The season of the data.
            - 'player': The identifier (usually MLBAM) for the player (batter or pitcher).
            - 'pas': The total plate appearances for the player (if a batter).
            - 'bfs': The total batters faced for the player (if a pitcher).
            - 'is_likely_pitcher': A binary indicator (1.0 or 0.0) denoting whether the player is more likely
                a pitcher.
            - 'is_likely_batter': A binary indicator (1.0 or 0.0) denoting whether the player is more likely
                a batter.
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


def load_pitch_quality_model_data(date_min: str = "2025-01-01", date_max: str = "2025-08-01") -> pd.DataFrame:
    """ """
    # load the pitch-level data
    data_df = load_pitch_data(date_min="2023-01-01", date_max="2023-08-01")

    # make the categorical response index
    data_df["categorical_response_idx"] = data_df["pitch_outcome_category"].replace(CATEGORICAL_RESPONSE_INDICES)

    # identify likely batters and hitters
    playing_totals = assign_probable_playing_totals(data_df)

    # identify real batters and pitchers
    valid_batters = playing_totals.query("is_likely_batter == 1").rename(columns={"player": "batter"})[
        ["season", "batter"]
    ]
    valid_pitchers = playing_totals.query("is_likely_pitcher == 1").rename(columns={"player": "pitcher"})[
        ["season", "pitcher"]
    ]

    # filter out pitchers batting, and batters pitching
    data_df = data_df.merge(valid_batters, on=["season", "batter"]).merge(valid_pitchers, on=["season", "pitcher"])

    return data_df
