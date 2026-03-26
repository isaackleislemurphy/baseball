import pandas as pd

import baseball.data.savant.pitch.constants as DATA_CONSTANTS
from baseball.duckdb.tables import TABLES
from baseball.utils.duckdb import query

INVALID_PITCH_FILTER = f"""
        -- filter to ensure a "real" pitch offering
        p.pitch_type in {DATA_CONSTANTS.PITCH_TYPES} AND
        -- filter to non-spring training games, i.e. competitive games
        p.game_type in {DATA_CONSTANTS.GAME_TYPES} AND
        -- filter out eephus pitches and "fastballs" from position players pitching
        p.release_speed >= {DATA_CONSTANTS.MIN_VELO} AND
        -- weird pitch types
        p.description not in {DATA_CONSTANTS.INVALID_PITCH_DESCRIPTIONS} AND
        -- no bunts
        p.is_bunt_attempt = 0 AND
        (p.events <> 'sac_bunt' OR p.events IS NULL) AND
        -- no catcher's interferences
        (p.events <> 'catcher_interf' OR p.events IS NULL)
"""


def load_pitch_data(
    date_min: str = "2020-01-01", date_max: str = "2025-12-31", addl_where_clause: str = ""
) -> pd.DataFrame:
    """
    Query cleaned, feature-engineered Statcast pitch-level data from DuckDB-backed parquet files.

    This function executes a single DuckDB SQL query over season-partitioned Statcast
    parquet files to produce a modeling-ready pitch-level table. The query applies
    domain-specific pitch validity filters, engineers batter- and pitcher-neutral
    features, expands count state indicators, and imputes missing xwOBA values for
    balls in play using historical averages by batted-ball type and hit location.

    Parameters
    ----------
    date_min : str, default "2020-01-01"
        Inclusive lower bound on `game_date` for pitches returned.
        Intended to support partial-season or rolling-window queries.
    date_max : str, default "2025-12-31"
        Inclusive upper bound on `game_date` for pitches returned.
    addl_where_clause : str, default = ""
        Additional SQL to include in your filtering, to save on memory etc.
        User expected to have  familiarity with this function and does so at their own risk.

    Returns
    -------
    pd.DataFrame
        A pitch-level DataFrame suitable for downstream modeling. The output includes:
        - Game, inning, and count state information
        - Pitcher- and batter-handedness indicators
        - Batter- and pitcher-neutralized release, movement, and location features
        - Expanded count dummy variables (e.g. "0_0", "3_2")
        - Pitch outcome labels and events
        - Observed and imputed xwOBA values for balls in play

    Notes
    -----
    - Invalid or non-competitive pitches are filtered out using
      `INVALID_PITCH_FILTER`, which excludes bunts, catcher interference, position-
      player pitches, spring training games, and malformed pitch types.
    - For balls in play with missing xwOBA, values are imputed using league-average
      xwOBA conditional on (bb_type, hit_location) computed within the query.
    - All joins and feature engineering are performed in SQL to minimize Python-side
      memory overhead and to leverage DuckDB’s vectorized execution.
    """

    sql = f"""
    WITH xwoba_impute AS (
        SELECT
            IFNULL(bb_type, 'not_recorded') bb_type,
            IFNULL(hit_location, 0) hit_location,
            AVG(xwoba) xwoba,
            AVG(woba_value) woba_value,
            COUNT(*) AS num_hit_type_obs
        FROM '{TABLES.savant.pitch}' p
        WHERE {INVALID_PITCH_FILTER} AND
            pitch_outcome_category = 'bip'
        GROUP BY
            bb_type,
            hit_location
    )
    SELECT
        -- game state info --
        p.season,
        p.game_date,
        p.game_pk,
        p.inning,
        p.inning_topbot,
        p.at_bat_number,
        p.pitch_number,

        -- personnel info --
        p.pitcher,
        p.batter,
        p.throws,
        p.bats,
        p.is_oppo_hand,

        -- count info --
        p.balls,
        p.strikes,
        IF(p.balls = 0 AND p.strikes = 0, 1, 0) "0_0",
        IF(p.balls = 0 AND p.strikes = 1, 1, 0) "0_1",
        IF(p.balls = 0 AND p.strikes = 2, 1, 0) "0_2",
        IF(p.balls = 1 AND p.strikes = 0, 1, 0) "1_0",
        IF(p.balls = 1 AND p.strikes = 1, 1, 0) "1_1",
        IF(p.balls = 1 AND p.strikes = 2, 1, 0) "1_2",
        IF(p.balls = 2 AND p.strikes = 0, 1, 0) "2_0",
        IF(p.balls = 2 AND p.strikes = 1, 1, 0) "2_1",
        IF(p.balls = 2 AND p.strikes = 2, 1, 0) "2_2",
        IF(p.balls = 3 AND p.strikes = 0, 1, 0) "3_0",
        IF(p.balls = 3 AND p.strikes = 1, 1, 0) "3_1",
        IF(p.balls = 3 AND p.strikes = 2, 1, 0) "3_2",

        -- pitch type info --
        p.pitch_type,
        p.pitch_group,

        -- release info --
        p.release_speed,
        p.arm_angle,
        p.release_pos_x,
        IF(p.bats = 'L', -p.release_pos_x, p.release_pos_x) release_pos_x_batter_neutral,
        IF(p.throws = 'L', -p.release_pos_x, p.release_pos_x) release_pos_x_pitcher_neutral,
        p.release_pos_y,
        p.release_pos_z,

        -- spin axis info --
        p.spin_axis,
        IF(p.throws = 'L', 360 - p.spin_axis, p.spin_axis) spin_axis_pitcher_neutral,
        COS(IF(p.throws = 'L', 360 - p.spin_axis, p.spin_axis) * PI() / 180) cos_spin_axis_pitcher_neutral,
        SIN(IF(p.throws = 'L', 360 - p.spin_axis, p.spin_axis) * PI() / 180) sin_spin_axis_pitcher_neutral,

        -- movement info --
        p.pfx_x,
        IF(p.bats = 'L', -p.pfx_x, p.pfx_x) pfx_x_batter_neutral,
        IF(p.throws = 'L', -p.pfx_x, p.pfx_x) pfx_x_pitcher_neutral,
        p.pfx_z,

        -- location --
        p.plate_x,
        IF(p.bats = 'L', -p.plate_x, p.plate_x) plate_x_batter_neutral,
        IF(p.throws = 'L', -p.plate_x, p.plate_x) plate_x_pitcher_neutral,
        p.plate_z,
        p.sz_top,
        p.sz_bot,

        -- outcomes --
        p.events,
        p.description,
        IF(p.pitch_outcome_category = 'bip' AND p.bb_type IS NULL, 'not_recorded', p.bb_type) bb_type,
        IF(p.pitch_outcome_category = 'bip' AND p.hit_location IS NULL, 0, p.hit_location) hit_location,
        p.pitch_outcome_category,

        -- woba outcomes --
        IF(p.pitch_outcome_category = 'bip' AND p.xwoba IS NULL, xwf.xwoba, p.xwoba) xwoba,
        p.xwoba xwoba_raw,
        xwf.xwoba AS xwoba_fill,
        p.woba_value

    FROM '{TABLES.savant.pitch}' p

    LEFT JOIN xwoba_impute xwf ON
        IF(p.pitch_outcome_category = 'bip' AND p.bb_type IS NULL, 'not_recorded', p.bb_type) = xwf.bb_type AND
        IF(p.pitch_outcome_category = 'bip' AND p.hit_location IS NULL, 0, p.hit_location) = xwf.hit_location

    WHERE p.game_date >= '{date_min}' AND
        p.game_date <= '{date_max}' AND
        {INVALID_PITCH_FILTER}
        {addl_where_clause}

    ORDER BY
        p.game_date,
        p.game_pk,
        p.pitcher,
        p.at_bat_number,
        p.pitch_number
    """
    df = query(sql)
    return df
