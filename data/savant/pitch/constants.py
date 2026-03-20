"""
Data-related constants.

Statcast-defined pitch mappings

        {
                "SL": "Slider",
                "FF": "4-Seam Fastball",
                "FC": "Cutter",
                "SI": "Sinker",
                "CH": "Changeup",
                "CU": "Curveball",
                "ST": "Sweeper",
                "KC": "Knuckle Curve",
                "FS": "Split-Finger",
                "PO": "Pitch Out",
                "SV": "Slurve",
                "KN": "Knuckleball",
                "EP": "Eephus",
                "FA": "Other",
                "FO": "Forkball",
                "CS": "Slow Curve",
                "SC": "Screwball"
        }
"""

import itertools
import os

# O.D. 2016: https://en.wikipedia.org/wiki/2016_Major_League_Baseball_season
MIN_STATCAST_DATE = "2016-04-03"

# Game types to include
GAME_TYPES = "W", "L", "D", "F", "R"

# don't like these statcast column names, so renaming them
# to something more intuitive
STATCAST_RENAMINGS = {
    "stand": "bats",
    "p_throws": "throws",
    "game_year": "season",
    "estimated_woba_using_speedangle": "xwoba",
    "estimated_slg_using_speedangle": "xslg",
    "estimated_ba_using_speedangle": "xba",
}

COUNTS = [f"{balls}_{strikes}" for balls, strikes in itertools.product(range(4), range(3))]


# map pitch types to pitch groupings.
FA_TYPES = "FF", "SI"
BB_TYPES = "FC", "SL", "ST", "CU", "KC", "SV"
OS_TYPES = "FS", "CH", "FO", "SC"
PITCH_TYPES = FA_TYPES + BB_TYPES + OS_TYPES

PITCH_GROUP_MAPPINGS = {
    **{key: "FA" for key in FA_TYPES},
    **{key: "BB" for key in BB_TYPES},
    **{key: "OS" for key in OS_TYPES},
}

# indices for pitch groups, if it ever comes in handy
PITCH_GROUP_INDICES = {"FA": 0, "BB": 1, "OS": 2}

# expect pitches to be thrown at least this hard
MIN_VELO = 66

PITCH_OUTCOME_CATEGORY_MAPPINGS = {
    "called_strike": "called_strike",
    "ball": "ball",
    "foul": "foul",
    "hit_into_play": "bip",
    "blocked_ball": "ball",
    "swinging_strike": "whiff",
    "swinging_strike_blocked": "whiff",
    "foul_tip": "foul",
    "hit_by_pitch": "hbp",
}

# if the `description` field is this, ignore the pitch
INVALID_PITCH_DESCRIPTIONS = ("intent_ball", "unknown_strike")


# where are we
DIR_PATH = os.environ.get("PYTHONPATH")
SAVANT_DUCK_DB_PARQUET_PATH = os.path.join(DIR_PATH, "baseball", "duckdb", "pitch", "savant")
SAVANT_DUCK_DB_PARQUET_FILENAME = "raw_savant_pitch_data_{season}.parquet"


# TODO: deprecatet these

# Where the partition cache file will be saved.
RAW_PITCH_CSV_FOLDER = "csvs"

RAW_PITCH_CSV_FILENAME = "raw-pitch-df.csv"
RAW_PITCH_PARQUET_FILENAME = "raw-pitch-df.parquet"

RAW_PITCH_CSV_PATH = os.path.join(DIR_PATH, RAW_PITCH_CSV_FOLDER, RAW_PITCH_CSV_FILENAME)
RAW_PITCH_PARQUET_PATH = os.path.join(DIR_PATH, RAW_PITCH_CSV_FOLDER, RAW_PITCH_PARQUET_FILENAME)
