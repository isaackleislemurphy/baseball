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
}


# map pitch types to pitch groupings.
FA_TYPES = "FF", "SI"
BB_TYPES = "FC", "SL", "ST", "CU", "KC", "SC", "SV"
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
