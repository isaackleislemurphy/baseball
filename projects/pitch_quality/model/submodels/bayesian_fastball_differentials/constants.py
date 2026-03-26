import os

FB_DIFF_COLS = ["release_speed", "release_pos_z", "pfx_x", "pfx_z", "arm_angle"]
DIM = len(FB_DIFF_COLS)
FA_TYPES = ("FF", "SI")
KEY_COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]

# filepath crap
DIR_PATH = os.environ.get("PYTHONPATH")
# player-season means and player-season-game means will live in this general folder
FASTBALL_DIFF_PATH = os.path.join(DIR_PATH, "baseball", "duckdb", "model_outputs", "smoothed_fa_shapes")

# ---------------------- #
# DUCK DB PATHS
# -----------------------#
# this will be the general database for player-season fastball shapes
SMOOTHED_FA_PLAYER_MEANS_DUCK_DB_PATH = os.path.join(FASTBALL_DIFF_PATH, "player_season")
# same deal, but for the player-season-game means
SMOOTHED_FA_PLAYER_GAME_MEANS_DUCK_DB_PATH = os.path.join(FASTBALL_DIFF_PATH, "player_season_game")


# ---------------------- #
# DUCK DB FULL FILENAMES
# -----------------------#
# specific filename for those files (one for each season)
SMOOTHED_FA_PLAYER_MEANS_DUCK_DB_FILENAME = os.path.join(
    SMOOTHED_FA_PLAYER_MEANS_DUCK_DB_PATH, "smoothed_player_means_{season}.parquet"
)
SMOOTHED_FA_PLAYER_GAME_MEANS_DUCK_DB_FILENAME = os.path.join(
    SMOOTHED_FA_PLAYER_GAME_MEANS_DUCK_DB_PATH, "smoothed_player_game_means_{season}.parquet"
)

# test pitchers: nola, wheeler, kerkering, hoff, zeus, strahm, banks, skenes, fairbanks
TEST_SUBSET_PITCHERS = (605400, 554430, 689147, 656046, 666200, 621381, 621383, 694973, 664126)
