""" """

import os

from baseball.data.savant.pitch.constants import SAVANT_DUCK_DB_PARQUET_FILENAME, SAVANT_DUCK_DB_PARQUET_PATH


def get_season_savant_duckdb_filepath(season: int) -> str:
    """Gets the DuckDB parquet filepath for a season's worth of pitch data from Savant."""
    return os.path.join(SAVANT_DUCK_DB_PARQUET_PATH, SAVANT_DUCK_DB_PARQUET_FILENAME.format(season=season))
