"""Chadwick-related constants"""

import os

DIR_PATH = os.environ.get("PYTHONPATH")
CHADWICK_DUCK_DB_PARQUET_PATH = os.path.join(DIR_PATH, "baseball", "duckdb", "chadwick", "ids", "chadwick_ids.parquet")

CHADWICK_PEOPLE_CSV_LINK = (
    "https://raw.githubusercontent.com/chadwickbureau/register/refs/heads/master/data/people-{i}.csv"
)
