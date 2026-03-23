"""Strategy-related constants"""

import os

import numpy as np

MAX_RUNS_PER_INNING = 15
GAME_STATES = [
    "---:0",
    "---:1",
    "---:2",
    "--3:0",
    "--3:1",
    "--3:2",
    "-2-:0",
    "-2-:1",
    "-2-:2",
    "-23:0",
    "-23:1",
    "-23:2",
    "1--:0",
    "1--:1",
    "1--:2",
    "1-3:0",
    "1-3:1",
    "1-3:2",
    "12-:0",
    "12-:1",
    "12-:2",
    "123:0",
    "123:1",
    "123:2",
    "---:3",
]
RUNS_ARRAY = np.arange(MAX_RUNS_PER_INNING + 1)

DB_PATH = os.path.join(os.environ.get("PYTHONPATH"), "baseball", "duckdb", "strategery")
RE24_PATH = os.path.join(DB_PATH, "re24")
PE288_PATH = os.path.join(DB_PATH, "pe288")
