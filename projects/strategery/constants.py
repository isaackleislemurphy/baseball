"""Strategy-related constants"""

import numpy as np

# roll it after this many runs
MAX_RUNS_PER_INNING = 15

# game's over when score gap gets this bad
MAX_SCORE_DIFFERENTIAL = 21

# baseball game states
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
]
GAME_STATES_FULL = GAME_STATES + ["---:3"]
RUNS_ARRAY = np.arange(MAX_RUNS_PER_INNING + 1)
