""" """

import os

CATEGORICAL_RESPONSE_INDICES = {
    "hbp": 0,
    "ball": 1,
    "called_strike": 2,
    "whiff": 3,
    "foul": 4,
    "bip": 5,
}


# ---------------------------------------------------------
# Constants for making reproducible train/test partitions
# ---------------------------------------------------------

# where are we
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed seed for the random number generator to ensure the train/test split
# is identical every time the code is run (reproducibility).
TRAINING_PARTITION_SEED = 2025

# The number of quantile bins to group pitchers into based on their arm angle.
# This ensures the split is stratified: we want a roughly equal distribution of
# slots acrosss both the training and test sets.
NUM_ARM_ANGLE_STRATIFICATION_BINS = 20

# Where the partition cache file will be saved.
PARTITION_CSV_FOLDER = "csvs"
PARTITION_CSV_FILENAME = "pitcher-id-partitions.csv"
PARTITION_CSV_PATH = os.path.join(SCRIPT_DIR, PARTITION_CSV_FOLDER, PARTITION_CSV_FILENAME)
