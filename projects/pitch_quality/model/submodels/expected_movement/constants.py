import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs: basic physical characteristics of the release
XMVMT_INPUTS = {
    "FF": ["arm_angle", "release_speed"],
    "SI": ["arm_angle", "release_speed"],
    "BB": ["arm_angle", "release_speed", "cos_spin_axis_pitcher_neutral", "sin_spin_axis_pitcher_neutral"],
    "OS": ["arm_angle", "release_speed"],
}
# have any/all xMvmt feature, for use in pandas column slicing and munging
XMVMT_INPUTS_UNION = sorted(list(set().union(*XMVMT_INPUTS.values())))
# XMVMT_INPUTS = ["arm_angle", "release_speed"]

# Outputs/targets: basic movement metrics to model. No SSW for now.
XMVMT_OUTPUTS = ["pfx_z", "pfx_x_pitcher_neutral"]

# Prefix for the named column, i.e. the expected movement predictions
XMVMT_PREFIX = "x_"
# Column names for the predictions
XMVMT_PRED_COLNAMES = [XMVMT_PREFIX + item for item in XMVMT_OUTPUTS]

# Minimum sample size to include a pitcher-season in the training set
MIN_XMVMT_PITCHES = 25

# models pickled along this path
XMVMT_OBJECT_PATH = os.path.join(SCRIPT_DIR.replace("submodels", "objects"), "xmvmt_models.pkl")

# parallel prediction configs; Exact GPs need to be chunked at prediction time
JOBLIB_BACKEND = "threading"  # outta my depth but this goes quicker

# DuckDB path
XMVMT_DUCK_DB_PARQUET_PATH = os.path.join(
    os.path.os.environ.get("PYTHONPATH"), "baseball", "duckdb", "model_outputs", "expected_fastball_movement"
)
