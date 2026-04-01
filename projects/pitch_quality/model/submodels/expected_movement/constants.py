import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XMVMT_OBJECT_PATH = os.path.join(SCRIPT_DIR.replace("submodels", "objects"), "xmvmt_models.pkl")
JOBLIB_BACKEND = "threading"  # outta my depth but this goes quicker
XMVMT_DUCK_DB_PARQUET_PATH = os.path.join(os.path.os.environ.get("PYTHONPATH"), "duckdb", "expected_fastball_movement")
