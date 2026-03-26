""" """

import pandas as pd

import duckdb


def query(sql: str) -> pd.DataFrame:
    """Helper to query from DuckDB"""
    con = duckdb.connect()
    df = con.execute(sql).df()
    return df
