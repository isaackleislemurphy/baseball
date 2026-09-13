""" """

import duckdb
import pandas as pd


def query(sql: str) -> pd.DataFrame:
    """
    Helper to query from DuckDB

    Parameters
    ----------
    sql : str
        SQL-like code to hit DuckDB

    Returns
    -------
    pd.DataFrame
        The dataframe queried via duckdb.
    """
    con = duckdb.connect()
    df = con.execute(sql).df()
    return df
