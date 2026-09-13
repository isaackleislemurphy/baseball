"""
Utils for ingesting Chadwick data / player IDs
"""

import pandas as pd

from baseball.duckdb.database import TABLES
from baseball.utils.duckdb import query


def query_chadwick_ids() -> pd.DataFrame:
    """
    Executes the DuckDB query to load the cached Chadwick player IDs into memory.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the locally cached Chadwick player IDs and names.
    """
    sql = f"""
    SELECT
        *
    FROM '{TABLES.chadwick.ids}'
    """
    return query(sql)


def get_bam_id_from_name(full_name: str) -> int:
    """
    Gets a player's MLBAM ID from their full name, e.g. "Rhys Hoskins" gives you 656555.

    Parameters
    ----------
    full_name : str
        The full name of the player, e.g. "Rhys Hoskins"

    Returns
    -------
    bam_id : int
        That player's MLBAM ID, e.g. 656555
    """
    # parse the name
    name_first, name_last = full_name.split(" ")

    # extract the corresponding BAM ID. TODO: error-handling here
    bam_id = (
        query_chadwick_ids()
        .sort_values(["name_first", "name_last", "mlb_played_last"], ascending=False)
        .query(f"name_last == '{name_last}' & name_first == '{name_first}'")
        .mlbam_id.iloc[0]
    )
    return bam_id
