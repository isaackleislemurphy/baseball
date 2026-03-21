"""
Used to pull player ID CSVs off of Chadwick baseball person database. Helpful for FG <--> BAM ID
mapping.

CSVs live here: https://github.com/chadwickbureau/register/tree/master/data
"""

import os
import string
import urllib

import pandas as pd
from pybaseball import chadwick_register
from unidecode import unidecode

from baseball.utils.duckdb import query

DIR_PATH = os.environ.get("PYTHONPATH")
CHADWICK_ID_PATH = os.path.join(DIR_PATH, "baseball", "duckdb", "chadwick", "ids")
CHADWICK_ID_PARQUET = os.path.join(CHADWICK_ID_PATH, "chadwick_ids.parquet")


CHADWICK_PEOPLE_CSV_LINK = (
    "https://raw.githubusercontent.com/chadwickbureau/register/refs/heads/master/data/people-{i}.csv"
)


def load_raw_chadwick_people_csvs() -> pd.DataFrame:
    """
    Chadwick stuffs its people data here:
        https://github.com/chadwickbureau/register/tree/master/data

    This script iterates through files, reads them in, and then smushes them into a
    jumbo ID mapper. Nothing is changed from the Chadwick format.

    Returns
    -------
    pd.DataFrame
        The compiled Chadwick people database, as is
    """
    # initialize list to store everything, and a counter
    id_df, iter = [], 0

    # first, read in numeric chadwicks
    while True:
        print(f"...loading Chadwick CSV indexed at {iter}")
        try:
            id_df += [pd.read_csv(CHADWICK_PEOPLE_CSV_LINK.format(i=iter))]
        except urllib.error.HTTPError:
            print(f"No Chadwick CSV indexed at {iter}...stopping at {iter - 1}")
            break
        iter += 1

    # second, read in letter-based chadwicks.
    letters, iter = string.ascii_lowercase, 0
    while True:
        print(f"...loading Chadwick CSV lettered at {letters[iter]}")
        try:
            id_df += [pd.read_csv(CHADWICK_PEOPLE_CSV_LINK.format(i=letters[iter]))]
        except urllib.error.HTTPError:
            print(f"No Chadwick CSV lettered at '{letters[iter]}'...stopping at {letters[iter - 1]}")
            break
        iter += 1

    # put it all together
    id_df = pd.concat(id_df, axis=0).reset_index(drop=True)
    # cast BAM to int
    id_df["key_mlbam"] = id_df["key_mlbam"].values.astype(int)

    return id_df


def cache_chadwick_ids() -> None:
    """
    Fetches Chadwick player IDs using pybaseball, cleans the data, and caches it locally.

    This function retrieves the Chadwick register, removes accent marks from the
    players' first and last names (via unidecode) to facilitate easier text matching,
    and saves the resulting DataFrame to a local Parquet file.

    Returns
    -------
    None
    """
    # load the IDs and names
    id_df = chadwick_register()

    # strip out accent marks from names to make lookups easier
    for col in ("name_last", "name_first"):
        id_df[col] = [unidecode(item) if isinstance(item, str) else item for item in id_df[col]]

    # save it to the "database"
    id_df.to_parquet(os.path.join(CHADWICK_ID_PARQUET))


def load_chadwick_ids_query() -> str:
    """
    Generates the DuckDB SQL query needed to load the cached Chadwick IDs.

    Returns
    -------
    str
        A SQL query string selecting all records from the cached Parquet file.
    """
    return f"""
    SELECT * FROM
    '{CHADWICK_ID_PARQUET}'
    """


def load_chadwick_ids() -> pd.DataFrame:
    """
    Executes the DuckDB query to load the cached Chadwick player IDs into memory.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the locally cached Chadwick player IDs and names.
    """
    return query(load_chadwick_ids_query())


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
        load_chadwick_ids()
        .sort_values(["name_first", "name_last", "mlb_played_last"], ascending=False)
        .query(f"name_last == '{name_last}' & name_first == '{name_first}'")
        .key_mlbam.iloc[0]
    )
    return bam_id


if __name__ == "__main__":
    cache_chadwick_ids()
