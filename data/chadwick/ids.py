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
    """ """
    # load the IDs and names
    id_df = chadwick_register()

    # strip out accent marks from names to make lookups easier
    for col in ("name_last", "name_first"):
        id_df[col] = [unidecode(item) if isinstance(item, str) else item for item in id_df[col]]

    # save it to the "database"
    id_df.to_parquet(os.path.join(CHADWICK_ID_PARQUET))


def load_chadwick_ids_query() -> str:
    """ """
    return f"""
    SELECT * FROM
    '{CHADWICK_ID_PARQUET}'
    """


def load_chadwick_ids() -> pd.DataFrame:
    """ """
    return query(load_chadwick_ids_query())


if __name__ == "__main__":
    cache_chadwick_ids()
