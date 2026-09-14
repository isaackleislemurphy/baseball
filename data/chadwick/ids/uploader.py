"""
Used to pull player ID CSVs off of Chadwick baseball person database. Helpful for FG <--> BAM ID
mapping.

CSVs live here: https://github.com/chadwickbureau/register/tree/master/data
"""

import string
import urllib

import pandas as pd
from pybaseball import chadwick_register
from unidecode import unidecode

from baseball.data.chadwick.ids.constants import CHADWICK_PEOPLE_CSV_LINK
from baseball.duckdb.database import write_parquet
from baseball.utils.logging import get_logger

LOGGER = get_logger()


def scrape_raw_chadwick_people_csvs() -> pd.DataFrame:
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

    # rename so that IDs are not ``key_<source>``, but rather ``source_<id>``
    id_df = id_df.rename(
        columns={col: col.replace("key_", "") + "_id" for col in id_df.columns if col.startswith("key_")}
    )

    return id_df


def upload_chadwick_ids() -> None:
    """
    Fetches Chadwick player IDs using pybaseball, cleans the data, and caches it locally.

    This function retrieves the Chadwick register, removes accent marks from the
    players' first and last names (via unidecode) to facilitate easier text matching,
    and saves the resulting DataFrame to a local Parquet file.

    Returns
    -------
    None
    """
    LOGGER.info("Beginning Chadwick upload.")

    # load the IDs and names
    id_df = chadwick_register()
    LOGGER.info("Chadwick IDs ingested")

    # strip out accent marks from names to make lookups easier
    for col in ("name_last", "name_first"):
        id_df[col] = [unidecode(item) if isinstance(item, str) else item for item in id_df[col]]
    LOGGER.info("Chadwick names converted to A-Z lettering.")

    # rename so that IDs are not ``key_<source>``, but rather ``source_<id>``
    id_df = id_df.rename(
        columns={col: col.replace("key_", "") + "_id" for col in id_df.columns if col.startswith("key_")}
    )
    LOGGER.info("ID columns renamed")

    write_parquet(id_df, "duckdb/table_config/chadwick__ids.yaml")
    LOGGER.info("Chadwick data written.")


def main() -> None:
    """Main function"""
    upload_chadwick_ids()


if __name__ == "__main__":
    main()
