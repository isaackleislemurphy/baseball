"""Generic, codebase-wide utils"""

import pickle
from pathlib import Path
from typing import Any
import pandas as pd


def get_train_test_cut_date(df: pd.DataFrame, date_col: str, pct_test: float = 0.25) -> Any:
    """Gets a cut date for train/test partitioning purposes.

    Given a dataframe of dates, with various observations corresponding to each such date, this function
    sorts the dates old --> new, and then identifies a date such that roughly `pct_test` of observations come
    after the cut date and `1 - pct_test` come before.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which you'll extract the partition date
    date_col : str
        The column in `dataframe` with the relevant dates
    pct_test : float, optional
        The percentage of "test" data you want to come after the cut date.
        Default is 0.25.

    Returns
    -------
    Any
        A date-like object to serve as the cut / partition point.

    """
    # sort number of observations per day
    dates = df.sort_values(date_col).assign(n_obs=1).groupby([date_col], as_index=False)["n_obs"].sum()
    # count what cumulative percentage of the data each day's observations represent
    dates["n_obs"] = dates["n_obs"].cumsum() / dates["n_obs"].sum()
    # find a cut date such that `pct_test` of the data can be withheld OOS
    return dates.query(f"n_obs <= {1 - pct_test}")[date_col].iloc[-1]


def write_pickled_object(obj: Any, path: str | Path) -> None:
    """
    Serialize an object to disk using pickle.

    - Creates parent directories if needed
    - Uses highest protocol for speed/size
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_object(path: str | Path) -> Any:
    """
    Load a pickled object from disk.
    """
    path = Path(path)

    with path.open("rb") as f:
        return pickle.load(f)
