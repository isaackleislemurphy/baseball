"""Generic, codebase-wide utils"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def str2bool(x: Any) -> bool:
    """
    Converts a value to its boolean equivalent.

    Parameters
    ----------
    x : Any
        The input value to convert. Expected types are bool, int, float, or
        a string representing a truth value.

    Returns
    -------
    bool
        The boolean representation of the input.

    Raises
    ------
    ValueError
        If the string cannot be parsed or the type is unsupported.
    """
    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return x > 0

    if isinstance(x, str):
        # Strip whitespace and standardize case for safer matching
        x_clean = x.strip().lower()

        if x_clean in {"t", "true", "y", "yes", "on", "1"}:
            return True
        elif x_clean in {"f", "false", "n", "no", "off", "0"}:
            return False

    # A more descriptive error message helps with debugging logs
    raise ValueError(f"Input to `str2bool()` makes no sense; cannot {type(x).__name__} {x!r} to bool.")


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


def write_pickled_object(obj: Any, path: str) -> None:
    """
    Serialize a Python object to disk using the highest pickle protocol.

    Parameters
    ----------
    obj : Any
        The Python object to be serialized.
    path : str
        The destination file path where the object will be saved.

    Returns
    -------
    None

    Notes
    -----
    - This function automatically creates any missing parent directories for the
      specified path using `mkdir(parents=True)`.
    - It uses `pickle.HIGHEST_PROTOCOL` to ensure the most efficient serialization
      available for the running Python version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_object(path: str) -> Any:
    """
    Deserialize and load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        The file path pointing to the pickled object.

    Returns
    -------
    Any
        The deserialized Python object.
    """
    path = Path(path)

    with path.open("rb") as f:
        return pickle.load(f)


def read_yaml(file_path: str) -> Any:
    """
    Read a YAML file and return its contents as a Python object.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def make_gitkeep(dir_path: str) -> None:
    """
    Create an empty .gitkeep file in the given directory so Git
    tracks the (otherwise empty) folder.

    Parameters
    -----------
    dir_path : str
        Directory that should be kept.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    gitkeep_path = dir_path / ".gitkeep"
    gitkeep_path.touch(exist_ok=True)
    return gitkeep_path
