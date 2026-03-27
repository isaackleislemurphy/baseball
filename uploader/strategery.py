"""
This script serves as a quasi-uploader/refresher function for all of the code in
`projects/strategery`. Specifically it:

1.) Re-calculates RE24 values, and writes them to `duckdb/strategery/re24` as a DuckDB-able parquet.
2.) Re-calculates win probabilities (on top of the refreshed RE24 values), and writes them to
    `duckdb/strategery/winprob` as a DuckDB-able parquet.

3.) TODO: Run the PE288 uploader

"""

from baseball.projects.strategery.run_expectancy import calculate_and_save_re24
from baseball.projects.strategery.win_probability import calculate_and_save_win_probs


def upload_strategery() -> None:
    """Runs the uploader"""
    calculate_and_save_re24()
    calculate_and_save_win_probs()


if __name__ == "__main__":
    upload_strategery()
