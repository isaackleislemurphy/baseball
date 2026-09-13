import os
from types import SimpleNamespace

from baseball.duckdb.registry import REGISTRY
from baseball.utils.general import make_gitkeep, read_yaml

DUCKDB_DIR = os.path.join(os.environ.get("PYTHONPATH"), "baseball", "duckdb")
DB_DIR = os.path.join(DUCKDB_DIR, "db")


def register_tables(gitkeep: bool = True) -> None:
    """
    Ensures filepaths are set up for duckdb table parquets, so that data
    can be stored in the appropriate spot.

    Parameters
    ----------
    gitkeep : bool, default = True
        If True, a `.gitkeep` file will be added to each registered table.
    """

    os.makedirs(DB_DIR, exist_ok=True)
    for item in REGISTRY:
        # pull in table configs
        table_config = read_yaml(os.path.join(DUCKDB_DIR, item))

        # configure the namespace, if not already done so.
        namespace_dir = os.path.join(DB_DIR, table_config["namespace"])
        os.makedirs(namespace_dir, exist_ok=True)

        # configure the specific table
        table_dir = os.path.join(namespace_dir, table_config["table_name"])
        os.makedirs(table_dir, exist_ok=True)

        # add a gitkeep, if desired
        if gitkeep and ".gitkeep" not in os.listdir(table_dir):
            make_gitkeep(table_dir)


class DuckNamespaceContainer:
    """
    Exposes the DuckDB parquet tables in `SCHEMA` as nested attributes, so a table
    is reachable as `TABLES.<namespace>.<table>` (e.g. `TABLES.chadwick.ids`).
    """

    def __init__(self) -> None:
        """init fn"""
        register_tables()

        # pull all the tables (flat)
        tables = [read_yaml(os.path.join(DUCKDB_DIR, table)) for table in REGISTRY]

        # pull all the namespaces
        namespaces = set([table["namespace"] for table in tables])

        # nest the tables within their namespaces
        tables_nested = {
            namespace: {
                tbl_config["table_name"]: self.parse_filepath(tbl_config)
                for tbl_config in tables
                if tbl_config["namespace"] == namespace
            }
            for namespace in namespaces
        }
        # make containers for each namespace
        for namespace, tbls in tables_nested.items():
            self.add_attribute(namespace, SimpleNamespace(**tbls))

    def parse_filepath(self, table_config: dict) -> str:
        """"""
        return os.path.join(
            DB_DIR,
            table_config["namespace"],  # namespace
            table_config["table_name"],  # table
            table_config["parquet_read_file"],  # parquets
        )

    def add_attribute(self, name, value):
        """
        Sets `name` as an attribute holding `value`.

        Parameters
        ----------
        name : str
            Attribute name (the namespace).
        value : Any
            Attribute value (the namespace's table container).

        Returns
        -------
        None
        """
        # name is a string, value is anything
        setattr(self, name, value)


if __name__ == "__main__":
    TABLES = DuckNamespaceContainer()
    breakpoint()

    print("complete")
