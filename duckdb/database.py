"""
DuckDB table registry and namespace access layer.

This module scaffolds the on-disk parquet storage for DuckDB tables and exposes
them as nested attributes for convenient access.

Tables are declared in ``registry.REGISTRY`` and configured via per-table YAML
files (see ``table_config/``). Each config specifies a ``namespace``,
``table_name``, ``parquet_write_file``, ``parquet_read_file``, and a ``schema``.

The module provides two entry points:

- ``register_tables`` : Ensures the directory structure
  (``db/<namespace>/<table_name>/``) exists for every registered table,
  optionally seeding each with a ``.gitkeep``.
- ``DuckNamespaceContainer`` : Reads all registered configs and exposes each
  table's parquet read path as ``TABLES.<namespace>.<table_name>``
  (e.g. ``TABLES.chadwick.ids``).

Attributes
----------
DUCKDB_DIR : str
    Absolute path to the ``baseball/duckdb`` package directory, derived from
    the ``PYTHONPATH`` environment variable.
DB_DIR : str
    Absolute path to the ``db`` directory where namespaced parquet tables are
    stored.

TABLES : DuckNamespaceContainer
    Container to organize tables/namespaces filepaths, so that you can query
    via something like:
        ```
        SELECT
            *
        FROM {TABLES.namespace.table}
        ```
    without having to manually insert the specific parquet paths
"""

import os
from string import Formatter
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


def make_read_path(table_config: dict) -> str:
    """
    Builds the absolute path to a table's parquet read file from its config.

    Parameters
    ----------
    table_config : dict
        Parsed table configuration containing the keys ``namespace``,
        ``table_name``, and ``parquet_read_file``.

    Returns
    -------
    str
        The joined filepath pointing at the table's parquet read file, of the
        form ``<DB_DIR>/<namespace>/<table_name>/<parquet_read_file>``.
    """
    return os.path.join(
        DB_DIR,
        table_config["namespace"],  # namespace
        table_config["table_name"],  # table
        table_config["parquet_read_file"],  # parquets
    )


def make_write_path(table_config: dict, **kwargs) -> str:
    """
    Builds the absolute path to a table's parquet write file from its config.

    Constructs the path from the table's ``namespace``, ``table_name``, and
    ``parquet_write_file``. If ``parquet_write_file`` is a ``str.format``
    template containing replacement fields (e.g. ``games_{year}.parquet``),
    the path is formatted using ``kwargs``.

    Parameters
    ----------
    table_config : dict
        Parsed table configuration containing the keys ``namespace``,
        ``table_name``, and ``parquet_write_file``.
    **kwargs
        Keyword arguments supplying values for any replacement fields present
        in ``parquet_write_file``.

    Returns
    -------
    str
        The joined filepath pointing at the table's parquet write file, of the
        form ``<DB_DIR>/<namespace>/<table_name>/<parquet_write_file>``, with
        any template fields substituted from ``kwargs``.
    """

    write_path = os.path.join(
        DB_DIR,
        table_config["namespace"],  # namespace
        table_config["table_name"],  # table
        table_config["parquet_write_file"],  # parquets
    )

    # check if we need to format
    if any(field_name is not None for _, field_name, _, _ in Formatter().parse(write_path)):
        write_path = write_path.format(**kwargs)

    return write_path


class DuckNamespaceContainer:
    """
    Exposes the DuckDB parquet tables declared in ``REGISTRY`` as nested
    attributes, so a table is reachable as ``TABLES.<namespace>.<table>``
    (e.g. ``TABLES.chadwick.ids``).
    """

    def __init__(self) -> None:
        """
        Initializes the container and populates it with namespaced tables.

        Registers table filepaths on disk, reads every table config listed in
        ``REGISTRY``, groups those tables by their namespace, and attaches each
        namespace to the instance as an attribute holding a ``SimpleNamespace``
        of its tables.

        Returns
        -------
        None
        """
        # make sure table paths are ready to go
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
        """
        Builds the absolute path to a table's parquet read file from its config.

        Parameters
        ----------
        table_config : dict
            Parsed table configuration containing the keys ``namespace``,
            ``table_name``, and ``parquet_read_file``.

        Returns
        -------
        str
            The joined filepath pointing at the table's parquet read file,
            of the form ``<DB_DIR>/<namespace>/<table_name>/<parquet_read_file>``.
        """
        return make_read_path(table_config)

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


TABLES = DuckNamespaceContainer()
