from graphlib import TopologicalSorter

from baseball.data.uploader.dependencies import DEPENDENCIES
from baseball.data.uploader.runners import RUNNERS
from baseball.utils.logging import get_logger

LOGGER = get_logger()


def run_uploader_dag(runners: dict, dependencies: dict) -> None:
    """
    Run every node in dependency order.

    Parameters
    ----------
    runners : dict
        Maps a node name to its zero-arg callable entry point.
    dependencies : dict
        Maps a node name to the set of node names that must run before it.

    Returns
    -------
    None
    """
    for name in TopologicalSorter(dependencies).static_order():
        LOGGER.info(f"Running node: {name}")
        runners[name]()
        LOGGER.info(f"Node complete: {name}")


if __name__ == "__main__":
    run_uploader_dag(RUNNERS, DEPENDENCIES)
