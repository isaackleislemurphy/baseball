"""Basic logging"""

import inspect
import logging
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that writes to stderr.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` from the calling module. If not passed,
        just goes to location where `get_logger()` is called.
    level : int, default logging.INFO
        Minimum level to emit.

    Returns
    -------
    logging.Logger
        A logger with a stream handler attached.
    """

    if name is None:
        # inspect the caller's frame to grab their __file__
        caller_frame = inspect.stack()[1]
        name = str(Path(caller_frame.filename).resolve()).split("baseball/")[-1]
        # name = Path(caller_frame.filename).stem

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # guard against adding duplicate handlers on re-import / repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger
