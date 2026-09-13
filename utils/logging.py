import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that writes to stderr.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` from the calling module.
    level : int, default logging.INFO
        Minimum level to emit.

    Returns
    -------
    logging.Logger
        A logger with a stream handler attached.
    """
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
