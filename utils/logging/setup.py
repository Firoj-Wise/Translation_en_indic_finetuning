"""
Logger factory with configurable handlers.

Outputs to both stdout and an optional rotating log file.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a named logger.

    Parameters
    ----------
    name : str
        Logger name (usually the pipeline stage, e.g. "ingestion").
    level : int
        Logging level. Default: INFO.
    log_file : str, optional
        If provided, also writes to this file path.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # ── Stdout handler (only on master process to avoid double logging) ────────────────
    if local_rank == 0:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    # ── File handler (optional) ───────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root to prevent duplicate messages
    logger.propagate = False
    return logger
