"""Logging utilities.

Provides a factory function that creates a configured :class:`logging.Logger`
instance writing both to the console and to a rotating file handler.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "agent.log",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Return a logger with a console handler and a rotating file handler.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__`` of the calling module).
    log_dir:
        Directory in which the log file will be created.
    log_file:
        Name of the log file inside *log_dir*.
    level:
        Logging level string (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    max_bytes:
        Maximum size of each log file in bytes before rotation.
    backup_count:
        Number of rotated files to keep.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the function is called multiple times
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def configure_from_config(cfg: dict) -> logging.Logger:
    """Convenience wrapper that reads logging config from the YAML config dict.

    Parameters
    ----------
    cfg:
        The top-level configuration dictionary loaded from ``config.yaml``.

    Returns
    -------
    logging.Logger
        Root agent logger named ``"league_agent"``.
    """
    log_cfg = cfg.get("logging", {})
    return get_logger(
        name="league_agent",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "agent.log"),
        level=log_cfg.get("level", "INFO"),
        max_bytes=log_cfg.get("max_bytes", 10 * 1024 * 1024),
        backup_count=log_cfg.get("backup_count", 5),
    )
