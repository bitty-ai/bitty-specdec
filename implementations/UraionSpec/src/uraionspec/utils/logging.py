"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "uraionspec",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name.
        level: Log level.
        log_file: Optional path to a log file.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def add_metric(name: str, value, den=1.0, tag: str = "train"):
    """Log a metric for tracking."""
    logger = logging.getLogger("uraionspec.metrics")
    if den > 0:
        logger.info(f"[{tag}] {name}: {value / den:.6f}")
    else:
        logger.info(f"[{tag}] {name}: {value:.6f} (den=0)")
