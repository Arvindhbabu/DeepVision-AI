"""
DeepVision AI

Logger Utility
"""

from pathlib import Path
import logging


def create_logger(
    log_dir="outputs/logs",
    log_name="train.log",
):
    """
    Create a reusable logger.
    """

    log_dir = Path(log_dir)

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("DeepVision")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_dir / log_name
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger