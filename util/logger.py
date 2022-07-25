""" This module contains the logger, that will be used in the project.

"""
import logging


def define_logger() -> logging.Logger:
    """Defines the logger, that will be used throughout the project.

    Args:
        None

    Returns:
        logger: The logger used throughout the project

    """
    # Build logger
    logging.basicConfig(
        format="%(asctime)s; %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG,
    )

    logger = logging.getLogger(__name__)

    return logger
