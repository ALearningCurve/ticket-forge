"""shared: Logging utilities for the TicketForge monorepo."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
  """Get a configured logger for the given module name.

  Args:
      name: The name of the logger, typically __name__ from the calling module.
      level: The logging level, defaults to INFO.

  Returns:
      A configured Logger instance.
  """
  logger = logging.getLogger(name)

  if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
      fmt="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  logger.setLevel(level)
  return logger
