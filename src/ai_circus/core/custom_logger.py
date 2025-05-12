"""
- Title:    Custom Logger
- Author:   Angel Martinez-tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# === Constants ===
LOG_DIR = Path("log")
FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%d---%H-%M-%S"

DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_SUBFOLDER: Path | str | None = None
DEFAULT_FILENAME_MODIFIER = ""
DEFAULT_SAVE_LOG = False
DEFAULT_SIMPLE_FORMAT = False

# === Log Format Templates ===
FORMAT_CONSOLE_VERBOSE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{file.name}:{line}</cyan> | "
    "<level>{message}</level>"
)
FORMAT_FILE_VERBOSE = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {file.name}:{line} | {message}"
FORMAT_CONSOLE_SIMPLE = "<level>{time:HH:mm:ss}</level> | <level>{level:<8}</level> | <level>{message}</level>"
FORMAT_FILE_SIMPLE = "{time:HH:mm:ss} | {level:<8} | {message}"


def init(
    level: str = DEFAULT_LOG_LEVEL,
    subfolder: Path | str | None = DEFAULT_SUBFOLDER,
    filename_modifier: str = DEFAULT_FILENAME_MODIFIER,
    save_log: bool = DEFAULT_SAVE_LOG,
    console_format: str = FORMAT_CONSOLE_VERBOSE,
    file_format: str = FORMAT_FILE_VERBOSE,
    simple_format: bool = DEFAULT_SIMPLE_FORMAT,
    force_filepath: Path | str | None = None,
) -> Any:
    """Initialize and configure the Loguru logger.

    Args:
        level (str): Logging level (e.g., "INFO", "DEBUG").
        subfolder (Optional[Path | str]): Subfolder under log directory.
        filename_modifier (str): String to append to the filename.
        save_log (bool): Whether to save logs to a file.
        console_format (str): Format for console logging.
        file_format (str): Format for file logging.
        simple_format (bool): Use simpler output formats.
        force_filepath (Optional[Path | str]): Explicit log file path.

    Returns:
        Logger: Configured Loguru logger instance.
    """
    logger.remove()

    if simple_format:
        console_format = FORMAT_CONSOLE_SIMPLE
        file_format = FORMAT_FILE_SIMPLE

    # Configure console output
    logger.add(sys.stdout, level=level, format=console_format)

    if save_log:
        log_filepath = _resolve_log_filepath(
            subfolder=subfolder,
            filename_modifier=filename_modifier,
            force_filepath=force_filepath,
        )
        try:
            log_filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.add(log_filepath, level=level, format=file_format)
            logger.debug(f"Logging to file: {log_filepath.resolve()}")
        except OSError as e:
            logger.error(f"Could not create log file: {e}")

    return logger


def get_logger(name: str) -> Any:
    """Bind and return a logger instance with a custom name.

    Args:
        name (str): Descriptive name to tag log messages.

    Returns:
        Logger: Bound Loguru logger instance.
    """
    return logger.bind(name=name)


def _resolve_log_filepath(
    subfolder: Path | str | None,
    filename_modifier: str,
    force_filepath: Path | str | None,
) -> Path:
    """Helper function to determine the log file path."""
    if force_filepath:
        return Path(force_filepath)

    timestamp = datetime.now(tz=UTC).strftime(FILENAME_TIMESTAMP_FORMAT)
    filename = f"{timestamp}"
    if filename_modifier:
        filename += f"_{filename_modifier}"
    filename += ".log"

    if subfolder:
        return LOG_DIR / Path(subfolder) / filename
    return LOG_DIR / filename


# === Example usage ===
if __name__ == "__main__":
    logger = init(level="INFO", save_log=True)
    logger.info("---- App Caller ----")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
