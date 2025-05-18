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
from pydantic import Field, ValidationError
from pydantic.dataclasses import dataclass

# === Constants ===
LOG_DIR = Path("log")
FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_LOG_LEVEL = "INFO"

# === Log Format Templates ===
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{file.name}:{line}</cyan> | "
    "<level>{message}</level>"
)
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {file.name}:{line} | {message}"


@dataclass
class LoggerConfig:
    """Configuration for the logger."""

    level: str = Field(default=DEFAULT_LOG_LEVEL, pattern=r"^(TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    save_to_file: bool = False
    subfolder: str | None = None
    filename_modifier: str = ""
    filepath: Path | None = None


def configure_logger(config: LoggerConfig | None = None, **kwargs: Any) -> Any:
    """Configure and return the Loguru logger.

    Args:
        config: Logger configuration. If provided, overrides keyword arguments.
        **kwargs: Keyword arguments for LoggerConfig (e.g., level, save_to_file).

    Returns:
        Configured Loguru logger instance.

    Raises:
        ValueError: If log file creation fails or configuration is invalid.
    """
    logger.remove()  # Reset existing handlers

    # Use provided config or create from kwargs
    if config is None:
        try:
            config = LoggerConfig(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid logger configuration: {e}") from e
    else:
        if kwargs:
            raise ValueError("Cannot provide both config and keyword arguments")

    # Configure console output
    logger.add(sys.stdout, level=config.level, format=CONSOLE_FORMAT)

    # Configure file output if enabled
    if config.save_to_file:
        log_filepath = _resolve_log_filepath(
            subfolder=config.subfolder,
            filename_modifier=config.filename_modifier,
            force_filepath=config.filepath,
        )
        try:
            log_filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.add(log_filepath, level=config.level, format=FILE_FORMAT)
            logger.debug(f"Logging to file: {log_filepath}")
        except OSError as e:
            logger.error(f"Failed to create log file: {e}")
            raise ValueError(f"Could not create log file: {e}") from e

    return logger


def get_logger(name: str) -> Any:
    """Return a logger instance with a custom name.

    Args:
        name: Descriptive name to tag log messages.

    Returns:
        Bound Loguru logger instance.
    """
    return logger.bind(name=name)


def _resolve_log_filepath(
    subfolder: str | None,
    filename_modifier: str,
    force_filepath: Path | None,
) -> Path:
    """Determine the log file path.

    Args:
        subfolder: Subfolder under log directory.
        filename_modifier: String to append to the filename.
        force_filepath: Explicit log file path.

    Returns:
        Resolved log file path.
    """
    if force_filepath:
        return Path(force_filepath)

    timestamp = datetime.now(tz=UTC).strftime(FILENAME_TIMESTAMP_FORMAT)
    filename = f"{timestamp}{f'_{filename_modifier}' if filename_modifier else ''}.log"

    return LOG_DIR / (Path(subfolder) / filename if subfolder else filename)


# === Example usage ===
if __name__ == "__main__":
    try:
        # Simplified single-line configuration
        logger = configure_logger(save_to_file=True, subfolder="test", filename_modifier="app")
        logger.info("Application started")
        logger.warning("This is a warning")
        logger.error("This is an error")
    except ValueError as e:
        print(f"Logger setup failed: {e}")  # noqa: T201

# === Example usage ina module that uses the logger ===
# if __name__ == "__main__":
#     # Example usage for testing
#     logger = get_logger(__name__)
#     logger.info("This is an info message")
