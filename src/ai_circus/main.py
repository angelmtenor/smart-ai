"""
- Title: Dummy module for ai-circus.
- Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

from ai_circus.core import logger
from ai_circus.core.info import info_system

log = logger.init(level="INFO")


def main() -> None:
    """Main function to execute when the module is run."""
    log.info("Starting the script...")
    info_system()


class SimpleClass:
    """A simple class for demonstration purposes."""

    def __init__(self, name: str) -> None:
        """Initialize the SimpleClass with a name."""
        self.name = name

    def greet(self) -> None:
        """Greet the user."""
        log.info(f"Hello, {self.name}!")


if __name__ == "__main__":
    main()
