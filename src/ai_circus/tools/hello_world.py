"""Tool "Hello World" for AI Circus.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

from dotenv import load_dotenv

from ai_circus.core import custom_logger
from ai_circus.core.info import info_system

# Initialize logger and load environment variables
logger = custom_logger.init(level="INFO")
load_dotenv()


class SimpleClass:
    """A simple class for demonstration purposes."""

    def __init__(self, name: str) -> None:
        """Initialize the class with a name."""
        self.name = name

    def greet(self) -> None:
        """Print a greeting message."""
        logger.info(f"Hello, {self.name}!")


def main() -> None:
    """Main function to demonstrate the functionality of the module."""
    info_system()

    simple = SimpleClass("AI Circus")
    simple.greet()


if __name__ == "__main__":
    main()
