"""Dummy module for smart-ai."""

from smart_ai.core import logger
from smart_ai.core.info import info_system

log = logger.init(level="INFO")


def main():
    """Main function to execute when the module is run."""
    log.info("Starting the script...")
    info_system()


if __name__ == "__main__":
    main()
