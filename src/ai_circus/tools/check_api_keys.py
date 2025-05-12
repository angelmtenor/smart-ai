"""Tool: Check API Keys and Fetch Data from APIs
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from ai_circus.core import custom_logger
from ai_circus.core.info import info_system

# Initialize logger and load environment variables
logger = custom_logger.init(level="INFO")
load_dotenv(override=True)


@dataclass
class APIConfig:
    """Configuration for API endpoints and request parameters."""

    name: str
    url: str | Callable[[str], str]
    method: str = "GET"
    headers: dict[str, str] | Callable[[str], dict[str, str]] | None = None
    json: dict | Callable[[str], dict] | None = None
    params: dict | Callable[[str], dict] | None = None


class APIClient:
    """Generic API client for making HTTP requests."""

    @staticmethod
    def fetch_data(config: APIConfig) -> dict | None:
        """Fetch data from an API with the given configuration."""
        try:
            with httpx.Client() as client:
                # Base request kwargs, excluding json for GET requests
                request_kwargs = {
                    "url": config.url,
                    "timeout": 10.0,
                    "headers": config.headers or {},
                    "params": config.params,
                }
                # Only include json for POST requests
                if config.method.upper() == "POST":
                    request_kwargs["json"] = config.json

                if config.method.upper() == "GET":
                    response = client.get(**request_kwargs)
                else:
                    response = client.post(**request_kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch {config.name} data: {e}")
            return None


def main() -> None:
    """Main function to execute API data fetching."""
    logger.info("Starting the script...")
    info_system()

    # API configurations
    api_configs = [
        APIConfig(
            name="OpenAI",
            url="https://api.openai.com/v1/models",
            headers=lambda key: {"Authorization": f"Bearer {key}"},
        ),
        APIConfig(
            name="Google",
            url=lambda key: f"https://www.googleapis.com/discovery/v1/apis?key={key}",
        ),
        APIConfig(
            name="Tavily",
            url="https://api.tavily.com/search",
            method="POST",
            json=lambda key: {"api_key": key, "query": "example search"},
        ),
    ]

    # Retrieve API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY", ""),
        "Google": os.getenv("GOOGLE_API_KEY", ""),
        "Tavily": os.getenv("TAVILY_API_KEY", ""),
    }

    # Initialize checklist to log the status of each API
    checklist = []

    # Fetch data from APIs
    client = APIClient()
    for config in api_configs:
        api_key = api_keys.get(config.name, "")
        if not api_key:
            logger.warning(f"Skipping {config.name} API due to missing API key")
            checklist.append(f"[ ] {config.name}: missing API key (skipped)")
            continue
        logger.info(f"Fetching data from {config.name} API...")
        # Dynamically resolve headers, json, or params if they're callable
        config.headers = config.headers(api_key) if callable(config.headers) else config.headers
        config.json = config.json(api_key) if callable(config.json) else config.json
        config.params = config.params(api_key) if callable(config.params) else config.params
        config.url = config.url(api_key) if callable(config.url) else config.url

        data = client.fetch_data(config)
        if data:
            key = "data" if config.name == "OpenAI" else "items" if config.name == "Google" else "results"
            logger.info(f"{config.name} data retrieved: {data.get(key, [])[:1]}")
            checklist.append(f"[✔] {config.name}: data retrieved")
        else:
            checklist.append(f"[ ] {config.name}: call failed")

    # Print checklist summary
    logger.info("Checklist summary:")
    for item in checklist:
        logger.info(item)


class SimpleClass:
    """A simple class for demonstration purposes."""

    def __init__(self, name: str) -> None:
        """Initialize the class with a name."""
        self.name = name

    def greet(self) -> None:
        """Print a greeting message."""
        logger.info(f"Hello, {self.name}!")


if __name__ == "__main__":
    main()
