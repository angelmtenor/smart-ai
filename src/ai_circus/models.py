"""
This module provides a function to initialize and return a language model (LLM)
based on the specified provider (OpenAI or Google). It uses environment variables to configure the LLM settings.
Author: Angel Martinez-Tenor, 2025
"""

from __future__ import annotations

import os
from typing import Literal, cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv(override=True)


# Configuration
DEFAULT_LLM_PROVIDER: Literal["openai", "google"] = cast(
    Literal["openai", "google"], os.getenv("DEFAULT_LLM_PROVIDER", "openai")
)
DEFAULT_LLM_MODEL: str = os.getenv(
    "DEFAULT_LLM_MODEL", "gpt-4o-mini" if DEFAULT_LLM_PROVIDER == "openai" else "gemini-2.0-pro"
)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "your-google-api-key")


def get_llm() -> ChatOpenAI | ChatGoogleGenerativeAI:
    """Initialize and return the appropriate LLM based on the provider."""
    if DEFAULT_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key":
            raise ValueError("OPENAI_API_KEY not set or invalid")
        return ChatOpenAI(model=DEFAULT_LLM_MODEL, api_key=SecretStr(OPENAI_API_KEY))
    elif DEFAULT_LLM_PROVIDER == "google":
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-google-api-key":
            raise ValueError("GOOGLE_API_KEY not set or invalid")
        return ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {DEFAULT_LLM_PROVIDER}")
