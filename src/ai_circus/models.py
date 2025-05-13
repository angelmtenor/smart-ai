"""
This module provides functions to initialize and return a language model (LLM)
and embedding model based on the specified provider (OpenAI or Google).
It uses environment variables for configuration.
Author: Angel Martinez-Tenor, 2025
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv(override=True)

# Configuration
DEFAULT_LLM_PROVIDER: Literal["openai", "google"] = os.getenv("DEFAULT_LLM_PROVIDER", "openai")  # type: ignore[attr-defined]
DEFAULT_LLM_MODEL: str = os.getenv(
    "DEFAULT_LLM_MODEL", "gpt-4o-mini" if DEFAULT_LLM_PROVIDER == "openai" else "gemini-2.0-flash"
)
API_KEYS: dict[Literal["openai", "google"], str] = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "google": os.getenv("GOOGLE_API_KEY", ""),
}


def get_llm(provider: Literal["openai", "google"] = DEFAULT_LLM_PROVIDER) -> ChatOpenAI | ChatGoogleGenerativeAI:
    """Initialize and return the LLM based on the provider."""
    api_key = API_KEYS.get(provider)
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY not set")

    if provider == "openai":
        return ChatOpenAI(model=DEFAULT_LLM_MODEL, api_key=SecretStr(api_key))
    if provider == "google":
        return ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, google_api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embeddings(provider: Literal["openai", "google"] = DEFAULT_LLM_PROVIDER) -> Embeddings:
    """
    Returns an embedding model based on the provider.

    Args:
        provider: Either "openai" or "google". Defaults to DEFAULT_LLM_PROVIDER.

    Returns:
        Embeddings: The selected embedding model.
    """
    api_key = API_KEYS.get(provider)
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY not set")

    if provider == "openai":
        return OpenAIEmbeddings(api_key=SecretStr(api_key))
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07", google_api_key=SecretStr(api_key)
        )
    raise ValueError(f"Invalid model provider: {provider}")
