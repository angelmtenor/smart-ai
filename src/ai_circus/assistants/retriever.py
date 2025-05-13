"""
Basic Retriever class for the AI Circus project using Qdrant as the vector database.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

from typing import Literal

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Qdrant

from ai_circus.core import custom_logger
from ai_circus.models import get_embeddings

logger = custom_logger.init(level="DEBUG")

# Configuration
MODEL_CHOICE = "openai"  # or "google"
QDRANT_PATH = "./qdrant_data"  # Local path for Qdrant storage (adjust as needed)


class Retriever:
    """Retriever class for indexing and querying documents using Qdrant."""

    def __init__(
        self, model_choice: Literal["openai", "google"] = MODEL_CHOICE, qdrant_path: str = QDRANT_PATH
    ) -> None:
        """Initialize the retriever with embeddings and Qdrant vector store."""
        self.embeddings: Embeddings = get_embeddings(model_choice)
        self.qdrant_path = qdrant_path
        self.vectorstore: Qdrant | None = None
        self.retriever = None
        logger.info(f"Retriever initialized with model: {model_choice} and Qdrant path: {qdrant_path}")

    def index_texts(self, texts: list[str]) -> None:
        """Index a list of texts into the Qdrant vector store."""
        try:
            self.vectorstore = Qdrant.from_texts(
                texts, embedding=self.embeddings, path=self.qdrant_path, prefer_grpc=True
            )
            if self.vectorstore is None:
                raise ValueError("Failed to initialize Qdrant vectorstore.")
            self.retriever = self.vectorstore.as_retriever()
            logger.info(f"Indexed {len(texts)} texts into Qdrant at {self.qdrant_path}")
        except Exception as e:
            logger.error(f"Failed to index texts: {e!s}")
            raise ValueError(f"Failed to index texts: {e!s}") from e


if __name__ == "__main__":
    # Example: Indexing sample data
    sample_texts = [
        "Python is a versatile programming language.",
        "Java is used for enterprise applications.",
    ]
    retriever = Retriever()
    retriever.index_texts(sample_texts)
