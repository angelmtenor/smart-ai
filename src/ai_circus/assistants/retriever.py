"""
Retriever module for indexing and querying documents using Qdrant or FAISS.
Author: Angel Martinez-Tenor, 2025.
"""

from __future__ import annotations

from typing import Literal

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, Qdrant
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient

from ai_circus.core import custom_logger
from ai_circus.models import get_embeddings

# Module-level constants
QDRANT_PATH: str = "./qdrant_data"  # Local path for Qdrant storage
DEFAULT_COLLECTION_NAME: str = "ai_circus_documents"  # Default collection name

logger = custom_logger.init(level="DEBUG")


class Retriever(BaseRetriever):
    """Retriever class for indexing and querying documents using Qdrant / FAISS, with optional hybrid BM25 retrieval."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        model_choice: Literal["openai", "google"] = "openai",
        vector_db: Literal["qdrant", "faiss"] = "qdrant",
        hybrid: bool = False,
        qdrant_path: str = QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        default_k: int = 4,
    ) -> None:
        """
        Initialize the retriever with embeddings and the chosen vector store.

        Args:
            embeddings (Embeddings, optional): Embeddings object to use. If None, created based on model_choice.
            model_choice (Literal["openai", "google"], optional): Model for embeddings if embeddings is None.
                Defaults to "openai".
            vector_db (Literal["qdrant", "faiss"], optional): Vector database to use. Defaults to "qdrant".
            hybrid (bool, optional): Whether to use hybrid retrieval with BM25. Defaults to False.
            qdrant_path (str, optional): Path for Qdrant local storage. Used only if vector_db is "qdrant".
            collection_name (str, optional): Name of the Qdrant collection. Used only if vector_db is "qdrant".
            default_k (int, optional): Default number of documents to retrieve. Defaults to 4.
        """
        super().__init__()  # Initialize BaseRetriever
        if embeddings is None:
            self.embeddings: Embeddings = get_embeddings(model_choice)
        else:
            self.embeddings = embeddings

        self.vector_db = vector_db
        self.hybrid = hybrid
        self.default_k = default_k

        if vector_db == "qdrant":
            client = QdrantClient(path=qdrant_path)
            self.vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=self.embeddings)
        elif vector_db == "faiss":
            self.vectorstore = FAISS.from_texts([], self.embeddings)
        else:
            raise ValueError(f"Unsupported vector_db: {vector_db}")

        if hybrid:
            self.documents: list[Document] = []

        logger.info(
            f"Retriever initialized with model: {model_choice if embeddings is None else 'custom'}, "
            f"vector_db: {vector_db}, hybrid: {hybrid}"
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Add a list of texts to the vector store and, if hybrid, to the documents list.

        Args:
            texts (list[str]): List of text documents to add.
            metadatas (list[dict], optional): List of metadata dictionaries for each text.
        """
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas or [{}] * len(texts), strict=False)
        ]
        self.vectorstore.add_documents(documents)
        if self.hybrid:
            self.documents.extend(documents)
        logger.info(f"Added {len(texts)} texts to the vector store")

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """
        Retrieve the top k relevant documents for the given query.

        Args:
            query (str): The query string.
            k (int, optional): Number of documents to retrieve. Defaults to 4.

        Returns:
            list[Document]: List of relevant documents.
        """
        if self.hybrid:
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            bm25_retriever = BM25Retriever.from_documents(self.documents, k=k)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],  # Equal weighting for vector and BM25
            )
            return ensemble_retriever.get_relevant_documents(query)
        else:
            return self.vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents using the default k value, as required by BaseRetriever.

        Args:
            query (str): The query string.

        Returns:
            list[Document]: List of relevant documents.
        """
        return self.retrieve(query, k=self.default_k)


if __name__ == "__main__":
    # Example usage for testing
    sample_texts = [
        "Python is a versatile programming language.",
        "Java is used for enterprise applications.",
    ]

    # Test with Qdrant (default)
    retriever_qdrant = Retriever()
    retriever_qdrant.add_texts(sample_texts)
    query = "programming language"
    results = retriever_qdrant.retrieve(query)
    for i, doc in enumerate(results):
        logger.info(f"Qdrant Result {i + 1}: {doc.page_content[:100]}...")

    # Test with FAISS
    retriever_faiss = Retriever(vector_db="faiss")
    retriever_faiss.add_texts(sample_texts)
    results = retriever_faiss.retrieve(query)
    for i, doc in enumerate(results):
        logger.info(f"FAISS Result {i + 1}: {doc.page_content[:100]}...")

    # Test with hybrid retrieval (using FAISS)
    retriever_hybrid = Retriever(vector_db="faiss", hybrid=True)
    retriever_hybrid.add_texts(sample_texts)
    results = retriever_hybrid.retrieve(query)
    for i, doc in enumerate(results):
        logger.info(f"Hybrid Result {i + 1}: {doc.page_content[:100]}...")
