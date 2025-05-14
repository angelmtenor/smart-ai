"""
Retriever module for indexing and querying documents using FAISS.
Author: Angel Martinez-Tenor, 2025.
"""

from __future__ import annotations

from typing import Literal

from langchain.embeddings.base import Embeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from ai_circus.core import custom_logger
from ai_circus.models import get_embeddings

logger = custom_logger.init(level="DEBUG")


class Retriever(BaseRetriever):
    """Retriever class for indexing and querying documents using FAISS, with optional hybrid BM25 retrieval."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        model_choice: Literal["openai", "google"] = "openai",
        hybrid: bool = False,
        default_k: int = 4,
    ) -> None:
        """
        Initialize the retriever with embeddings and FAISS vector store.

        Args:
            embeddings (Embeddings, optional): Embeddings object to use. If None, created based on model_choice.
            model_choice (Literal["openai", "google"], optional): Model for embeddings if embeddings is None.
                Defaults to "openai".
            hybrid (bool, optional): Whether to use hybrid retrieval with BM25. Defaults to False.
            default_k (int, optional): Default number of documents to retrieve. Defaults to 4.
        """
        super().__init__()  # Initialize BaseRetriever
        if embeddings is None:
            object.__setattr__(self, "embeddings", get_embeddings(model_choice))
        else:
            object.__setattr__(self, "embeddings", embeddings)

        # Set additional attributes using object.__setattr__
        object.__setattr__(self, "hybrid", hybrid)
        object.__setattr__(self, "default_k", default_k)

        # Initialize empty FAISS index without texts
        object.__setattr__(self, "vectorstore", FAISS.from_texts([""], self.embeddings))
        self.vectorstore.delete([self.vectorstore.index_to_docstore_id[0]])  # Remove dummy document

        if hybrid:
            object.__setattr__(self, "documents", [])

        logger.info(
            f"Retriever initialized with model: {model_choice if embeddings is None else 'custom'}, "
            f"vector_db: faiss, hybrid: {hybrid}"
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Add a list of texts to the FAISS vector store and, if hybrid, to the documents list.

        Args:
            texts (list[str]): List of text documents to add.
            metadatas (list[dict], optional): List of metadata dictionaries for each text.
        """
        if not texts:
            logger.warning("No texts provided to add_texts")
            return

        if metadatas is None:
            metadatas = [{}] * len(texts)
        if len(texts) != len(metadatas):
            raise ValueError(f"Length of texts ({len(texts)}) must match metadatas ({len(metadatas)})")

        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas, strict=True)]
        self.vectorstore.add_documents(documents)
        if self.hybrid:
            self.documents.extend(documents)
        logger.info(f"Added {len(texts)} texts to the FAISS vector store")

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """
        Retrieve the top k relevant documents for the given query.

        Args:
            query (str): The query string.
            k (int, optional): Number of documents to retrieve. Defaults to 4.

        Returns:
            list[Document]: List of relevant documents.
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        if self.hybrid:
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            bm25_retriever = BM25Retriever.from_documents(self.documents, k=k)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],  # Equal weighting for vector and BM25
            )
            return ensemble_retriever.invoke(query)
        else:
            return self.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        Implement abstract retrieval method required by BaseRetriever.

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

    # Test with FAISS
    retriever_faiss = Retriever()
    retriever_faiss.add_texts(sample_texts)
    query = "programming language"
    results = retriever_faiss.retrieve(query)
    for i, doc in enumerate(results):
        logger.info(f"FAISS Result {i + 1}: {doc.page_content[:100]}...")

    # Test with hybrid retrieval (using FAISS)
    retriever_hybrid = Retriever(hybrid=True)
    retriever_hybrid.add_texts(sample_texts)
    results = retriever_hybrid.retrieve(query)
    for i, doc in enumerate(results):
        logger.info(f"Hybrid Result {i + 1}: {doc.page_content[:100]}...")
