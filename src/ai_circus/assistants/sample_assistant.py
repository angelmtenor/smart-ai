"""
Sample Assistant for Intent Detection and Document Retrieval.
Author: Angel Martinez-Tenor, 2025.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypedDict

from langgraph.graph.state import CompiledStateGraph

from ai_circus.assistants.document_extractor import DocumentExtractor
from ai_circus.assistants.intent_detector_graph import GraphState, build_graph
from ai_circus.assistants.retriever import Retriever
from ai_circus.core import custom_logger

# Configuration constants
CHUNK_SIZE: int = 5000
CHUNK_OVERLAP: int = 100
SAMPLE_FILE_PATH: str = "scenarios/python_development/documents/15_software_engineering_principles.docx"

logger = custom_logger.init(level="DEBUG")


class DocumentChunk(TypedDict):
    """Type definition for document chunks with metadata."""

    page_content: str
    metadata: dict


def extract_document_chunks(file_path: str, chunk_size: int, chunk_overlap: int) -> Sequence[DocumentChunk]:
    """Extract and chunk documents from the specified file.

    Args:
        file_path: Path to the document file.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        list[DocumentChunk]: List of extracted document chunks with metadata.

    Raises:
        ValueError: If document extraction fails.
    """
    try:
        extractor = DocumentExtractor()
        chunks = extractor.extract_text(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="fast",
            languages=["eng"],
            include_metadata=True,
        )
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunks]
    except Exception as e:
        logger.error(f"Failed to extract document chunks from {file_path}: {e}")
        raise ValueError(f"Document extraction failed: {e}") from e


def initialize_document_retriever(chunks: list[DocumentChunk]) -> Retriever:
    """Initialize and populate the retriever with document chunks.

    Args:
        chunks: List of document chunks with metadata.

    Returns:
        Retriever: Initialized retriever instance.

    Raises:
        ValueError: If retriever initialization fails.
    """
    try:
        retriever = Retriever()
        retriever.add_texts(
            texts=[chunk["page_content"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
        )
        logger.info("Retriever initialized with document chunks")
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize document retriever: {e}")
        raise ValueError(f"Retriever initialization failed: {e}") from e


def process_user_query(
    graph: CompiledStateGraph,
    query: str,
    conversation_history: list[dict[str, str]],
    document_path: str,
    chunk_index: int | None = None,
) -> tuple[str, list[dict[str, str]], dict, dict]:
    """Process a user query through the assistant graph and return the response, updated history, intent output, and
      response output.

    Args:
        graph: The compiled intent detection graph.
        query: The user query.
        conversation_history: List of previous conversation exchanges.
        document_path: Path to the document for metadata logging.
        chunk_index: The document chunk index for metadata logging (optional).

    Returns:
        tuple[str, list[dict[str, str]], dict, dict]: The response text, updated conversation history, intent output,
          and response output.

    Raises:
        ValueError: If query processing fails.
    """
    try:
        state = GraphState(user_input=query, history=conversation_history)
        result = graph.invoke(state)
        state = GraphState(**result)
        response = state.response_output.get("response", "")

        # Log metadata for the round
        metadata = {
            "conversation_round": len(conversation_history) + 1,
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "response_length": len(response),
            "document_path": document_path or "N/A",
            "chunk_index": chunk_index if chunk_index is not None else "N/A",
            "history_length": len(conversation_history),
            "intent_output": state.intent_output,
        }
        logger.debug(f"Round {metadata['conversation_round']} metadata: {json.dumps(metadata, indent=2)}")

        return response, state.history, state.intent_output, state.response_output
    except Exception as e:
        logger.error(f"Failed to process user query '{query[:50]}...': {e}")
        raise ValueError(f"Query processing failed: {e}") from e


def run_assistant_workflow() -> None:
    """Orchestrate the assistant workflow with multiple test conversations, logging intent results, assistant response,
    and document filename.

    Raises:
        ValueError: If any step in the workflow fails.
    """
    logger.info("Starting assistant workflow")

    try:
        # Extract document chunks
        document_chunks = extract_document_chunks(
            file_path=SAMPLE_FILE_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info(f"Document extraction completed for {SAMPLE_FILE_PATH}")

        # Initialize retriever
        retriever = initialize_document_retriever(list(document_chunks))
        logger.info("Retriever setup completed")

        # Build assistant graph
        assistant_graph = build_graph(retriever)
        logger.info("Assistant graph built successfully")

        # Define test conversations
        test_conversations = [
            [
                "What is the DRY principle in software engineering?",
                "How does KISS complement DRY?",
                "Give an example of violating DRY.",
            ],
            [
                "Explain the SOLID principles.",
                "How does the Single Responsibility Principle improve code?",
                "What's an example of a class violating SOLID?",
            ],
            [
                "What does the Law of Demeter mean?",
                "I love writing clean code!",
                "What's the weather like today?",
            ],
        ]

        # Run test conversations
        for conv_index, queries in enumerate(test_conversations, 1):
            logger.info(f"Starting conversation {conv_index} with {len(queries)} queries")
            conversation_history: list[dict[str, str]] = []
            for query_index, query in enumerate(queries, 1):
                logger.debug(f"Conversation {conv_index}, Round {query_index}: Processing query: {query[:50]}...")
                response, conversation_history, intent_output, response_output = process_user_query(
                    graph=assistant_graph,
                    query=query,
                    conversation_history=conversation_history,
                    document_path=SAMPLE_FILE_PATH,
                    chunk_index=0,  # Simplified: Use first chunk index
                )
                # Log intent results, assistant response, and document filename
                round_info = {
                    "conversation": conv_index,
                    "round": query_index,
                    "query": query,
                    "intent_output": intent_output,
                    "assistant_response": response_output.get("response", ""),
                    "response_length": len(response),
                    "document_filename": SAMPLE_FILE_PATH,
                }
                logger.info(f"Round {query_index} results: {json.dumps(round_info, indent=2)}")
            logger.info(f"Completed conversation {conv_index}")

        logger.info("All test conversations completed successfully")

    except ValueError as e:
        logger.critical(f"Workflow failed: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in workflow: {e}")
        raise ValueError(f"Unexpected workflow failure: {e}") from e


if __name__ == "__main__":
    run_assistant_workflow()
