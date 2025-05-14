"""
Sample Assistant that demonstrates document extraction, retrieval, and intent detection graph.
Author: Angel Martinez-Tenor, 2025.
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from ai_circus.assistants.document_extractor import DocumentExtractor
from ai_circus.assistants.intent_detector_graph import GraphState, build_graph, log_round
from ai_circus.assistants.retriever import Retriever
from ai_circus.core import custom_logger

# Configuration constants
CHUNK_SIZE: int = 5000
CHUNK_OVERLAP: int = 100
SAMPLE_FILE: str = "scenarios/python_development/documents/15_software_engineering_principles.docx"

logger = custom_logger.init(level="DEBUG")


def extract_documents(file_path: str) -> list:
    """Extract and chunk documents from the specified file.

    Args:
        file_path (str): Path to the document file.

    Returns:
        List: List of extracted document chunks with metadata.

    Raises:
        Exception: If document extraction fails.
    """
    try:
        extractor = DocumentExtractor()
        return extractor.extract_text(
            file_path,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            strategy="fast",
            languages=["eng"],
            include_metadata=True,
        )
    except Exception as e:
        logger.error(f"Failed to extract documents: {e}")
        raise


def initialize_retriever(documents: list) -> Retriever:
    """Initialize and populate the retriever with documents.

    Args:
        documents (List): List of document chunks with metadata.

    Returns:
        Retriever: Initialized retriever instance.

    Raises:
        Exception: If retriever initialization fails.
    """
    try:
        retriever = Retriever()
        retriever.add_texts([doc.page_content for doc in documents], [doc.metadata for doc in documents])
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise


def process_query(
    graph: CompiledStateGraph, query: str, history: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    """Run a query through the assistant graph and return the response and updated history.

    Args:
        graph (CompiledStateGraph): The compiled intent detection graph.
        query (str): The user query.
        history (List[dict[str, str]]): Conversation history.

    Returns:
        tuple[str, List[dict[str, str]]]: The response and updated history.

    Raises:
        Exception: If query processing fails.
    """
    try:
        state = GraphState(user_input=query, history=history)
        result = graph.invoke(state)
        state = GraphState(**result)
        response = state.response_output.get("response")
        logger.info(f"Assistant response: {response}")
        log_round(len(history) + 1, state, history)
        return response, state.history
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise


def main() -> None:
    """Main function to orchestrate the assistant workflow with multiple test conversations."""
    logger.info("Starting assistant workflow")

    try:
        # Extract documents
        documents = extract_documents(SAMPLE_FILE)
        logger.info(f"Extracted {len(documents)} document chunks")

        # Initialize retriever
        retriever = initialize_retriever(documents)
        logger.info("Retriever initialized successfully")

        # Build assistant graph
        assistant_graph = build_graph(retriever)
        logger.info("Assistant graph built successfully")

        # Define test conversations with questions related to the 15 Software Engineering Principles
        conversations = [
            # Conversation 1: Focus on retrieval-based questions about principles
            [
                "What is the DRY principle in software engineering?",
                "How does KISS complement DRY?",
                "Give an example of violating DRY.",
            ],
            # Conversation 2: Mix of retrieval and follow-up questions
            [
                "Explain the SOLID principles.",
                "How does the Single Responsibility Principle improve code?",
                "What’s an example of a class violating SOLID?",
            ],
            # Conversation 3: Retrieval, chit-chat, and out-of-scope
            [
                "What does the Law of Demeter mean?",
                "I love writing clean code!",
                "What’s the weather like today?",
            ],
        ]

        # Run test conversations
        for i, queries in enumerate(conversations, 1):
            logger.info(f"Starting test conversation {i}")
            history: list[dict[str, str]] = []
            for j, query in enumerate(queries, 1):
                logger.info(f"Processing query {j} in conversation {i}: {query}")
                response, history = process_query(assistant_graph, query, history)
                logger.info(f"Response {j}: {response}")

        logger.info("All test conversations completed successfully")

    except Exception as e:
        logger.critical(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()
