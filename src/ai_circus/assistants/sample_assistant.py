"""Sample Assistant that demonstrates document extraction, retrieval, and intent detection graph."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from ai_circus.assistants.document_extractor import DocumentExtractor
from ai_circus.assistants.intent_detector_graph import GraphState, build_graph
from ai_circus.assistants.retriever import Retriever
from ai_circus.core import custom_logger

# Configuration constants
CHUNK_SIZE: int = 5000
CHUNK_OVERLAP: int = 100
SAMPLE_FILE = "scenarios/python_development/documents/15_software_engineering_principles.docx"

logger = custom_logger.init(level="DEBUG")


def extract_documents(file_path: str) -> list:
    """Extract and chunk documents from the specified file."""
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
    """Initialize and populate the retriever with documents."""
    try:
        retriever = Retriever()
        retriever.add_texts([doc.page_content for doc in documents], [doc.metadata for doc in documents])
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise


def process_query(graph: CompiledStateGraph, query: str) -> str:
    """Run a query through the assistant graph and return the response."""
    try:
        state = GraphState(user_input=query, history=[])
        result = graph.invoke(state)
        response = result["response_output"].get("response")
        logger.info(f"Assistant response: {response}")
        return response
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise


def main() -> None:
    """Main function to orchestrate the assistant workflow."""
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

        # Process sample query
        user_query = "What are the main principles of software engineering?"
        response = process_query(assistant_graph, user_query)
        logger.info(f"Response: {response}")

    except Exception as e:
        logger.critical(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()
