"""Sample Assistant that demonstrates document extraction, retrieval, and intent detection graph."""

from __future__ import annotations
import logging

from ai_circus.assistants.document_extractor import DocumentExtractor, CHUNK_SIZE, CHUNK_OVERLAP
from ai_circus.assistants.retriever import Retriever
from ai_circus.assistants.intent_detector_graph import build_graph, GraphState

# 1. Extract and chunk a sample docx file
extractor = DocumentExtractor()
sample_file = "scenarios/python_development/documents/15_software_engineering_principles.docx"
documents = extractor.extract_text(
    sample_file,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    strategy="fast",
    languages=["eng"],
    include_metadata=True,
)

# 2. Create a retriever and add the split documents
retriever = Retriever()
retriever.add_texts([doc.page_content for doc in documents], [doc.metadata for doc in documents])

# 3. Build the assistant graph
assistant_graph = build_graph()

# 4. Example: Run a query through the assistant graph
if __name__ == "__main__":
    user_input = "What are the main principles of software engineering?"
    state = GraphState(user_input=user_input, history=[])
    result = assistant_graph.invoke(state)
    logging.info(f"Assistant response: {result['response_output'].get('response')}")
