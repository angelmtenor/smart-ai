"""
Document extraction module for the AI Circus project using the unstructured package.
Author: Angel Martinez-Tenor, 2025.

Dependencies:
- unstructured[pdf,docx] (install with `pip install "unstructured[pdf,docx]"`)
- langchain-core (install with `pip install langchain-core`)
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.auto import partition

from ai_circus.core import custom_logger

# Module-level constants
CHUNK_SIZE: int = 5000  # Maximum characters per chunk
CHUNK_OVERLAP: int = 100  # Overlapping characters between chunks
OCR_LANGUAGES: tuple[str, ...] = ("eng",)  # Languages for OCR
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".md", ".txt")

logger = custom_logger.init(level="DEBUG")


class DocumentExtractor:
    """Class for extracting and chunking text from various document formats using unstructured."""

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Split text into chunks with specified size and overlap.

        Args:
            text (str): Input text to chunk.
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Overlapping characters between chunks.

        Returns:
            list[str]: List of text chunks.
        """
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            logger.error(f"Invalid chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
            raise ValueError("Chunk size must be positive, and overlap must be non-negative and less than chunk size")

        chunks = []
        start = 0
        text = " ".join(text.split())  # Normalize whitespace
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():  # Only include non-empty chunks
                chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks

    def extract_text(
        self,
        file_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str | None = None,
        languages: list[str] | None = None,
        include_metadata: bool = True,
    ) -> list[Document]:
        """
        Extract and chunk text from the specified file, returning langchain Document objects with metadata.

        Args:
            file_path (str): Path to the document file.
            chunk_size (int, optional): Maximum characters per chunk. Defaults to CHUNK_SIZE.
            chunk_overlap (int, optional): Overlapping characters between chunks. Defaults to CHUNK_OVERLAP.
            strategy (str, optional): Partitioning strategy (e.g., "auto", "fast", "ocr_only"). Defaults to "auto".
            languages (list[str], optional): Languages for OCR (e.g., ["eng"]). Defaults to OCR_LANGUAGES.
            include_metadata (bool, optional): Whether to include metadata in the output. Defaults to True.

        Returns:
            list[Document]: List of langchain Document objects, each containing a text chunk and metadata.

        Raises:
            FileNotFoundError: If the file does not exist or is not a file.
            ValueError: If the file type is not supported, extraction fails, or chunk parameters are invalid.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.error(f"File not found or is not a file: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")

        # Use provided parameters or fall back to module-level defaults
        chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        strategy = strategy if strategy is not None else "auto"
        languages = languages if languages is not None else list(OCR_LANGUAGES)

        try:
            logger.info(f"Extracting text from {file_path} with strategy={strategy}, languages={languages}")
            elements = partition(str(path), strategy=strategy, languages=languages)
            logger.debug(f"Extracted {len(elements)} elements from {file_path}")

            documents = []
            global_chunk_index = 0
            for element in elements:
                if hasattr(element, "text") and element.text:
                    chunks = self._chunk_text(element.text, chunk_size, chunk_overlap)
                    base_metadata = {"source": file_path}
                    if include_metadata and hasattr(element, "metadata"):
                        base_metadata.update(element.metadata.to_dict())
                    for i, chunk in enumerate(chunks):
                        metadata = base_metadata.copy()
                        metadata["chunk_index"] = i
                        metadata["total_chunks_from_element"] = len(chunks)
                        metadata["global_chunk_index"] = global_chunk_index
                        documents.append(Document(page_content=chunk, metadata=metadata))
                        global_chunk_index += 1
            logger.info(f"Extracted and chunked {len(documents)} Document objects from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise ValueError(f"Failed to extract text: {e}") from e


if __name__ == "__main__":
    # Example usage for testing
    extractor = DocumentExtractor()
    sample_file = "scenarios/python_development/documents/15_software_engineering_principles.docx"
    try:
        documents = extractor.extract_text(
            sample_file,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            strategy="fast",
            languages=["eng"],
            include_metadata=True,
        )
        logger.info(f"Extracted {len(documents)} Document objects")
        for i, doc in enumerate(documents[:5]):
            logger.info(f"Document {i + 1}:")
            logger.info(f"  Content (first 100 chars): {doc.page_content[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")
    except Exception as e:
        logger.error(f"Error: {e}")
