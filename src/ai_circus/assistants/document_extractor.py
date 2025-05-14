"""
Document extraction module for the AI Circus project using the unstructured package.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

Dependencies:
- unstructured[pdf,docx] (install with `pip install "unstructured[pdf,docx]"`)
- poppler and tesseract for PDF processing
"""

from __future__ import annotations

from pathlib import Path

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
            List[str]: List of text chunks.
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
    ) -> list[str]:
        """
        Extract and chunk text from the specified file.

        Args:
            file_path (str): Path to the document file.
            chunk_size (int, optional): Maximum characters per chunk. Defaults to CHUNK_SIZE.
            chunk_overlap (int, optional): Overlapping characters between chunks. Defaults to CHUNK_OVERLAP.
            strategy (str, optional): Partitioning strategy (e.g., "auto", "fast", "ocr_only").
                Defaults to PARTITION_STRATEGY.
            languages (list[str], optional): Languages for OCR (e.g., ["eng"]). Defaults to OCR_LANGUAGES.

        Returns:
            List[str]: List of text chunks extracted and chunked from the document.

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

            # Combine text elements into a single string
            texts = [element.text for element in elements if hasattr(element, "text") and element.text]
            combined_text = " ".join(texts)
            logger.debug(f"Combined text length: {len(combined_text)} characters")

            # Chunk the combined text
            chunks = self._chunk_text(combined_text, chunk_size, chunk_overlap)
            logger.info(f"Extracted and chunked {len(chunks)} text chunks from {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise ValueError(f"Failed to extract text: {e}") from e


if __name__ == "__main__":
    # Example usage for testing
    extractor = DocumentExtractor()
    sample_file = "scenarios/python_development/documents/15_software_engineering_principles.docx"
    try:
        texts = extractor.extract_text(
            sample_file,
            chunk_size=CHUNK_SIZE,  # Override default for testing
            chunk_overlap=CHUNK_OVERLAP,
            strategy="fast",
            languages=["eng"],
        )
        logger.info(f"Extracted {len(texts)} text chunks")
        for i, text in enumerate(texts[:5]):
            logger.info(f"Chunk {i + 1}: {text[:100]}...")
    except Exception as e:
        logger.error(f"Error: {e}")
