"""
Chunker implementations for document chunking.

This module provides various chunking strategies for splitting documents
into semantically coherent chunks optimized for retrieval.
"""

from abc import ABC, abstractmethod
from typing import List
import re

from .config import Chunk, Document


class Chunker(ABC):
    """
    Abstract base class for chunkers.

    A chunker is responsible for splitting documents into chunks
    that are semantically coherent and optimized for retrieval.
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks with metadata
        """
        pass


class FixedSizeChunker(Chunker):
    """
    Fixed-size chunking with overlap.

    Simple and predictable chunking strategy that splits documents
    into fixed-size chunks with optional overlap.

    Attributes:
        chunk_size: Target chunk size in characters
        overlap: Overlap between consecutive chunks

    Example:
        >>> chunker = FixedSizeChunker(chunk_size=512, overlap=50)
        >>> chunks = chunker.chunk(Document(id="doc1", text="..."))
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between consecutive chunks
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into fixed-size chunks.

        Args:
            document: Document to chunk

        Returns:
            List of fixed-size chunks
        """
        text = document.text
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "document_id": document.id,
                        "chunk_index": chunk_id,
                        "start_char": start,
                        "end_char": end,
                        "char_count": len(chunk_text),
                    },
                )
            )

            start = end - self.overlap
            chunk_id += 1

        return chunks


class SemanticChunker(Chunker):
    """
    Semantic chunking at natural boundaries.

    Splits documents at natural boundaries like paragraphs and sections
    to preserve semantic coherence.

    Attributes:
        min_size: Minimum chunk size in characters
        max_size: Maximum chunk size in characters
        similarity_threshold: Similarity threshold for merging paragraphs

    Example:
        >>> chunker = SemanticChunker(min_size=100, max_size=2000)
        >>> chunks = chunker.chunk(Document(id="doc1", text="..."))
    """

    def __init__(
        self,
        min_size: int = 100,
        max_size: int = 2000,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize semantic chunker.

        Args:
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters
            similarity_threshold: Threshold for merging paragraphs (not implemented)
        """
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        if max_size <= min_size:
            raise ValueError(f"max_size must be greater than min_size")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1")

        self.min_size = min_size
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into semantic chunks.

        Args:
            document: Document to chunk

        Returns:
            List of semantic chunks
        """
        paragraphs = self._split_paragraphs(document.text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for para in paragraphs:
            para_size = len(para)

            # Check if paragraph starts new chunk
            if current_size > 0 and current_size + para_size > self.max_size:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "document_id": document.id,
                            "chunk_index": chunk_id,
                            "paragraph_count": len(current_chunk),
                            "char_count": len(chunk_text),
                        },
                    )
                )
                current_chunk = []
                current_size = 0
                chunk_id += 1

            current_chunk.append(para)
            current_size += para_size

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "document_id": document.id,
                        "chunk_index": chunk_id,
                        "paragraph_count": len(current_chunk),
                        "char_count": len(chunk_text),
                    },
                )
            )

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newline (paragraph boundary)
        paragraphs = re.split(r"\n\n+", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]


class HierarchicalChunker(Chunker):
    """
    Hierarchical semantic chunking (HiChunk).

    Creates multi-level chunks preserving document structure:
    - Level 0: Document summary
    - Level 1: Sections
    - Level 2: Paragraphs

    Enables coarse-to-fine retrieval for better accuracy.

    Research:
    - HiChunk Framework (arXiv 2025): 15-25% improvement
    - Optimal leaf_size: ~1,800 characters (Snowflake 2025)

    Attributes:
        leaf_size: Target size for leaf chunks (paragraphs)
        max_levels: Maximum hierarchy depth

    Example:
        >>> chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        >>> chunks = chunker.chunk(Document(id="doc1", text="..."))
    """

    def __init__(self, leaf_size: int = 1800, max_levels: int = 3):
        """
        Initialize hierarchical chunker.

        Args:
            leaf_size: Target size for leaf chunks in characters
            max_levels: Maximum hierarchy depth
        """
        if leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {leaf_size}")
        if max_levels < 2:
            raise ValueError(f"max_levels must be at least 2, got {max_levels}")

        self.leaf_size = leaf_size
        self.max_levels = max_levels

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Create hierarchical chunks.

        Args:
            document: Document to chunk

        Returns:
            List of hierarchical chunks with parent-child relationships
        """
        chunks = []
        doc_id = document.id

        # Level 0: Document summary
        doc_summary = self._summarize_document(document)
        chunks.append(
            Chunk(
                text=doc_summary,
                metadata={
                    "document_id": doc_id,
                    "level": 0,
                    "type": "document",
                    "title": document.metadata.get("title", ""),
                },
                level=0,
            )
        )

        # Level 1+: Sections and paragraphs
        sections = self._split_sections(document.text)
        for section_idx, section in enumerate(sections):
            section_id = f"{doc_id}-L1-{section_idx}"

            # Create section chunk (Level 1)
            section_text = section["content"]
            chunks.append(
                Chunk(
                    text=section_text,
                    metadata={
                        "document_id": doc_id,
                        "level": 1,
                        "type": "section",
                        "section_title": section["title"],
                        "section_index": section_idx,
                        "parent_id": f"{doc_id}-L0",
                    },
                    parent_id=f"{doc_id}-L0",
                    level=1,
                )
            )

            # Level 2: Paragraphs within section (if max_levels >= 2)
            if self.max_levels >= 2:
                paragraphs = self._split_paragraphs(section_text)
                for para_idx, para in enumerate(paragraphs):
                    chunks.append(
                        Chunk(
                            text=para,
                            metadata={
                                "document_id": doc_id,
                                "level": 2,
                                "type": "paragraph",
                                "parent_id": section_id,
                                "paragraph_index": para_idx,
                            },
                            parent_id=section_id,
                            level=2,
                        )
                    )

        return chunks

    def _summarize_document(self, document: Document) -> str:
        """Generate document summary for Level 0 chunk."""
        # Simple implementation: first paragraph or first N characters
        lines = document.text.split("\n")
        first_line = lines[0] if lines else ""

        if len(first_line) > 500:
            return first_line[:500] + "..."
        return first_line or document.text[:500]

    def _split_sections(self, text: str) -> List[dict]:
        """Split text into sections based on headers."""
        sections = []
        current_section = {"title": "Introduction", "content": ""}

        for line in text.split("\n"):
            # Detect markdown headers (# ## ###)
            if line.strip().startswith("#"):
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "title": line.lstrip("#").strip(),
                    "content": "",
                }
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections if sections else [{"title": "Main", "content": text}]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\n+", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]
