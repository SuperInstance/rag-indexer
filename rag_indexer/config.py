"""
Configuration and data classes for rag-indexer.

This module defines the core data structures used throughout the library.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class Document:
    """
    A document to be indexed.

    Attributes:
        id: Unique document identifier
        text: Document text content
        metadata: Optional metadata (author, date, topic, etc.)
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Document text cannot be empty")


@dataclass
class Chunk:
    """
    A chunk of text with metadata.

    Attributes:
        text: Chunk text content
        metadata: Chunk metadata (document_id, position, level, etc.)
        parent_id: Parent chunk ID (for hierarchical chunking)
        level: Hierarchy level (0 = document, 1 = section, 2 = paragraph, etc.)
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    level: int = 0

    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

        # Add timestamp if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            level=data.get("level", 0),
        )


@dataclass
class Result:
    """
    A retrieval result with score and metadata.

    Attributes:
        text: Result text content
        score: Relevance score (0-1, higher = better)
        metadata: Result metadata (document_id, provenance, etc.)
        provenance: Source document/section identifier
    """

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[str] = None

    def __post_init__(self):
        """Validate result after initialization."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Result":
        """Create result from dictionary."""
        return cls(
            text=data["text"],
            score=data["score"],
            metadata=data.get("metadata", {}),
            provenance=data.get("provenance"),
        )


@dataclass
class IndexerStats:
    """
    Statistics about an indexer.

    Attributes:
        num_documents: Number of indexed documents
        num_chunks: Number of indexed chunks
        index_size_mb: Index size in megabytes
        avg_chunk_size: Average chunk size in characters
        indexing_time_sec: Total indexing time in seconds
    """

    num_documents: int = 0
    num_chunks: int = 0
    index_size_mb: float = 0.0
    avg_chunk_size: float = 0.0
    indexing_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "index_size_mb": self.index_size_mb,
            "avg_chunk_size": self.avg_chunk_size,
            "indexing_time_sec": self.indexing_time_sec,
        }
