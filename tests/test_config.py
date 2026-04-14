"""
Tests for rag_indexer.config module.

Covers Document, Chunk, Result, and IndexerStats dataclasses.
"""

import pytest
from rag_indexer.config import Document, Chunk, Result, IndexerStats


# ===========================================================================
# Document tests
# ===========================================================================

class TestDocument:
    """Tests for the Document dataclass."""

    def test_create_document_with_required_fields(self):
        doc = Document(id="doc1", text="Hello world")
        assert doc.id == "doc1"
        assert doc.text == "Hello world"
        assert doc.metadata == {}

    def test_create_document_with_metadata(self):
        meta = {"title": "Test", "category": "science"}
        doc = Document(id="doc1", text="Hello", metadata=meta)
        assert doc.metadata == meta

    def test_document_empty_id_raises(self):
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            Document(id="", text="text")

    def test_document_whitespace_id_no_raise(self):
        # Whitespace-only IDs are truthy in Python, so no ValueError is raised
        doc = Document(id="   ", text="text")
        assert doc.id == "   "

    def test_document_empty_text_raises(self):
        with pytest.raises(ValueError, match="Document text cannot be empty"):
            Document(id="doc1", text="")

    def test_document_whitespace_text_raises(self):
        with pytest.raises(ValueError, match="Document text cannot be empty"):
            Document(id="doc1", text="   \n\t  ")

    def test_document_non_empty_text_succeeds(self):
        doc = Document(id="d1", text="  some text  ")
        assert doc.text == "  some text  "

    def test_document_mutation_allowed(self):
        doc = Document(id="d1", text="hello")
        doc.text = "updated"
        assert doc.text == "updated"


# ===========================================================================
# Chunk tests
# ===========================================================================

class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_create_chunk_minimal(self):
        chunk = Chunk(text="Sample text")
        assert chunk.text == "Sample text"
        assert chunk.parent_id is None
        assert chunk.level == 0

    def test_create_chunk_full(self):
        chunk = Chunk(
            text="Content",
            metadata={"doc_id": "d1"},
            parent_id="parent_1",
            level=2,
        )
        assert chunk.metadata["doc_id"] == "d1"
        assert chunk.parent_id == "parent_1"
        assert chunk.level == 2

    def test_chunk_empty_text_raises(self):
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk(text="")

    def test_chunk_whitespace_text_raises(self):
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk(text="  \n  ")

    def test_chunk_timestamp_auto_added(self):
        chunk = Chunk(text="Hello")
        assert "timestamp" in chunk.metadata

    def test_chunk_timestamp_not_overwritten(self):
        ts = "2024-01-01T00:00:00"
        chunk = Chunk(text="Hello", metadata={"timestamp": ts})
        assert chunk.metadata["timestamp"] == ts

    def test_chunk_to_dict(self):
        chunk = Chunk(text="abc", parent_id="p1", level=1)
        d = chunk.to_dict()
        assert d["text"] == "abc"
        assert d["parent_id"] == "p1"
        assert d["level"] == 1
        assert "metadata" in d

    def test_chunk_from_dict(self):
        data = {"text": "abc", "metadata": {"k": "v"}, "parent_id": "p1", "level": 2}
        chunk = Chunk.from_dict(data)
        assert chunk.text == "abc"
        assert chunk.parent_id == "p1"
        assert chunk.level == 2

    def test_chunk_from_dict_defaults(self):
        # from_dict bypasses __post_init__, so no auto-timestamp
        chunk = Chunk.from_dict({"text": "hello"})
        assert chunk.parent_id is None
        assert chunk.level == 0
        # from_dict passes metadata={} but __post_init__ adds timestamp
        assert "timestamp" in chunk.metadata

    def test_chunk_roundtrip(self):
        original = Chunk(text="round trip", metadata={"x": 1}, parent_id="p", level=3)
        restored = Chunk.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.parent_id == original.parent_id
        assert restored.level == original.level


# ===========================================================================
# Result tests
# ===========================================================================

class TestResult:
    """Tests for the Result dataclass."""

    def test_create_result_minimal(self):
        result = Result(text="result text", score=0.95)
        assert result.text == "result text"
        assert result.score == pytest.approx(0.95)
        assert result.metadata == {}
        assert result.provenance is None

    def test_create_result_full(self):
        result = Result(
            text="text",
            score=0.8,
            metadata={"source": "doc1"},
            provenance="chunk_42",
        )
        assert result.metadata["source"] == "doc1"
        assert result.provenance == "chunk_42"

    def test_result_score_zero_allowed(self):
        result = Result(text="t", score=0.0)
        assert result.score == 0.0

    def test_result_score_one_allowed(self):
        result = Result(text="t", score=1.0)
        assert result.score == 1.0

    def test_result_score_negative_raises(self):
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Result(text="t", score=-0.1)

    def test_result_score_above_one_raises(self):
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Result(text="t", score=1.5)

    def test_result_to_dict(self):
        result = Result(text="r", score=0.5, provenance="src")
        d = result.to_dict()
        assert d["text"] == "r"
        assert d["score"] == 0.5
        assert d["provenance"] == "src"

    def test_result_from_dict(self):
        data = {"text": "r", "score": 0.7, "metadata": {"k": "v"}, "provenance": "p"}
        result = Result.from_dict(data)
        assert result.text == "r"
        assert result.score == 0.7
        assert result.provenance == "p"

    def test_result_from_dict_defaults(self):
        result = Result.from_dict({"text": "r", "score": 0.5})
        assert result.metadata == {}
        assert result.provenance is None

    def test_result_roundtrip(self):
        original = Result(text="data", score=0.88, metadata={"k": 1}, provenance="p")
        restored = Result.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.score == original.score
        assert restored.provenance == original.provenance


# ===========================================================================
# IndexerStats tests
# ===========================================================================

class TestIndexerStats:
    """Tests for the IndexerStats dataclass."""

    def test_default_values(self):
        stats = IndexerStats()
        assert stats.num_documents == 0
        assert stats.num_chunks == 0
        assert stats.index_size_mb == 0.0
        assert stats.avg_chunk_size == 0.0
        assert stats.indexing_time_sec == 0.0

    def test_custom_values(self):
        stats = IndexerStats(num_documents=10, num_chunks=50, indexing_time_sec=2.5)
        assert stats.num_documents == 10
        assert stats.num_chunks == 50
        assert stats.indexing_time_sec == 2.5

    def test_to_dict(self):
        stats = IndexerStats(num_documents=5, num_chunks=20)
        d = stats.to_dict()
        assert d["num_documents"] == 5
        assert d["num_chunks"] == 20
        assert d["index_size_mb"] == 0.0
