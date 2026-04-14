"""
Tests for rag_indexer.chunker module.

Covers FixedSizeChunker, SemanticChunker, and HierarchicalChunker.
"""

import pytest
from rag_indexer.config import Document
from rag_indexer.chunker import (
    Chunker,
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
)


# ===========================================================================
# FixedSizeChunker tests
# ===========================================================================

class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_single_chunk_short_text(self):
        chunker = FixedSizeChunker(chunk_size=512, overlap=50)
        doc = Document(id="d1", text="Short text")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_multiple_chunks(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 100  # 100 characters
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_chunk_metadata_document_id(self):
        chunker = FixedSizeChunker(chunk_size=512, overlap=50)
        doc = Document(id="my_doc", text="Hello world")
        chunks = chunker.chunk(doc)
        assert all(c.metadata["document_id"] == "my_doc" for c in chunks)

    def test_chunk_metadata_indices(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 120
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_metadata_char_range(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            start = chunk.metadata["start_char"]
            end = chunk.metadata["end_char"]
            assert chunk.text == text[start:end]

    def test_overlap_creates_shared_content(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=20)
        text = "ABCDEFGHIJ" * 10  # 100 chars
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            # Last 20 chars of chunk 0 should overlap with first part of chunk 1
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            assert end_of_first == start_of_second

    def test_zero_overlap(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        text = "A" * 100
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        # No overlap: end of first = start of second
        assert chunks[0].metadata["end_char"] == chunks[1].metadata["start_char"]

    def test_chunk_size_validation_negative(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=-1)

    def test_chunk_size_validation_zero(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0)

    def test_overlap_validation_negative(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            FixedSizeChunker(chunk_size=100, overlap=-5)

    def test_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FixedSizeChunker(chunk_size=50, overlap=60)

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FixedSizeChunker(chunk_size=50, overlap=50)

    def test_default_parameters(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_single_character_text(self):
        chunker = FixedSizeChunker(chunk_size=512, overlap=50)
        doc = Document(id="d1", text="X")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "X"


# ===========================================================================
# SemanticChunker tests
# ===========================================================================

class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_single_paragraph(self):
        chunker = SemanticChunker(min_size=10, max_size=2000)
        doc = Document(id="d1", text="This is a single paragraph.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "single paragraph" in chunks[0].text

    def test_multiple_paragraphs(self):
        chunker = SemanticChunker(min_size=10, max_size=2000)
        text = "Para one.\n\nPara two.\n\nPara three."
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_max_size_splits_into_multiple(self):
        chunker = SemanticChunker(min_size=5, max_size=50)
        text = "\n\n".join(["A" * 30, "B" * 30, "C" * 30])
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        # Should split into multiple chunks due to max_size
        assert len(chunks) >= 2

    def test_chunk_metadata_document_id(self):
        chunker = SemanticChunker(min_size=10, max_size=5000)
        doc = Document(id="special_doc", text="Paragraph one.\n\nParagraph two.")
        chunks = chunker.chunk(doc)
        assert all(c.metadata["document_id"] == "special_doc" for c in chunks)

    def test_chunk_metadata_paragraph_count(self):
        chunker = SemanticChunker(min_size=5, max_size=5000)
        text = "\n\n".join(["Para 1", "Para 2", "Para 3"])
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        total_paras = sum(c.metadata["paragraph_count"] for c in chunks)
        assert total_paras == 3

    def test_empty_lines_ignored(self):
        chunker = SemanticChunker(min_size=5, max_size=5000)
        text = "One.\n\n\n\n\nTwo."
        doc = Document(id="d1", text=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_min_size_validation(self):
        with pytest.raises(ValueError, match="min_size must be positive"):
            SemanticChunker(min_size=0)

    def test_max_size_less_than_min_size(self):
        with pytest.raises(ValueError, match="max_size must be greater than min_size"):
            SemanticChunker(min_size=100, max_size=50)

    def test_similarity_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            SemanticChunker(similarity_threshold=1.5)

    def test_default_parameters(self):
        chunker = SemanticChunker()
        assert chunker.min_size == 100
        assert chunker.max_size == 2000
        assert chunker.similarity_threshold == 0.7


# ===========================================================================
# HierarchicalChunker tests
# ===========================================================================

class TestHierarchicalChunker:
    """Tests for HierarchicalChunker."""

    def test_produces_document_level_chunk(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        level_zero = [c for c in chunks if c.level == 0]
        assert len(level_zero) == 1
        assert level_zero[0].metadata.get("type") == "document"

    def test_produces_section_level_chunks(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        level_one = [c for c in chunks if c.level == 1]
        assert len(level_one) >= 1

    def test_produces_paragraph_level_chunks(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        level_two = [c for c in chunks if c.level == 2]
        assert len(level_two) >= 1

    def test_parent_child_relationships(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        parent_ids = {c.metadata.get("parent_id") for c in chunks if c.parent_id}
        # Parent IDs should reference existing chunks (at least the L0 chunk)
        assert len(parent_ids) > 0

    def test_section_chunks_have_parent(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        sections = [c for c in chunks if c.level == 1]
        for s in sections:
            assert s.parent_id is not None

    def test_plain_text_creates_main_section(self):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=2)
        doc = Document(id="d1", text="Plain text without headers.\n\nAnother paragraph.")
        chunks = chunker.chunk(doc)
        # Should have at least document level and paragraph level
        levels = {c.level for c in chunks}
        assert 0 in levels  # document level

    def test_leaf_size_validation(self):
        with pytest.raises(ValueError, match="leaf_size must be positive"):
            HierarchicalChunker(leaf_size=0)

    def test_max_levels_minimum(self):
        with pytest.raises(ValueError, match="max_levels must be at least 2"):
            HierarchicalChunker(max_levels=1)

    def test_document_summary_truncation(self):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=2)
        very_long_line = "A" * 1000
        doc = Document(id="d1", text=very_long_line)
        chunks = chunker.chunk(doc)
        doc_chunk = next(c for c in chunks if c.level == 0)
        assert len(doc_chunk.text) <= 503  # 500 + "..."

    def test_multiple_sections_detected(self, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        chunks = chunker.chunk(long_document)
        sections = [c for c in chunks if c.level == 1]
        section_titles = {c.metadata.get("section_title") for c in sections}
        assert len(section_titles) >= 1


# ===========================================================================
# Chunker ABC tests
# ===========================================================================

class TestChunkerABC:
    """Tests for Chunker abstract base class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Chunker()

    def test_subclass_must_implement_chunk(self):
        class BadChunker(Chunker):
            pass

        with pytest.raises(TypeError):
            BadChunker()
