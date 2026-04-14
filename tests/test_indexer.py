"""
Tests for rag_indexer.indexer module.

Covers Indexer class: init, index, search, update, delete, stats, benchmark.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from rag_indexer.indexer import Indexer
from rag_indexer.config import Document, Result, IndexerStats


# ===========================================================================
# Helper
# ===========================================================================

def _make_indexer(embedder=None, vector_store=None, chunker=None, retriever=None):
    """Create an Indexer with mocked or provided dependencies."""
    if embedder is None:
        embedder = MagicMock()
        embedder.dimension = 384
        embedder.embed_batch.return_value = MagicMock()

    if vector_store is None:
        vector_store = MagicMock()
        vector_store.size_mb.return_value = 0.01

    if chunker is None:
        chunker = MagicMock()
        from rag_indexer.config import Chunk
        chunker.chunk.return_value = [
            Chunk(text="chunk text", metadata={"id": "c1", "document_id": "d1"})
        ]

    if retriever is None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            Result(text="result", score=0.9, provenance="c1")
        ]

    return Indexer(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        retriever=retriever,
    )


# ===========================================================================
# Indexer.__init__ tests
# ===========================================================================

class TestIndexerInit:
    """Tests for Indexer initialization."""

    def test_initializes_components(self):
        idx = _make_indexer()
        assert idx.embedder is not None
        assert idx.vector_store is not None
        assert idx.chunker is not None
        assert idx.retriever is not None

    def test_initial_stats(self):
        idx = _make_indexer()
        stats = idx.stats()
        assert stats.num_documents == 0
        assert stats.num_chunks == 0
        assert stats.indexing_time_sec == 0.0

    def test_not_indexed_flag(self):
        idx = _make_indexer()
        assert idx._indexed is False


# ===========================================================================
# Indexer.index tests
# ===========================================================================

class TestIndexerIndex:
    """Tests for Indexer.index method."""

    def test_index_calls_chunker(self):
        chunker = MagicMock()
        from rag_indexer.config import Chunk
        chunker.chunk.return_value = [Chunk(text="c", metadata={"id": "c1"})]
        idx = _make_indexer(chunker=chunker)

        docs = [Document(id="d1", text="Hello")]
        idx.index(docs)
        chunker.chunk.assert_called_once()

    def test_index_calls_embed_batch(self):
        embedder = MagicMock()
        embedder.dimension = 384
        embedder.embed_batch.return_value = [[0.0] * 384]
        idx = _make_indexer(embedder=embedder)

        docs = [Document(id="d1", text="Hello")]
        idx.index(docs)
        embedder.embed_batch.assert_called_once()

    def test_index_calls_vector_store_add(self):
        vector_store = MagicMock()
        vector_store.size_mb.return_value = 0.01
        idx = _make_indexer(vector_store=vector_store)

        docs = [Document(id="d1", text="Hello")]
        idx.index(docs)
        vector_store.add.assert_called_once()

    def test_index_updates_stats(self):
        idx = _make_indexer()
        docs = [Document(id="d1", text="Hello"), Document(id="d2", text="World")]
        idx.index(docs)
        stats = idx.stats()
        assert stats.num_documents == 2
        assert stats.num_chunks >= 1
        assert stats.indexing_time_sec > 0

    def test_index_sets_indexed_flag(self):
        idx = _make_indexer()
        docs = [Document(id="d1", text="Hello")]
        idx.index(docs)
        assert idx._indexed is True

    def test_index_multiple_batches_accumulates(self):
        idx = _make_indexer()
        docs1 = [Document(id="d1", text="One")]
        docs2 = [Document(id="d2", text="Two")]
        idx.index(docs1)
        idx.index(docs2)
        stats = idx.stats()
        assert stats.num_documents == 2

    def test_index_empty_document_list_raises(self):
        """Empty document list triggers ZeroDivisionError in avg_chunk_size."""
        idx = _make_indexer()
        with pytest.raises(ZeroDivisionError):
            idx.index([])

    def test_index_calls_build_sparse_index_for_hybrid(self):
        retriever = MagicMock()
        retriever.build_sparse_index = MagicMock()
        idx = _make_indexer(retriever=retriever)

        docs = [Document(id="d1", text="Hello")]
        idx.index(docs)
        retriever.build_sparse_index.assert_called_once()


# ===========================================================================
# Indexer.search tests
# ===========================================================================

class TestIndexerSearch:
    """Tests for Indexer.search method."""

    def test_search_raises_when_not_indexed(self):
        idx = _make_indexer()
        with pytest.raises(RuntimeError, match="No documents indexed"):
            idx.search("query")

    def test_search_returns_results(self):
        idx = _make_indexer()
        idx.index([Document(id="d1", text="Hello")])
        results = idx.search("query", k=5)
        assert isinstance(results, list)

    def test_search_passes_correct_k(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        idx = _make_indexer(retriever=retriever)
        idx.index([Document(id="d1", text="Hello")])

        idx.search("query", k=3)
        call_kwargs = retriever.retrieve.call_args[1]
        assert call_kwargs["k"] == 3

    def test_search_with_filter_applies_filter(self):
        from rag_indexer.config import Chunk
        chunker = MagicMock()
        chunker.chunk.return_value = [
            Chunk(text="chunk", metadata={"id": "c1", "document_id": "d1"})
        ]
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            Result(text="result", score=0.9, metadata={"category": "music"}, provenance="c1"),
            Result(text="other", score=0.8, metadata={"category": "marine"}, provenance="c2"),
        ]
        idx = _make_indexer(chunker=chunker, retriever=retriever)
        idx.index([Document(id="d1", text="Hello")])

        results = idx.search("query", k=5, filter={"category": "music"})
        assert len(results) == 1
        assert results[0].metadata["category"] == "music"

    def test_search_with_non_matching_filter(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            Result(text="result", score=0.9, metadata={"cat": "a"}),
        ]
        idx = _make_indexer(retriever=retriever)
        idx.index([Document(id="d1", text="Hello")])

        results = idx.search("query", k=5, filter={"cat": "nonexistent"})
        assert len(results) == 0

    def test_search_without_filter(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [Result(text="r", score=0.9)]
        idx = _make_indexer(retriever=retriever)
        idx.index([Document(id="d1", text="Hello")])

        results = idx.search("query", k=5)
        assert len(results) == 1


# ===========================================================================
# Indexer.update tests
# ===========================================================================

class TestIndexerUpdate:
    """Tests for Indexer.update method."""

    def test_update_calls_index(self):
        idx = _make_indexer()
        doc = Document(id="d1", text="Updated text")
        idx.update(doc)
        assert idx._indexed is True
        stats = idx.stats()
        assert stats.num_documents == 1


# ===========================================================================
# Indexer.delete tests
# ===========================================================================

class TestIndexerDelete:
    """Tests for Indexer.delete method."""

    def test_delete_not_implemented(self):
        idx = _make_indexer()
        with pytest.raises(NotImplementedError):
            idx.delete("doc1")


# ===========================================================================
# Indexer.save tests
# ===========================================================================

class TestIndexerSave:
    """Tests for Indexer.save method."""

    def test_save_calls_vector_store_save(self):
        vector_store = MagicMock()
        vector_store.size_mb.return_value = 0.01
        idx = _make_indexer(vector_store=vector_store)
        idx.save("/tmp/test.index")
        vector_store.save.assert_called_once_with("/tmp/test.index")


# ===========================================================================
# Indexer.load tests
# ===========================================================================

class TestIndexerLoad:
    """Tests for Indexer.load classmethod."""

    def test_load_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Indexer.load("/tmp/test.index")


# ===========================================================================
# Indexer.stats tests
# ===========================================================================

class TestIndexerStatsMethod:
    """Tests for Indexer.stats method."""

    def test_stats_returns_indexer_stats(self):
        idx = _make_indexer()
        stats = idx.stats()
        assert isinstance(stats, IndexerStats)

    def test_stats_calls_size_mb(self):
        vector_store = MagicMock()
        vector_store.size_mb.return_value = 42.5
        idx = _make_indexer(vector_store=vector_store)
        stats = idx.stats()
        assert stats.index_size_mb == 42.5


# ===========================================================================
# Indexer._apply_filter tests
# ===========================================================================

class TestApplyFilter:
    """Tests for Indexer._apply_filter internal method."""

    def test_filter_matching_key(self):
        idx = _make_indexer()
        results = [
            Result(text="a", score=0.9, metadata={"cat": "music"}),
            Result(text="b", score=0.8, metadata={"cat": "marine"}),
        ]
        filtered = idx._apply_filter(results, {"cat": "music"})
        assert len(filtered) == 1
        assert filtered[0].text == "a"

    def test_filter_no_match(self):
        idx = _make_indexer()
        results = [Result(text="a", score=0.9, metadata={"cat": "music"})]
        filtered = idx._apply_filter(results, {"cat": "nonexistent"})
        assert filtered == []

    def test_filter_multiple_keys(self):
        idx = _make_indexer()
        results = [
            Result(text="a", score=0.9, metadata={"cat": "music", "lang": "en"}),
            Result(text="b", score=0.8, metadata={"cat": "music", "lang": "fr"}),
        ]
        filtered = idx._apply_filter(results, {"cat": "music", "lang": "en"})
        assert len(filtered) == 1
        assert filtered[0].text == "a"

    def test_filter_missing_key(self):
        idx = _make_indexer()
        results = [Result(text="a", score=0.9, metadata={})]
        filtered = idx._apply_filter(results, {"missing_key": "val"})
        assert filtered == []
