"""
Tests for rag_indexer.retriever module.

Covers DenseRetriever, HybridRetriever, RecursiveRetriever.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from rag_indexer.config import Chunk, Result
from rag_indexer.retriever import (
    Retriever,
    DenseRetriever,
    HybridRetriever,
    RecursiveRetriever,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_embedder(dimension=384):
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.dimension = dimension

    def embed_fn(text):
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    embedder.embed.side_effect = embed_fn
    embedder.embed_batch.side_effect = lambda texts: np.array([embed_fn(t) for t in texts])
    return embedder


def _make_vector_store(dimension=384, num_items=10):
    """Create a mock vector store with some data."""
    store = MagicMock()
    store.size_mb.return_value = 0.01

    # Pre-populate with some items
    items = []
    for i in range(num_items):
        rng = np.random.RandomState(i)
        vec = rng.randn(dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        items.append({
            "id": f"chunk_{i}",
            "score": max(0.0, 1.0 - i * 0.05),
            "text": f"Chunk text {i}",
            "metadata": {"document_id": f"doc_{i % 3}", "chunk_index": i},
        })

    store.search.return_value = items
    return store


# ===========================================================================
# Retriever ABC tests
# ===========================================================================

class TestRetrieverABC:
    """Tests for Retriever abstract base class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Retriever()

    def test_subclass_must_implement_retrieve(self):
        class BadRetriever(Retriever):
            pass

        with pytest.raises(TypeError):
            BadRetriever()


# ===========================================================================
# DenseRetriever tests
# ===========================================================================

class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_default_retrieval_k_is_none(self):
        retriever = DenseRetriever()
        assert retriever.retrieval_k is None

    def test_custom_retrieval_k(self):
        retriever = DenseRetriever(retrieval_k=50)
        assert retriever.retrieval_k == 50

    def test_retrieve_returns_results(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        results = retriever.retrieve("test query", k=5, vector_store=store, embedder=embedder)
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_retrieve_calls_embedder(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        retriever.retrieve("query", k=3, vector_store=store, embedder=embedder)
        embedder.embed.assert_called_once_with("query")

    def test_retrieve_calls_search(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        retriever.retrieve("query", k=3, vector_store=store, embedder=embedder)
        store.search.assert_called_once()

    def test_retrieve_uses_retrieval_k(self):
        retriever = DenseRetriever(retrieval_k=20)
        embedder = _make_embedder()
        store = _make_vector_store()

        retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        call_args = store.search.call_args
        # retrieval_k (20) should be used, not k (5)
        assert call_args[1].get("k") == 20 or call_args[0][1] == 20

    def test_retrieve_truncates_to_k(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = _make_vector_store(num_items=20)

        results = retriever.retrieve("query", k=3, vector_store=store, embedder=embedder)
        assert len(results) <= 3

    def test_retrieve_result_types(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        results = retriever.retrieve("query", k=3, vector_store=store, embedder=embedder)
        for r in results:
            assert isinstance(r, Result)
            assert isinstance(r.text, str)
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_empty_store(self):
        retriever = DenseRetriever()
        embedder = _make_embedder()
        store = MagicMock()
        store.search.return_value = []

        results = retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        assert results == []


# ===========================================================================
# HybridRetriever tests
# ===========================================================================

class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_default_weights(self):
        retriever = HybridRetriever()
        assert retriever.dense_weight == 0.7
        assert retriever.sparse_weight == 0.3

    def test_custom_weights(self):
        retriever = HybridRetriever(dense_weight=0.6, sparse_weight=0.4)
        assert retriever.dense_weight == 0.6
        assert retriever.sparse_weight == 0.4

    def test_weights_sum_not_one_raises(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            HybridRetriever(dense_weight=0.5, sparse_weight=0.6)

    def test_dense_weight_out_of_range(self):
        with pytest.raises(ValueError, match="dense_weight must be between 0 and 1"):
            HybridRetriever(dense_weight=1.5, sparse_weight=-0.5)

    def test_sparse_weight_out_of_range(self):
        with pytest.raises(ValueError, match="sparse_weight must be between 0 and 1"):
            HybridRetriever(dense_weight=0.5, sparse_weight=-0.1)

    def test_weights_within_tolerance(self):
        # 0.7 + 0.3 = 1.0 should pass
        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
        assert retriever is not None

    def test_initializes_sparse_index_none(self):
        retriever = HybridRetriever()
        assert retriever.sparse_matrix is None

    def test_build_sparse_index(self):
        retriever = HybridRetriever()
        chunks = [
            Chunk(text="First chunk about jazz music", metadata={"id": "c1"}),
            Chunk(text="Second chunk about fishing boats", metadata={"id": "c2"}),
        ]
        retriever.build_sparse_index(chunks)
        assert retriever.sparse_matrix is not None
        assert retriever.sparse_vectorizer is not None

    def test_build_sparse_index_empty(self):
        retriever = HybridRetriever()
        with pytest.raises(Exception):
            retriever.build_sparse_index([])

    def test_sparse_search_without_index(self):
        retriever = HybridRetriever()
        results = retriever._sparse_search("query", k=5)
        assert results == []

    def test_sparse_search_with_index(self):
        retriever = HybridRetriever()
        chunks = [
            Chunk(text="jazz music and blues", metadata={"id": "c1"}),
            Chunk(text="fishing boats and ocean", metadata={"id": "c2"}),
            Chunk(text="classical symphony concerts", metadata={"id": "c3"}),
        ]
        retriever.build_sparse_index(chunks)
        results = retriever._sparse_search("jazz blues", k=2)
        assert len(results) > 0
        # First result should be about jazz
        assert results[0]["id"] == "c1"

    def test_weighted_fusion_strategy(self):
        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3, fusion_strategy="weighted")
        dense = [{"id": "c1", "score": 0.9, "text": "a", "metadata": {}}]
        sparse = [{"id": "c1", "score": 0.6}]
        fused = retriever._fuse_scores(dense, sparse, strategy="weighted")
        assert len(fused) == 1
        expected = 0.7 * 0.9 + 0.3 * 0.6
        assert fused[0].score == pytest.approx(expected, rel=1e-3)

    def test_rrf_fusion_strategy(self):
        retriever = HybridRetriever(fusion_strategy="rrf")
        dense = [
            {"id": "c1", "score": 0.9, "text": "a", "metadata": {}},
            {"id": "c2", "score": 0.7, "text": "b", "metadata": {}},
        ]
        sparse = [{"id": "c2", "score": 0.8}]
        fused = retriever._fuse_scores(dense, sparse, strategy="rrf")
        # Both c1 and c2 should be present (c2 has sparse match)
        assert len(fused) == 2

    def test_fusion_unknown_strategy(self):
        retriever = HybridRetriever()
        dense = [{"id": "c1", "score": 0.9, "text": "a", "metadata": {}}]
        sparse = []
        fused = retriever._fuse_scores(dense, sparse, strategy="unknown")
        # Should fall back to dense score * dense_weight
        assert fused[0].score == pytest.approx(0.9 * 0.7)

    def test_fusion_dense_only_results(self):
        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
        dense = [
            {"id": "c1", "score": 0.9, "text": "a", "metadata": {}},
            {"id": "c2", "score": 0.8, "text": "b", "metadata": {}},
        ]
        sparse = []
        fused = retriever._fuse_scores(dense, sparse)
        assert len(fused) == 2
        # Scores should be scaled by dense_weight
        assert fused[0].score == pytest.approx(0.9 * 0.7)

    def test_fusion_results_sorted_descending(self):
        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
        dense = [
            {"id": "c1", "score": 0.5, "text": "low", "metadata": {}},
            {"id": "c2", "score": 0.9, "text": "high", "metadata": {}},
        ]
        sparse = [{"id": "c2", "score": 0.8}]
        fused = retriever._fuse_scores(dense, sparse)
        assert fused[0].score >= fused[1].score

    def test_retrieve_with_sparse_index(self):
        retriever = HybridRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        chunks = [
            Chunk(text=f"chunk {i} text", metadata={"id": f"c{i}"})
            for i in range(10)
        ]
        retriever.build_sparse_index(chunks)

        results = retriever.retrieve("test", k=3, vector_store=store, embedder=embedder)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, Result)


# ===========================================================================
# RecursiveRetriever tests
# ===========================================================================

class TestRecursiveRetriever:
    """Tests for RecursiveRetriever."""

    def test_default_parameters(self):
        retriever = RecursiveRetriever()
        assert retriever.levels == 2
        assert retriever.top_parents == 3
        assert retriever.children_per_parent == 5

    def test_custom_parameters(self):
        retriever = RecursiveRetriever(levels=3, top_parents=5, children_per_parent=10)
        assert retriever.levels == 3
        assert retriever.top_parents == 5
        assert retriever.children_per_parent == 10

    def test_levels_below_minimum_raises(self):
        with pytest.raises(ValueError, match="levels must be at least 2"):
            RecursiveRetriever(levels=1)

    def test_top_parents_below_minimum_raises(self):
        with pytest.raises(ValueError, match="top_parents must be at least 1"):
            RecursiveRetriever(top_parents=0)

    def test_children_per_parent_below_minimum_raises(self):
        with pytest.raises(ValueError, match="children_per_parent must be at least 1"):
            RecursiveRetriever(children_per_parent=0)

    def test_retrieve_calls_search_with_filter(self):
        retriever = RecursiveRetriever(levels=2, top_parents=2, children_per_parent=3)
        embedder = _make_embedder()
        store = _make_vector_store()

        retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        # Should call search at least once for parent level
        assert store.search.call_count >= 1

    def test_retrieve_returns_results(self):
        retriever = RecursiveRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        results = retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        assert isinstance(results, list)

    def test_retrieve_results_are_result_objects(self):
        retriever = RecursiveRetriever()
        embedder = _make_embedder()
        store = _make_vector_store()

        results = retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        for r in results:
            assert isinstance(r, Result)

    def test_retrieve_truncates_to_k(self):
        retriever = RecursiveRetriever(levels=2, top_parents=5, children_per_parent=5)
        embedder = _make_embedder()
        store = _make_vector_store(num_items=30)

        results = retriever.retrieve("query", k=3, vector_store=store, embedder=embedder)
        assert len(results) <= 3

    def test_retrieve_empty_store(self):
        retriever = RecursiveRetriever()
        embedder = _make_embedder()
        store = MagicMock()
        store.search.return_value = []

        results = retriever.retrieve("query", k=5, vector_store=store, embedder=embedder)
        assert results == []
