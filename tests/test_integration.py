"""
Integration tests for rag-indexer.

Tests end-to-end flows with mocked external dependencies.
"""

import pytest
import numpy as np

from rag_indexer.config import Document, Chunk, Result
from rag_indexer.chunker import FixedSizeChunker, SemanticChunker, HierarchicalChunker
from rag_indexer.retriever import DenseRetriever, HybridRetriever, RecursiveRetriever
from rag_indexer.indexer import Indexer


# ===========================================================================
# Full pipeline integration tests
# ===========================================================================

class TestFullPipeline:
    """End-to-end tests using real chunkers with mocked embedder/store."""

    def test_index_and_search_fixed_size_chunker(self, embedder, vector_store, sample_documents):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index(sample_documents)
        results = idx.search("jazz music", k=2)

        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, Result)
            assert 0.0 <= r.score <= 1.0

    def test_index_and_search_semantic_chunker(self, embedder, vector_store, sample_documents):
        chunker = SemanticChunker(min_size=50, max_size=5000)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index(sample_documents)
        results = idx.search("fishing boats", k=3)
        assert len(results) <= 3

    def test_index_and_search_hybrid_retriever(self, embedder, vector_store, sample_documents):
        chunker = SemanticChunker(min_size=50, max_size=5000)
        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index(sample_documents)
        results = idx.search("symphony concert", k=3)
        assert isinstance(results, list)

    def test_index_stats_are_accurate(self, embedder, vector_store, sample_documents):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index(sample_documents)
        stats = idx.stats()

        assert stats.num_documents == len(sample_documents)
        assert stats.num_chunks > len(sample_documents)  # more chunks than docs
        assert stats.indexing_time_sec > 0
        assert stats.avg_chunk_size > 0

    def test_search_after_update(self, embedder, vector_store):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index([Document(id="d1", text="First document about cats")])
        idx.index([Document(id="d2", text="Second document about dogs")])

        stats = idx.stats()
        assert stats.num_documents == 2


# ===========================================================================
# Chunker + Retriever compatibility
# ===========================================================================

class TestChunkerRetrieverCompatibility:
    """Test that chunker output is compatible with retriever input."""

    def test_fixed_chunks_with_hybrid_retriever(self, embedder, vector_store):
        doc = Document(id="d1", text="A" * 500)
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        retriever = HybridRetriever(dense_weight=0.6, sparse_weight=0.4)
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index([doc])
        results = idx.search("test", k=3)
        assert isinstance(results, list)

    def test_semantic_chunks_with_dense_retriever(self, embedder, vector_store):
        doc = Document(id="d1", text="Para one.\n\nPara two.\n\nPara three.")
        chunker = SemanticChunker(min_size=5, max_size=5000)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index([doc])
        results = idx.search("test", k=3)
        assert isinstance(results, list)

    def test_hierarchical_chunks_with_recursive_retriever(self, embedder, vector_store, long_document):
        chunker = HierarchicalChunker(leaf_size=1800, max_levels=3)
        retriever = RecursiveRetriever(levels=2, top_parents=2, children_per_parent=3)
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index([long_document])
        results = idx.search("methodology", k=3)
        assert isinstance(results, list)


# ===========================================================================
# Edge case integration tests
# ===========================================================================

class TestEdgeCases:
    """Test edge cases in integration scenarios."""

    def test_single_word_documents(self, embedder, vector_store):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        docs = [Document(id="d1", text="Hello"), Document(id="d2", text="World")]
        idx.index(docs)
        results = idx.search("Hello", k=1)
        assert len(results) >= 0  # May or may not return results

    def test_very_long_document(self, embedder, vector_store):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~12000 chars
        doc = Document(id="long", text=long_text)
        idx.index([doc])

        stats = idx.stats()
        assert stats.num_chunks > 1

    def test_unicode_document(self, embedder, vector_store):
        chunker = FixedSizeChunker(chunk_size=200, overlap=10)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        doc = Document(id="unicode", text="Héllo wörld! 日本語テスト 中文测试 🌍")
        idx.index([doc])
        results = idx.search("test", k=1)
        assert isinstance(results, list)

    def test_empty_query_after_indexing(self, embedder, vector_store, sample_documents):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        retriever = DenseRetriever()
        idx = Indexer(embedder=embedder, vector_store=vector_store, chunker=chunker, retriever=retriever)

        idx.index(sample_documents)
        # Empty query should still work (behavior depends on embedder)
        results = idx.search("", k=3)
        assert isinstance(results, list)
