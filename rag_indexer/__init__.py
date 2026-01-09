"""
rag-indexer: Retrieval-Augmented Generation Indexing for equilibrium-tokens

A high-performance Python/Go library providing advanced RAG indexing capabilities,
combining semantic similarity search with intelligent chunking and hybrid retrieval.

Core Components:
- Indexer: Main interface for document indexing and retrieval
- Chunker: Document chunking strategies (fixed, semantic, hierarchical)
- Retriever: Retrieval strategies (dense, hybrid, recursive)

Example:
    >>> from rag_indexer import Indexer, SemanticChunker, HybridRetriever
    >>> from embeddings_engine import EmbeddingsEngine
    >>> from vector_navigator import VectorStore
    >>>
    >>> embedder = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
    >>> vector_store = VectorStore(dimension=384)
    >>> chunker = SemanticChunker()
    >>> retriever = HybridRetriever(dense_weight=0.7)
    >>>
    >>> indexer = Indexer(embedder=embedder, vector_store=vector_store,
    ...                   chunker=chunker, retriever=retriever)
    >>>
    >>> documents = ["Document 1 text...", "Document 2 text..."]
    >>> indexer.index(documents)
    >>>
    >>> results = indexer.search("query", k=10)
    >>> for result in results:
    ...     print(f"[{result.score:.3f}] {result.text}")
"""

__version__ = "1.0.0"

# Core abstractions
from .indexer import Indexer
from .chunker import (
    Chunker,
    Chunk,
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
)
from .retriever import (
    Retriever,
    Result,
    DenseRetriever,
    HybridRetriever,
    RecursiveRetriever,
)
from .config import Document

__all__ = [
    # Version
    "__version__",
    # Core
    "Indexer",
    "Document",
    # Chunkers
    "Chunker",
    "Chunk",
    "FixedSizeChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    # Retrievers
    "Retriever",
    "Result",
    "DenseRetriever",
    "HybridRetriever",
    "RecursiveRetriever",
]
