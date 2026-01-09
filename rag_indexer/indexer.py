"""
Main Indexer interface for rag-indexer.

The Indexer provides the primary API for document indexing and retrieval.
"""

import time
from typing import List, Dict, Any, Optional

from .config import Document, Chunk, Result, IndexerStats
from .chunker import Chunker
from .retriever import Retriever
from vector_navigator import VectorStore
from embeddings_engine import EmbeddingsEngine


class Indexer:
    """
    Main RAG indexing interface.

    The Indexer coordinates document indexing and retrieval by combining:
    - Chunker: Split documents into chunks
    - Embedder: Convert text to embeddings
    - VectorStore: Store and search embeddings
    - Retriever: Retrieve and rank relevant chunks

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
        >>> indexer = Indexer(
        ...     embedder=embedder,
        ...     vector_store=vector_store,
        ...     chunker=chunker,
        ...     retriever=retriever,
        ... )
        >>>
        >>> documents = ["Document 1...", "Document 2..."]
        >>> indexer.index(documents)
        >>>
        >>> results = indexer.search("query", k=10)
    """

    def __init__(
        self,
        embedder: EmbeddingsEngine,
        vector_store: VectorStore,
        chunker: Chunker,
        retriever: Retriever,
    ):
        """
        Initialize indexer.

        Args:
            embedder: Embeddings engine for text-to-embedding conversion
            vector_store: Vector store for storage and search
            chunker: Chunker for document splitting
            retriever: Retriever for result ranking
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker
        self.retriever = retriever

        self._stats = IndexerStats()
        self._indexed = False

    def index(self, documents: List[Document]) -> None:
        """
        Index documents into the vector store.

        Pipeline:
        1. Chunk documents
        2. Embed chunks
        3. Store in vector store
        4. Build sparse index (if hybrid retriever)

        Args:
            documents: List of documents to index
        """
        start_time = time.time()

        # Step 1: Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        # Step 2: Embed chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # Step 3: Store in vector store
        self.vector_store.add(
            vectors=embeddings,
            metadata=[chunk.metadata for chunk in all_chunks],
            ids=[chunk.metadata.get("id", f"chunk_{i}") for i, chunk in enumerate(all_chunks)],
        )

        # Step 4: Build sparse index if hybrid retriever
        if hasattr(self.retriever, "build_sparse_index"):
            self.retriever.build_sparse_index(all_chunks)

        # Update stats
        self._stats.num_documents += len(documents)
        self._stats.num_chunks += len(all_chunks)
        self._stats.avg_chunk_size = sum(len(c.text) for c in all_chunks) / len(all_chunks)
        self._stats.indexing_time_sec += time.time() - start_time
        self._indexed = True

    def index_batch(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = False,
        save_interval: int = None,
    ) -> None:
        """
        Index documents in batches.

        Args:
            documents: List of documents to index
            batch_size: Number of documents per batch
            show_progress: Show progress bar
            save_interval: Save index every N batches
        """
        from tqdm import tqdm

        num_docs = len(documents)
        num_batches = (num_docs + batch_size - 1) // batch_size

        iterator = range(0, num_docs, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Indexing", total=num_batches)

        for i in iterator:
            batch = documents[i : i + batch_size]
            self.index(batch)

            # Save if requested
            if save_interval and (i // batch_size + 1) % save_interval == 0:
                self.save(f"/tmp/rag_indexer_checkpoint_{i}.index")

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Result]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of ranked results (descending by score)
        """
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call index() first.")

        # Retrieve using retriever strategy
        results = self.retriever.retrieve(
            query=query,
            k=k,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

        # Apply filter if provided
        if filter:
            results = self._apply_filter(results, filter)

        return results

    def update(self, document: Document) -> None:
        """
        Update a document in the index.

        Strategy:
        - Mark old chunks as deleted
        - Index new chunks
        - Rebuild sparse index if needed

        Args:
            document: Document to update
        """
        # Implementation depends on vector store capabilities
        # For now, just re-index
        self.index([document])

    def delete(self, document_id: str) -> None:
        """
        Delete a document from the index.

        Args:
            document_id: ID of document to delete
        """
        # Implementation depends on vector store capabilities
        raise NotImplementedError("Delete not yet implemented")

    def save(self, path: str) -> None:
        """
        Save index to disk.

        Args:
            path: Path to save index
        """
        self.vector_store.save(path)

    @classmethod
    def load(cls, path: str) -> "Indexer":
        """
        Load index from disk.

        Args:
            path: Path to load index from

        Returns:
            Loaded Indexer instance
        """
        # Implementation depends on serialization strategy
        raise NotImplementedError("Load not yet implemented")

    def stats(self) -> IndexerStats:
        """
        Get indexer statistics.

        Returns:
            IndexerStats object with indexing metrics
        """
        # Update index size
        self._stats.index_size_mb = self.vector_store.size_mb()

        return self._stats

    def benchmark_indexing(
        self,
        documents: List[Document],
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark indexing performance.

        Args:
            documents: Test documents
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with performance metrics
        """
        times = []
        for _ in range(num_runs):
            # Create fresh indexer
            vector_store = VectorStore(dimension=self.embedder.dimension)

            start = time.time()
            self.index(documents)
            times.append(time.time() - start)

        import numpy as np

        return {
            "mean_sec": np.mean(times),
            "p50_sec": np.percentile(times, 50),
            "p99_sec": np.percentile(times, 99),
            "docs_per_sec": len(documents) / np.mean(times),
        }

    def benchmark_retrieval(
        self,
        queries: List[str],
        k: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark retrieval performance.

        Args:
            queries: Test queries
            k: Number of results
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with performance metrics
        """
        import numpy as np

        times = []
        for _ in range(num_runs):
            query = queries[_ % len(queries)]

            start = time.time()
            self.search(query, k=k)
            times.append((time.time() - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
        }

    def _apply_filter(self, results: List[Result], filter: Dict[str, Any]) -> List[Result]:
        """Apply metadata filter to results."""
        filtered = []
        for result in results:
            match = True
            for key, value in filter.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered
