"""
Basic usage example for rag-indexer.

This example demonstrates how to:
1. Create an indexer with semantic chunking and hybrid retrieval
2. Index a collection of documents
3. Retrieve relevant context for a query
"""

from rag_indexer import (
    Indexer,
    SemanticChunker,
    HybridRetriever,
    Document,
)
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore, IndexType


def main():
    """Run basic usage example."""

    # Step 1: Setup components
    print("Setting up rag-indexer...")

    embedder = EmbeddingsEngine(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # Use "cuda" for GPU acceleration
    )

    vector_store = VectorStore(
        dimension=384,  # Embedding dimension for all-MiniLM-L6-v2
        index_type=IndexType.HNSW,  # Use HNSW for fast approximate search
        M=16,
        efConstruction=200,
    )

    chunker = SemanticChunker(
        min_size=100,
        max_size=2000,
    )

    retriever = HybridRetriever(
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    # Step 2: Create indexer
    print("Creating indexer...")
    indexer = Indexer(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        retriever=retriever,
    )

    # Step 3: Index documents
    print("\nIndexing documents...")

    documents = [
        Document(
            id="doc1",
            text="""
            Fishing boats navigate calm waters at sunrise.
            These vessels are designed for commercial fishing operations
            in coastal waters and the open ocean. Modern fishing boats
            are equipped with advanced navigation systems, sonar,
            and GPS technology to locate fish populations efficiently.
            """,
            metadata={
                "title": "Fishing Boats",
                "category": "marine",
                "sentiment": "positive",
            },
        ),
        Document(
            id="doc2",
            text="""
            Jazz music originated in New Orleans in the late 19th century.
            This genre combines elements of blues, ragtime, and brass band
            music. Jazz is characterized by swing, blue notes, complex chords,
            and improvisation. Famous jazz musicians include Louis Armstrong,
            Miles Davis, and John Coltrane.
            """,
            metadata={
                "title": "Jazz History",
                "category": "music",
                "sentiment": "neutral",
            },
        ),
        Document(
            id="doc3",
            text="""
            Symphony pops concerts feature popular classical music selections.
            These performances blend traditional symphonic repertoire with
            contemporary pieces, film scores, and Broadway show tunes.
            Pops concerts are designed to be accessible and entertaining
            for diverse audiences.
            """,
            metadata={
                "title": "Symphony Pops",
                "category": "music",
                "sentiment": "positive",
            },
        ),
    ]

    indexer.index(documents)
    print(f"Indexed {len(documents)} documents")

    # Step 4: Search
    print("\nSearching for relevant context...")

    query = "Tell me about calm waters on the ocean"
    print(f"Query: {query}\n")

    results = indexer.search(query, k=3)

    # Step 5: Display results
    print("Retrieved results:")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result.score:.3f}")
        print(f"  Text: {result.text[:200]}...")
        print(f"  Metadata: {result.metadata}")
        print(f"  Provenance: {result.provenance}")

    # Step 6: Display stats
    print("\n" + "-" * 80)
    print("Indexer Statistics:")
    stats = indexer.stats()
    print(f"  Documents: {stats.num_documents}")
    print(f"  Chunks: {stats.num_chunks}")
    print(f"  Avg chunk size: {stats.avg_chunk_size:.0f} characters")
    print(f"  Indexing time: {stats.indexing_time_sec:.2f} seconds")


if __name__ == "__main__":
    main()
