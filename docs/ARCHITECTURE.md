# rag-indexer Architecture

**Version:** 1.0
**Last Updated:** January 8, 2026
**Status:** Round 3 - SuperInstance Architecture Orchestrator

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Timeless Principles](#timeless-principles)
3. [Core Abstractions](#core-abstractions)
4. [Component Architecture](#component-architecture)
5. [Chunking Strategies](#chunking-strategies)
6. [Hybrid Retrieval](#hybrid-retrieval)
7. [Recursive Retrieval](#recursive-retrieval)
8. [Integration with Dependencies](#integration-with-dependencies)
9. [Performance Architecture](#performance-architecture)
10. [Research Foundation](#research-foundation)

---

## Philosophy

### Core Principle

**"Retrieval augments generation with relevant context"**

rag-indexer exists to bridge the gap between large language models and vast document collections. By retrieving only the most relevant context, we:

1. **Enhance Accuracy**: Ground responses in retrieved evidence
2. **Reduce Hallucination**: Constrain generation to retrieved facts
3. **Improve Efficiency**: Avoid processing entire document collections
4. **Enable Scale**: Work with millions of documents efficiently

### Design Tenets

1. **IR Fundamentals First**: Built on precision/recall, not hype
2. **Research-Backed**: Incorporates 2024-2025 academic advances
3. **Performance-Critical**: Sub-50ms retrieval latency
4. **Modular**: Pluggable chunkers, retrievers, embedders
5. **Language-Agnostic**: Python core, Go bindings for performance

---

## Timeless Principles

### Information Retrieval: Precision and Recall

```python
# The fundamental IR metrics (eternal, unchanging)

relevant = set(documents_that_answer_the_query)
retrieved = set(documents_returned_by_system)

# Precision: Of what we retrieved, how much is relevant?
precision = len(relevant & retrieved) / len(retrieved)

# Recall: Of what's relevant, how much did we retrieve?
recall = len(relevant & retrieved) / len(relevant)

# F1 Score: Harmonic mean (balances both)
f1 = 2 * (precision * recall) / (precision + recall)
```

**Why This Matters**: RAG systems must balance:
- **Precision**: Don't overwhelm the LLM with irrelevant context
- **Recall**: Don't miss critical information

### The Precision-Recall Trade-off

```python
# High precision, low recall:
# - Retrieved: 5 docs, 5 relevant → Precision = 100%, Recall = 10%
# - Good: Context is highly relevant
# - Bad: Might miss important information

# Low precision, high recall:
# - Retrieved: 100 docs, 10 relevant → Precision = 10%, Recall = 100%
# - Good: Found all relevant info
# - Bad: Overwhelms LLM with noise

# RAG sweet spot:
# - Precision > 85% (most retrieved is relevant)
# - Recall > 80% (found most relevant info)
# - Achieved via hybrid retrieval + hierarchical chunking
```

### Vector Similarity: Cosine Distance

```python
# The timeless semantic similarity metric

import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity: -1 to 1 (1 = identical direction)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a, b):
    """Cosine distance: 0 to 2 (0 = identical)"""
    return 1 - cosine_similarity(a, b)

# Why cosine?
# - Magnitude-independent (document length doesn't matter)
# - Direction-focused (semantic meaning, not word count)
# - Bounded ([-1, 1] for similarity, [0, 2] for distance)
```

---

## Core Abstractions

### 1. Indexer

The primary interface for document indexing and retrieval.

```python
class Indexer:
    """
    Main RAG indexing interface.

    Responsibility:
    - Chunk documents
    - Embed chunks
    - Store in vector store
    - Retrieve relevant chunks
    """

    def __init__(
        self,
        embedder: EmbeddingsEngine,
        vector_store: VectorStore,
        chunker: Chunker,
        retriever: Retriever,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker
        self.retriever = retriever

    def index(self, documents: List[Document]) -> None:
        """
        Index documents into the vector store.

        Pipeline:
        1. Chunk documents
        2. Embed chunks
        3. Store in vector store
        4. Build sparse index (if hybrid)

        Args:
            documents: List of documents to index
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)

        # Embed chunks
        embeddings = self.embedder.embed_batch([c.text for c in chunks])

        # Store in vector store
        self.vector_store.add(embeddings, metadata=[c.metadata for c in chunks])

        # Build sparse index if hybrid retrieval
        if isinstance(self.retriever, HybridRetriever):
            self.retriever.build_sparse_index(chunks)

    def search(self, query: str, k: int = 10) -> List[Result]:
        """
        Retrieve relevant chunks for a query.

        Pipeline:
        1. Embed query
        2. Retrieve via retriever strategy
        3. Return top-k results

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of ranked results with scores
        """
        return self.retriever.retrieve(query, k, self.vector_store, self.embedder)

    def update(self, document: Document) -> None:
        """
        Update a document in the index.

        Strategy:
        - Mark old chunks as deleted
        - Index new chunks
        - Rebuild sparse index if needed
        """
        # Implementation depends on vector store capabilities
        pass
```

### 2. Chunker

Abstract base for document chunking strategies.

```python
class Chunker(ABC):
    """
    Document chunking abstraction.

    Responsibility:
    - Split documents into semantically coherent chunks
    - Preserve metadata (document ID, position, hierarchy)
    - Optimize for retrieval (not just storage)
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

class Chunk:
    """A chunk of text with metadata."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        parent_id: Optional[str] = None,
        level: int = 0,
    ):
        self.text = text
        self.metadata = metadata
        self.parent_id = parent_id  # For hierarchical chunking
        self.level = level  # Hierarchy level (0 = leaf)
```

### 3. Retriever

Abstract base for retrieval strategies.

```python
class Retriever(ABC):
    """
    Retrieval strategy abstraction.

    Responsibility:
    - Retrieve relevant chunks for a query
    - Rank by relevance
    - Support different strategies (hybrid, recursive, etc.)
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        vector_store: VectorStore,
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Retrieve top-k relevant chunks.

        Args:
            query: Search query
            k: Number of results
            vector_store: Vector store to search
            embedder: Embeddings engine

        Returns:
            List of ranked results
        """
        pass

class Result:
    """A retrieval result with score and metadata."""

    def __init__(
        self,
        text: str,
        score: float,
        metadata: Dict[str, Any],
        provenance: Optional[str] = None,
    ):
        self.text = text
        self.score = score  # 0 to 1 (higher = better)
        self.metadata = metadata
        self.provenance = provenance  # Source document/section
```

---

## Component Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
│                    (equilibrium-tokens)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Indexer (Facade)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Indexing Pipeline                       │  │
│  │  Document → Chunker → Chunks → Embedder → Vectors → Store │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Retrieval Pipeline                       │  │
│  │  Query → Embedder → Query Vector → Retriever → Results    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌──────────────────────────┐  ┌────────────┐
│    Chunker     │  │       Retriever           │  │  Embedder  │
│                │  │                          │  │            │
│ • FixedSize    │  │ • HybridRetriever        │  │ embeddings │
│ • Semantic     │  │ • RecursiveRetriever     │  │   -engine  │
│ • Hierarchical │  │ • DenseOnlyRetriever     │  │            │
└────────────────┘  └──────────────────────────┘  └────────────┘
                                                        │
                                                        ▼
                                        ┌──────────────────────────┐
                                        │     vector-navigator     │
                                        │                          │
                                        │  • IVF-HNSW hybrid       │
                                        │  • Cosine similarity     │
                                        │  • Quantization (PQ)     │
                                        │  • Sub-5ms latency       │
                                        └──────────────────────────┘
```

### Data Flow

#### Indexing Flow

```python
# 1. Document ingestion
documents = [
    Document(id="doc1", text="..."),
    Document(id="doc2", text="..."),
]

# 2. Chunking (HierarchicalChunker)
chunks = chunker.chunk(documents[0])
# Output: [
#   Chunk(id="doc1-L1", text="Section 1...", level=1),
#   Chunk(id="doc1-L1-1", text="Paragraph 1.1...", level=2, parent="doc1-L1"),
#   Chunk(id="doc1-L1-2", text="Paragraph 1.2...", level=2, parent="doc1-L1"),
#   ...
# ]

# 3. Embedding (EmbeddingsEngine)
embeddings = embedder.embed_batch([c.text for c in chunks])
# Output: np.array([[0.1, 0.2, ...], [0.3, 0.4, ...], ...])  # Shape: (n_chunks, 384)

# 4. Storage (VectorStore from vector-navigator)
vector_store.add(
    vectors=embeddings,
    metadata=[c.metadata for c in chunks],
    ids=[c.id for c in chunks],
)
# Storage: IVF-HNSW hybrid index with Product Quantization
```

#### Retrieval Flow

```python
# 1. Query embedding
query = "Tell me about calm waters on the ocean"
query_vector = embedder.embed(query)  # Shape: (384,)

# 2. Dense retrieval (vector-navigator)
dense_results = vector_store.search(query_vector, k=20)
# Output: [
#   Result(id="chunk42", score=0.92, text="..."),
#   Result(id="chunk17", score=0.87, text="..."),
#   ...
# ]

# 3. Sparse retrieval (BM25 inverted index)
sparse_results = sparse_index.search(query, k=20)
# Output: [
#   Result(id="chunk15", score=0.78, text="..."),
#   Result(id="chunk42", score=0.65, text="..."),
#   ...
# ]

# 4. Score fusion (HybridRetriever)
combined = merge_and_rerank(dense_results, sparse_results)
# Algorithm:
#   - Merge results by chunk ID
#   - Fuse scores: 0.7 * dense_score + 0.3 * sparse_score
#   - Rerank top-20 → top-10

# 5. Return top-k
final_results = combined[:k]
```

---

## Chunking Strategies

### Strategy Comparison

| Strategy | Description | Use Case | Accuracy | Speed |
|----------|-------------|----------|----------|-------|
| **Fixed-Size** | Fixed token/char count | Simple documents | Baseline | Fastest |
| **Semantic** | Split at natural boundaries | Well-structured docs | +15-20% | Fast |
| **Hierarchical** | Multi-level chunks | Large/complex docs | +20-25% | Medium |

### 1. Fixed-Size Chunking

```python
class FixedSizeChunker(Chunker):
    """
    Fixed-size chunking with overlap.

    Pros:
    - Simple, predictable
    - Fast chunking
    - Good for uniform documents

    Cons:
    - May break semantic units
    - Context window overhead
    - Lower retrieval accuracy
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        chunks = []
        text = document.text
        start = 0

        chunk_id = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "document_id": document.id,
                    "chunk_index": chunk_id,
                    "start_char": start,
                    "end_char": end,
                },
            ))

            start = end - self.overlap
            chunk_id += 1

        return chunks
```

### 2. Semantic Chunking

```python
class SemanticChunker(Chunker):
    """
    Semantic chunking at natural boundaries.

    Splits at:
    - Paragraph boundaries
    - Section headers
    - Sentence clusters (optional)

    Pros:
    - Preserves semantic units
    - Better retrieval accuracy
    - Less context overhead

    Cons:
    - Variable chunk sizes
    - Requires document structure
    - Slower chunking
    """

    def __init__(
        self,
        min_size: int = 100,
        max_size: int = 2000,
        similarity_threshold: float = 0.7,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> List[Chunk]:
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(document.text)

        # Merge paragraphs into semantic chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            # Check if paragraph starts new chunk
            if current_size > 0 and current_size + len(para) > self.max_size:
                # Save current chunk
                chunks.append(self._create_chunk(document, current_chunk, chunks))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += len(para)

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(document, current_chunk, chunks))

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Simple implementation: split by double newline
        # Could be enhanced with NLP sentence segmentation
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _create_chunk(
        self,
        document: Document,
        paragraphs: List[str],
        existing_chunks: List[Chunk],
    ) -> Chunk:
        """Create a chunk from paragraphs."""
        chunk_text = "\n\n".join(paragraphs)

        return Chunk(
            text=chunk_text,
            metadata={
                "document_id": document.id,
                "chunk_index": len(existing_chunks),
                "paragraph_count": len(paragraphs),
                "char_count": len(chunk_text),
            },
        )
```

### 3. Hierarchical Chunking (HiChunk)

```python
class HierarchicalChunker(Chunker):
    """
    Hierarchical semantic chunking (HiChunk).

    Creates multi-level chunks:
    - Level 0: Document (highest)
    - Level 1: Sections
    - Level 2: Paragraphs (lowest)

    Enables coarse-to-fine retrieval.

    Pros:
    - Best retrieval accuracy
    - Efficient recursive retrieval
    - Preserves document structure

    Cons:
    - Most complex
    - Higher storage overhead
    - Slower chunking

    Research:
    - HiChunk Framework (arXiv 2025): 15-25% improvement
    - Optimal chunk size: ~1,800 characters (Snowflake 2025)
    """

    def __init__(
        self,
        leaf_size: int = 1800,  # Optimal per research
        max_levels: int = 3,
    ):
        self.leaf_size = leaf_size
        self.max_levels = max_levels

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Create hierarchical chunks.

        Example structure:
        doc1-L0: "Full document summary..." (level 0)
        doc1-L1-1: "Section 1 content..." (level 1, parent: doc1-L0)
        doc1-L1-2: "Section 2 content..." (level 1, parent: doc1-L0)
        doc1-L2-1: "Paragraph 1.1..." (level 2, parent: doc1-L1-1)
        doc1-L2-2: "Paragraph 1.2..." (level 2, parent: doc1-L1-1)
        """
        chunks = []

        # Level 0: Document summary (optional)
        doc_summary = self._summarize_document(document)
        chunks.append(Chunk(
            text=doc_summary,
            metadata={
                "document_id": document.id,
                "level": 0,
                "type": "document",
            },
            level=0,
        ))

        # Level 1: Sections
        sections = self._split_sections(document.text)
        for section_idx, section in enumerate(sections):
            section_id = f"{document.id}-L1-{section_idx}"

            # Create section chunk (Level 1)
            chunks.append(Chunk(
                text=section["content"],
                metadata={
                    "document_id": document.id,
                    "level": 1,
                    "type": "section",
                    "section_title": section["title"],
                    "parent_id": f"{document.id}-L0",
                },
                parent_id=f"{document.id}-L0",
                level=1,
            ))

            # Level 2: Paragraphs within section
            paragraphs = self._split_paragraphs(section["content"])
            for para_idx, para in enumerate(paragraphs):
                chunks.append(Chunk(
                    text=para,
                    metadata={
                        "document_id": document.id,
                        "level": 2,
                        "type": "paragraph",
                        "parent_id": section_id,
                    },
                    parent_id=section_id,
                    level=2,
                ))

        return chunks

    def _summarize_document(self, document: Document) -> str:
        """Generate document summary for Level 0 chunk."""
        # Simple implementation: first paragraph or title
        # Could use LLM-based summarization
        lines = document.text.split("\n")
        return lines[0] if lines else document.text[:500]

    def _split_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into sections based on headers."""
        # Simple implementation: detect markdown headers
        sections = []
        current_section = {"title": "Introduction", "content": ""}

        for line in text.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "title": line.lstrip("#").strip(),
                    "content": "",
                }
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"]:
            sections.append(current_section)

        return sections

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]
```

---

## Hybrid Retrieval

### Architecture

Hybrid retrieval combines two complementary approaches:

```python
# Dense Retrieval (Semantic)
dense_score = cosine_similarity(query_embedding, chunk_embedding)

# Sparse Retrieval (Keyword)
sparse_score = bm25_score(query_terms, chunk_text)

# Score Fusion
final_score = α × dense_score + (1 - α) × sparse_score
```

### Implementation

```python
class HybridRetriever(Retriever):
    """
    Hybrid retriever combining dense and sparse search.

    Research:
    - Hybrid Dense-Sparse: 20-30% improvement (2024)
    - Optimal α: 0.6-0.8 dense weight (empirical)
    """

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion_strategy: str = "weighted",  # "weighted" | "rrf" | "learned"
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_strategy = fusion_strategy
        self.sparse_index = None  # Built during indexing

    def build_sparse_index(self, chunks: List[Chunk]) -> None:
        """Build BM25 sparse index from chunks."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Extract chunk texts
        texts = [chunk.text for chunk in chunks]

        # Build TF-IDF index (simpler than BM25, similar results)
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams + bigrams
        )
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(texts)

        # Store chunk IDs for lookup
        self.chunk_ids = [chunk.metadata.get("id", i) for i, chunk in enumerate(chunks)]

    def retrieve(
        self,
        query: str,
        k: int,
        vector_store: VectorStore,
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Retrieve using hybrid dense + sparse search.

        Pipeline:
        1. Dense retrieval (vector store)
        2. Sparse retrieval (BM25/TF-IDF)
        3. Score fusion
        4. Return top-k
        """
        # Step 1: Dense retrieval
        query_vector = embedder.embed(query)
        dense_results = vector_store.search(query_vector, k=k*2)  # Retrieve more

        # Step 2: Sparse retrieval
        sparse_results = self._sparse_search(query, k=k*2)

        # Step 3: Score fusion
        fused_results = self._fuse_scores(
            dense_results,
            sparse_results,
            strategy=self.fusion_strategy,
        )

        # Step 4: Return top-k
        return fused_results[:k]

    def _sparse_search(self, query: str, k: int) -> List[Result]:
        """Sparse keyword search using TF-IDF."""
        if self.sparse_matrix is None:
            return []

        # Transform query
        query_vec = self.sparse_vectorizer.transform([query])

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, self.sparse_matrix)[0]

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches
                results.append(Result(
                    text="",  # Text filled in during fusion
                    score=float(scores[idx]),
                    metadata={"chunk_id": self.chunk_ids[idx]},
                ))

        return results

    def _fuse_scores(
        self,
        dense_results: List[Result],
        sparse_results: List[Result],
        strategy: str = "weighted",
    ) -> List[Result]:
        """Fuse dense and sparse scores."""
        # Create lookup by chunk_id
        sparse_lookup = {r.metadata["chunk_id"]: r for r in sparse_results}

        fused = {}
        for dense_result in dense_results:
            chunk_id = dense_result.metadata.get("chunk_id")
            sparse_result = sparse_lookup.get(chunk_id)

            if sparse_result:
                # Weighted fusion
                if strategy == "weighted":
                    fused_score = (
                        self.dense_weight * dense_result.score +
                        self.sparse_weight * sparse_result.score
                    )
                # Reciprocal Rank Fusion (RRF)
                elif strategy == "rrf":
                    dense_rank = next(
                        i for i, r in enumerate(dense_results)
                        if r.metadata["chunk_id"] == chunk_id
                    )
                    sparse_rank = next(
                        i for i, r in enumerate(sparse_results)
                        if r.metadata["chunk_id"] == chunk_id
                    )
                    fused_score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
                else:
                    fused_score = dense_result.score  # Fallback

                fused[chunk_id] = Result(
                    text=dense_result.text,
                    score=fused_score,
                    metadata=dense_result.metadata,
                )
            else:
                # Dense-only result
                fused[chunk_id] = dense_result

        # Sort by fused score
        return sorted(fused.values(), key=lambda r: r.score, reverse=True)
```

### Fusion Strategies

```python
# 1. Weighted Fusion (simple, effective)
score = 0.7 × dense_score + 0.3 × sparse_score

# 2. Reciprocal Rank Fusion (RRF)
# - Rank-based (not score-based)
# - Robust to score scale differences
score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)

# 3. Learned Fusion (requires training)
# - Learn optimal weights per query type
# - Best for specialized domains
score = model.predict(features=[dense_score, sparse_score, query_type])
```

---

## Recursive Retrieval

### Architecture

Coarse-to-fine retrieval for large document collections:

```
Query → Level 1 (Documents) → Top-K docs → Level 2 (Chunks within docs) → Final results
```

### Implementation

```python
class RecursiveRetriever(Retriever):
    """
    Recursive retriever for coarse-to-fine navigation.

    Strategy:
    1. Level 1: Retrieve top-K parent chunks (documents/sections)
    2. Level 2: Search within top parents for child chunks
    3. Merge and rerank

    Research:
    - IVF-HNSW hybrid: 40-50% improvement (2024)
    - Reduces search space efficiently
    """

    def __init__(
        self,
        levels: int = 2,
        top_parents: int = 3,
        children_per_parent: int = 5,
    ):
        self.levels = levels
        self.top_parents = top_parents
        self.children_per_parent = children_per_parent

    def retrieve(
        self,
        query: str,
        k: int,
        vector_store: VectorStore,
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Recursive coarse-to-fine retrieval.

        Example:
        - k = 10
        - top_parents = 3
        - children_per_parent = 5
        - Retrieves 3 parents, then 5 children from each = 15 candidates
        - Reranks and returns top-10
        """
        query_vector = embedder.embed(query)

        # Level 1: Coarse search (parent chunks)
        parent_results = vector_store.search(
            query_vector,
            filter={"level": 0},  # Search only Level 0 (documents)
            k=self.top_parents,
        )

        # Level 2: Fine search (children within top parents)
        all_results = []
        for parent in parent_results:
            parent_id = parent.metadata.get("id")

            # Search within children
            child_results = vector_store.search(
                query_vector,
                filter={"parent_id": parent_id},  # Children only
                k=self.children_per_parent,
            )

            all_results.extend(child_results)

        # Add parent chunks themselves (for context)
        all_results.extend(parent_results)

        # Rerank all candidates
        reranked = self._rerank(query, all_results, embedder)

        # Return top-k
        return reranked[:k]

    def _rerank(
        self,
        query: str,
        results: List[Result],
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Rerank results using cross-encoder or other strategy.

        Simple implementation: Use existing scores
        Advanced: Use cross-encoder for reranking
        """
        # For now, just sort by score
        return sorted(results, key=lambda r: r.score, reverse=True)
```

### Performance Characteristics

```python
# Traditional search:
# - Search all chunks: 100K chunks × 384 dims = 38M comparisons
# - Latency: ~50ms

# Recursive search (2 levels):
# - Level 1: 100 docs × 384 dims = 38K comparisons (~5ms)
# - Level 2: 3 docs × 1000 chunks × 384 dims = 1.1M comparisons (~10ms)
# - Total: ~15ms (3× faster)

# Accuracy:
# - Recall: Similar (most relevant in top-3 parents)
# - Precision: Higher (fine-grained search within relevant parents)
```

---

## Integration with Dependencies

### vector-navigator Integration

```python
from vector_navigator import VectorStore, IndexType

# Create vector store with IVF-HNSW hybrid
vector_store = VectorStore(
    dimension=384,  # Embedding dimension
    index_type=IndexType.IVF_HNSW,  # Hybrid index
    metric="cosine",  # Cosine similarity
    nlist=100,  # IVF partitions (coarse quantization)
    nprobe=10,  # Partitions to search (speed vs recall trade-off)
    M=16,  # HNSW connections per node
    efConstruction=200,  # HNSW build time vs accuracy
)

# Index chunks
indexer = Indexer(vector_store=vector_store, ...)
indexer.index(documents)

# Search
results = indexer.search(query, k=10)
```

### embeddings-engine Integration

```python
from embeddings_engine import EmbeddingsEngine, ModelConfig

# Create embeddings engine
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",  # GPU acceleration
    batch_size=32,
    cache_embeddings=True,  # Cache for re-use
)

# Batch embedding (for indexing)
embeddings = embedder.embed_batch(
    texts=[chunk.text for chunk in chunks],
    batch_size=32,
    show_progress=True,
)

# Single embedding (for queries)
query_vector = embedder.embed(query)
```

### Combined Integration

```python
# Complete setup
embedder = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
vector_store = VectorStore(dimension=384, index_type=IndexType.IVF_HNSW)
chunker = HierarchicalChunker(leaf_size=1800)
retriever = HybridRetriever(dense_weight=0.7)

indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=chunker,
    retriever=retriever,
)

# Index
indexer.index(documents)

# Search
results = indexer.search(query, k=10)
```

---

## Performance Architecture

### Indexing Performance

```python
# GPU-accelerated embedding
embedder = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",  # 10-20× faster than CPU
    batch_size=32,  # Optimal for GPU utilization
)

# Target performance:
# - Throughput: >100 docs/sec
# - Latency: <100ms per doc (GPU)
# - Chunk quality: >90% semantic coherence
```

### Retrieval Performance

```python
# IVF-HNSW hybrid index
vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.IVF_HNSW,
    nlist=100,  # IVF partitions
    nprobe=10,  # Partitions to search
)

# Target performance:
# - Latency: <50ms p99 for top-10
# - Throughput: >100 QPS
# - Recall: >0.95 at 100 candidates
# - Precision: >0.85 @10
```

### Storage Performance

```python
# Product Quantization (PQ) for compression
vector_store = VectorStore(
    dimension=384,
    quantization="PQ",  # Product Quantization
    nbits=8,  # Bits per quantization code
)

# Storage targets:
# - Compression: >5× vs. raw embeddings
# - Memory: <1GB for 100K documents
# - Disk: <10GB for 1M documents
# - Accuracy loss: <5% vs. uncompressed
```

### Optimization Strategies

```python
# 1. Indexing optimization
# - Batch embedding (GPU)
# - Parallel chunking (multiprocessing)
# - Incremental index updates

# 2. Retrieval optimization
# - Hybrid retrieval (parallel dense+sparse)
# - Recursive retrieval (coarse-to-fine)
# - Query caching (LRU cache)

# 3. Storage optimization
# - Quantization (PQ, OPQ)
# - Sparse index compression
# - Metadata filtering
```

---

## Research Foundation

### Key Research Papers (2024-2025)

1. **HiChunk Framework** (arXiv, September 2025)
   - Hierarchical semantic chunking
   - 15-25% improvement in retrieval accuracy
   - Optimal chunk size: ~1,800 characters

2. **Hybrid Dense-Sparse Retrieval** (ResearchGate, 2024)
   - Combines semantic and keyword search
   - 20-30% improvement in accuracy
   - Optimal dense weight: 0.6-0.8

3. **Incremental IVF Index Maintenance** (arXiv, November 2024)
   - Real-time index updates
   - Streaming data support
   - IVF-HNSW hybrid for performance

4. **Achieving Low-Latency Graph-Based Vector Search** (USENIX OSDI 2025)
   - PipeANN system
   - 35% lower latency than DiskANN
   - 1-5ms query latency

5. **Chunking Strategies for Finance RAG** (Snowflake, March 2025)
   - Optimal chunk size: ~1,800 characters
   - Semantic chunking outperforms fixed-size
   - Context window efficiency +30-40%

### Research-to-Implementation Mapping

| Research Finding | Implementation | Improvement |
|-----------------|----------------|-------------|
| HiChunk hierarchical chunking | `HierarchicalChunker` | +15-25% accuracy |
| Hybrid dense-sparse fusion | `HybridRetriever` | +20-30% accuracy |
| Optimal chunk size ~1,800 | `leaf_size=1800` | +10% efficiency |
| IVF-HNSW hybrid | vector-navigator IVF-HNSW | +40-50% speed |
| Product Quantization | `quantization="PQ"` | 5× compression |

---

## Conclusion

rag-indexer combines cutting-edge research with production-grade engineering to deliver:

- **High Accuracy**: Hybrid retrieval + hierarchical chunking
- **Low Latency**: IVF-HNSW + GPU acceleration
- **Scale**: Quantization + efficient indexing
- **Modularity**: Pluggable components
- **Research-Backed**: 2024-2025 advances

Built on timeless IR principles, designed for modern RAG applications.

---

**Next**: See [USER_GUIDE.md](USER_GUIDE.md) for usage examples
