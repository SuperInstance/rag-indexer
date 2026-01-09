# rag-indexer User Guide

**Version:** 1.0
**Last Updated:** January 8, 2026

---

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Configuration](#configuration)
5. [Performance Tuning](#performance-tuning)
6. [Integration with equilibrium-tokens](#integration-with-equilibrium-tokens)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Python Installation

```bash
# Install from PyPI
pip install rag-indexer

# Or install with specific dependencies
pip install rag-indexer[gpu]  # GPU support
pip install rag-indexer[dev]  # Development dependencies
```

### Go Installation

```bash
# Install Go bindings
go get github.com/equilibrium-tokens/rag-indexer-go
```

### Dependencies

rag-indexer requires:
- Python 3.8+
- vector-navigator (installed automatically)
- embeddings-engine (installed automatically)
- Optional: CUDA for GPU acceleration

### Verifying Installation

```python
import rag_indexer

print(rag_indexer.__version__)  # Should print version
```

---

## Basic Usage

### Quick Start

```python
from rag_indexer import Indexer, SemanticChunker, HybridRetriever
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

# Step 1: Setup components
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",  # Use "cpu" if no GPU
)
vector_store = VectorStore(dimension=384)
chunker = SemanticChunker()
retriever = HybridRetriever(dense_weight=0.7)

# Step 2: Create indexer
indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=chunker,
    retriever=retriever,
)

# Step 3: Index documents
documents = [
    "Document 1: Fishing boats navigate calm waters at sunrise.",
    "Document 2: Jazz music originated in New Orleans in the late 19th century.",
    "Document 3: The symphony pops concert features popular classical music.",
]
indexer.index(documents)

# Step 4: Search
query = "Tell me about calm waters"
results = indexer.search(query, k=3)

# Step 5: Use results
for result in results:
    print(f"[{result.score:.3f}] {result.text}")
```

### Working with Document Objects

```python
from rag_indexer import Document

# Create structured documents
documents = [
    Document(
        id="doc1",
        text="Fishing boats navigate calm waters...",
        metadata={
            "title": "Fishing Boats",
            "author": "John Doe",
            "date": "2025-01-01",
            "category": "marine",
        },
    ),
    Document(
        id="doc2",
        text="Jazz music originated in New Orleans...",
        metadata={
            "title": "Jazz History",
            "author": "Jane Smith",
            "date": "2025-01-02",
            "category": "music",
        },
    ),
]

indexer.index(documents)

# Search with metadata filter
query = "music in New Orleans"
results = indexer.search(query, k=5, filter={"category": "music"})
```

### Saving and Loading Indexes

```python
# Save index to disk
indexer.save("/path/to/index")

# Load index from disk
indexer = Indexer.load("/path/to/index")
```

---

## Advanced Usage

### Chunking Strategies

#### 1. Fixed-Size Chunking

```python
from rag_indexer import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,  # Tokens per chunk
    overlap=50,      # Overlap between chunks
)

indexer = Indexer(chunker=chunker, ...)
```

**Use cases:**
- Simple documents without structure
- Predictable memory usage
- Fast chunking required

#### 2. Semantic Chunking

```python
from rag_indexer import SemanticChunker

chunker = SemanticChunker(
    min_size=100,                  # Minimum chunk size
    max_size=2000,                 # Maximum chunk size
    similarity_threshold=0.7,      # Merge similar paragraphs
)

indexer = Indexer(chunker=chunker, ...)
```

**Use cases:**
- Well-structured documents (articles, reports)
- Better retrieval accuracy required
- Preserving semantic boundaries important

#### 3. Hierarchical Chunking (HiChunk)

```python
from rag_indexer import HierarchicalChunker

chunker = HierarchicalChunker(
    leaf_size=1800,      # Optimal per research
    max_levels=3,        # Document → Section → Paragraph
)

indexer = Indexer(chunker=chunker, ...)
```

**Use cases:**
- Large documents (>10 pages)
- Complex document structure
- Coarse-to-fine retrieval needed
- Best retrieval accuracy required

**Example: Recursive Retrieval with Hierarchical Chunking**

```python
from rag_indexer import HierarchicalChunker, RecursiveRetriever

chunker = HierarchicalChunker(leaf_size=1800)
retriever = RecursiveRetriever(
    levels=2,
    top_parents=3,
    children_per_parent=5,
)

indexer = Indexer(chunker=chunker, retriever=retriever, ...)

# Search uses coarse-to-fine strategy
results = indexer.search("query", k=10)
```

### Hybrid Search Configuration

```python
from rag_indexer import HybridRetriever

# Basic hybrid retrieval
retriever = HybridRetriever(
    dense_weight=0.7,    # Semantic search weight
    sparse_weight=0.3,   # Keyword search weight
)

# Advanced: Reciprocal Rank Fusion (RRF)
retriever = HybridRetriever(
    dense_weight=0.7,
    sparse_weight=0.3,
    fusion_strategy="rrf",  # "weighted" | "rrf" | "learned"
)

# Create indexer
indexer = Indexer(retriever=retriever, ...)
```

**Tuning hybrid weights:**

```python
# Semantic-heavy queries (abstract concepts)
retriever = HybridRetriever(dense_weight=0.8, sparse_weight=0.2)

# Keyword-heavy queries (specific terms)
retriever = HybridRetriever(dense_weight=0.5, sparse_weight=0.5)

# Balanced approach (default)
retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
```

### Retrieval Strategies

#### Dense-Only Retrieval (Fastest)

```python
from rag_indexer import DenseRetriever

retriever = DenseRetriever()
indexer = Indexer(retriever=retriever, ...)

# Latency: ~10-20ms
# Accuracy: Baseline
```

#### Hybrid Retrieval (Balanced)

```python
from rag_indexer import HybridRetriever

retriever = HybridRetriever(dense_weight=0.7)
indexer = Indexer(retriever=retriever, ...)

# Latency: ~20-30ms
# Accuracy: +20-30% vs dense-only
```

#### Recursive Retrieval (Most Accurate)

```python
from rag_indexer import RecursiveRetriever

retriever = RecursiveRetriever(
    levels=2,
    top_parents=3,
    children_per_parent=5,
)
indexer = Indexer(retriever=retriever, ...)

# Latency: ~30-50ms
# Accuracy: +40-50% vs dense-only (for large docs)
```

### Multi-Document Indexing

```python
# Index from directory
import os

documents = []
for filename in os.listdir("/path/to/docs"):
    with open(os.path.join("/path/to/docs", filename)) as f:
        text = f.read()

        documents.append(Document(
            id=filename,
            text=text,
            metadata={"source": filename},
        ))

indexer.index(documents)
```

### Batch Indexing

```python
# For large collections, batch indexing is more efficient

documents = load_large_corpus()  # 100K+ documents

# Index in batches
indexer.index_batch(
    documents,
    batch_size=1000,
    show_progress=True,
    save_interval=10000,  # Save every 10K docs
)
```

### Updating Indexes

```python
# Add new documents
new_documents = [
    Document(id="doc_new", text="New content..."),
]
indexer.index(new_documents)

# Update existing document
updated_doc = Document(id="doc1", text="Updated content...")
indexer.update(updated_doc)

# Delete document
indexer.delete("doc1")
```

---

## Configuration

### Embedder Configuration

```python
from embeddings_engine import EmbeddingsEngine, ModelConfig

# Fast model (English only)
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
    batch_size=32,
)

# Multilingual model
embedder = EmbeddingsEngine(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda",
    batch_size=32,
)

# Large model (better accuracy, slower)
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-mpnet-base-v2",
    device="cuda",
    batch_size=16,  # Smaller batch for larger model
)
```

### Vector Store Configuration

```python
from vector_navigator import VectorStore, IndexType

# For small collections (<10K docs)
vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.HNSW,  # Pure HNSW (simpler)
    M=16,
    efConstruction=200,
)

# For large collections (>10K docs)
vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.IVF_HNSW,  # Hybrid (faster at scale)
    nlist=100,   # IVF partitions
    nprobe=10,   # Partitions to search (tune for speed vs recall)
)

# For memory-constrained environments
vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.IVF_HNSW,
    quantization="PQ",  # Product Quantization (5× compression)
    nbits=8,
)
```

### Complete Configuration Example

```python
from rag_indexer import Indexer, HierarchicalChunker, HybridRetriever
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore, IndexType

# Production configuration
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
    batch_size=32,
    cache_embeddings=True,
)

vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.IVF_HNSW,
    nlist=100,
    nprobe=10,
    quantization="PQ",
    nbits=8,
)

chunker = HierarchicalChunker(
    leaf_size=1800,
    max_levels=3,
)

retriever = HybridRetriever(
    dense_weight=0.7,
    sparse_weight=0.3,
    fusion_strategy="weighted",
)

indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=chunker,
    retriever=retriever,
)
```

---

## Performance Tuning

### Indexing Performance

```python
# 1. Use GPU for embedding
embedder = EmbeddingsEngine(device="cuda")  # 10-20× faster

# 2. Increase batch size (if memory allows)
embedder = EmbeddingsEngine(
    device="cuda",
    batch_size=64,  # Default: 32
)

# 3. Parallel chunking (for CPU-bound chunking)
chunker = SemanticChunker(
    num_workers=4,  # Use 4 CPU cores
)

# 4. Batch indexing
indexer.index_batch(documents, batch_size=1000)
```

### Retrieval Performance

```python
# 1. Tune IVF nprobe (speed vs recall trade-off)
vector_store = VectorStore(
    index_type=IndexType.IVF_HNSW,
    nlist=100,
    nprobe=5,   # Lower = faster (but lower recall)
    nprobe=20,  # Higher = slower (but higher recall)
)

# 2. Use dense-only retrieval (fastest)
retriever = DenseRetriever()

# 3. Reduce retrieval candidates
retriever = HybridRetriever(
    dense_weight=0.7,
    retrieval_k=10,  # Retrieve fewer candidates
)
```

### Memory Optimization

```python
# 1. Use Product Quantization
vector_store = VectorStore(
    quantization="PQ",
    nbits=8,  # 8 bits = 5× compression
)

# 2. Stream from disk (for very large collections)
indexer = Indexer(
    vector_store=vector_store,
    storage_backend="disk",  # Instead of "memory"
)

# 3. Clear embedding cache
embedder = EmbeddingsEngine(cache_embeddings=False)
```

### Benchmarking

```python
from rag_indexer import benchmark

# Benchmark indexing
indexing_stats = indexer.benchmark_indexing(
    documents=test_docs,
    num_runs=10,
)
print(f"Indexing throughput: {indexing_stats['docs_per_sec']:.1f} docs/sec")

# Benchmark retrieval
retrieval_stats = indexer.benchmark_retrieval(
    queries=test_queries,
    k=10,
    num_runs=100,
)
print(f"Retrieval latency p50: {retrieval_stats['latency_p50']:.1f}ms")
print(f"Retrieval latency p99: {retrieval_stats['latency_p99']:.1f}ms")
```

---

## Integration with equilibrium-tokens

### Basic Integration

```python
from equilibrium_tokens import Orchestrator
from rag_indexer import Indexer, SemanticChunker, HybridRetriever

# Setup RAG indexer
indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=SemanticChunker(),
    retriever=HybridRetriever(dense_weight=0.7),
)

# Index conversation history
conversations = [
    Document(id="conv1", text="User: Tell me about jazz...\nAI: Jazz originated in..."),
    Document(id="conv2", text="User: What about fishing boats?\nAI: Fishing boats are..."),
]
indexer.index(conversations)

# Setup orchestrator
orchestrator = Orchestrator()

# Query with context
query = "Tell me more about jazz music"
results = indexer.search(query, k=5)
context = [r.text for r in results]

# Set context and generate
orchestrator.set_context(context)
response = orchestrator.generate(query)
print(response)
```

### Context Window Management

```python
# Estimate token usage
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4

# Dynamic context selection
max_context_tokens = 2000  # Reserve space for response

results = indexer.search(query, k=20)  # Retrieve more

selected_context = []
total_tokens = 0

for result in results:
    tokens = estimate_tokens(result.text)
    if total_tokens + tokens > max_context_tokens:
        break
    selected_context.append(result.text)
    total_tokens += tokens

print(f"Selected {len(selected_context)} chunks ({total_tokens} tokens)")
```

### Sentiment-Aware Retrieval

```python
# Index with sentiment metadata
from textblob import TextBlob

documents = []
for doc in raw_documents:
    sentiment = TextBlob(doc.text).sentiment.polarity  # -1 to 1

    documents.append(Document(
        id=doc.id,
        text=doc.text,
        metadata={
            "sentiment": sentiment,
            "sentiment_label": "positive" if sentiment > 0 else "negative",
        },
    ))

indexer.index(documents)

# Search with sentiment filter
query = "calm peaceful waters"
results = indexer.search(
    query,
    k=10,
    filter={"sentiment_label": "positive"},  # Only positive sentiment
)
```

### Conversation Memory Management

```python
class ConversationMemory:
    """Manage conversation context with RAG."""

    def __init__(self, indexer: Indexer, max_turns: int = 10):
        self.indexer = indexer
        self.max_turns = max_turns
        self.conversation_id = f"conv_{uuid.uuid4()}"

    def add_turn(self, user_message: str, ai_response: str):
        """Add conversation turn to index."""
        turn = Document(
            id=f"{self.conversation_id}_{len(self.indexer)}",
            text=f"User: {user_message}\nAI: {ai_response}",
            metadata={
                "conversation_id": self.conversation_id,
                "turn_number": len(self.indexer),
            },
        )
        self.indexer.index([turn])

    def get_context(self, query: str, k: int = 5):
        """Retrieve relevant conversation context."""
        results = self.indexer.search(
            query,
            k=k,
            filter={"conversation_id": self.conversation_id},
        )
        return [r.text for r in results]

# Usage
memory = ConversationMemory(indexer)

memory.add_turn("Tell me about jazz", "Jazz originated in...")
memory.add_turn("What about fishing boats?", "Fishing boats are...")

# Get context for next response
query = "Tell me more about jazz"
context = memory.get_context(query, k=3)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory During Indexing

```python
# Solution: Reduce batch size
embedder = EmbeddingsEngine(
    device="cuda",
    batch_size=16,  # Reduce from 32
)

# Solution: Use CPU instead
embedder = EmbeddingsEngine(device="cpu")

# Solution: Batch indexing
indexer.index_batch(documents, batch_size=100)
```

#### 2. Slow Retrieval

```python
# Solution: Reduce nprobe
vector_store = VectorStore(
    index_type=IndexType.IVF_HNSW,
    nprobe=5,  # Reduce from 10
)

# Solution: Use dense-only retrieval
retriever = DenseRetriever()

# Solution: Disable quantization decoding
vector_store = VectorStore(
    quantization=None,  # Disable PQ
)
```

#### 3. Poor Retrieval Accuracy

```python
# Solution: Use hierarchical chunking
chunker = HierarchicalChunker(leaf_size=1800)

# Solution: Use hybrid retrieval
retriever = HybridRetriever(dense_weight=0.7)

# Solution: Increase nprobe
vector_store = VectorStore(
    index_type=IndexType.IVF_HNSW,
    nprobe=20,  # Increase from 10
)

# Solution: Use larger embedding model
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-mpnet-base-v2",  # Larger model
)
```

#### 4. CUDA Out of Memory

```python
# Solution: Clear GPU cache
import torch
torch.cuda.empty_cache()

# Solution: Use CPU
embedder = EmbeddingsEngine(device="cpu")

# Solution: Reduce batch size
embedder = EmbeddingsEngine(device="cuda", batch_size=8)
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check index stats
print(f"Documents indexed: {indexer.stats()['num_documents']}")
print(f"Chunks indexed: {indexer.stats()['num_chunks']}")
print(f"Index size: {indexer.stats()['index_size_mb']} MB")

# Analyze retrieval results
results = indexer.search(query, k=10)
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Score: {result.score:.3f}")
    print(f"  Text: {result.text[:100]}...")
    print(f"  Metadata: {result.metadata}")
```

### Performance Profiling

```python
import time

# Profile indexing
start = time.time()
indexer.index(documents)
print(f"Indexing took: {time.time() - start:.2f}s")

# Profile retrieval
start = time.time()
results = indexer.search(query, k=10)
print(f"Retrieval took: {time.time() - start*1000:.1f}ms")

# Profile with multiple runs
import numpy as np

times = []
for _ in range(100):
    start = time.time()
    indexer.search(query, k=10)
    times.append((time.time() - start) * 1000)

print(f"Mean latency: {np.mean(times):.1f}ms")
print(f"P50 latency: {np.percentile(times, 50):.1f}ms")
print(f"P99 latency: {np.percentile(times, 99):.1f}ms")
```

---

## Best Practices

### 1. Chunking Strategy Selection

- **Fixed-Size**: Simple docs, predictable memory
- **Semantic**: Most use cases (default)
- **Hierarchical**: Large/complex docs, best accuracy

### 2. Retrieval Strategy Selection

- **Dense-Only**: Speed critical, small collections
- **Hybrid**: Balanced performance (default)
- **Recursive**: Large collections, accuracy critical

### 3. Performance Optimization

- Use GPU for embedding (10-20× faster)
- Use IVF-HNSW for large collections (>10K docs)
- Use PQ quantization for memory-constrained environments
- Tune nprobe for speed vs recall trade-off

### 4. Accuracy Optimization

- Use hierarchical chunking (+15-25% accuracy)
- Use hybrid retrieval (+20-30% accuracy)
- Use recursive retrieval for large docs (+40-50% accuracy)
- Increase retrieval candidates and rerank

### 5. Integration with LLMs

- Estimate token usage to avoid overflow
- Use dynamic context selection
- Track provenance for citations
- Manage conversation memory with RAG

---

**Next**: See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for development setup
