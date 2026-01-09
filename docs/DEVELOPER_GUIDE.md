# rag-indexer Developer Guide

**Version:** 1.0
**Last Updated:** January 8, 2026

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding New Chunkers](#adding-new-chunkers)
4. [Adding New Retrievers](#adding-new-retrievers)
5. [Testing Strategy](#testing-strategy)
6. [Benchmarking Methodology](#benchmarking-methodology)
7. [Release Process](#release-process)

---

## Development Setup

### Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU)
- Go 1.18+ (for Go bindings)

# Python dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### Local Development

```bash
# Clone repository
git clone https://github.com/equilibrium-tokens/rag-indexer.git
cd rag-indexer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check rag_indexer/
black rag_indexer/
mypy rag_indexer/
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Benchmark tests
pytest tests/benchmarks/ -v

# With coverage
pytest --cov=rag_indexer --cov-report=html
```

### Code Style

```bash
# Format code
black rag_indexer/
ruff check --fix rag_indexer/

# Type checking
mypy rag_indexer/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

## Project Structure

```
rag-indexer/
├── rag_indexer/                 # Main Python package
│   ├── __init__.py
│   ├── indexer.py               # Indexer interface
│   ├── chunker.py               # Chunker implementations
│   ├── retriever.py             # Retriever implementations
│   ├── config.py                # Configuration classes
│   └── utils.py                 # Utility functions
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_indexer.py
│   ├── integration/             # Integration tests
│   │   ├── test_full_pipeline.py
│   │   └── test_vector_navigator_integration.py
│   └── benchmarks/              # Benchmark tests
│       ├── bench_indexing.py
│       └── bench_retrieval.py
│
├── examples/                    # Usage examples
│   ├── basic_usage.py
│   ├── advanced_chunking.py
│   └── equilibrium_integration.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   └── INTEGRATION.md
│
├── go/                          # Go bindings
│   ├── rag_indexer.go
│   ├── rag_indexer_test.go
│   └── cgo/                     # C bridge code
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Adding New Chunkers

### Step 1: Create Chunker Class

```python
# rag_indexer/chunker.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .config import Chunk, Document

class Chunker(ABC):
    """Base class for chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        pass

class YourCustomChunker(Chunker):
    """
    Your custom chunking strategy.

    Description:
    - What makes this chunker unique
    - When to use it
    - Performance characteristics
    """

    def __init__(
        self,
        param1: type,
        param2: type = default_value,
    ):
        self.param1 = param1
        self.param2 = param2

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Implement your chunking logic here.

        Example:
        1. Parse document structure
        2. Identify chunk boundaries
        3. Create Chunk objects with metadata
        4. Return list of chunks
        """
        chunks = []

        # Your chunking logic
        # ...

        return chunks
```

### Step 2: Add Tests

```python
# tests/unit/test_chunker.py

import pytest
from rag_indexer import YourCustomChunker, Document

def test_custom_chunker_basic():
    """Test basic chunking functionality."""
    chunker = YourCustomChunker(param1="value")

    document = Document(id="test", text="Test document text")
    chunks = chunker.chunk(document)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

def test_custom_chunker_edge_cases():
    """Test edge cases."""
    chunker = YourCustomChunker(param1="value")

    # Empty document
    doc = Document(id="empty", text="")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 0

    # Very short document
    doc = Document(id="short", text="Short")
    chunks = chunker.chunk(doc)
    # Assert expected behavior

def test_custom_chunker_metadata():
    """Test that metadata is preserved."""
    chunker = YourCustomChunker(param1="value")

    doc = Document(
        id="test",
        text="Test text",
        metadata={"key": "value"},
    )
    chunks = chunker.chunk(doc)

    # Assert metadata is in chunks
    assert all("key" in c.metadata for c in chunks)
```

### Step 3: Update Exports

```python
# rag_indexer/__init__.py

from .chunker import (
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
    YourCustomChunker,  # Add here
)

__all__ = [
    "FixedSizeChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "YourCustomChunker",  # Add here
    # ... other exports
]
```

### Step 4: Add Documentation

```python
# Add docstring to class with:
# - Description
# - Usage example
# - Parameters
# - Performance characteristics
# - When to use this chunker

# Update docs/ARCHITECTURE.md with new chunker section
# Update docs/USER_GUIDE.md with usage example
```

---

## Adding New Retrievers

### Step 1: Create Retriever Class

```python
# rag_indexer/retriever.py

from abc import ABC, abstractmethod
from typing import List
from .config import Result
from vector_navigator import VectorStore
from embeddings_engine import EmbeddingsEngine

class Retriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        vector_store: VectorStore,
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            k: Number of results to return
            vector_store: Vector store to search
            embedder: Embeddings engine

        Returns:
            List of ranked results
        """
        pass

class YourCustomRetriever(Retriever):
    """
    Your custom retrieval strategy.

    Description:
    - What makes this retriever unique
    - When to use it
    - Performance characteristics
    """

    def __init__(
        self,
        param1: type,
        param2: type = default_value,
    ):
        self.param1 = param1
        self.param2 = param2

    def retrieve(
        self,
        query: str,
        k: int,
        vector_store: VectorStore,
        embedder: EmbeddingsEngine,
    ) -> List[Result]:
        """
        Implement your retrieval logic here.

        Example:
        1. Embed query
        2. Search vector store
        3. Apply custom scoring/ranking
        4. Return top-k results
        """
        # Your retrieval logic
        # ...

        return results
```

### Step 2: Add Tests

```python
# tests/unit/test_retriever.py

import pytest
from rag_indexer import YourCustomRetriever, Indexer
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

@pytest.fixture
def setup_indexer():
    """Setup indexer with test data."""
    embedder = EmbeddingsEngine(device="cpu")
    vector_store = VectorStore(dimension=384)

    # Add test data
    # ...

    return Indexer(
        embedder=embedder,
        vector_store=vector_store,
        retriever=YourCustomRetriever(param1="value"),
    )

def test_custom_retriever_basic(setup_indexer):
    """Test basic retrieval."""
    indexer = setup_indexer

    results = indexer.search("test query", k=5)

    assert len(results) <= 5
    assert all(isinstance(r, Result) for r in results)
    assert all(0 <= r.score <= 1 for r in results)

def test_custom_retriever_ranking(setup_indexer):
    """Test that results are properly ranked."""
    indexer = setup_indexer

    results = indexer.search("test query", k=10)

    # Assert results are sorted by score (descending)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

def test_custom_retriever_empty_query(setup_indexer):
    """Test edge case: empty query."""
    indexer = setup_indexer

    results = indexer.search("", k=5)

    # Assert expected behavior
    # (empty results, error, etc.)
```

### Step 3: Update Exports

```python
# rag_indexer/__init__.py

from .retriever import (
    DenseRetriever,
    HybridRetriever,
    RecursiveRetriever,
    YourCustomRetriever,  # Add here
)

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "RecursiveRetriever",
    "YourCustomRetriever",  # Add here
    # ... other exports
]
```

### Step 4: Add Benchmarks

```python
# tests/benchmarks/bench_custom_retriever.py

import pytest
from rag_indexer import YourCustomRetriever
from benchmarks.utils import setup_test_indexer

def test_custom_retriever_performance(benchmark):
    """Benchmark custom retriever performance."""
    indexer = setup_test_indexer(
        retriever=YourCustomRetriever(param1="value"),
        num_docs=1000,
    )

    # Benchmark retrieval
    results = benchmark(indexer.search, "test query", k=10)

    # Assert performance targets
    assert len(results) == 10
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_indexer.py

import pytest
from rag_indexer import Indexer, Document
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

@pytest.fixture
def mock_embedder(mocker):
    """Mock embedder for fast tests."""
    mock = mocker.Mock(spec=EmbeddingsEngine)
    mock.embed.return_value = [0.1] * 384
    mock.embed_batch.return_value = [[0.1] * 384] * 10
    return mock

@pytest.fixture
def mock_vector_store(mocker):
    """Mock vector store for fast tests."""
    mock = mocker.Mock(spec=VectorStore)
    mock.search.return_value = []
    return mock

def test_indexer_index(mock_embedder, mock_vector_store):
    """Test indexing functionality."""
    indexer = Indexer(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )

    documents = [Document(id="1", text="Test")]
    indexer.index(documents)

    # Assert embedder was called
    mock_embedder.embed_batch.assert_called_once()

    # Assert vector store was updated
    mock_vector_store.add.assert_called_once()
```

### Integration Tests

```python
# tests/integration/test_full_pipeline.py

import pytest
from rag_indexer import Indexer, SemanticChunker, HybridRetriever
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

@pytest.mark.integration
def test_full_pipeline():
    """Test full indexing and retrieval pipeline."""
    # Setup real components
    embedder = EmbeddingsEngine(device="cpu")
    vector_store = VectorStore(dimension=384)
    chunker = SemanticChunker()
    retriever = HybridRetriever()

    indexer = Indexer(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        retriever=retriever,
    )

    # Index documents
    documents = [
        Document(id="1", text="Fishing boats navigate calm waters."),
        Document(id="2", text="Jazz music originated in New Orleans."),
    ]
    indexer.index(documents)

    # Search
    results = indexer.search("calm waters", k=1)

    # Assert results
    assert len(results) == 1
    assert "calm waters" in results[0].text.lower()
    assert results[0].score > 0.5
```

### Benchmark Tests

```python
# tests/benchmarks/bench_indexing.py

import pytest
import time
from rag_indexer import Indexer, Document

@pytest.mark.benchmark
def test_indexing_throughput():
    """Benchmark indexing throughput."""
    indexer = Indexer(...)

    # Generate test documents
    documents = [
        Document(id=str(i), text=f"Test document {i} " * 100)
        for i in range(1000)
    ]

    # Time indexing
    start = time.time()
    indexer.index(documents)
    duration = time.time() - start

    # Assert performance target
    throughput = len(documents) / duration
    assert throughput > 100, f"Throughput: {throughput:.1f} docs/sec"

@pytest.mark.benchmark
def test_retrieval_latency():
    """Benchmark retrieval latency."""
    indexer = Indexer(...)
    indexer.index(test_documents)

    # Time retrieval
    import numpy as np
    times = []
    for _ in range(100):
        start = time.time()
        indexer.search("test query", k=10)
        times.append((time.time() - start) * 1000)

    # Assert performance target
    p99_latency = np.percentile(times, 99)
    assert p99_latency < 50, f"P99 latency: {p99_latency:.1f}ms"
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=rag_indexer --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html

# Coverage targets:
# - Unit tests: >90% coverage
# - Integration tests: Core paths covered
# - Overall: >80% coverage
```

---

## Benchmarking Methodology

### Benchmark Framework

```python
# tests/benchmarks/framework.py

import time
import numpy as np
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Benchmark result metrics."""

    name: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

class Benchmark:
    """Benchmark framework."""

    def __init__(self, warmup_runs: int = 5):
        self.warmup_runs = warmup_runs

    def run(
        self,
        func: Callable,
        *args,
        num_runs: int = 100,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run benchmark.

        Args:
            func: Function to benchmark
            *args: Function arguments
            num_runs: Number of benchmark runs
            **kwargs: Function keyword arguments

        Returns:
            BenchmarkResult with metrics
        """
        # Warmup
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Calculate metrics
        return BenchmarkResult(
            name=func.__name__,
            runs=num_runs,
            mean_ms=np.mean(times),
            p50_ms=np.percentile(times, 50),
            p95_ms=np.percentile(times, 95),
            p99_ms=np.percentile(times, 99),
            min_ms=np.min(times),
            max_ms=np.max(times),
    )
```

### Benchmarking Indexing

```python
# tests/benchmarks/bench_indexing.py

from benchmarks.framework import Benchmark
from rag_indexer import Indexer, Document

def benchmark_indexing():
    """Benchmark indexing performance."""

    def setup_indexer(num_docs: int) -> Indexer:
        embedder = EmbeddingsEngine(device="cuda")
        vector_store = VectorStore(dimension=384)
        chunker = SemanticChunker()
        retriever = HybridRetriever()

        return Indexer(
            embedder=embedder,
            vector_store=vector_store,
            chunker=chunker,
            retriever=retriever,
        )

    # Generate test documents
    def generate_documents(num_docs: int) -> List[Document]:
        return [
            Document(
                id=str(i),
                text=f"Test document {i}. " * 100,  # ~1000 chars
            )
            for i in range(num_docs)
        ]

    benchmark = Benchmark()

    # Test different scales
    for num_docs in [100, 1000, 10000]:
        indexer = setup_indexer(num_docs)
        documents = generate_documents(num_docs)

        result = benchmark.run(
            indexer.index,
            documents,
            num_runs=10,
        )

        print(f"\nIndexing {num_docs} documents:")
        print(f"  Mean: {result.mean_ms:.1f}ms")
        print(f"  P99: {result.p99_ms:.1f}ms")
        print(f"  Throughput: {num_docs / (result.mean_ms / 1000):.1f} docs/sec")
```

### Benchmarking Retrieval

```python
# tests/benchmarks/bench_retrieval.py

def benchmark_retrieval():
    """Benchmark retrieval performance."""

    indexer = setup_test_indexer(num_docs=10000)
    queries = [
        "test query one",
        "test query two",
        "jazz music",
        "fishing boats",
    ]

    benchmark = Benchmark()

    # Test different k values
    for k in [5, 10, 20, 50]:
        results = []
        for query in queries:
            result = benchmark.run(
                indexer.search,
                query,
                k,
                num_runs=100,
            )
            results.append(result)

        # Aggregate results
        mean_latency = np.mean([r.mean_ms for r in results])
        p99_latency = np.max([r.p99_ms for r in results])

        print(f"\nRetrieval (k={k}):")
        print(f"  Mean latency: {mean_latency:.1f}ms")
        print(f"  P99 latency: {p99_latency:.1f}ms")
```

### Benchmarking Accuracy

```python
# tests/benchmarks/bench_accuracy.py

def benchmark_accuracy():
    """Benchmark retrieval accuracy."""

    # Setup test data with ground truth
    test_queries = [
        {
            "query": "jazz music origins",
            "relevant_doc_ids": ["doc2", "doc5", "doc8"],
        },
        {
            "query": "fishing boats",
            "relevant_doc_ids": ["doc1", "doc3"],
        },
        # ... more queries
    ]

    indexer = setup_test_indexer(documents)

    # Calculate metrics
    precisions = []
    recalls = []

    for test_case in test_queries:
        results = indexer.search(test_case["query"], k=10)
        retrieved_ids = [r.metadata["document_id"] for r in results]

        relevant_ids = set(test_case["relevant_doc_ids"])
        retrieved_set = set(retrieved_ids)

        # Precision and Recall
        precision = len(relevant_ids & retrieved_set) / len(retrieved_set)
        recall = len(relevant_ids & retrieved_set) / len(relevant_ids)

        precisions.append(precision)
        recalls.append(recall)

    # Print metrics
    print(f"\nAccuracy Metrics:")
    print(f"  Precision@10: {np.mean(precisions):.3f}")
    print(f"  Recall@10: {np.mean(recalls):.3f}")
```

### Performance Targets

```python
# tests/benchmarks/verify_targets.py

def verify_performance_targets():
    """Verify that performance targets are met."""

    # Indexing targets
    assert indexing_throughput > 100, "Indexing too slow"
    assert indexing_latency_p99 < 100, "Indexing latency too high"

    # Retrieval targets
    assert retrieval_latency_p99 < 50, "Retrieval latency too high"
    assert precision_at_10 > 0.85, "Precision too low"
    assert recall_at_100 > 0.80, "Recall too low"

    # Storage targets
    assert compression_ratio > 5, "Compression too low"
    assert memory_usage < 1_000_000_000, "Memory usage too high (1GB)"

    print("✅ All performance targets met")
```

---

## Release Process

### Version Bumping

```bash
# Update version in pyproject.toml
sed -i 's/version = "1.0.0"/version = "1.1.0"/' pyproject.toml

# Update __version__ in __init__.py
sed -i 's/__version__ = "1.0.0"/__version__ = "1.1.0"/' rag_indexer/__init__.py

# Commit changes
git add pyproject.toml rag_indexer/__init__.py
git commit -m "Bump version to 1.1.0"
```

### Changelog Generation

```bash
# Generate changelog from commits
git log --oneline v1.0.0..HEAD > CHANGELOG.md

# Or use towncrier (recommended)
pip install towncrier
towncrier build --version 1.1.0
```

### Building Release

```bash
# Build Python package
python -m build

# Build Go bindings
cd go
go build -o ../dist/rag-indexer-go
cd ..

# Verify build
twine check dist/*
```

### Testing Release

```bash
# Test installation from built package
pip install dist/rag_indexer-1.1.0-py3-none-any.whl

# Run smoke tests
python -c "import rag_indexer; print(rag_indexer.__version__)"

# Run full test suite
pytest tests/
```

### Publishing

```bash
# Publish to PyPI
twine upload dist/rag_indexer-1.1.0-py3-none-any.whl

# Create Git tag
git tag v1.1.0
git push origin v1.1.0

# Create GitHub release (via gh CLI)
gh release create v1.1.0 --notes "Release 1.1.0: Add custom chunker support"
```

### Post-Release

```bash
# Update documentation
# Update examples
# Announce release (blog, Twitter, etc.)

# Start next development cycle
git checkout -b develop
```

---

## Contributing Guidelines

### Code Review Checklist

- [ ] Code follows style guide (Black, Ruff)
- [ ] Type hints included (mypy clean)
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] Backwards compatible (or major version bump)
- [ ] Changelog updated

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Benchmarks run and documented

## Performance Impact
- Indexing throughput: X docs/sec
- Retrieval latency: X ms
- Memory usage: X MB

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
```

---

**Next**: See [INTEGRATION.md](INTEGRATION.md) for equilibrium-tokens integration
