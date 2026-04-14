"""
Shared test configuration for rag-indexer tests.

Provides mock objects for external dependencies (embeddings_engine, vector_navigator)
and common fixtures for testing.
"""

import sys
import types
from typing import List, Dict, Any

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Mock external packages so they can be imported at collection time
# ---------------------------------------------------------------------------

def _create_mock_embeddings_engine():
    """Create a mock embeddings_engine module."""
    module = types.ModuleType("embeddings_engine")

    class MockEmbeddingsEngine:
        def __init__(self, model=None, device=None, dimension=None):
            self.model = model or "all-MiniLM-L6-v2"
            self.device = device or "cpu"
            self.dimension = dimension or 384
            self._call_count = 0

        def embed(self, text: str) -> np.ndarray:
            self._call_count += 1
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec

        def embed_batch(self, texts: List[str]) -> np.ndarray:
            return np.array([self.embed(t) for t in texts])

    module.EmbeddingsEngine = MockEmbeddingsEngine
    return module


def _create_mock_vector_navigator():
    """Create a mock vector_navigator module."""
    module = types.ModuleType("vector_navigator")

    class IndexType:
        FLAT = "flat"
        HNSW = "hnsw"
        IVF = "ivf"

    class MockVectorStore:
        def __init__(self, dimension=384, index_type=None, M=16, efConstruction=200):
            self.dimension = dimension
            self.index_type = index_type or IndexType.FLAT
            self.M = M
            self.efConstruction = efConstruction
            self._vectors: List[np.ndarray] = []
            self._metadata: List[Dict[str, Any]] = []
            self._ids: List[str] = []

        def add(self, vectors, metadata=None, ids=None):
            if vectors is None:
                return
            for i, v in enumerate(vectors):
                self._vectors.append(np.array(v))
                self._metadata.append(metadata[i] if metadata and i < len(metadata) else {})
                self._ids.append(ids[i] if ids and i < len(ids) else f"chunk_{len(self._vectors)}")

        def search(self, query_vector, k=10, filter=None):
            if not self._vectors:
                return []
            query = np.array(query_vector)
            scores = []
            for i, vec in enumerate(self._vectors):
                norm = np.linalg.norm(vec) * np.linalg.norm(query)
                if norm == 0:
                    score = 0.0
                else:
                    score = float(np.dot(vec, query) / norm)
                # Clamp cosine similarity to [0, 1] for RAG compatibility
                score = max(0.0, min(1.0, score))
                scores.append((score, i))
            scores.sort(key=lambda x: x[0], reverse=True)

            results = []
            for score, idx in scores[:k]:
                item = {
                    "id": self._ids[idx],
                    "score": score,
                    "text": self._metadata[idx].get("text", ""),
                    "metadata": self._metadata[idx],
                }
                if filter:
                    match = all(
                        self._metadata[idx].get(k) == v
                        for k, v in filter.items()
                    )
                    if not match:
                        continue
                results.append(item)
            return results

        def size_mb(self) -> float:
            total = sum(v.nbytes for v in self._vectors)
            return total / (1024 * 1024)

        def save(self, path: str):
            pass

    module.VectorStore = MockVectorStore
    module.IndexType = IndexType
    return module


# Install mock modules before any rag_indexer imports
if "embeddings_engine" not in sys.modules:
    sys.modules["embeddings_engine"] = _create_mock_embeddings_engine()

if "vector_navigator" not in sys.modules:
    sys.modules["vector_navigator"] = _create_mock_vector_navigator()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder():
    """Provide a mock EmbeddingsEngine."""
    from embeddings_engine import EmbeddingsEngine
    return EmbeddingsEngine(model="test-model", device="cpu", dimension=384)


@pytest.fixture
def vector_store():
    """Provide a mock VectorStore."""
    from vector_navigator import VectorStore
    return VectorStore(dimension=384)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from rag_indexer.config import Document

    return [
        Document(
            id="doc1",
            text=(
                "Fishing boats navigate calm waters at sunrise. These vessels are designed "
                "for commercial fishing operations in coastal waters and the open ocean. "
                "Modern fishing boats are equipped with advanced navigation systems."
            ),
            metadata={"title": "Fishing Boats", "category": "marine"},
        ),
        Document(
            id="doc2",
            text=(
                "Jazz music originated in New Orleans in the late 19th century. This genre "
                "combines elements of blues, ragtime, and brass band music. Jazz is "
                "characterized by swing, blue notes, complex chords, and improvisation."
            ),
            metadata={"title": "Jazz History", "category": "music"},
        ),
        Document(
            id="doc3",
            text=(
                "Symphony pops concerts feature popular classical music selections. These "
                "performances blend traditional symphonic repertoire with contemporary pieces, "
                "film scores, and Broadway show tunes."
            ),
            metadata={"title": "Symphony Pops", "category": "music"},
        ),
    ]


@pytest.fixture
def long_document():
    """Provide a long document for chunking tests."""
    from rag_indexer.config import Document

    text = (
        "# Introduction\n\n"
        "This is the introduction paragraph. It contains some initial content about the topic.\n\n"
        "# Background\n\n"
        "The background section provides historical context and prior work. "
        "It spans multiple sentences to provide depth and detail about the research area.\n\n"
        "# Methodology\n\n"
        "Our methodology follows a systematic approach. First, we collect the data. "
        "Second, we preprocess it using standard techniques. Third, we apply our model.\n\n"
        "# Results\n\n"
        "The results show significant improvements across all metrics. "
        "Precision improved by 20%, recall by 15%, and F1 score by 18%. "
        "These improvements are statistically significant with p < 0.01.\n\n"
        "# Conclusion\n\n"
        "We conclude that our approach is effective and scalable. "
        "Future work will explore additional optimizations and broader applications."
    )
    return Document(id="long_doc", text=text, metadata={"title": "Research Paper"})
