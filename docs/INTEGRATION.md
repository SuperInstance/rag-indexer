# rag-indexer Integration with equilibrium-tokens

**Version:** 1.0
**Last Updated:** January 8, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Context Indexing Workflow](#context-indexing-workflow)
3. [Retrieval for Conversation Context](#retrieval-for-conversation-context)
4. [Performance Optimization](#performance-optimization)
5. [Advanced Patterns](#advanced-patterns)
6. [Complete Examples](#complete-examples)

---

## Overview

rag-indexer enables equilibrium-tokens to maintain and retrieve relevant context from large conversation histories and document collections. This integration follows the equilibrium-tokens architecture where:

- **Frozen embeddings** represent conversational territories
- **Sentiment-weighted basins** organize context by emotional tone
- **Dynamic retrieval** provides relevant context without overwhelming the LLM

### Key Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                  equilibrium-tokens                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Orchestrator │  │  Basins      │  │ Context      │      │
│  │              │  │              │  │ Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      rag-indexer                             │
│  • Index conversation history                               │
│  • Retrieve relevant context                                │
│  • Manage sentiment-weighted basins                         │
│  • Optimize for sub-50ms retrieval                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Dependencies (vector-navigator, embeddings)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Context Indexing Workflow

### 1. Initial Setup

```python
from equilibrium_tokens import Orchestrator, ContextManager, BasinManager
from rag_indexer import Indexer, HierarchicalChunker, HybridRetriever
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

# Setup RAG indexer
embedder = EmbeddingsEngine(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
)

vector_store = VectorStore(
    dimension=384,
    index_type=IndexType.IVF_HNSW,
    nlist=100,
    nprobe=10,
)

chunker = HierarchicalChunker(
    leaf_size=1800,  # Optimal per research
    max_levels=3,
)

retriever = HybridRetriever(
    dense_weight=0.7,
    sparse_weight=0.3,
)

# Create indexer
context_indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=chunker,
    retriever=retriever,
)

# Setup equilibrium-tokens
orchestrator = Orchestrator()
basin_manager = BasinManager()
context_manager = ContextManager(indexer=context_indexer)
```

### 2. Indexing Conversation History

```python
from equilibrium_tokens import Conversation, Message

# Load conversation history
conversations = load_conversation_history()  # Your data loading function

# Index conversations
for conversation in conversations:
    # Convert to RAG documents
    doc = Document(
        id=conversation.id,
        text=format_conversation(conversation),
        metadata={
            "conversation_id": conversation.id,
            "user_id": conversation.user_id,
            "timestamp": conversation.timestamp,
            "sentiment": conversation.sentiment,  # -1.0 to 1.0
            "topic": conversation.topic,
        },
    )

    # Index with RAG
    context_indexer.index([doc])

    # Also create sentiment-weighted basin
    basin_manager.create_basin(
        basin_id=f"basin_{conversation.id}",
        embedding=embedder.embed(doc.text),
        sentiment_weight=conversation.sentiment,
        metadata={
            "conversation_id": conversation.id,
            "timestamp": conversation.timestamp,
        },
    )
```

### 3. Indexing Document Collections

```python
# Index external knowledge bases
documents = load_document_collections()  # PDFs, articles, etc.

# Batch index
context_indexer.index_batch(
    documents,
    batch_size=1000,
    show_progress=True,
)

# Create separate basins for different topics
for doc in documents:
    topic = doc.metadata.get("topic", "general")

    basin_manager.create_basin(
        basin_id=f"basin_{topic}",
        embedding=embedder.embed(doc.text),
        sentiment_weight=0.0,  # Neutral for documents
        metadata={
            "document_id": doc.id,
            "topic": topic,
        },
    )
```

### 4. Real-Time Indexing

```python
# Index new conversations as they happen
async def handle_conversation_turn(
    user_message: str,
    ai_response: str,
    conversation_id: str,
):
    """Index conversation turn in real-time."""

    # Format as document
    doc = Document(
        id=f"{conversation_id}_{len(messages)}",
        text=f"User: {user_message}\nAI: {ai_response}",
        metadata={
            "conversation_id": conversation_id,
            "timestamp": time.time(),
            "sentiment": analyze_sentiment(user_message),
        },
    )

    # Index immediately
    context_indexer.index([doc])

    # Update basin
    basin_manager.update_basin(
        basin_id=f"basin_{conversation_id}",
        new_embedding=embedder.embed(doc.text),
        sentiment_shift=analyze_sentiment(user_message),
    )
```

---

## Retrieval for Conversation Context

### 1. Basic Context Retrieval

```python
# Retrieve context for new query
query = "Tell me more about jazz music"

# Search RAG index
results = context_indexer.search(
    query,
    k=5,
    filter={
        "conversation_id": current_conversation_id,
    },
)

# Format context for orchestrator
context = [
    {
        "text": result.text,
        "score": result.score,
        "provenance": result.metadata.get("conversation_id"),
    }
    for result in results
]

# Set context in orchestrator
orchestrator.set_context(context)

# Generate response
response = orchestrator.generate(query)
print(response)
```

### 2. Sentiment-Aware Retrieval

```python
# Retrieve context with sentiment weighting
query = "I'm feeling stressed about work"

# Analyze query sentiment
query_sentiment = analyze_sentiment(query)  # Returns -1.0 to 1.0

# Retrieve with sentiment filter
results = context_indexer.search(
    query,
    k=10,
    filter={
        "sentiment_label": "positive" if query_sentiment < 0 else "neutral",
    },
)

# Alternatively, retrieve similar sentiment basins
basin_results = basin_manager.search(
    query_embedding=embedder.embed(query),
    sentiment_weight=query_sentiment,  # Prefer similar sentiment
    k=5,
)

# Combine RAG and basin results
combined_results = merge_and_rank(results, basin_results)

# Use in orchestrator
orchestrator.set_context([r.text for r in combined_results])
```

### 3. Multi-Context Retrieval

```python
# Retrieve from multiple sources
query = "Tell me about jazz and fishing boats"

# Conversation history
conv_results = context_indexer.search(
    query,
    k=3,
    filter={"source": "conversation"},
)

# Document collection
doc_results = context_indexer.search(
    query,
    k=3,
    filter={"source": "documents"},
)

# Knowledge base
kb_results = context_indexer.search(
    query,
    k=3,
    filter={"source": "knowledge_base"},
)

# Merge and deduplicate
all_results = conv_results + doc_results + kb_results
merged = deduplicate_by_document_id(all_results)

# Select top results by score
final_results = sorted(merged, key=lambda r: r.score, reverse=True)[:10]

# Use in orchestrator
orchestrator.set_context([r.text for r in final_results])
```

### 4. Temporal Context Retrieval

```python
# Retrieve recent + relevant context
query = "What did we discuss about fishing?"

# Recent conversation history (last 10 turns)
recent_results = context_indexer.search(
    query,
    k=5,
    filter={
        "conversation_id": current_conversation_id,
        "timestamp": {"gt": time.time() - 3600},  # Last hour
    },
)

# Historical relevant context (all time)
historical_results = context_indexer.search(
    query,
    k=5,
    filter={"conversation_id": current_conversation_id},
)

# Prioritize recent, but include historical if relevant
combined = []
combined.extend([r for r in recent_results if r.score > 0.7])
combined.extend([r for r in historical_results if r.score > 0.8])

# Limit total context
final_results = combined[:10]

# Use in orchestrator
orchestrator.set_context([r.text for r in final_results])
```

---

## Performance Optimization

### 1. Context Window Management

```python
class ContextWindowManager:
    """Manage context window to avoid overflow."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.tokenizer = get_tokenizer()  # Your tokenizer

    def select_context(
        self,
        results: List[Result],
        query: str,
    ) -> List[str]:
        """
        Select context that fits in window.

        Strategy:
        1. Prioritize high-score results
        2. Estimate token usage
        3. Truncate if needed
        """
        selected = []
        total_tokens = 0
        query_tokens = len(self.tokenizer.encode(query))

        for result in results:
            # Estimate tokens for this result
            result_tokens = len(self.tokenizer.encode(result.text))

            # Check if fits
            if total_tokens + query_tokens + result_tokens > self.max_tokens:
                break

            selected.append(result.text)
            total_tokens += result_tokens

        return selected

# Usage
manager = ContextWindowManager(max_tokens=2000)
results = context_indexer.search(query, k=20)
context = manager.select_context(results, query)

orchestrator.set_context(context)
```

### 2. Query Caching

```python
from functools import lru_cache

class CachedRetriever:
    """Cache frequent queries."""

    def __init__(self, indexer: Indexer, cache_size: int = 1000):
        self.indexer = indexer
        self.cache_size = cache_size

    @lru_cache(maxsize=1000)
    def search(self, query: str, k: int) -> List[Result]:
        """Search with caching."""
        return self.indexer.search(query, k)

# Usage
retriever = CachedRetriever(context_indexer)

# First call: cache miss (full search)
results1 = retriever.search("jazz music", k=5)

# Second call: cache hit (instant)
results2 = retriever.search("jazz music", k=5)
```

### 3. Parallel Retrieval

```python
import asyncio

async def parallel_retrieval(
    queries: List[str],
    indexer: Indexer,
    k: int = 5,
) -> Dict[str, List[Result]]:
    """Retrieve for multiple queries in parallel."""

    async def retrieve(query: str) -> Tuple[str, List[Result]]:
        results = await asyncio.to_thread(indexer.search, query, k)
        return query, results

    # Parallel retrieval
    tasks = [retrieve(q) for q in queries]
    results = await asyncio.gather(*tasks)

    # Format as dict
    return {query: result for query, result in results}

# Usage
queries = [
    "jazz music",
    "fishing boats",
    "symphony pops",
]

all_results = await parallel_retrieval(queries, context_indexer)
```

### 4. Incremental Indexing

```python
# For real-time conversations, index incrementally
class RealTimeIndexer:
    """Index new messages as they arrive."""

    def __init__(self, indexer: Indexer, batch_size: int = 10):
        self.indexer = indexer
        self.batch_size = batch_size
        self.pending = []

    def add_message(self, message: Message):
        """Add message to pending batch."""
        doc = Document(
            id=message.id,
            text=message.text,
            metadata={
                "conversation_id": message.conversation_id,
                "timestamp": message.timestamp,
            },
        )

        self.pending.append(doc)

        # Flush if batch full
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """Index pending messages."""
        if self.pending:
            self.indexer.index(self.pending)
            self.pending = []

# Usage
indexer = RealTimeIndexer(context_indexer)

# Add messages as they arrive
for message in conversation_stream:
    indexer.add_message(message)
```

---

## Advanced Patterns

### 1. Hierarchical Context Selection

```python
class HierarchicalContextSelector:
    """Select context at multiple granularity levels."""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def select_context(
        self,
        query: str,
        conversation_id: str,
    ) -> Dict[str, List[str]]:
        """
        Select context at multiple levels:

        Returns:
            {
                "overview": [...],      # High-level summary (1-2 chunks)
                "details": [...],       # Relevant details (3-5 chunks)
                "examples": [...],      # Specific examples (2-3 chunks)
            }
        """
        results = {}

        # Level 1: Overview (document-level chunks)
        overview_results = self.indexer.search(
            query,
            k=2,
            filter={
                "conversation_id": conversation_id,
                "level": 0,  # Document level
            },
        )
        results["overview"] = [r.text for r in overview_results]

        # Level 2: Details (section-level chunks)
        detail_results = self.indexer.search(
            query,
            k=5,
            filter={
                "conversation_id": conversation_id,
                "level": 1,  # Section level
            },
        )
        results["details"] = [r.text for r in detail_results]

        # Level 3: Examples (paragraph-level chunks)
        example_results = self.indexer.search(
            query,
            k=3,
            filter={
                "conversation_id": conversation_id,
                "level": 2,  # Paragraph level
            },
        )
        results["examples"] = [r.text for r in example_results]

        return results

# Usage
selector = HierarchicalContextSelector(context_indexer)
context = selector.select_context(query, conversation_id)

# Use structured context
orchestrator.set_structured_context(context)
```

### 2. Dynamic Context Adaptation

```python
class DynamicContextAdapter:
    """Adapt context based on query complexity."""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)."""
        # Simple heuristics
        complexity = 0.0

        # Length
        complexity += min(len(query.split()) / 50, 0.3)

        # Question words
        question_words = ["how", "why", "what", "explain", "describe"]
        if any(qw in query.lower() for qw in question_words):
            complexity += 0.3

        # Technical terms
        technical_words = ["algorithm", "implementation", "architecture"]
        if any(tw in query.lower() for tw in technical_words):
            complexity += 0.2

        return min(complexity, 1.0)

    def retrieve_adaptive(
        self,
        query: str,
        conversation_id: str,
    ) -> List[str]:
        """Retrieve context adapted to query complexity."""
        complexity = self.estimate_complexity(query)

        # Simple queries: Less context
        if complexity < 0.3:
            k = 3
            level_filter = {"level": 0}  # Overview only

        # Medium queries: Moderate context
        elif complexity < 0.7:
            k = 7
            level_filter = {"level": [0, 1]}  # Overview + details

        # Complex queries: Full context
        else:
            k = 15
            level_filter = None  # All levels

        # Retrieve
        results = self.indexer.search(
            query,
            k=k,
            filter={
                "conversation_id": conversation_id,
                **(level_filter or {}),
            },
        )

        return [r.text for r in results]

# Usage
adapter = DynamicContextAdapter(context_indexer)

# Simple query (k=3)
context1 = adapter.retrieve_adaptive("Hi there", conversation_id)

# Complex query (k=15)
context2 = adapter.retrieve_adaptive(
    "How does the recursive retrieval algorithm work in hierarchical indexing?",
    conversation_id,
)
```

### 3. Multi-Turn Context Accumulation

```python
class MultiTurnContextAccumulator:
    """Accumulate context across conversation turns."""

    def __init__(self, indexer: Indexer, max_turns: int = 5):
        self.indexer = indexer
        self.max_turns = max_turns
        self.turn_history = []

    def add_turn(self, query: str, response: str):
        """Add conversation turn to history."""
        self.turn_history.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
        })

        # Limit history
        if len(self.turn_history) > self.max_turns:
            self.turn_history = self.turn_history[-self.max_turns:]

    def get_accumulated_context(
        self,
        current_query: str,
        conversation_id: str,
    ) -> List[str]:
        """
        Get context considering conversation history.

        Strategy:
        1. Retrieve relevant to current query
        2. Also retrieve relevant to previous turns
        3. Merge and deduplicate
        """
        all_results = []

        # Current query
        current_results = self.indexer.search(
            current_query,
            k=5,
            filter={"conversation_id": conversation_id},
        )
        all_results.extend(current_results)

        # Previous turns (with decay)
        for i, turn in enumerate(reversed(self.turn_history)):
            weight = 1.0 - (i * 0.2)  # Decay weight

            if weight > 0:
                turn_results = self.indexer.search(
                    turn["query"],
                    k=3,
                    filter={"conversation_id": conversation_id},
                )

                # Adjust scores
                for result in turn_results:
                    result.score *= weight

                all_results.extend(turn_results)

        # Merge and deduplicate
        merged = deduplicate_by_document_id(all_results)

        # Return top results
        top_results = sorted(merged, key=lambda r: r.score, reverse=True)[:10]

        return [r.text for r in top_results]

# Usage
accumulator = MultiTurnContextAccumulator(context_indexer)

# Turn 1
accumulator.add_turn("Tell me about jazz", "Jazz originated...")
context1 = accumulator.get_accumulated_context("Tell me about jazz", conv_id)

# Turn 2
accumulator.add_turn("What about instruments?", "Common instruments...")
context2 = accumulator.get_accumulated_context("What about instruments?", conv_id)
# Context includes relevant from Turn 1
```

---

## Complete Examples

### Example 1: Customer Support Chatbot

```python
from equilibrium_tokens import Orchestrator
from rag_indexer import Indexer, SemanticChunker, HybridRetriever
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

# Setup
embedder = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
vector_store = VectorStore(dimension=384, index_type=IndexType.IVF_HNSW)
chunker = SemanticChunker()
retriever = HybridRetriever(dense_weight=0.7)

indexer = Indexer(
    embedder=embedder,
    vector_store=vector_store,
    chunker=chunker,
    retriever=retriever,
)
orchestrator = Orchestrator()

# Index knowledge base
kb_docs = load_knowledge_base("customer_support_kb/*.md")
indexer.index_batch(kb_docs, batch_size=100)

# Handle customer query
def handle_customer_query(query: str, conversation_id: str):
    """Handle customer query with RAG context."""

    # Retrieve relevant knowledge base articles
    kb_results = indexer.search(
        query,
        k=5,
        filter={"source": "knowledge_base"},
    )

    # Retrieve conversation history
    conv_results = indexer.search(
        query,
        k=3,
        filter={"conversation_id": conversation_id},
    )

    # Combine context
    context = [r.text for r in (kb_results + conv_results)]

    # Generate response
    orchestrator.set_context(context)
    response = orchestrator.generate(query)

    # Index this turn for future context
    indexer.index([
        Document(
            id=f"{conversation_id}_{int(time.time())}",
            text=f"Customer: {query}\nAgent: {response}",
            metadata={"conversation_id": conversation_id},
        ),
    ])

    return response

# Usage
response = handle_customer_query(
    "How do I reset my password?",
    conversation_id="conv_123",
)
print(response)
```

### Example 2: Document QA System

```python
# Index document collection
documents = load_documents("technical_docs/*.pdf")
indexer.index_batch(documents, batch_size=50)

# Handle document QA
def answer_question(query: str) -> Dict[str, Any]:
    """Answer question with document evidence."""

    # Retrieve relevant documents
    results = indexer.search(query, k=10)

    # Format with citations
    context_with_citations = []
    for result in results:
        context_with_citations.append({
            "text": result.text,
            "score": result.score,
            "source": result.metadata.get("source"),
            "page": result.metadata.get("page_number"),
        })

    # Generate answer
    orchestrator.set_context([c["text"] for c in context_with_citations])
    answer = orchestrator.generate(query)

    return {
        "answer": answer,
        "sources": context_with_citations,
    }

# Usage
qa_result = answer_question("What is the performance limit?")
print(qa_result["answer"])
print("\nSources:")
for source in qa_result["sources"]:
    print(f"  - {source['source']} (page {source['page']})")
```

### Example 3: Code Assistant

```python
# Index codebase with documentation
code_docs = extract_code_and_docs("/path/to/codebase")
indexer.index_batch(code_docs, batch_size=100)

# Handle code query
def handle_code_query(query: str, language: str = "python") -> str:
    """Search codebase by natural language."""

    # Retrieve relevant code/documentation
    results = indexer.search(
        query,
        k=10,
        filter={
            "language": language,
        },
    )

    # Format context
    context = []
    for result in results:
        if result.metadata.get("type") == "code":
            context.append(f"Code:\n{result.text}")
        else:
            context.append(f"Documentation:\n{result.text}")

    # Generate explanation
    orchestrator.set_context(context)
    explanation = orchestrator.generate(
        f"Explain how to: {query}\n"
        f"Provide code examples in {language}.",
    )

    return explanation

# Usage
explanation = handle_code_query(
    "initialize a vector store with HNSW index",
    language="python",
)
print(explanation)
```

---

## Best Practices

### 1. Context Selection

- Use **sentiment-aware filtering** for emotionally sensitive queries
- Prioritize **recent conversation history** for multi-turn dialogues
- Use **hierarchical chunking** for large document collections
- Implement **context window management** to avoid overflow

### 2. Performance

- **Cache frequent queries** to reduce retrieval latency
- Use **parallel retrieval** for multiple queries
- Implement **incremental indexing** for real-time conversations
- **Tune nprobe** in IVF-HNSW for speed vs recall trade-off

### 3. Accuracy

- Use **hybrid retrieval** (dense + sparse) for best accuracy
- Implement **recursive retrieval** for large documents
- Use **hierarchical chunking** to preserve document structure
- **Rerank results** with cross-encoder if accuracy critical

### 4. Integration

- **Index conversations in real-time** as they happen
- **Maintain separate indexes** for different data sources
- Use **sentiment-weighted basins** for context organization
- **Track provenance** for citation and debugging

---

**Next**: See [USER_GUIDE.md](USER_GUIDE.md) for more usage examples
