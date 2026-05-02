# Conversational RAG System with LangGraph Orchestration

An intelligent conversational RAG (Retrieval-Augmented Generation) system that maintains persistent, context-aware conversations across multiple users using LangGraph-based agentic workflows.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│         (Chat interface + Session management)           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                        │
│          /chat   /sessions   /documents/upload           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              LangGraph Orchestrator                      │
│    ┌──────────────────────────────────────────────┐     │
│    │  Query Understanding → Query Rewriting        │     │
│    │         ↓                    ↓                │     │
│    │  Retrieval Router → Context Synthesis          │     │
│    │         ↓                    ↓                │     │
│    │  Conv. Summarizer → Memory Manager            │     │
│    └──────────────────────────────────────────────┘     │
└──────┬──────────────────────────────────┬───────────────┘
       │                                  │
┌──────▼──────────┐            ┌──────────▼──────────┐
│    ChromaDB     │            │      SQLite         │
│  (Vector Store) │            │  (Sessions/Memory)  │
└─────────────────┘            └─────────────────────┘
```

## LangGraph Agent Flow

The system orchestrates 6 specialized agents through a LangGraph state machine:

| Agent | Purpose | Routing |
|-------|---------|---------|
| **Query Understanding** | Classifies intent (factual, follow-up, clarification, comparison, multi-part, greeting, off-topic) and decides if conversation history is needed | Entry point → conditional routing |
| **Query Rewriting** | Reformulates ambiguous queries by incorporating conversation context (resolves pronouns like "it", "that") | Only triggered if query needs rewriting |
| **Retrieval Router** | Selects optimal search strategy: semantic, keyword, or hybrid based on query characteristics | Skipped for greetings/off-topic |
| **Context Synthesis** | Combines retrieved documents with conversation history to generate the final LLM response | Always runs |
| **Conversation Summarizer** | Periodically summarizes long conversations to maintain context within token limits | Runs after every response |
| **Memory Manager** | Identifies important information (preferences, decisions, context) to persist across sessions | Final node before END |

### Conditional Routing Logic

```
START → Query Understanding
    ├─ greeting/off_topic → Context Synthesis (skip retrieval)
    ├─ needs_rewrite=true → Query Rewriting → Retrieval Router
    └─ clear query → Retrieval Router
                        ↓
              Context Synthesis → Conversation Summarizer → Memory Manager → END
```

## Memory Management Strategy

### Short-term (per-session)
- **Sliding window**: Last N messages kept in active memory (configurable, default 5 exchanges)
- **Summary buffer**: When conversation exceeds threshold (default 10 messages), a summary is generated and stored

### Long-term (cross-session)
- **User memory**: Preferences, project context, decisions, and issues stored in SQLite
- **Relevance scoring**: Memories ranked by recency and relevance
- **Pruning**: Old low-relevance memories pruned when exceeding limit (default 50)

### Memory Types
| Type | Example | Persistence |
|------|---------|-------------|
| `preference` | "User prefers Python over Java" | Cross-session |
| `context` | "Working on an e-commerce API" | Cross-session |
| `decision` | "Chose PostgreSQL for the database" | Cross-session |
| `issue` | "Having CORS problems with FastAPI" | Cross-session |

## Setup Instructions

### Prerequisites
- Python 3.10+
- 4GB+ RAM (for embedding model)

### 1. Clone and install

```bash
git clone <repository-url>
cd conversational-rag

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API key (Groq is free and recommended)
```

**Supported LLM providers:**
| Provider | Model | Cost |
|----------|-------|------|
| Groq | `llama-3.3-70b-versatile` | Free tier available |
| OpenAI | `gpt-4o-mini` or `gpt-4o` | Paid |
| Ollama | Any local model | Free (local) |

### 3. Ingest sample documents

```bash
python ingest_docs.py
```

### 4. Start the API server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Docker (alternative)

```bash
cp .env.example .env
# Edit .env

docker-compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

## API Documentation

Interactive docs available at `http://localhost:8000/docs` (Swagger) or `/redoc`.

### Endpoints

#### POST `/api/v1/sessions`
Create a new conversation session.

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "user_id": "alice",
  "created_at": "2025-01-01T00:00:00"
}
```

#### POST `/api/v1/chat`
Send a message and receive a RAG-powered response.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a1b2c3d4-...",
    "user_id": "alice",
    "message": "How do I create a FastAPI endpoint with dependency injection?"
  }'
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "response": "To create a FastAPI endpoint with dependency injection...",
  "sources": [
    {"title": "fastapi_guide.md", "source": "...", "relevance": 0.87}
  ],
  "agent_log": [
    {"agent": "query_understanding", "intent": "factual", ...},
    {"agent": "retrieval_router", "strategy": "semantic", ...},
    {"agent": "orchestrator", "total_latency_seconds": 2.3}
  ]
}
```

#### POST `/api/v1/documents/upload`
Upload a document for RAG ingestion.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@my_document.pdf"
```

#### GET `/api/v1/sessions/{user_id}`
List all sessions for a user.

#### GET `/api/v1/sessions/{session_id}/messages`
Get conversation history for a session.

#### GET `/api/v1/stats`
System statistics (vector store count, etc.).

## Sample Conversation Flows

### Flow 1: Multi-turn technical question

```
User: How do I set up authentication in FastAPI?
Bot:  [Retrieves FastAPI guide → explains OAuth2PasswordBearer, JWT tokens]

User: What about dependency injection for the database?
Bot:  [Query understanding detects follow-up → rewriter adds "FastAPI" context
       → retrieves DI section → synthesizes with auth context]

User: Can you compare that with Django's approach?
Bot:  [Detects comparison intent → hybrid search for both → synthesized comparison]
```

### Flow 2: Cross-session memory

```
Session 1:
User: I'm building a REST API with FastAPI and PostgreSQL
Bot:  [Stores memory: context="Building REST API with FastAPI + PostgreSQL"]

Session 2:
User: What indexing strategy should I use?
Bot:  [Loads user memory → knows they use PostgreSQL → retrieves PostgreSQL guide
       → recommends B-tree and GIN indexes relevant to their stack]
```

### Flow 3: Conversation summarization

```
[After 10+ messages in a session]
Bot:  [Summarizer agent triggers → produces summary:
       "User is building an e-commerce API. Discussed FastAPI routing,
        PostgreSQL setup, and Docker deployment. Decided to use
        connection pooling with PgBouncer."]
```

## Performance Benchmarks

Measured on sample_docs (5 documents, ~200 chunks):

| Metric | Value |
|--------|-------|
| Document ingestion (5 docs) | ~8-12s |
| First query (cold start) | ~3-5s |
| Subsequent queries | ~1.5-3s |
| Memory overhead per session | ~2KB |
| Vector search (top-5) | ~50ms |
| Embedding generation | ~100ms per query |

*Latency depends on LLM provider. Groq is fastest (~0.5s), OpenAI ~1-2s.*

## Project Structure

```
conversational-rag/
├── main.py                      # FastAPI application entry point
├── config.py                    # Environment configuration
├── streamlit_app.py             # Streamlit chat UI
├── ingest_docs.py               # Document ingestion script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment template
│
├── agents/                      # LangGraph agent nodes
│   ├── orchestrator.py          # Graph builder + RAGPipeline class
│   ├── query_understanding.py   # Intent classification
│   ├── query_rewriting.py       # Context-aware reformulation
│   ├── retrieval_router.py      # Search strategy selection
│   ├── context_synthesis.py     # Document + history merging
│   ├── conversation_summarizer.py  # Long conversation compression
│   └── memory_manager.py        # Long-term memory decisions
│
├── core/                        # Core infrastructure
│   ├── models.py                # Pydantic data models
│   ├── memory_store.py          # SQLite session/memory persistence
│   ├── vector_store.py          # ChromaDB with hybrid search
│   ├── document_processor.py    # PDF/MD/HTML ingestion pipeline
│   ├── embeddings.py            # Sentence-transformer embeddings
│   └── llm_provider.py          # Multi-provider LLM factory
│
├── api/                         # FastAPI routes
│   └── routes.py                # Chat, session, upload endpoints
│
├── sample_docs/                 # 5 technical documents for demo
│   ├── fastapi_guide.md
│   ├── docker_guide.md
│   ├── langchain_rag_guide.md
│   ├── python_best_practices.md
│   └── postgresql_guide.md
│
└── tests/                       # Test suite
    ├── test_core.py             # Model + memory store + processor tests
    └── test_api.py              # API endpoint tests
```

## Key Design Decisions

1. **ChromaDB over Pinecone/Weaviate**: Zero-config local setup, great for prototyping. Swap-ready via `VectorStore` abstraction.
2. **SQLite over PostgreSQL**: Simpler deployment. The `MemoryStore` class is database-agnostic — swap the connector for production.
3. **sentence-transformers for embeddings**: Runs locally, no API costs. The `all-MiniLM-L6-v2` model is fast and produces good 384-dim embeddings.
4. **Groq as default LLM**: Free tier, extremely fast inference (~500ms). Easily switchable to OpenAI or Ollama.
5. **Hybrid search with RRF**: Reciprocal Rank Fusion combines semantic understanding with keyword precision.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
