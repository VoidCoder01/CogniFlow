# CogniFlow — Conversational RAG with LangGraph Orchestration

An intelligent conversational RAG (Retrieval-Augmented Generation) system that maintains persistent, context-aware conversations across multiple users using LangGraph-based agentic workflows.

## Architecture diagrams

- Mermaid diagrams (LangGraph flow, memory): [docs/architecture.md](docs/architecture.md)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│         (Chat interface + Session management)           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                       │
│          /chat   /sessions   /documents/upload          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              LangGraph Orchestrator                     │
│    ┌──────────────────────────────────────────────┐     │
│    │  Query Understanding → Query Rewriting       │     │
│    │         ↓                    ↓               │     │
│    │  Retrieval Router → Context Synthesis        │     │
│    │         ↓                    ↓               │     │
│    │  Conv. Summarizer → Memory Manager           │     │
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
| **Query Decomposer** | Splits multi-part questions into sub-queries for retrieval | Runs when intent is `multi_part` (before the retrieval router) |
| **Retrieval Router** | Selects optimal search strategy: semantic, keyword, or hybrid based on query characteristics | Skipped for greetings/off-topic |
| **Context Synthesis** | Combines retrieved documents with conversation history to generate the final LLM response | Always runs |
| **Conversation Summarizer** | Periodically summarizes long conversations to maintain context within token limits | Runs when message count ≥ `summary_threshold` (see config) |
| **Memory Manager** | Identifies important information (preferences, decisions, context) to persist across sessions | Final node before END |

### Conditional Routing Logic

```
START → Query Understanding
    ├─ greeting/off_topic → Context Synthesis (skip retrieval)
    ├─ needs_rewrite=true → Query Rewriting
    │       ├─ multi_part → Query Decomposer → Retrieval Router
    │       └─ otherwise → Retrieval Router
    ├─ multi_part → Query Decomposer → Retrieval Router
    └─ clear query → Retrieval Router
                        ↓
              Context Synthesis → Conversation Summarizer → Memory Manager → END
```

## Memory Management Strategy

### Short-term (per-session)
- **Sliding window**: Last N messages kept in active memory (`MEMORY_WINDOW_SIZE` user–assistant pairs; default **10** pairs → up to 20 messages loaded for the graph)
- **Summary buffer**: When enough messages exist (`SUMMARY_THRESHOLD`, default **2**), rolling summaries are generated and stored on `sessions.summary`

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

### Memory pruning strategies

Controlled by **`MEMORY_PRUNING_STRATEGY`** (default: `relevance`):

| Strategy | Behavior | Config value |
|----------|----------|--------------|
| **Relevance** | Keep rows with highest `relevance_score` (within the prune budget) | `relevance` |
| **Sliding window** | Keep the **N** most recent memories by `created_at` | `sliding` |
| **Summary buffer** | Fold the oldest memories beyond the budget into one consolidated **summary** row | `summary` |

Set via `.env`, for example:

```
MEMORY_PRUNING_STRATEGY=sliding
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- 4GB+ RAM (for local CPU embeddings; API embeddings need less)

**No GPU required.** Chat uses cloud LLMs via API keys (Gemini is the default provider; also OpenAI, Anthropic, Groq, OpenRouter). Local retrieval embeddings use **SentenceTransformer on CPU** by default (`EMBEDDING_DEVICE=cpu`). To avoid loading PyTorch for embeddings entirely, set `EMBEDDING_BACKEND=openai` and `OPENAI_API_KEY` (then clear or rename your Chroma data and re-ingest so vector dimensions match). `LLM_PROVIDER=ollama` is optional and assumes a capable local machine.

### 1. Clone and install

```bash
git clone <repository-url>
cd CogniFlow

python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY for Gemini (default), or switch LLM_PROVIDER and the matching key
```

**Supported LLM providers** (default: **Gemini**):
| Provider | Model | Cost |
|----------|-------|------|
| Gemini | e.g. `gemini-2.5-flash` (`GEMINI_MODEL`; `GOOGLE_API_KEY`) | Per Google AI pricing |
| OpenRouter | e.g. `google/gemma-2-9b-it` (set `OPENROUTER_MODEL`) | Per OpenRouter / model |
| Groq | `llama-3.3-70b-versatile` | Free tier available |
| OpenAI | `gpt-4o-mini` or `gpt-4o` | Paid |
| Anthropic | Claude (see `ANTHROPIC_MODEL`) | Paid |
| Ollama | Any local model | Free (local) |

By default, set `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey) and optionally `GEMINI_MODEL` (default `gemini-2.5-flash`). For **OpenRouter** (e.g. Gemma), use `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY` from [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Ingest sample documents

```bash
python ingest_docs.py sample_docs
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

By default **`STREAMLIT_SHORT_MESSAGE_SYNC_CHARS=0`**: the UI always uses **`POST /api/v1/chat/stream`** (SSE + `httpx` streaming) for token-by-token display. Set it to a positive number (e.g. **120**) if you want short prompts to use sync **`POST /api/v1/chat`** instead (slightly less overhead, no live tokens).

**Reverse proxies (production):** for `/api/v1/chat/stream`, disable buffering and allow long reads, e.g. Nginx:

```nginx
location /api/v1/chat/stream {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
    chunked_transfer_encoding on;
}
```

### Docker (alternative)

```bash
cp .env.example .env
# Edit .env — at minimum set GOOGLE_API_KEY (or your LLM provider keys). For browser UIs, set ALLOWED_ORIGINS if you changed frontend URLs.

# Compose V2 (recommended; ships with Docker Desktop / recent Docker Engine):
docker compose up --build

# Foreground with logs (default). For background:
# docker compose up --build -d

# Older installs with the standalone compose binary:
# docker-compose up --build
```

- API: http://localhost:8000/docs  
- UI: http://localhost:8501  

Services: **api** (FastAPI), **streamlit** (UI), **chroma** (vector DB). Data persists in the `cogniflow_data` volume.

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
  "latency_seconds": 2.3,
  "postprocess_latency_seconds": null,
  "conversation_summary": "",
  "agent_log": [
    {"node": "query_understanding", "intent": "factual", "elapsed_seconds": 0.42},
    {"node": "retrieval_router", "retrieval_strategy": "semantic", "elapsed_seconds": 0.35},
    {"node": "orchestrator", "total_latency_seconds": 2.3},
    {"node": "pipeline", "timed_node_steps": 6, "elapsed_seconds_sum_nodes": 1.85}
  ]
}
```

Each graph node appends **`elapsed_seconds`** (per-node CPU time). The final **`pipeline`** row sums those node times. On **`POST /chat`**, **`latency_seconds`** matches a full **`graph.invoke`** (reply plus summarizer and memory manager inside the graph). On **`POST /chat/stream`**, **`latency_seconds`** is wall time **until the assistant reply has finished streaming**; **`postprocess_latency_seconds`** is the extra time for summarizer + memory manager after that; the **`orchestrator`** log row includes **`response_latency_seconds`**, **`postprocess_latency_seconds`**, and **`total_latency_seconds`** (their sum). Rolling **`/metrics`** chat latency uses the stream path’s **`latency_seconds`** (response-ready time).

Questions about **chat meta** (e.g. your name, **document/file names**, what you uploaded) **skip vector retrieval** and go straight to context synthesis using conversation history — faster and fewer LLM calls on the retrieval path.

#### POST `/api/v1/documents/upload`
Upload a document for RAG ingestion.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@my_document.pdf"
```

#### GET `/api/v1/users/{user_id}/sessions`
List all sessions for a user (newest first).

#### GET `/api/v1/sessions/{session_id}/messages`
Get conversation history for a session.

#### GET `/api/v1/sessions/{session_id}/agent-logs`

Retrieve per-message agent decision logs for explainability (from assistant message metadata).

```bash
curl http://localhost:8000/api/v1/sessions/{session_id}/agent-logs
```

Response:

```json
{
  "session_id": "a1b2c3d4-...",
  "agent_logs": [
    {
      "message_id": "msg-uuid",
      "timestamp": "2025-01-01T00:00:00",
      "agent_log": [
        {"node": "query_understanding", "intent": "factual", "elapsed_seconds": 0.42},
        {"node": "retrieval_router", "retrieval_strategy": "semantic", "elapsed_seconds": 0.35}
      ]
    }
  ]
}
```

#### GET `/api/v1/stats`
System statistics (vector store counts, SQLite row counts, embedding model). If Chroma cannot open the on-disk catalog (upgrade skew, corruption, missing `_type` in config), **`vector_store.status`** is **`unavailable`** with a **`detail`** string and **`count`: 0** — the endpoint still returns **200** so the UI can load; delete **`CHROMA_PERSIST_DIR`** and re-ingest to restore counts. Dependencies pin **`posthog<6`** with Chroma because PostHog 6 broke telemetry `capture()` (see [chroma #4966](https://github.com/chroma-core/chroma/issues/4966)).

#### GET `/api/v1/metrics`
Rolling chat latency (average, p95) and request counters.

#### POST `/api/v1/chat/stream`

Server-Sent Events with **token-level streaming** during synthesis.

1. **Pre-synthesis** — Query understanding, rewriting, decomposition (when applicable), and retrieval run synchronously (~0.5–1.5s typical; depends on LLM/embeddings).
2. **Streaming synthesis** — Tokens from `model.stream()` are emitted as SSE events as they are generated (~1–3s typical).
3. **Post-synthesis** — Conversation summarizer and memory manager run after the full reply (~0.3–0.8s typical).

SSE event types:

- **`token`** — Partial assistant text (incremental).
- **`done`** — Final payload aligned with `/chat` (`response`, `sources`, `agent_log`, `latency_seconds`, `postprocess_latency_seconds`, `conversation_summary`, …). On this route, **`latency_seconds`** is time until the streamed reply is complete, not including summarizer/memory-manager work (see **`postprocess_latency_seconds`**).
- **`error`** — Error detail.

#### GET `/api/v1/health`
Liveness probe.

## Sample Conversation Flows

### Flow 1 — Multi-turn technical Q&A

Three turns show how **query rewriting** resolves anaphora and how **comparison intent** widens retrieval.

| Turn | User | What happens internally |
|------|------|-------------------------|
| 1 | “How do I set up authentication in FastAPI?” | Intent: factual. Retrieval: semantic/keyword on `sample_docs/fastapi_guide.md`. Response covers `OAuth2PasswordBearer`, token URL, and password flow patterns. |
| 2 | “What about dependency injection for the database?” | Intent: **follow_up**. Rewriter expands “the database” using prior turn → *FastAPI database dependency injection* (session + `Depends`, lifespan for engine). Retrieval targets DI sections; answer ties back to auth (e.g. `get_current_user` + DB session). |
| 3 | “Can you compare that with Django?” | Intent: **comparison**. Rewriter binds “that” to **FastAPI’s DI + auth style**. Retrieval router favors **hybrid** (RRF) to pull both FastAPI-oriented chunks and framework-contrast material (e.g. Django middleware / ORM patterns). Synthesis produces a side-by-side comparison, not a single-doc paraphrase. |

```text
User: How do I set up authentication in FastAPI?
Assistant: [OAuth2 routes, JWT outline, cites fastapi_guide.md]

User: What about dependency injection for the database?
Assistant: [Rewritten query includes FastAPI context → DI / SessionLocal / Depends]

User: Can you compare that with Django?
Assistant: [comparison intent → hybrid retrieval → FastAPI vs Django auth & DI contrast]
```

### Flow 2 — Cross-session memory

| Phase | Session | User | System behavior |
|-------|---------|------|-----------------|
| A | Session 1 | “I’m building a REST API with FastAPI and PostgreSQL.” | Memory manager persists e.g. `(context)` *Uses FastAPI + PostgreSQL for a REST API*. |
| B | Session 2 (new `session_id`, same `user_id`) | “What indexing strategy should I use?” | **User memory** is loaded into `user_memory_context` before retrieval. The assistant infers **PostgreSQL** from stored memory (not from this message alone), retrieves `postgresql_guide.md` (B-tree, GIN, partial, FTS), and answers in that stack context. |

Without cross-session memory, the question is ambiguous (Redis? MySQL?). With memory, the retriever stays anchored to **PostgreSQL**.

### Flow 3 — Conversation summarization

When the session exceeds the configured message threshold (default: **10 messages**), the **conversation summarizer** runs after the response. It produces a **rolling summary** stored on the session (e.g. goals, decisions, open questions). Subsequent turns inject this summary so the model retains gist without sending the full transcript.

Summary length is influenced by **`SUMMARY_COMPRESSION_RATIO`** (default **`0.3`**, i.e. target ~30% of the approximate raw conversation volume passed into the summarizer). Lower values produce more aggressive compression.

```text
[Messages 1–10+: routing, Docker, PostgreSQL indexes, env vars, …]

Summarizer output (illustrative):
"The user is building an e-commerce API with FastAPI and PostgreSQL.
Discussed indexing (B-tree vs GIN), Docker Compose networking, and .env handling.
Open: whether to add PgBouncer in staging."

Next user message: shorter history window + summary → stays within token budget.
```

## Performance Benchmarks

Representative timings on a typical dev machine with `sample_docs` (5–7 documents) ingested:

| Metric | Value |
|--------|-------|
| Document ingestion (5-7 docs) | ~8-12s |
| First query (cold start, embedding model load) | ~3-5s |
| Subsequent queries | ~1.5-3s |
| Memory overhead per session | ~2KB |
| Vector search latency (top-5) | ~50ms |
| Embedding generation per query | ~100ms |

Latency depends on LLM provider. Groq is fastest (~0.5s), OpenAI ~1-2s.

## Project Structure

```
CogniFlow/
├── main.py                      # FastAPI application entry point
├── config.py                    # Environment configuration
├── streamlit_app.py             # Streamlit chat UI
├── ingest_docs.py               # Document ingestion script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment template
├── docs/
│   └── architecture.md         # Mermaid architecture diagrams
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
│   ├── routes.py                # Chat, session, upload, metrics
│   ├── deps.py                  # Singletons (memory, vector, orchestrator)
│   └── metrics.py               # Rolling latency / counters
│
├── sample_docs/                 # Sample technical documents (ingest for RAG)
│   ├── fastapi_guide.md
│   ├── docker_guide.md
│   ├── langchain_rag_guide.md
│   ├── python_best_practices.md
│   ├── postgresql_guide.md
│   ├── git_workflow.md
│   └── api_security.md
│
└── tests/                       # Test suite
    ├── test_core.py             # Model + memory store + processor tests
    └── test_api.py              # API endpoint tests
```

## Key Design Decisions

1. **ChromaDB over Pinecone/Weaviate**: Zero-config local setup, great for prototyping. Swap-ready via `VectorStore` abstraction.
2. **SQLite over PostgreSQL**: Simpler deployment. The `MemoryStore` class is database-agnostic — swap the connector for production.
3. **sentence-transformers for embeddings**: Runs locally, no API costs. The `all-MiniLM-L6-v2` model is fast and produces good 384-dim embeddings.
4. **Gemini as default LLM**: Native Google Generative AI (`gemini-2.5-flash` by default). Switch via `LLM_PROVIDER` to OpenRouter, Groq, OpenAI, Anthropic, or Ollama.
5. **Hybrid search with RRF**: Reciprocal Rank Fusion combines semantic understanding with keyword precision.

## Running Tests

Use a virtualenv with all dependencies from `requirements.txt` (including `python-multipart` for upload routes):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
```

Coverage (targets **agents**, **core**, **api**; HTML report under `htmlcov/`):

```bash
make cov
# or:
pytest tests/ --cov=agents --cov=core --cov=api --cov-report=term-missing --cov-report=html -v
```

## Performance benchmarks

Measured on this repo’s dev setup (single machine; numbers vary with CPU, disk, and LLM latency):

| Metric | Observed |
|--------|----------|
| `/api/v1/health` (warm, local) | ~11–15 ms total HTTP time (`curl -w '%{time_total}'`) |
| Full `/chat` pipeline | Dominated by LLM + embedding latency; inspect `agent_log` `elapsed_seconds` per node |
| SQLite session messages | Typically sub-ms per query on local SSD |
| Chroma semantic query | Depends on collection size and embedding backend (`local` vs OpenAI) |

Reproduce health timing: `curl -s -o /dev/null -w "%{time_total}s\n" http://127.0.0.1:8000/api/v1/health`

## License

MIT
