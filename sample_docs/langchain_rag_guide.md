# RAG with LangChain — Architecture and Operations

Retrieval-Augmented Generation (RAG) grounds language model outputs in your documents. LangChain provides composable building blocks for loaders, splitters, embeddings, vector stores, retrievers, memory, and evaluation harnesses.

## Architecture overview

A typical RAG pipeline:

1. **Ingest** documents → split → embed → index.
2. **Query** → retrieve relevant chunks → assemble prompt → LLM answer with citations.

```text
Documents → Loader → Splitter → Embeddings → VectorStore
                                              ↑
User query → Retriever ──────────────────────┘ → LLM
```

Decouple components so you can swap Chroma, FAISS, or managed vector DBs without rewriting orchestration logic.

## Document loading

Loaders read PDFs, HTML, Markdown, and databases. Normalize to `Document` objects with metadata (source, title).

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
```

For PDFs, consider `PyPDFLoader` or `UnstructuredPDFLoader` depending on layout complexity.

## Text chunking strategies

Chunking trades context coherence against retrieval precision.

- **Fixed size** with overlap preserves boundary continuity.
- **Recursive** splitting respects paragraphs and headings.
- **Markdown-aware** splitting keeps sections intact.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " "],
)
chunks = splitter.split_documents(docs)
```

Tune `chunk_size` to your embedding model limits and latency budget.

## Embedding models

Embeddings map text to dense vectors for similarity search. Options include OpenAI, Cohere, local `sentence-transformers`, and multi-lingual models.

```python
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = emb.embed_documents([c.page_content for c in chunks])
```

Track embedding dimension consistency with your vector index.

## Vector stores

Vector stores persist embeddings with metadata filters. Chroma and FAISS are common for prototypes; managed stores (Pinecone, Weaviate) add ops features.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

store = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), persist_directory="./chroma")
```

## Retrieval strategies

### Semantic search

Dense retrieval using embedding similarity (cosine, inner product).

### Keyword search

BM25-style sparse retrieval excels at exact tokens and rare entities.

### Hybrid search

Combine semantic and keyword with **Reciprocal Rank Fusion (RRF)** or weighted scores to reduce single-mode failures.

```python
# Pseudocode: merge ranked lists
def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)
    return sorted(scores, key=scores.get, reverse=True)
```

### Multi-query retrieval

Generate query variants to improve recall on ambiguous questions; deduplicate results.

## Conversation memory

Chat history can be summarized or windowed to fit context limits. LangChain provides buffer, summary, and entity memories—choose based on task length and privacy.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=4, return_messages=True)
```

For RAG, inject retrieved docs into the prompt template rather than relying on memory alone.

## Evaluation metrics

Measure both retrieval and generation:

- **Context precision/recall**: Did retrieved chunks contain the answer?
- **Faithfulness**: Are claims supported by context?
- **Answer relevance**: Does the response address the question?

Use labeled QA pairs and LLM-as-judge sparingly with human spot checks.

## Production notes

- Refresh embeddings when documents change; version your index.
- Log retrieval traces for debugging bad answers.
- Rate-limit embedding APIs and batch where possible.

LangChain evolves quickly; pin versions and consult the migration guide when upgrading major releases.

## LCEL and Runnables

LangChain Expression Language chains steps with `|` composition. Retrievers and prompts become pipeline stages.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only the context.\n{context}"),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | llm
)
```

## Metadata filtering

Store structured metadata (source, section, product_id) at index time and filter during retrieval to reduce irrelevant hits.

```python
# Conceptual Chroma filter
store.similarity_search(
    query,
    k=5,
    where={"product_id": "SKU-123"},
)
```

## Parent document retriever

For small chunks with large parent context, retrieve child chunks but return parent documents to the LLM for richer grounding.

## Streaming responses

Stream tokens to improve perceived latency in UIs:

```python
for chunk in chain.stream({"question": "What is RAG?"}):
    print(chunk.content, end="", flush=True)
```

## Failure modes

- **Lost in the middle**: Models underweight middle context; place critical facts at start/end.
- **Hallucination**: Enforce cite-or-refuse policies and low temperature for factual tasks.
- **Stale index**: Automate re-ingestion on document updates.

## Offline evaluation harness

Build a JSONL dataset `{question, expected_sources, reference_answer}` and script precision@k / exact match for regression testing when you change chunking or embeddings.
