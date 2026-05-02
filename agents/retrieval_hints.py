"""Heuristics for when a user turn should hit session-scoped document retrieval."""


def query_suggests_document_lookup(q: str) -> bool:
    """True if the message likely needs the session's indexed documents (not pure small talk)."""
    s = (q or "").strip().lower()
    if not s:
        return False
    hints = (
        "upload",
        "uploaded",
        "document",
        "documents",
        "file",
        "files",
        "pdf",
        "markdown",
        "html",
        "indexed",
        "index",
        "chunk",
        "chunks",
        "passage",
        "section",
        "page",
        "cite",
        "citation",
        "source",
        "readme",
        "manual",
        "spec",
        "according",
        "extract",
        "summarize",
        "summary",
        "what does",
        "what is in",
        "where does",
        "which page",
        "in my",
        "the doc",
        "this doc",
        "that doc",
        "rag",
        "knowledge base",
        "kb",
    )
    if any(h in s for h in hints):
        return True
    if "?" in s and len(s) > 35:
        return True
    return False
