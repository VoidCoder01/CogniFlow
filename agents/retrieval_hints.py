"""Heuristics for when a user turn should hit session-scoped document retrieval."""


def query_is_conversation_meta_not_docs(q: str) -> bool:
    """
    True when the user is asking about the chat itself — name, what they said, etc.
    These should be answered from conversation history, not rejected for weak doc retrieval.
    """
    s = (q or "").strip().lower()
    if not s:
        return False
    needles = (
        "my name",
        "what's my name",
        "what is my name",
        "whats my name",
        "who am i",
        "what did i say",
        "what i said",
        "did i say",
        "what did i ask",
        "what i asked",
        "what was my question",
        "what were my questions",
        "questions did i",
        "questions i asked",
        "did i ask you",
        "have i asked you",
        "have i asked",
        "recap my questions",
        "summarize my questions",
        "remind me what",
        "what we talked",
        "our conversation",
        "earlier i said",
        "you know my name",
        "remember my name",
        "tell me my name",
        "do you know my name",
    )
    return any(n in s for n in needles)


def query_prefers_conversation_over_vector(q: str) -> bool:
    """
    True when the answer should come from chat context (filenames, uploads, session meta),
    not vector search — e.g. 'what document did I upload', 'what is the file called'.
    """
    s = (q or "").strip().lower()
    if not s:
        return False
    needles = (
        "document name",
        "name of the document",
        "name of document",
        "what's the document",
        "what is the document",
        "what documents",
        "my documents",
        "name of the file",
        "name of file",
        "file name",
        "filename",
        "what file did i",
        "which file did i",
        "which file have i",
        "which files",
        "what files",
        "my files",
        "which document did i",
        "what did i upload",
        "what have i uploaded",
        "my uploads",
        "did i upload",
        "files did i upload",
        "file did i upload",
        "what files did i",
        "list my uploads",
        "files i attached",
        "file i attached",
        "what did we upload",
        "uploaded files",
        "names of the files",
        "name of the pdf",
        "name of my pdf",
    )
    return any(n in s for n in needles)


def query_answer_from_chat_skip_retrieval(q: str) -> bool:
    """True if we should skip the retrieval router and go straight to synthesis."""
    return query_is_conversation_meta_not_docs(q) or query_prefers_conversation_over_vector(
        q
    )


def session_has_indexed_docs(session_id: str) -> bool:
    """True if this chat session has at least one chunk in the vector store."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    try:
        from core.vector_store import VectorStore

        stats = VectorStore().get_collection_stats(session_id=sid)
        return int(stats.get("count") or 0) > 0
    except Exception:
        return False


def user_has_indexed_docs(user_id: str) -> bool:
    """True if this user has at least one chunk (any session) with user_id metadata."""
    uid = (user_id or "").strip()
    if not uid:
        return False
    try:
        from core.vector_store import VectorStore

        stats = VectorStore().get_collection_stats(user_id=uid)
        return int(stats.get("count") or 0) > 0
    except Exception:
        return False


def corpus_available_for_chat(session_id: str, user_id: str) -> bool:
    """RAG corpus for this turn: current session uploads plus same-user uploads from other chats."""
    return session_has_indexed_docs(session_id) or user_has_indexed_docs(user_id)


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
