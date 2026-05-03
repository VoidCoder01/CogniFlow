from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """Rewrite the user's query into a standalone search query for a technical knowledge base.

RULES:
1. Resolve pronouns and references using the conversation history.
2. Term preservation (critical for lexical/embedding match): keep product names, version numbers, error codes, API names, file paths, and distinctive tokens exactly as written in the user's message unless history explicitly substitutes a synonym. Do not replace them with generic paraphrases (e.g. keep "CORS" not "cross-origin resource sharing", keep "OAuth2" not "authentication protocol").
3. Keep the rewritten query concise — a single sentence or phrase, not a paragraph.
4. Output ONLY the rewritten query text. No quotes, preamble, or explanation.

EXAMPLES:
History: "Tell me about FastAPI middleware" → User: "How does it handle errors?"
Rewrite: "How does FastAPI middleware handle errors?"

History: "I'm using PostgreSQL 15" → User: "What about connection pooling for it?"
Rewrite: "PostgreSQL 15 connection pooling"

History: (none) → User: "What is CORS?"
Rewrite: "What is CORS?" """


def query_rewriting_node(state: CogniFlowState) -> dict[str, Any]:
    """Rewrite elliptical user queries using recent conversation context."""
    t0 = time.perf_counter()
    model = get_chat_model()
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-10:]
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Recent conversation:\n{hist_snip or '(empty)'}\n\n"
                f"Original query:\n{state.get('user_query', '')}\n\n"
                "Preserve distinctive technical tokens from the query and history verbatim in the rewrite "
                "unless history clearly renames them."
            )
        ),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = with_log_timing(
        {"node": "query_rewriting", "rewritten_query": text}, t0
    )
    logger.debug("query_rewriting: %s", log_entry)
    return {
        "rewritten_query": text,
        "agent_log": [log_entry],
    }
