from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.prompt_suite import MASTER_ROUTER_SYSTEM
from agents.schemas import QueryRouterResult
from core.llm_provider import get_chat_model
from core.models import QueryIntent

logger = logging.getLogger(__name__)

_ROUTER_INTENTS = frozenset(
    {
        "factual_doc",
        "general_knowledge",
        "follow_up",
        "meta",
        "preference",
        "session_recall",
        "multi_part",
    }
)


def _wants_contextual_rewrite_for_retrieval(user_q: str, hist_snip: str, intent: str) -> bool:
    """True when a factual_doc turn should rewrite the search query using chat context."""
    if intent != "factual_doc":
        return False
    if not (hist_snip and str(hist_snip).strip() and str(hist_snip).strip() != "(empty)"):
        return False
    q = f" {(user_q or '').strip().lower()} "
    pronouns = (" it ", " that ", " this ", " they ", " them ", " those ", " the same ")
    return any(p in q for p in pronouns)


def _normalize_intent(raw: str) -> str:
    """Normalize intent labels from models or tests onto stable snake_case strings."""
    v = (raw or "").strip().lower().replace("-", "_")
    if v in _ROUTER_INTENTS:
        return v
    allowed = {e.value for e in QueryIntent}
    if v in allowed:
        return v
    aliases = {
        "followup": "follow_up",
        "offtopic": "off_topic",
        "multi part": "multi_part",
    }
    if v in aliases:
        return aliases[v]
    return "general_knowledge"


def _coerce_router_output(
    out: QueryRouterResult, user_q: str, hist_snip: str
) -> dict[str, Any]:
    """Align model output with routing constraints; unknown intent → no retrieval."""
    raw = (out.intent or "").strip().lower().replace("-", "_").replace(" ", "_")
    intent = raw if raw in _ROUTER_INTENTS else "general_knowledge"
    needs_retrieval = bool(out.needs_retrieval)
    needs_memory = bool(out.needs_memory)
    response_style = out.response_style
    if raw not in _ROUTER_INTENTS:
        needs_retrieval = False
        needs_memory = False
    if intent == "session_recall":
        needs_retrieval = False
        needs_memory = False
    elif intent == "meta":
        needs_retrieval = False
        needs_memory = True
    elif intent == "preference":
        needs_retrieval = False
        needs_memory = True
    elif intent == "general_knowledge":
        needs_retrieval = False
    elif intent == "factual_doc":
        needs_retrieval = True
    elif intent == "multi_part":
        # Compound questions: decomposer → per-sub-query retrieval (see orchestrator)
        needs_retrieval = True
        needs_memory = False
    elif intent == "follow_up":
        # Suite routing: conversation-first (memory / history), not vector retrieval
        needs_memory = True
        needs_retrieval = False
    needs_history = needs_memory or intent in (
        "follow_up",
        "meta",
        "preference",
        "session_recall",
    )
    needs_rewrite = needs_retrieval and _wants_contextual_rewrite_for_retrieval(
        user_q, hist_snip, intent
    )
    return {
        "query_intent": intent,
        "needs_retrieval": needs_retrieval,
        "needs_memory": needs_memory,
        "response_style": response_style,
        "needs_history": needs_history,
        "needs_rewrite": needs_rewrite,
    }


def _state_from_heuristic(user_q: str, hist_snip: str) -> dict[str, Any]:
    """Map legacy heuristic triple to full router-shaped state (fallback path)."""
    intent_old, nh, _nrw = _heuristic_classify(user_q, hist_snip)
    style = "short"
    if intent_old == "greeting":
        return {
            "query_intent": "greeting",
            "needs_retrieval": False,
            "needs_memory": False,
            "response_style": style,
            "needs_history": False,
            "needs_rewrite": False,
        }
    if intent_old == "follow_up":
        return {
            "query_intent": "follow_up",
            "needs_retrieval": False,
            "needs_memory": True,
            "response_style": style,
            "needs_history": True,
            "needs_rewrite": False,
        }
    if intent_old in ("comparison", "factual"):
        return {
            "query_intent": "factual_doc",
            "needs_retrieval": True,
            "needs_memory": False,
            "response_style": style,
            "needs_history": nh,
            "needs_rewrite": False,
        }
    if intent_old == "general_knowledge":
        return {
            "query_intent": "general_knowledge",
            "needs_retrieval": False,
            "needs_memory": False,
            "response_style": style,
            "needs_history": nh,
            "needs_rewrite": False,
        }
    if intent_old == "multi_part":
        return {
            "query_intent": "multi_part",
            "needs_retrieval": True,
            "needs_memory": False,
            "response_style": style,
            "needs_history": False,
            "needs_rewrite": False,
        }
    return {
        "query_intent": "factual_doc",
        "needs_retrieval": True,
        "needs_memory": False,
        "response_style": style,
        "needs_history": nh,
        "needs_rewrite": False,
    }


_GREETING_TOKENS = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "hiya",
        "yo",
        "howdy",
        "sup",
        "wassup",
        "there",
        "dude",
        "mate",
        "bro",
        "sis",
        "thanks",
        "thank",
        "you",
        "bye",
        "cya",
        "later",
        "cheers",
        "morning",
        "afternoon",
        "evening",
        "good",
        "gm",
        "gn",
        "up",
        "whats",
        "what's",
        "doing",
        "going",
        "nice",
        "everyone",
    }
)


def _looks_like_lightweight_greeting(query: str) -> bool:
    """True for short casual hellos — skip the QU LLM (saves a round-trip to the provider)."""
    q = (query or "").strip().lower()
    if not q or len(q) > 96:
        return False
    if "?" in q:
        return False
    technical = (
        "error",
        "stack",
        "api",
        "code",
        "docker",
        "kubernetes",
        "k8s",
        "how do",
        "how to",
        "what is",
        "what are",
        "explain",
        "fix",
        "debug",
        "import ",
        "http",
        "sql",
        "database",
        "function",
        "class ",
        "def ",
        "endpoint",
        "upload",
        "document",
        "file",
    )
    if any(t in q for t in technical):
        return False
    words = [w.strip(".,!'\"") for w in q.split() if w.strip(".,!'\"")]
    if not words or len(words) > 8:
        return False
    for w in words:
        if w in _GREETING_TOKENS:
            continue
        if len(w) <= 2 and w.isalpha():
            continue
        return False
    return True


def _heuristic_classify(query: str, history_snippet: str) -> tuple[str, bool, bool]:
    """Rule-based fallback when LLM structured output fails."""
    q = (query or "").strip().lower()
    if not q or q in ("hi", "hello", "hey", "thanks", "thank you", "bye"):
        return "greeting", False, False
    if _looks_like_lightweight_greeting(q):
        return "greeting", False, False
    pronouns = ("it", "that", "this", "they", "them", "those", "the same")
    has_pronoun_ref = any(f" {p} " in f" {q} " for p in pronouns)
    has_history = bool(history_snippet and history_snippet.strip() != "(empty)")
    if has_pronoun_ref and has_history:
        return "follow_up", True, True
    if "compare" in q or "vs" in q or "versus" in q or "difference between" in q:
        return "comparison", True, False
    if q.count("?") > 1 or (" and " in q and "?" in q):
        return "multi_part", False, False
    _doc_signals = (
        "your document",
        "the document",
        "the pdf",
        "uploaded file",
        "my upload",
        "in the file",
        "according to the",
        "from my doc",
        "indexed",
        "in my readme",
        "the readme",
        "section ",
        "page ",
        "in the spec",
        "the spec",
        "what does my",
        "from the pdf",
    )
    if any(s in q for s in _doc_signals):
        return "factual", False, False
    return "general_knowledge", False, False


def query_understanding_node(state: CogniFlowState) -> dict[str, Any]:
    """Classify the user turn via an intelligent router (retrieval vs memory vs general knowledge)."""
    t0 = time.perf_counter()
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-8:]
    )
    user_q = state.get("user_query", "")

    if _looks_like_lightweight_greeting(user_q):
        log_entry = with_log_timing(
            {
                "node": "query_understanding",
                "intent": "greeting",
                "needs_retrieval": False,
                "needs_memory": False,
                "response_style": "short",
                "needs_history": False,
                "needs_rewrite": False,
                "fast_path": True,
            },
            t0,
        )
        return {
            "query_intent": "greeting",
            "needs_retrieval": False,
            "needs_memory": False,
            "response_style": "short",
            "needs_history": False,
            "needs_rewrite": False,
            "agent_log": [log_entry],
        }

    try:
        model = get_chat_model().with_structured_output(QueryRouterResult)
        messages = [
            SystemMessage(content=MASTER_ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Recent conversation (optional, for disambiguation):\n"
                    f"{hist_snip or '(empty)'}\n\n"
                    f'User query:\n"{user_q}"\n\n'
                    "Choose response_style: use \"short\" unless the user clearly asks for depth, "
                    'step-by-step detail, or a long answer — then use \"detailed\".'
                )
            ),
        ]
        out: QueryRouterResult = model.invoke(messages)
        patch = _coerce_router_output(out, str(user_q), hist_snip)
    except Exception as exc:
        logger.warning(
            "query_understanding structured output failed, using heuristic: %s",
            exc,
            exc_info=True,
        )
        patch = _state_from_heuristic(user_q, hist_snip)
        if patch.get("query_intent") == "factual_doc":
            patch = {
                **patch,
                "needs_rewrite": _wants_contextual_rewrite_for_retrieval(
                    str(user_q), hist_snip, "factual_doc"
                ),
            }

    log_entry = with_log_timing(
        {
            "node": "query_understanding",
            "intent": patch["query_intent"],
            "needs_retrieval": patch["needs_retrieval"],
            "needs_memory": patch["needs_memory"],
            "response_style": patch["response_style"],
            "needs_history": patch["needs_history"],
            "needs_rewrite": patch["needs_rewrite"],
        },
        t0,
    )
    logger.debug("query_understanding: %s", log_entry)
    return {**patch, "agent_log": [log_entry]}

