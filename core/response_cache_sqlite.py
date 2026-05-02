"""SQLite persistence for ``response_cache`` (exact + semantic rows, LRU eviction)."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
import uuid
from array import array
from pathlib import Path
from typing import Any

_conn_lock = threading.Lock()


def _ensure_parent(path: str) -> None:
    p = Path(path).expanduser().resolve()
    parent = p.parent
    if parent and str(parent) not in (".", ""):
        parent.mkdir(parents=True, exist_ok=True)


def _connect(path: str) -> sqlite3.Connection:
    _ensure_parent(path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS response_cache (
          id TEXT PRIMARY KEY NOT NULL,
          session_id TEXT NOT NULL,
          user_id TEXT NOT NULL,
          context_fp TEXT NOT NULL DEFAULT '',
          norm_hash TEXT NOT NULL,
          embedding BLOB,
          response TEXT NOT NULL,
          sources_json TEXT NOT NULL,
          conversation_summary TEXT NOT NULL DEFAULT '',
          query_preview TEXT NOT NULL DEFAULT '',
          created_at REAL NOT NULL,
          accessed_at REAL NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_rc_exact
          ON response_cache(session_id, user_id, context_fp, norm_hash);
        CREATE INDEX IF NOT EXISTS ix_rc_sess ON response_cache(session_id);
        CREATE INDEX IF NOT EXISTS ix_rc_scope
          ON response_cache(session_id, user_id, context_fp);
        """
    )
    conn.commit()


def _pack_embedding(vec: list[float]) -> bytes | None:
    if not vec:
        return None
    a = array("f", vec)
    return a.tobytes()


def _unpack_embedding(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def truncate_table(path: str) -> None:
    """Tests / dev reset: drop all cached rows (keeps schema)."""
    if not path or not Path(path).exists():
        return
    with _conn_lock:
        conn = _connect(path)
        try:
            _init_schema(conn)
            conn.execute("DELETE FROM response_cache")
            conn.commit()
        finally:
            conn.close()


def get_exact_row(
    path: str,
    session_id: str,
    user_id: str,
    context_fp: str,
    norm_hash: str,
) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    nh = norm_hash
    now = time.time()
    with _conn_lock:
        conn = _connect(path)
        try:
            _init_schema(conn)
            row = conn.execute(
                """
                SELECT response, sources_json, conversation_summary
                FROM response_cache
                WHERE session_id=? AND user_id=? AND context_fp=? AND norm_hash=?
                """,
                (sid, uid, ctx, nh),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE response_cache SET accessed_at=? WHERE session_id=? AND user_id=? AND context_fp=? AND norm_hash=?",
                (now, sid, uid, ctx, nh),
            )
            conn.commit()
            return {
                "response": row["response"] or "",
                "sources": json.loads(row["sources_json"] or "[]"),
                "conversation_summary": row["conversation_summary"] or "",
            }
        finally:
            conn.close()


def semantic_best(
    path: str,
    session_id: str,
    user_id: str,
    context_fp: str,
    qvec: list[float],
    min_similarity: float,
    cosine_fn: Any,
) -> tuple[dict[str, Any], float] | None:
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    now = time.time()
    best_row: sqlite3.Row | None = None
    best_sim = -1.0
    best_key: tuple[str, str, str, str] | None = None

    with _conn_lock:
        conn = _connect(path)
        try:
            _init_schema(conn)
            rows = conn.execute(
                """
                SELECT session_id, user_id, context_fp, norm_hash,
                       embedding, response, sources_json, conversation_summary
                FROM response_cache
                WHERE session_id=? AND user_id=? AND context_fp=?
                  AND embedding IS NOT NULL AND length(embedding) >= 16
                """,
                (sid, uid, ctx),
            ).fetchall()

            for row in rows:
                emb = _unpack_embedding(row["embedding"])
                sim = cosine_fn(qvec, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row
                    best_key = (
                        row["session_id"],
                        row["user_id"],
                        row["context_fp"],
                        row["norm_hash"],
                    )

            if best_row is None or best_sim < min_similarity:
                return None

            conn.execute(
                """
                UPDATE response_cache SET accessed_at=?
                WHERE session_id=? AND user_id=? AND context_fp=? AND norm_hash=?
                """,
                (now, best_key[0], best_key[1], best_key[2], best_key[3]),
            )
            conn.commit()
            out = {
                "response": best_row["response"] or "",
                "sources": json.loads(best_row["sources_json"] or "[]"),
                "conversation_summary": best_row["conversation_summary"] or "",
            }
            return out, best_sim
        finally:
            conn.close()


def upsert_row(
    path: str,
    *,
    session_id: str,
    user_id: str,
    context_fp: str,
    norm_hash: str,
    embedding: list[float],
    response: str,
    sources: list[dict[str, Any]],
    conversation_summary: str,
    query_preview: str,
    max_rows: int,
) -> None:
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    now = time.time()
    blob = _pack_embedding(embedding)
    sources_json = json.dumps(list(sources or []), ensure_ascii=False)
    row_id = str(uuid.uuid4())

    with _conn_lock:
        conn = _connect(path)
        try:
            _init_schema(conn)
            conn.execute(
                """
                INSERT INTO response_cache (
                  id, session_id, user_id, context_fp, norm_hash, embedding,
                  response, sources_json, conversation_summary, query_preview,
                  created_at, accessed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, user_id, context_fp, norm_hash) DO UPDATE SET
                  embedding=excluded.embedding,
                  response=excluded.response,
                  sources_json=excluded.sources_json,
                  conversation_summary=excluded.conversation_summary,
                  query_preview=excluded.query_preview,
                  accessed_at=excluded.accessed_at
                """,
                (
                    row_id,
                    sid,
                    uid,
                    ctx,
                    norm_hash,
                    blob,
                    response or "",
                    sources_json,
                    conversation_summary or "",
                    query_preview or "",
                    now,
                    now,
                ),
            )
            conn.commit()

            cnt = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
            overflow = int(cnt) - max_rows
            if overflow > 0:
                victims = conn.execute(
                    """
                    SELECT id FROM response_cache
                    ORDER BY accessed_at ASC, created_at ASC
                    LIMIT ?
                    """,
                    (overflow,),
                ).fetchall()
                for v in victims:
                    conn.execute("DELETE FROM response_cache WHERE id=?", (v["id"],))
                conn.commit()
        finally:
            conn.close()


def invalidate_session(path: str, session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    if not Path(path).exists():
        return
    with _conn_lock:
        conn = _connect(path)
        try:
            _init_schema(conn)
            conn.execute("DELETE FROM response_cache WHERE session_id=?", (sid,))
            conn.commit()
        finally:
            conn.close()
