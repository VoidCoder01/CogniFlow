from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from config import settings
from core.models import ChatMessage, MessageRole, Session


class PruningStrategy:
    """Long-term memory compaction modes."""

    RELEVANCE = "relevance"
    SLIDING_WINDOW = "sliding"
    SUMMARY_BUFFER = "summary"


class MemoryStore:
    """SQLite persistence for sessions, chat messages, and user-scoped memories."""

    def __init__(self, db_path: str | None = None):
        path = db_path or settings.sqlite_db_path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection (WAL, Row factory)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the SQLite connection for the current thread."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._local.conn = None

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary    TEXT DEFAULT '',
                    metadata   TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    metadata   TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS user_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT NOT NULL,
                    memory_type     TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    metadata        TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                    ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_memory_user_id
                    ON user_memory(user_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """Insert a new chat session row."""
        session = Session(user_id=user_id)
        if session_id:
            session.session_id = session_id

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at, updated_at, summary, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.user_id,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.summary,
                json.dumps(session.metadata),
            ),
        )
        conn.commit()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Load session metadata and hydrate ``messages`` from storage."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if row is None:
            return None

        session = Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            summary=row["summary"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )
        session.messages = self.get_messages(session_id)
        return session

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """Return session rows for a user (newest ``updated_at`` first)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_peer_session_summaries(
        self,
        user_id: str,
        exclude_session_id: str,
        limit: int = 6,
    ) -> list[dict]:
        """Non-empty rolling summaries from this user's other chats (cross-session context)."""
        uid = (user_id or "").strip()
        ex = (exclude_session_id or "").strip()
        if not uid or not ex:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT session_id, summary, updated_at FROM sessions
            WHERE user_id = ? AND session_id != ?
              AND TRIM(COALESCE(summary, '')) != ''
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (uid, ex, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def update_session_summary(self, session_id: str, summary: str):
        """Persist rolling summary text on the session row."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
            (summary, datetime.utcnow().isoformat(), session_id),
        )
        conn.commit()

    def add_message(self, session_id: str, message: ChatMessage):
        """Append a chat message and bump the parent session ``updated_at``."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                session_id,
                message.role.value,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.metadata),
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        conn.commit()

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[ChatMessage]:
        """Return messages in chronological order (optional cap)."""
        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
        params: tuple = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()

        return [
            ChatMessage(
                id=row["id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def get_recent_messages(self, session_id: str, n: int = 5) -> list[ChatMessage]:
        """Return the ``n`` most recent messages in thread order."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM messages WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (session_id, n),
        ).fetchall()

        messages = [
            ChatMessage(
                id=row["id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]
        return list(reversed(messages))

    def get_message_count(self, session_id: str) -> int:
        """Count messages stored for a session."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()

        return row[0]

    def store_user_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict = None,
    ):
        """Insert a durable user memory row."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO user_memory (user_id, memory_type, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                content,
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return user memory rows ordered for display (relevance unless using sliding elsewhere)."""
        if memory_type is not None:
            query = """
                SELECT * FROM user_memory
                WHERE user_id = ? AND memory_type = ?
                ORDER BY relevance_score DESC, created_at DESC
                LIMIT ?
            """
            params = (user_id, memory_type, limit)
        else:
            query = """
                SELECT * FROM user_memory
                WHERE user_id = ?
                ORDER BY relevance_score DESC, created_at DESC
                LIMIT ?
            """
            params = (user_id, limit)

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def prune_old_memories(
        self,
        user_id: str,
        keep_count: int = 50,
        strategy: str | None = None,
    ):
        """Trim ``user_memory`` using configured or requested strategy."""
        strat = (strategy or settings.memory_pruning_strategy).lower().strip()
        conn = self._get_conn()

        if strat == PruningStrategy.SLIDING_WINDOW:
            conn.execute(
                """
                DELETE FROM user_memory
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM user_memory
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (user_id, user_id, keep_count),
            )
            conn.commit()
            return

        if strat == PruningStrategy.SUMMARY_BUFFER:
            rows = conn.execute(
                """
                SELECT id, memory_type, content FROM user_memory
                WHERE user_id = ?
                ORDER BY created_at ASC
                """,
                (user_id,),
            ).fetchall()
            if len(rows) <= keep_count:
                return
            n_fold = len(rows) - keep_count
            to_fold = rows[:n_fold]
            if not to_fold:
                return
            parts = [
                f"[{r['memory_type']}] {(r['content'] or '').strip()}" for r in to_fold
            ]
            blob = (
                "Consolidated older memories:\n"
                + "\n".join(parts)[:12000]
            ).strip()
            ids = tuple(int(r["id"]) for r in to_fold)
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"DELETE FROM user_memory WHERE user_id = ? AND id IN ({placeholders})",
                (user_id, *ids),
            )
            conn.execute(
                """
                INSERT INTO user_memory (user_id, memory_type, content, created_at, metadata)
                VALUES (?, 'summary', ?, ?, '{}')
                """,
                (user_id, blob, datetime.utcnow().isoformat()),
            )
            conn.commit()
            return

        conn.execute(
            """
            DELETE FROM user_memory
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM user_memory
                WHERE user_id = ?
                ORDER BY relevance_score DESC, created_at DESC
                LIMIT ?
            )
            """,
            (user_id, user_id, keep_count),
        )
        conn.commit()

    def table_counts(self) -> dict[str, int]:
        """Cheap sanity counts for ops dashboards."""
        conn = self._get_conn()
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        memories = conn.execute("SELECT COUNT(*) FROM user_memory").fetchone()[0]
        return {
            "sessions": int(sessions),
            "messages": int(messages),
            "user_memory_rows": int(memories),
        }
