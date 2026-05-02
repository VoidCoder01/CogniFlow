from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional
from uuid import uuid4

from config import settings
from core.models import ChatMessage, MessageRole, Session


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        path = db_path or settings.sqlite_db_path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = path
        self._init_db()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = self._get_conn()
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

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        session = Session(user_id=user_id)
        if session_id:
            session.session_id = session_id

        conn = self._get_conn()
        try:
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
        finally:
            conn.close()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def update_session_summary(self, session_id: str, summary: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, datetime.utcnow().isoformat(), session_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, message: ChatMessage):
        conn = self._get_conn()
        try:
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
        finally:
            conn.close()

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[ChatMessage]:
        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
        params: tuple = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)

        conn = self._get_conn()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM messages WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, n),
            ).fetchall()
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()

        return row[0]

    # ------------------------------------------------------------------
    # Long-term user memory
    # ------------------------------------------------------------------

    def store_user_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict = None,
    ):
        conn = self._get_conn()
        try:
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
        finally:
            conn.close()

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
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
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def prune_old_memories(self, user_id: str, keep_count: int = 50):
        conn = self._get_conn()
        try:
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
        finally:
            conn.close()
