"""Tests for MemoryStore pruning strategies."""

from __future__ import annotations

import pytest

from core.memory_store import MemoryStore, PruningStrategy


@pytest.fixture
def ms(tmp_path):
    return MemoryStore(db_path=str(tmp_path / "mem.db"))


def test_prune_sliding_window_keeps_recent(ms: MemoryStore):
    uid = "u_slide"
    for i in range(55):
        ms.store_user_memory(uid, "context", f"m{i}")
    ms.prune_old_memories(uid, keep_count=50, strategy=PruningStrategy.SLIDING_WINDOW)
    conn = ms._get_conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (uid,)
    ).fetchone()[0]
    assert int(n) == 50


def test_prune_summary_buffer_folds_old(ms: MemoryStore):
    uid = "u_sum"
    for i in range(55):
        ms.store_user_memory(uid, "context", f"m{i}")
    ms.prune_old_memories(uid, keep_count=50, strategy=PruningStrategy.SUMMARY_BUFFER)
    conn = ms._get_conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (uid,)
    ).fetchone()[0]
    assert int(n) == 51
    row = conn.execute(
        "SELECT COUNT(*) FROM user_memory WHERE user_id = ? AND memory_type = 'summary'",
        (uid,),
    ).fetchone()[0]
    assert int(row) >= 1


def test_default_prune_relevance(ms: MemoryStore):
    uid = "u_rel"
    for i in range(55):
        ms.store_user_memory(uid, "context", f"m{i}")
    ms.prune_old_memories(uid, keep_count=50)
    conn = ms._get_conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (uid,)
    ).fetchone()[0]
    assert int(n) == 50
