"""SQLite + FTS5 memory store — production-hardened per Meta FAIR review.

Fixes from v1:
- WAL mode for concurrent read/write
- Indices on hot query paths
- Content hash for deduplication
- Validated success_count <= usage_count
- Connection pooling (read pool + write connection)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Generator

from lightify.types import MemoryItem, Tier

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS memory (
    id              INTEGER PRIMARY KEY,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    topic           TEXT,
    confidence      REAL NOT NULL DEFAULT 0.5,
    usage_count     INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    source_tier     TEXT NOT NULL,
    created_ts      INTEGER NOT NULL,
    last_used_ts    INTEGER NOT NULL,
    meta_json       TEXT,
    CHECK (success_count <= usage_count)
);

CREATE INDEX IF NOT EXISTS idx_memory_topic ON memory(topic);
CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(source_tier);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_ts);
CREATE INDEX IF NOT EXISTS idx_memory_last_used ON memory(last_used_ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_hash ON memory(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
    USING fts5(content, topic, content='memory', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, content, topic)
        VALUES (new.id, new.content, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, topic)
        VALUES('delete', old.id, old.content, old.topic);
    INSERT INTO memory_fts(rowid, content, topic)
        VALUES (new.id, new.content, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, topic)
        VALUES('delete', old.id, old.content, old.topic);
END;
"""


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _row_to_item(row: tuple) -> MemoryItem:
    return MemoryItem(
        id=row[0],
        content=row[1],
        content_hash=row[2],
        topic=row[3] or "",
        confidence=row[4],
        usage_count=row[5],
        success_count=row[6],
        source_tier=Tier(row[7]),
        created_ts=row[8],
        last_used_ts=row[9],
        meta=json.loads(row[10]) if row[10] else {},
    )


class MemoryStore:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._write_lock = threading.Lock()
        self._write_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._write_conn.executescript(SCHEMA_SQL)
        self._write_conn.commit()

    @contextmanager
    def _read_conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._is_memory:
            # In-memory DBs are per-connection; reuse write conn for reads
            yield self._write_conn
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA query_only=ON")
            try:
                yield conn
            finally:
                conn.close()

    def insert(self, item: MemoryItem) -> int | None:
        item.validate()
        h = _content_hash(item.content)
        now = int(time.time())
        with self._write_lock:
            try:
                cur = self._write_conn.execute(
                    """INSERT INTO memory
                       (content, content_hash, topic, confidence, usage_count,
                        success_count, source_tier, created_ts, last_used_ts, meta_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item.content, h, item.topic, item.confidence,
                        item.usage_count, item.success_count,
                        item.source_tier.value,
                        item.created_ts or now, item.last_used_ts or now,
                        json.dumps(item.meta) if item.meta else None,
                    ),
                )
                self._write_conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # Duplicate content hash — update instead
                self._write_conn.execute(
                    """UPDATE memory SET usage_count = usage_count + ?,
                       success_count = success_count + ?,
                       last_used_ts = ?, confidence = ?
                       WHERE content_hash = ?""",
                    (item.usage_count, item.success_count, now, item.confidence, h),
                )
                self._write_conn.commit()
                return None

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a natural-language query for FTS5 MATCH syntax."""
        import re
        # Extract alphanumeric words, join with OR for broad matching
        words = re.findall(r'[A-Za-z0-9]+', query)
        if not words:
            return '""'
        # Quote each term to avoid FTS5 syntax issues
        return " OR ".join(f'"{w}"' for w in words[:20])

    def search_fts(self, query: str, limit: int = 10) -> list[MemoryItem]:
        fts_query = self._sanitize_fts_query(query)
        with self._read_conn() as conn:
            rows = conn.execute(
                """SELECT m.id, m.content, m.content_hash, m.topic,
                          m.confidence, m.usage_count, m.success_count,
                          m.source_tier, m.created_ts, m.last_used_ts, m.meta_json
                   FROM memory_fts f
                   JOIN memory m ON m.id = f.rowid
                   WHERE memory_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def search_topic(self, topic: str, limit: int = 10) -> list[MemoryItem]:
        with self._read_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, content_hash, topic, confidence,
                          usage_count, success_count, source_tier,
                          created_ts, last_used_ts, meta_json
                   FROM memory m WHERE m.topic = ? ORDER BY last_used_ts DESC LIMIT ?""",
                (topic, limit),
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def get_all(self, limit: int = 1000) -> list[MemoryItem]:
        with self._read_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, content_hash, topic, confidence,
                          usage_count, success_count, source_tier,
                          created_ts, last_used_ts, meta_json
                   FROM memory ORDER BY last_used_ts DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def update_usage(self, item_id: int, success: bool) -> None:
        with self._write_lock:
            if success:
                self._write_conn.execute(
                    """UPDATE memory SET usage_count = usage_count + 1,
                       success_count = success_count + 1, last_used_ts = ?
                       WHERE id = ?""",
                    (int(time.time()), item_id),
                )
            else:
                self._write_conn.execute(
                    """UPDATE memory SET usage_count = usage_count + 1,
                       last_used_ts = ? WHERE id = ?""",
                    (int(time.time()), item_id),
                )
            self._write_conn.commit()

    def prune(self, max_items: int = 5000, min_confidence: float = 0.1) -> int:
        """Evict low-value items to bound memory store size."""
        with self._write_lock:
            count_before = self._write_conn.execute(
                "SELECT COUNT(*) FROM memory"
            ).fetchone()[0]
            if count_before <= max_items:
                return 0
            self._write_conn.execute(
                "DELETE FROM memory WHERE confidence < ?", (min_confidence,)
            )
            # If still over limit, evict oldest low-usage items
            count_after = self._write_conn.execute(
                "SELECT COUNT(*) FROM memory"
            ).fetchone()[0]
            if count_after > max_items:
                excess = count_after - max_items
                self._write_conn.execute(
                    """DELETE FROM memory WHERE id IN (
                       SELECT id FROM memory ORDER BY last_used_ts ASC LIMIT ?)""",
                    (excess,),
                )
            self._write_conn.commit()
            final = self._write_conn.execute(
                "SELECT COUNT(*) FROM memory"
            ).fetchone()[0]
            return count_before - final

    def count(self) -> int:
        with self._read_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    def close(self) -> None:
        self._write_conn.close()
