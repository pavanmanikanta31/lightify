"""Generate synthetic benchmark datasets for Lightify evaluation.

Creates:
- Memory items with known topics, confidence levels, and relationships
- Query profiles with known difficulty, expected answers, and characteristics
- Planted contradictions for MCD testing
"""
from __future__ import annotations

import random
import time

from lightify.models.simulated import QueryProfile
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import MemoryItem, Tier

TOPICS = [
    "python", "rust", "javascript", "database", "security",
    "caching", "networking", "testing", "deployment", "api",
]

# Seed data: each entry is (content, topic, tier, usage, success)
SEED_ITEMS = [
    # High-confidence, well-used items
    ("Python GIL prevents true thread parallelism; use multiprocessing or asyncio for CPU-bound concurrency", "python", Tier.FRONTIER, 100, 95),
    ("Rust ownership model guarantees memory safety at compile time without garbage collection", "rust", Tier.FRONTIER, 80, 78),
    ("PostgreSQL supports JSONB indexing with GIN indexes for efficient document queries", "database", Tier.FRONTIER, 60, 55),
    ("TLS 1.3 reduces handshake to 1-RTT and supports 0-RTT resumption for repeat connections", "security", Tier.FRONTIER, 50, 48),
    ("Redis supports 5 data structures: strings, lists, sets, sorted sets, and hashes", "caching", Tier.MID, 90, 85),
    ("REST APIs should use HTTP status codes correctly: 200 OK, 201 Created, 404 Not Found", "api", Tier.MID, 70, 65),

    # Medium-confidence items
    ("JavaScript event loop processes microtasks before macrotasks in each iteration", "javascript", Tier.MID, 30, 20),
    ("Docker containers share the host kernel; VMs have their own kernel", "deployment", Tier.MID, 25, 18),
    ("Unit tests should be fast, isolated, and deterministic", "testing", Tier.SMALL, 40, 30),
    ("TCP uses 3-way handshake: SYN, SYN-ACK, ACK", "networking", Tier.SMALL, 35, 25),

    # Low-confidence, newer items
    ("Python 3.13 introduced experimental JIT compilation", "python", Tier.SMALL, 5, 3),
    ("SQLite WAL mode allows concurrent readers with a single writer", "database", Tier.SMALL, 8, 4),

    # Planted contradictions for MCD testing
    ("Redis supports 5 data structures for caching", "caching", Tier.MID, 20, 15),
    ("Redis supports 8 data structures including streams and bitmaps", "caching", Tier.SMALL, 10, 5),
    ("Python GIL allows true parallel threading for IO-bound tasks", "python", Tier.SMALL, 3, 1),
    # ^ contradicts the first item about GIL preventing parallelism

    # Code-containing items (for compression testing)
    ("FastAPI endpoint example:\n```python\n@app.get('/items/{id}')\nasync def read_item(id: int):\n    return {'item_id': id}\n```", "api", Tier.MID, 15, 12),
    ("Rust error handling pattern:\n```rust\nfn parse(s: &str) -> Result<i32, ParseIntError> {\n    s.parse::<i32>()\n}\n```", "rust", Tier.MID, 12, 10),
]

# Query profiles for benchmarking
BENCHMARK_QUERIES = [
    # Easy factual lookups (should route to Tier-1)
    QueryProfile(difficulty=0.1, requires_reasoning=False, multi_hop=False,
                 expected_answer="Python GIL prevents true thread parallelism"),
    QueryProfile(difficulty=0.1, requires_reasoning=False, multi_hop=False,
                 expected_answer="Rust ownership guarantees memory safety"),
    QueryProfile(difficulty=0.15, requires_reasoning=False, multi_hop=False,
                 expected_answer="TLS 1.3 reduces handshake to 1-RTT"),
    QueryProfile(difficulty=0.2, requires_reasoning=False, multi_hop=False,
                 expected_answer="Redis supports strings, lists, sets, sorted sets, hashes"),

    # Moderate tasks (should route to Tier-2)
    QueryProfile(difficulty=0.4, requires_reasoning=True, multi_hop=False,
                 expected_answer="Use asyncio for IO concurrency, multiprocessing for CPU parallelism"),
    QueryProfile(difficulty=0.45, requires_reasoning=True, multi_hop=False,
                 expected_answer="PostgreSQL JSONB with GIN indexes"),
    QueryProfile(difficulty=0.5, requires_reasoning=True, multi_hop=False,
                 expected_answer="Docker shares kernel; VMs isolated"),
    QueryProfile(difficulty=0.55, requires_reasoning=True, multi_hop=False,
                 expected_answer="Event loop: microtasks before macrotasks"),

    # Hard multi-hop reasoning (should route to Tier-3)
    QueryProfile(difficulty=0.7, requires_reasoning=True, multi_hop=True,
                 expected_answer="Compare Rust ownership with Python GC for memory safety tradeoffs"),
    QueryProfile(difficulty=0.75, requires_reasoning=True, multi_hop=True,
                 expected_answer="TLS 1.3 + REST API design for secure endpoints"),
    QueryProfile(difficulty=0.8, requires_reasoning=True, multi_hop=True,
                 expected_answer="Redis caching + PostgreSQL persistence in hybrid architecture"),
    QueryProfile(difficulty=0.9, requires_reasoning=True, multi_hop=True,
                 expected_answer="Full-stack security: TLS + auth + input validation + rate limiting"),

    # Conflict-triggering queries (should detect MCD conflicts)
    QueryProfile(difficulty=0.5, requires_reasoning=True, multi_hop=False,
                 expected_answer="Redis supports 5 core + 3 extended data structures"),
    QueryProfile(difficulty=0.6, requires_reasoning=True, multi_hop=False,
                 expected_answer="GIL prevents thread parallelism but allows IO concurrency"),
]

QUERY_STRINGS = [
    # Easy
    "How does Python handle threading and the GIL?",
    "What is Rust's approach to memory safety?",
    "What improvements does TLS 1.3 bring?",
    "What data structures does Redis support?",
    # Moderate
    "When should I use asyncio vs multiprocessing in Python?",
    "How do I efficiently query JSON data in PostgreSQL?",
    "What's the difference between Docker containers and VMs?",
    "How does the JavaScript event loop process tasks?",
    # Hard multi-hop
    "Compare Rust and Python approaches to memory management and safety",
    "Design a secure REST API with TLS 1.3 best practices",
    "Architecture a caching layer with Redis and PostgreSQL for persistence",
    "What's a comprehensive security strategy for a web application?",
    # Conflict-triggering
    "How many data structures does Redis actually support?",
    "Can Python achieve true parallelism with the GIL?",
]


def seed_memory(store: MemoryStore) -> int:
    """Insert seed data into memory store. Returns count inserted."""
    now = int(time.time())
    count = 0
    for content, topic, tier, usage, success in SEED_ITEMS:
        age_offset = random.randint(0, 30 * 86400)  # 0-30 days old
        item = MemoryItem(
            content=content,
            topic=topic,
            source_tier=tier,
            usage_count=usage,
            success_count=success,
            created_ts=now - age_offset,
            last_used_ts=now - random.randint(0, 7 * 86400),
        )
        result = store.insert(item)
        if result is not None:
            count += 1
    return count
