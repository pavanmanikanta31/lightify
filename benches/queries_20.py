"""20 diverse evaluation queries for the Lightify benchmark pipeline.

Categories:
  - easy_factual  : Single-hop recall, should be answerable by local model
  - code          : Code generation / explanation tasks
  - architecture  : System design / multi-concern reasoning
  - multi_hop     : Requires synthesizing 2+ facts
  - conflict      : Contains planted contradictions in memory store
  - cold_knowledge: Topics unlikely to be in the memory store

Each query carries metadata for oracle construction and analysis:
  - category      : One of the categories above
  - keywords      : Terms a correct answer MUST mention (used for keyword scoring)
  - expected_tier : Cheapest tier we expect the oracle to assign ("local", "sonnet", "opus")
  - difficulty    : 0.0 (trivial) to 1.0 (very hard)
  - has_contradiction : True if memory store has conflicting info for this query
"""
from __future__ import annotations

EVAL_QUERIES: list[dict] = [
    # ── Easy factual (5) ── expect local can handle ──────────────────────
    {
        "id": "easy-1",
        "category": "easy_factual",
        "query": "What prevents true thread parallelism in CPython?",
        "keywords": ["GIL", "global interpreter lock", "threading"],
        "expected_tier": "local",
        "difficulty": 0.10,
        "has_contradiction": False,
    },
    {
        "id": "easy-2",
        "category": "easy_factual",
        "query": "What is Rust's ownership model?",
        "keywords": ["ownership", "borrow", "compile", "memory safety"],
        "expected_tier": "local",
        "difficulty": 0.10,
        "has_contradiction": False,
    },
    {
        "id": "easy-3",
        "category": "easy_factual",
        "query": "What is the TCP three-way handshake?",
        "keywords": ["SYN", "SYN-ACK", "ACK", "connection"],
        "expected_tier": "local",
        "difficulty": 0.10,
        "has_contradiction": False,
    },
    {
        "id": "easy-4",
        "category": "easy_factual",
        "query": "What are HTTP status codes 200, 201, and 404?",
        "keywords": ["200", "OK", "201", "Created", "404", "Not Found"],
        "expected_tier": "local",
        "difficulty": 0.05,
        "has_contradiction": False,
    },
    {
        "id": "easy-5",
        "category": "easy_factual",
        "query": "What is the difference between unit tests and integration tests?",
        "keywords": ["unit", "integration", "isolated", "component"],
        "expected_tier": "local",
        "difficulty": 0.15,
        "has_contradiction": False,
    },

    # ── Code (3) ── need to produce valid code ───────────────────────────
    {
        "id": "code-1",
        "category": "code",
        "query": "Write a Python FastAPI GET endpoint that takes an integer ID and returns a JSON item",
        "keywords": ["app", "get", "async", "def", "int", "return"],
        "expected_tier": "sonnet",
        "difficulty": 0.35,
        "has_contradiction": False,
    },
    {
        "id": "code-2",
        "category": "code",
        "query": "Show me how to handle errors in Rust using the Result type with pattern matching",
        "keywords": ["Result", "Ok", "Err", "match"],
        "expected_tier": "sonnet",
        "difficulty": 0.40,
        "has_contradiction": False,
    },
    {
        "id": "code-3",
        "category": "code",
        "query": "Write a JavaScript async function that fetches JSON from a URL with timeout and retry logic",
        "keywords": ["async", "await", "fetch", "timeout", "retry", "try", "catch"],
        "expected_tier": "sonnet",
        "difficulty": 0.45,
        "has_contradiction": False,
    },

    # ── Architecture (3) ── multi-concern system design ──────────────────
    {
        "id": "arch-1",
        "category": "architecture",
        "query": "Design a caching strategy using Redis with PostgreSQL as the persistence layer, including cache invalidation",
        "keywords": ["Redis", "cache", "PostgreSQL", "invalidation", "TTL", "write"],
        "expected_tier": "opus",
        "difficulty": 0.70,
        "has_contradiction": False,
    },
    {
        "id": "arch-2",
        "category": "architecture",
        "query": "How would you architect a rate-limited API gateway with authentication, logging, and circuit breaking?",
        "keywords": ["rate limit", "auth", "circuit breaker", "gateway", "log"],
        "expected_tier": "opus",
        "difficulty": 0.75,
        "has_contradiction": False,
    },
    {
        "id": "arch-3",
        "category": "architecture",
        "query": "Design a distributed task queue with at-least-once delivery, dead-letter handling, and priority scheduling",
        "keywords": ["queue", "delivery", "dead-letter", "priority", "worker", "retry"],
        "expected_tier": "opus",
        "difficulty": 0.80,
        "has_contradiction": False,
    },

    # ── Multi-hop (3) ── requires combining 2+ concepts ──────────────────
    {
        "id": "multi-1",
        "category": "multi_hop",
        "query": "Compare Python and Rust approaches to memory management for a high-throughput data pipeline",
        "keywords": ["GIL", "ownership", "garbage", "throughput", "safety"],
        "expected_tier": "opus",
        "difficulty": 0.65,
        "has_contradiction": False,
    },
    {
        "id": "multi-2",
        "category": "multi_hop",
        "query": "How do TLS 1.3 improvements interact with HTTP/2 multiplexing to reduce web latency?",
        "keywords": ["TLS", "1.3", "handshake", "HTTP/2", "multiplex", "RTT"],
        "expected_tier": "sonnet",
        "difficulty": 0.55,
        "has_contradiction": False,
    },
    {
        "id": "multi-3",
        "category": "multi_hop",
        "query": "Explain how Docker's shared-kernel architecture affects security compared to VM isolation, and what mitigations exist",
        "keywords": ["kernel", "isolation", "VM", "container", "namespace", "seccomp"],
        "expected_tier": "opus",
        "difficulty": 0.60,
        "has_contradiction": False,
    },

    # ── Conflict / contradiction (3) ── memory has conflicting data ──────
    {
        "id": "conflict-1",
        "category": "conflict",
        "query": "How many data structures does Redis support?",
        "keywords": ["strings", "lists", "sets", "sorted", "hashes", "streams"],
        "expected_tier": "sonnet",
        "difficulty": 0.50,
        "has_contradiction": True,
        "contradiction_note": "Memory says both '5 data structures' and '8 data structures including streams and bitmaps'",
    },
    {
        "id": "conflict-2",
        "category": "conflict",
        "query": "Can Python achieve true thread parallelism with the GIL?",
        "keywords": ["GIL", "parallelism", "multiprocessing", "threading"],
        "expected_tier": "sonnet",
        "difficulty": 0.50,
        "has_contradiction": True,
        "contradiction_note": "Memory says GIL 'prevents' parallelism AND 'allows true parallel threading for IO-bound tasks'",
    },
    {
        "id": "conflict-3",
        "category": "conflict",
        "query": "Is Python's GIL being removed? What is the current status of nogil/free-threading?",
        "keywords": ["GIL", "free-threading", "PEP", "3.13", "nogil"],
        "expected_tier": "sonnet",
        "difficulty": 0.55,
        "has_contradiction": True,
        "contradiction_note": "Memory mentions Python 3.13 JIT but not free-threading; may generate stale or conflicting info",
    },

    # ── Cold knowledge (3) ── topics NOT in memory store ─────────────────
    {
        "id": "cold-1",
        "category": "cold_knowledge",
        "query": "What is quantum annealing and how does it differ from gate-based quantum computing?",
        "keywords": ["quantum", "annealing", "optimization", "gate", "qubit"],
        "expected_tier": "opus",
        "difficulty": 0.80,
        "has_contradiction": False,
    },
    {
        "id": "cold-2",
        "category": "cold_knowledge",
        "query": "Explain the CAP theorem and how modern distributed databases like CockroachDB and Spanner handle it",
        "keywords": ["CAP", "consistency", "availability", "partition", "Spanner"],
        "expected_tier": "opus",
        "difficulty": 0.70,
        "has_contradiction": False,
    },
    {
        "id": "cold-3",
        "category": "cold_knowledge",
        "query": "What is a Bloom filter, what are its false-positive guarantees, and when would you use a Cuckoo filter instead?",
        "keywords": ["Bloom", "false positive", "hash", "Cuckoo", "deletion"],
        "expected_tier": "sonnet",
        "difficulty": 0.60,
        "has_contradiction": False,
    },
]

# Convenience accessors
QUERY_COUNT = len(EVAL_QUERIES)
CATEGORIES = sorted(set(q["category"] for q in EVAL_QUERIES))
CONTRADICTION_QUERIES = [q for q in EVAL_QUERIES if q.get("has_contradiction")]

assert QUERY_COUNT == 20, f"Expected 20 queries, got {QUERY_COUNT}"
assert len(CONTRADICTION_QUERIES) == 3, (
    f"Expected 3 contradiction queries, got {len(CONTRADICTION_QUERIES)}"
)
