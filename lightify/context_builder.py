"""Context builder — filter, rank (normalized), compress, format.

Fixes from v1:
- Topic matching uses word-boundary check, not substring
- Scoring uses Jaccard similarity (normalized), not raw count
- Integrates code-block-aware compression
"""
from __future__ import annotations

import re
from typing import Any

from lightify.compression import compress
from lightify.confidence import compute_confidence
from lightify.types import ContextCapsule, MemoryItem

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _tokenize(s: str) -> set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(s or ""))


def filter_items(items: list[MemoryItem], query: str) -> list[MemoryItem]:
    """Filter items by topic (word-boundary) or content keyword overlap."""
    query_lower = query.lower()
    query_tokens = _tokenize(query)
    out: list[MemoryItem] = []

    for item in items:
        topic = item.topic.lower()
        # Word-boundary topic match (not substring)
        if topic and re.search(rf'\b{re.escape(topic)}\b', query_lower):
            out.append(item)
        elif query_tokens & _tokenize(item.content):
            out.append(item)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def score_item(item: MemoryItem, query: str) -> float:
    """Score item relevance — all components normalized to [0, 1]."""
    import time

    q_tokens = _tokenize(query)
    c_tokens = _tokenize(item.content)

    # Jaccard similarity (normalized, replaces raw keyword count)
    keyword_score = _jaccard(q_tokens, c_tokens)

    # Recency score (hour-scale decay)
    now = int(time.time())
    hours_since = max(0, (now - item.last_used_ts)) / 3600.0
    recency_score = 1.0 / (1.0 + hours_since)

    # Confidence score
    conf_score = compute_confidence(item, calibrated=False)

    # Weighted combination — all in [0, 1]
    return 0.5 * keyword_score + 0.3 * conf_score + 0.2 * recency_score


def build_context(
    query: str,
    candidates: list[MemoryItem],
    top_k: int = 5,
) -> ContextCapsule:
    """Full Step-3 pipeline: filter → rank → compress → format."""
    filtered = filter_items(candidates, query)
    ranked = sorted(filtered, key=lambda it: score_item(it, query), reverse=True)[:top_k]

    compressed = [compress(item.content) for item in ranked if item.content]
    confidences = [compute_confidence(item, calibrated=False) for item in ranked]
    context_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Coverage: fraction of query tokens found in context
    q_tokens = _tokenize(query)
    all_c_tokens: set[str] = set()
    for item in ranked:
        all_c_tokens |= _tokenize(item.content)
    coverage = len(q_tokens & all_c_tokens) / len(q_tokens) if q_tokens else 0.0

    # Build prompt with structural separation (Anthropic reviewer's recommendation)
    ctx_block = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(compressed))
    prompt = (
        f"<context confidence=\"{context_conf:.2f}\">\n{ctx_block}\n</context>\n\n"
        f"<query>\n{query}\n</query>"
    )

    return ContextCapsule(
        prompt=prompt,
        raw_items=ranked,
        compressed_items=compressed,
        context_confidence=context_conf,
        num_items=len(ranked),
        coverage=coverage,
    )
