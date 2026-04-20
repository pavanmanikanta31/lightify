"""MCD — Memory Conflict Detection.

Detects contradictions across retrieved memory items using:
1. Negation patterns (X vs not-X)
2. Numerical disagreements
3. Semantic opposition (antonym pairs)

When conflicts are detected, confidence is penalized and tier is escalated.
"""
from __future__ import annotations

import re
from lightify.types import MemoryItem, ContextCapsule

# Common antonym pairs for basic semantic opposition
_ANTONYMS = {
    "true": "false", "yes": "no", "enable": "disable", "enabled": "disabled",
    "allow": "deny", "allowed": "denied", "success": "failure",
    "active": "inactive", "valid": "invalid", "safe": "unsafe",
    "increase": "decrease", "up": "down", "fast": "slow",
    "open": "closed", "start": "stop", "add": "remove",
}
# Build reverse mapping
_ANTONYMS.update({v: k for k, v in _ANTONYMS.items()})

_NEGATION_RE = re.compile(r'\b(not|no|never|cannot|don\'t|doesn\'t|isn\'t|aren\'t|won\'t|shouldn\'t)\b', re.I)
_NUMBER_RE = re.compile(r'\b(\d+(?:\.\d+)?)\b')


def _extract_claims(text: str) -> list[dict]:
    """Extract simple factual claims from text."""
    claims = []
    sentences = re.split(r'[.!?\n]', text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        has_negation = bool(_NEGATION_RE.search(sent))
        numbers = _NUMBER_RE.findall(sent)
        words = set(re.findall(r'[a-z]+', sent.lower()))
        claims.append({
            "text": sent,
            "negated": has_negation,
            "numbers": [float(n) for n in numbers],
            "words": words,
        })
    return claims


def _claims_conflict(c1: dict, c2: dict) -> str | None:
    """Check if two claims conflict. Returns reason string or None."""
    overlap = c1["words"] & c2["words"]
    if len(overlap) < 2:
        return None  # Not about the same topic

    # Negation conflict: same topic, one negated
    if c1["negated"] != c2["negated"] and len(overlap) >= 2:
        return f"negation: '{c1['text'][:50]}' vs '{c2['text'][:50]}'"

    # Numerical disagreement: same topic, different numbers
    if c1["numbers"] and c2["numbers"] and len(overlap) >= 2:
        for n1 in c1["numbers"]:
            for n2 in c2["numbers"]:
                if n1 != n2 and min(n1, n2) > 0:
                    ratio = max(n1, n2) / min(n1, n2)
                    if ratio > 1.5:  # >50% difference
                        return f"numerical: {n1} vs {n2} in '{c1['text'][:30]}'"

    # Antonym conflict
    for word in c1["words"]:
        antonym = _ANTONYMS.get(word)
        if antonym and antonym in c2["words"] and len(overlap) >= 2:
            return f"antonym: {word}/{antonym}"

    return None


def detect_conflicts(items: list[MemoryItem]) -> list[tuple[int, int, str]]:
    """Detect pairwise conflicts among memory items.

    Returns list of (item_i_id, item_j_id, reason) tuples.
    """
    conflicts: list[tuple[int, int, str]] = []
    item_claims = [(item, _extract_claims(item.content)) for item in items]

    for i, (item_i, claims_i) in enumerate(item_claims):
        for j, (item_j, claims_j) in enumerate(item_claims):
            if j <= i:
                continue
            for ci in claims_i:
                for cj in claims_j:
                    reason = _claims_conflict(ci, cj)
                    if reason:
                        conflicts.append((
                            item_i.id or i,
                            item_j.id or j,
                            reason,
                        ))
    return conflicts


def apply_conflict_penalties(capsule: ContextCapsule) -> ContextCapsule:
    """Detect conflicts in capsule and penalize confidence."""
    conflicts = detect_conflicts(capsule.raw_items)
    capsule.conflicts = conflicts

    if conflicts:
        penalty = min(0.3, 0.1 * len(conflicts))
        capsule.context_confidence = max(0.0, capsule.context_confidence - penalty)

    return capsule
