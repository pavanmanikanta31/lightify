"""Caveman-style compression with code-block awareness + SECR.

Fixes from v1:
- Preserves fenced code blocks, inline code, URLs, paths
- Only compresses prose segments
- SECR: learns shorthand rules from repeated patterns
"""
from __future__ import annotations

import re
from collections import Counter

STOPWORDS = frozenset({
    "the", "is", "are", "was", "were", "a", "an", "this", "that", "it",
    "in", "on", "for", "of", "to", "and", "with", "when", "where",
    "please", "thanks", "thank", "sure", "happy", "just", "really",
    "very", "also", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "shall", "must", "need", "let", "here", "there",
})

# Regex to identify protected regions (code blocks, inline code, URLs, paths)
_FENCED_CODE = re.compile(r'```[\s\S]*?```')
_INLINE_CODE = re.compile(r'`[^`]+`')
_URL = re.compile(r'https?://\S+')
_PATH = re.compile(r'(?:^|(?<=\s))[/~][\w./-]+|(?:^|(?<=\s))\w+(?:/\w+)+(?:\.\w+)?', re.MULTILINE)


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Find character spans that should not be compressed."""
    spans: list[tuple[int, int]] = []
    for pat in [_FENCED_CODE, _INLINE_CODE, _URL, _PATH]:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort(key=lambda s: s[0])
    # Merge overlapping spans
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _compress_prose(text: str) -> str:
    """Compress prose by removing stopwords and deduplicating."""
    tokens = re.findall(r"[A-Za-z0-9_./:-]+", text)
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        tl = tok.lower()
        if tl in STOPWORDS:
            continue
        if tl not in seen:
            out.append(tok)
            seen.add(tl)
    return " ".join(out)


def compress(text: str) -> str:
    """Compress text while preserving code blocks, URLs, and paths."""
    if not text:
        return ""

    protected = _find_protected_spans(text)
    if not protected:
        return _compress_prose(text)

    # Build output by alternating compressed prose and protected regions
    parts: list[str] = []
    prev_end = 0
    for start, end in protected:
        if prev_end < start:
            prose = text[prev_end:start]
            parts.append(_compress_prose(prose))
        parts.append(text[start:end])
        prev_end = end
    if prev_end < len(text):
        parts.append(_compress_prose(text[prev_end:]))

    return " ".join(p for p in parts if p.strip())


class SECREngine:
    """Self-Evolving Compression Rules — learns shorthands from patterns."""

    def __init__(self):
        self._phrase_counter: Counter[str] = Counter()
        self._rules: dict[str, str] = {}  # long phrase -> shorthand
        self._min_occurrences = 5
        self._min_phrase_len = 3  # words

    def observe(self, text: str) -> None:
        """Feed text to learn repeated patterns."""
        words = text.lower().split()
        for n in range(self._min_phrase_len, min(8, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                self._phrase_counter[phrase] += 1

    def evolve(self) -> int:
        """Generate new shorthand rules from frequent phrases. Returns count of new rules."""
        new_rules = 0
        for phrase, count in self._phrase_counter.most_common(100):
            if count < self._min_occurrences:
                break
            if phrase in self._rules:
                continue
            words = phrase.split()
            shorthand = "".join(w[0].upper() for w in words if len(w) > 2)
            if len(shorthand) >= 2 and len(shorthand) < len(phrase):
                self._rules[phrase] = f"[{shorthand}]"
                new_rules += 1
        return new_rules

    def apply(self, text: str) -> str:
        """Apply learned shorthand rules to compress text."""
        result = text
        for phrase, shorthand in self._rules.items():
            result = result.replace(phrase, shorthand)
        return result

    @property
    def rules(self) -> dict[str, str]:
        return dict(self._rules)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "phrases_observed": len(self._phrase_counter),
            "rules_learned": len(self._rules),
        }
