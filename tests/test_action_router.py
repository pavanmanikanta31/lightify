"""Tests for per-action tier routing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.action_router import (
    ActionDecision,
    classify_action,
    combine_tiers,
)
from lightify.types import Tier


def assert_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_bash_patterns_route_small():
    queries = [
        "ls -la /tmp",
        "kubectl get pods",
        "git status",
        "docker ps --filter 'status=running'",
        "what does the -a flag do",
        "how do i list files",
    ]
    for q in queries:
        d = classify_action(q)
        assert_eq(d.suggested_tier, Tier.SMALL, f"bash-like: {q!r}")
        assert_eq(d.action_class, "bash_or_shell_like", f"bash-like: {q!r}")


def test_lookup_patterns_route_small():
    queries = [
        "what is TLS?",
        "define idempotent",
        "who is the CEO of Anthropic",
        "when was Python 3 released",
        "how many bytes in a kilobyte",
    ]
    for q in queries:
        d = classify_action(q)
        assert_eq(d.suggested_tier, Tier.SMALL, f"lookup: {q!r}")
        assert_eq(d.action_class, "short_lookup", f"lookup: {q!r}")


def test_reasoning_patterns_route_mid():
    queries = [
        "compare CDDR and RouteLLM approaches",
        "why does prompt compression reduce quality when retrieval fails",
        "how would you design a retry budget for cascading tiers",
        "explain in detail the routing policy",
        "pros and cons of local inference",
    ]
    for q in queries:
        d = classify_action(q)
        assert_eq(d.suggested_tier, Tier.MID, f"reasoning: {q!r}")
        assert_eq(d.action_class, "reasoning", f"reasoning: {q!r}")


def test_code_routes_mid_unless_scope_hint():
    small_code = "write a function that reverses a string"
    d = classify_action(small_code)
    assert_eq(d.suggested_tier, Tier.MID)
    assert_eq(d.action_class, "code")

    large_code = "refactor the entire authentication module"
    d = classify_action(large_code)
    assert_eq(d.suggested_tier, Tier.FRONTIER)
    assert_eq(d.action_class, "large_code")


def test_unknown_defers_to_cddr():
    d = classify_action("bananas are good")
    assert_eq(d.action_class, "unknown")


def test_combine_tiers_cheap_action_downgrades():
    # CDDR picked MID but action is clearly bash — we want SMALL.
    out = combine_tiers(Tier.MID, Tier.SMALL, "bash_or_shell_like")
    assert_eq(out, Tier.SMALL)


def test_combine_tiers_cheap_action_respects_frontier_only_if_cddr_cheaper():
    # If CDDR is already SMALL, keep it.
    out = combine_tiers(Tier.SMALL, Tier.SMALL, "bash_or_shell_like")
    assert_eq(out, Tier.SMALL)
    # If CDDR picked FRONTIER (e.g. conflicts present) and action says SMALL,
    # SMALL wins under the cheap-action policy.
    out = combine_tiers(Tier.FRONTIER, Tier.SMALL, "bash_or_shell_like")
    assert_eq(out, Tier.SMALL)


def test_combine_tiers_expensive_action_takes_max():
    # For code/reasoning, we take max (most expensive) so we don't downgrade.
    out = combine_tiers(Tier.SMALL, Tier.MID, "code")
    assert_eq(out, Tier.MID)
    out = combine_tiers(Tier.SMALL, Tier.FRONTIER, "large_code")
    assert_eq(out, Tier.FRONTIER)
    out = combine_tiers(Tier.MID, Tier.FRONTIER, "reasoning")
    assert_eq(out, Tier.FRONTIER)
    # If CDDR already picked higher, keep CDDR.
    out = combine_tiers(Tier.FRONTIER, Tier.MID, "code")
    assert_eq(out, Tier.FRONTIER)


if __name__ == "__main__":
    funcs = [
        test_bash_patterns_route_small,
        test_lookup_patterns_route_small,
        test_reasoning_patterns_route_mid,
        test_code_routes_mid_unless_scope_hint,
        test_unknown_defers_to_cddr,
        test_combine_tiers_cheap_action_downgrades,
        test_combine_tiers_cheap_action_respects_frontier_only_if_cddr_cheaper,
        test_combine_tiers_expensive_action_takes_max,
    ]
    passed = 0
    failed = 0
    for f in funcs:
        try:
            f()
            print(f"  ok    {f.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {f.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
