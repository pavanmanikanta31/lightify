"""Structural comparison: Lightify vs LangGraph on the same 200-query set.

LangGraph ships as a graph runtime with no cost-aware routing primitive.
Three policies that a LangGraph user might hand-code as graph edges:

  1. Single-tier (default): every node calls one model (e.g. Opus). This is
     the Quickstart example; it has no routing. We use Always-Opus as the
     representative.

  2. Complexity routing: branch on a keyword heuristic -- "code" -> Sonnet,
     everything else -> Haiku. This is the common "first pass" pattern
     described in the LangGraph examples repo. We simulate this as a
     regex classifier.

  3. User-built AI-gateway: external request-level routing via an
     Envoy/Pydantic-AI-style gateway that the user wires manually. We simulate
     this as RouteLLM-style difficulty scoring based on query length
     (proxy commonly used in published gateway demos).

Lightify is run exactly as in run_n200_routing.py (CDDR + action_router).

This comparison measures *what a LangGraph user can achieve without adding
Lightify as a dependency*; it is not a claim that LangGraph itself is
unable to route (of course it is, if the user writes the router). The
paper's claim is that Lightify ships the router as a runtime primitive.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.router import Router
from lightify.types import ContextCapsule, Tier


TIER_RANK = {Tier.SMALL: 0, Tier.MID: 1, Tier.FRONTIER: 2}
NAME_TO_TIER = {"SMALL": Tier.SMALL, "MID": Tier.MID, "FRONTIER": Tier.FRONTIER}
TIER_COST = {Tier.SMALL: 0.000, Tier.MID: 0.019, Tier.FRONTIER: 0.076}


def langgraph_default(q: str) -> Tier:
    """Quickstart: one model for everything. We choose Opus as the
    conservative default used in LangGraph documentation examples."""
    return Tier.FRONTIER


_CODE_KW = re.compile(r"\b(write|generate|refactor|fix|implement|code|function|class|test)\b", re.I)


def langgraph_complexity(q: str) -> Tier:
    """Hand-coded keyword branch a LangGraph user might write."""
    if _CODE_KW.search(q):
        return Tier.MID  # Sonnet
    return Tier.SMALL  # Haiku (cloud, but cheapest tier in this policy)


def langgraph_gateway(q: str) -> Tier:
    """Gateway-style difficulty proxy using query length (common demo approach)."""
    if len(q) < 30:
        return Tier.SMALL
    if len(q) < 80:
        return Tier.MID
    return Tier.FRONTIER


def synth_capsule(row: dict) -> ContextCapsule:
    cat = row["category"]
    phi = {
        "bash_like": random.uniform(0.50, 0.85),
        "short_lookup": random.uniform(0.50, 0.85),
        "reasoning": random.uniform(0.30, 0.55),
        "code": random.uniform(0.30, 0.55),
        "large_code": random.uniform(0.25, 0.45),
        "conflict": random.uniform(0.30, 0.55),
        "cold_knowledge": random.uniform(0.02, 0.18),
    }[cat]
    coverage = min(1.0, phi + random.uniform(-0.1, 0.1))
    conflicts = [(0, 1, "planted contradiction")] if row.get("has_contradiction") else []
    return ContextCapsule(
        prompt="",
        context_confidence=phi,
        num_items=3,
        coverage=max(0.0, coverage),
        sufficient=phi >= 0.45 and coverage >= 0.3,
        conflicts=conflicts,
    )


def score(decisions: list[Tier], oracle: list[Tier]) -> dict:
    n = len(oracle)
    ra = sum(1 for d, o in zip(decisions, oracle) if d == o) / n
    osr = sum(1 for d, o in zip(decisions, oracle)
              if TIER_RANK[d] > TIER_RANK[o]) / n
    usr = sum(1 for d, o in zip(decisions, oracle)
              if TIER_RANK[d] < TIER_RANK[o]) / n
    avg_cost = sum(TIER_COST[d] for d in decisions) / n
    return {"ra": ra, "osr": osr, "usr": usr, "avg_cost": avg_cost}


def main(dataset: str = "queries_200.json"):
    random.seed(42)
    rows = json.loads((Path(__file__).parent / "datasets" / "synthetic" / dataset).read_text())
    oracle = [NAME_TO_TIER[r["oracle_tier"]] for r in rows]
    caps = [synth_capsule(r) for r in rows]

    lightify_router = Router(enable_action_routing=True)
    lightify_decisions = [lightify_router.route(c, query=r["query"]).tier
                          for c, r in zip(caps, rows)]

    policies = {
        "LangGraph default (Opus only)": [langgraph_default(r["query"]) for r in rows],
        "LangGraph keyword-branch": [langgraph_complexity(r["query"]) for r in rows],
        "LangGraph + gateway (length)": [langgraph_gateway(r["query"]) for r in rows],
        "Lightify (CDDR + action)": lightify_decisions,
    }

    print(f"Structural comparison on N={len(rows)} synthetic queries\n")
    print(f"{'Policy':<35} {'RA':>6} {'OSR':>6} {'USR':>6} {'$/query':>10}")
    print("-" * 70)
    for name, decs in policies.items():
        s = score(decs, oracle)
        print(f"{name:<35} {s['ra']:>6.3f} {s['osr']:>6.3f} {s['usr']:>6.3f} {s['avg_cost']:>10.4f}")

    out_path = Path(__file__).parent / f"results_langgraph_{Path(dataset).stem}.json"
    out_path.write_text(json.dumps(
        {name: score(decs, oracle) for name, decs in policies.items()},
        indent=2,
    ))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="queries_200.json")
    args = p.parse_args()
    main(dataset=args.dataset)
