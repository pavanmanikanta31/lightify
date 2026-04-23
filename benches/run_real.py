"""Run routing + LangGraph comparison on the hand-curated real query set.

This is a smaller, legitimacy-focused companion to the synthetic bench.
Queries are hand-written to match realistic phrasing patterns; no templates.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.router import Router
from lightify.types import ContextCapsule, Tier


TIER_RANK = {Tier.SMALL: 0, Tier.MID: 1, Tier.FRONTIER: 2}
NAME_TO_TIER = {"SMALL": Tier.SMALL, "MID": Tier.MID, "FRONTIER": Tier.FRONTIER}
TIER_COST = {Tier.SMALL: 0.000, Tier.MID: 0.019, Tier.FRONTIER: 0.076}


def synth_capsule(row: dict, rng: random.Random) -> ContextCapsule:
    cat = row["category"]
    phi = {
        "bash_like": rng.uniform(0.50, 0.85),
        "short_lookup": rng.uniform(0.50, 0.85),
        "reasoning": rng.uniform(0.30, 0.55),
        "code": rng.uniform(0.30, 0.55),
        "large_code": rng.uniform(0.25, 0.45),
        "conflict": rng.uniform(0.30, 0.55),
        "cold_knowledge": rng.uniform(0.02, 0.18),
    }[cat]
    coverage = min(1.0, phi + rng.uniform(-0.1, 0.1))
    conflicts = [(0, 1, "")] if row.get("has_contradiction") else []
    return ContextCapsule(
        context_confidence=phi,
        num_items=3,
        coverage=max(0.0, coverage),
        sufficient=phi >= 0.45 and coverage >= 0.3,
        conflicts=conflicts,
    )


_CODE_KW = re.compile(
    r"\b(write|generate|refactor|fix|implement|code|function|class|test)\b", re.I)


def lg_default(q): return Tier.FRONTIER
def lg_keyword(q): return Tier.MID if _CODE_KW.search(q) else Tier.SMALL
def lg_gateway(q):
    if len(q) < 30: return Tier.SMALL
    if len(q) < 80: return Tier.MID
    return Tier.FRONTIER


def score(decisions, oracle):
    n = len(oracle)
    return {
        "ra": sum(1 for d, o in zip(decisions, oracle) if d == o) / n,
        "osr": sum(1 for d, o in zip(decisions, oracle)
                   if TIER_RANK[d] > TIER_RANK[o]) / n,
        "usr": sum(1 for d, o in zip(decisions, oracle)
                   if TIER_RANK[d] < TIER_RANK[o]) / n,
        "avg_cost": sum(TIER_COST[d] for d in decisions) / n,
    }


def main():
    path = Path(__file__).parent / "datasets" / "real" / "queries_real.json"
    rows = json.loads(path.read_text())
    n = len(rows)
    rng = random.Random(42)
    oracle = [NAME_TO_TIER[r["oracle_tier"]] for r in rows]
    caps = [synth_capsule(r, rng) for r in rows]

    r_cddr = Router(enable_action_routing=False)
    r_act = Router(enable_action_routing=True)
    cddr = [r_cddr.route(c, query=r["query"]).tier for c, r in zip(caps, rows)]
    act = [r_act.route(c, query=r["query"]).tier for c, r in zip(caps, rows)]

    policies = {
        "Lightify CDDR":              cddr,
        "Lightify CDDR + action":     act,
        "LangGraph default (Opus)":   [lg_default(r["query"]) for r in rows],
        "LangGraph keyword-branch":   [lg_keyword(r["query"]) for r in rows],
        "LangGraph length gateway":   [lg_gateway(r["query"]) for r in rows],
    }

    print(f"HAND-CURATED REAL QUERIES  N = {n}  (100% unique)\n")
    print(f"{'Policy':<28} {'RA':>6} {'OSR':>6} {'USR':>6} {'$/q':>10}")
    print("-" * 62)
    out = {}
    for name, decs in policies.items():
        s = score(decs, oracle)
        out[name] = s
        print(f"{name:<28} {s['ra']:>6.3f} {s['osr']:>6.3f} {s['usr']:>6.3f} {s['avg_cost']:>10.4f}")
    print()

    # per-category breakdown for CDDR vs CDDR+action
    per = defaultdict(lambda: {"n":0,"c":0,"a":0})
    for r, tc, ta, o in zip(rows, cddr, act, oracle):
        d = per[r["category"]]
        d["n"]+=1; d["c"]+=int(tc==o); d["a"]+=int(ta==o)
    print(f"{'category':<17}{'N':>4}{'CDDR RA':>10}{'+act RA':>10}{'delta':>8}")
    print("-"*49)
    for cat in ["bash_like","short_lookup","code","reasoning","large_code","conflict","cold_knowledge"]:
        d = per[cat]
        cra = d["c"]/d["n"]; ara = d["a"]/d["n"]
        print(f"{cat:<17}{d['n']:>4}{cra:>10.3f}{ara:>10.3f}{ara-cra:>+8.3f}")

    # savings summary
    opus = out["LangGraph default (Opus)"]["avg_cost"]
    print(f"\nSavings vs Opus baseline (${opus:.4f}):")
    for name in ("Lightify CDDR", "Lightify CDDR + action"):
        cm = out[name]["avg_cost"]
        print(f"  {name:<28}{(opus-cm)/opus*100:+.1f}%  (${opus-cm:.4f}/query)")

    (Path(__file__).parent / "results_real.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> benches/results_real.json")


if __name__ == "__main__":
    main()
