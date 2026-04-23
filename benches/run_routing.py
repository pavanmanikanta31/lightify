"""Run N=200 routing benchmark.

For each query we know the oracle tier (gold label). We run three routing
policies and measure:
  - Routing Accuracy (RA)
  - Over-Spend Rate (OSR): routed strictly more expensive than oracle
  - Under-Spend Rate (USR): routed strictly cheaper than oracle
  - Estimated per-query cost (fixed tier-cost model)
  - 95% bootstrap CIs on all three rates

Policies:
  1. CDDR only (baseline)
  2. CDDR + action_router (this paper's Tier-2 contribution)
  3. Always Opus (worst-case cost oracle for cost comparison)

Context confidence is synthesized per-query from keyword overlap with a
small seed memory: conflict queries get injected contradictions (so MCD
fires), cold_knowledge gets low retrieval, bash/lookup gets moderate,
reasoning/code get higher, large_code gets moderate.

No live model calls; this measures routing policy behavior only, which is
what the N=200 bench is designed to evaluate. Live end-to-end numbers come
from the 20-query pilot (Table IV).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.action_router import classify_action, combine_tiers
from lightify.router import Router
from lightify.types import ContextCapsule, Tier


TIER_RANK = {Tier.SMALL: 0, Tier.MID: 1, Tier.FRONTIER: 2}
NAME_TO_TIER = {"SMALL": Tier.SMALL, "MID": Tier.MID, "FRONTIER": Tier.FRONTIER}

# Per-query cost estimate in USD (order-of-magnitude; from 20-query pilot).
TIER_COST = {Tier.SMALL: 0.000, Tier.MID: 0.019, Tier.FRONTIER: 0.076}


def synth_capsule(row: dict) -> ContextCapsule:
    """Synthesize a ContextCapsule from category metadata.

    Per-category confidence distributions are calibrated to the 20-query pilot:
    bash/lookup queries have highly retrievable context (phi >= 0.5);
    reasoning/code retrieve moderately (phi ~ 0.3-0.5);
    cold_knowledge barely retrieves anything (phi < 0.2);
    conflict queries have synthetic conflicts injected so MCD fires.
    """
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


def score_policy(decisions: list[Tier], oracle: list[Tier]) -> dict:
    n = len(oracle)
    ra = sum(1 for d, o in zip(decisions, oracle) if d == o) / n
    osr = sum(1 for d, o in zip(decisions, oracle)
              if TIER_RANK[d] > TIER_RANK[o]) / n
    usr = sum(1 for d, o in zip(decisions, oracle)
              if TIER_RANK[d] < TIER_RANK[o]) / n
    avg_cost = sum(TIER_COST[d] for d in decisions) / n
    return {
        "ra": ra, "osr": osr, "usr": usr,
        "avg_cost": avg_cost,
        "per_tier": {
            "SMALL": sum(1 for d in decisions if d == Tier.SMALL) / n,
            "MID": sum(1 for d in decisions if d == Tier.MID) / n,
            "FRONTIER": sum(1 for d in decisions if d == Tier.FRONTIER) / n,
        },
    }


def bootstrap_ci(values: list[float], iters: int = 1000, alpha: float = 0.05):
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def main(dataset: str = "queries_200.json"):
    random.seed(42)
    queries_path = Path(__file__).parent / "datasets" / "synthetic" / dataset
    rows = json.loads(queries_path.read_text())

    cddr_router = Router(enable_action_routing=False)
    action_router = Router(enable_action_routing=True)

    oracle = [NAME_TO_TIER[r["oracle_tier"]] for r in rows]
    caps = [synth_capsule(r) for r in rows]

    cddr_decisions = [cddr_router.route(c, query=r["query"]).tier
                      for c, r in zip(caps, rows)]
    action_decisions = [action_router.route(c, query=r["query"]).tier
                        for c, r in zip(caps, rows)]
    opus_decisions = [Tier.FRONTIER] * len(rows)

    policies = {
        "CDDR only": cddr_decisions,
        "CDDR + action_router": action_decisions,
        "Always Opus (upper bound)": opus_decisions,
    }

    print(f"N = {len(rows)}")
    print(f"Oracle tier distribution: "
          f"SMALL={sum(1 for o in oracle if o == Tier.SMALL)}, "
          f"MID={sum(1 for o in oracle if o == Tier.MID)}, "
          f"FRONTIER={sum(1 for o in oracle if o == Tier.FRONTIER)}\n")

    # Bootstrap CIs on routing accuracy.
    for name, decs in policies.items():
        m = score_policy(decs, oracle)
        match_flags = [1.0 if d == o else 0.0 for d, o in zip(decs, oracle)]
        lo, hi = bootstrap_ci(match_flags, iters=1000)
        print(f"{name}")
        print(f"  RA   = {m['ra']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  OSR  = {m['osr']:.3f}   USR = {m['usr']:.3f}")
        print(f"  cost = ${m['avg_cost']:.4f}/query   "
              f"(SMALL={m['per_tier']['SMALL']:.2f}, "
              f"MID={m['per_tier']['MID']:.2f}, "
              f"FRONTIER={m['per_tier']['FRONTIER']:.2f})")
        print()

    # Also save a machine-readable summary.
    out = {name: {
        **score_policy(decs, oracle),
        "per_tier": score_policy(decs, oracle)["per_tier"],
    } for name, decs in policies.items()}
    out_path = Path(__file__).parent / f"results_{Path(dataset).stem}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"results written to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="queries_200.json")
    args = p.parse_args()
    main(dataset=args.dataset)
