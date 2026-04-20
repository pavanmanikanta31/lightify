"""Real benchmark: Claude WITH Lightify vs WITHOUT Lightify.

Runs diverse queries through:
1. Raw Claude Opus (baseline — no context, no routing)
2. Lightify-augmented Claude (context retrieval + compression + routing)

Logs: tokens, latency, cost, tier used, response quality, conflicts detected.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightify.models.claude_cli import invoke_claude
from lightify.pipeline_real import RealLightifyPipeline
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import Tier
from benches.generate_data import seed_memory

# ── Benchmark queries (diverse categories) ────────────────────────────────

BENCHMARKS = [
    # Category 1: Factual lookup (should be cheap with context)
    {
        "id": "fact-1",
        "category": "factual",
        "query": "What prevents true thread parallelism in Python?",
        "expected_keywords": "GIL global interpreter lock parallelism threading",
    },
    {
        "id": "fact-2",
        "category": "factual",
        "query": "How does Rust guarantee memory safety?",
        "expected_keywords": "ownership borrow checker compile time garbage",
    },

    # Category 2: Code/technical (benefits from examples in context)
    {
        "id": "code-1",
        "category": "code",
        "query": "Show me how to create a FastAPI GET endpoint that takes an integer parameter",
        "expected_keywords": "app get async def return",
    },
    {
        "id": "code-2",
        "category": "code",
        "query": "How do you handle errors in Rust using Result types?",
        "expected_keywords": "Result Ok Err parse",
    },

    # Category 3: Architecture reasoning (multi-hop, needs multiple context pieces)
    {
        "id": "arch-1",
        "category": "architecture",
        "query": "Compare Python and Rust approaches to memory management for a high-throughput data pipeline",
        "expected_keywords": "GIL ownership garbage collector performance safety",
    },
    {
        "id": "arch-2",
        "category": "architecture",
        "query": "Design a caching strategy using Redis with PostgreSQL as the persistence layer",
        "expected_keywords": "Redis cache PostgreSQL write-through invalidation TTL",
    },

    # Category 4: Security (benefits from security context in memory)
    {
        "id": "sec-1",
        "category": "security",
        "query": "What are the key improvements in TLS 1.3 over TLS 1.2?",
        "expected_keywords": "handshake RTT 0-RTT resumption cipher",
    },

    # Category 5: Conflict resolution (tests MCD — memory has contradicting items)
    {
        "id": "conflict-1",
        "category": "conflict",
        "query": "How many data structures does Redis support?",
        "expected_keywords": "strings lists sets sorted hashes streams",
    },
]


def percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


def run_benchmark():
    print("=" * 80)
    print("LIGHTIFY REAL BENCHMARK: Claude WITH vs WITHOUT Lightify")
    print("=" * 80)
    print(f"Queries: {len(BENCHMARKS)}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "lightify_memory.db")
    store = MemoryStore(db_path)
    if store.count() == 0:
        print("[setup] Seeding memory store...")
        seed_memory(store)
    print(f"[setup] Memory store: {store.count()} items")
    print()

    pipeline = RealLightifyPipeline(store)

    results_without = []
    results_sonnet = []
    results_with = []

    for i, bench in enumerate(BENCHMARKS):
        qid = bench["id"]
        query = bench["query"]
        category = bench["category"]
        expected = bench["expected_keywords"]

        print(f"[{i+1}/{len(BENCHMARKS)}] {qid} ({category})")
        print(f"  Query: {query[:70]}...")

        # ── WITHOUT Lightify (Opus baseline) ──────────────────────────
        print(f"  [opus]     Running raw Claude Opus...", end=" ", flush=True)
        r_without = pipeline.run_without_lightify(query)
        print(f"done ({r_without.total_latency_ms:.0f}ms, "
              f"{r_without.total_tokens_in}+{r_without.total_tokens_out} tok)")

        # ── ALWAYS SONNET (trivial baseline) ──────────────────────────
        print(f"  [sonnet]   Running always-Sonnet...", end=" ", flush=True)
        r_sonnet_raw = invoke_claude(prompt=query, tier=Tier.MID, max_turns=1, timeout_s=60)
        print(f"done ({r_sonnet_raw.latency_ms:.0f}ms, "
              f"{r_sonnet_raw.tokens_in}+{r_sonnet_raw.tokens_out} tok)")

        # ── WITH Lightify ─────────────────────────────────────────────
        print(f"  [with]    Running Lightify pipeline...", end=" ", flush=True)
        r_with = pipeline.run_with_lightify(query)
        tiers_str = "→".join(t.value for t in r_with.tiers_attempted)
        conflicts = len(r_with.capsule.conflicts) if r_with.capsule else 0
        print(f"done ({r_with.total_latency_ms:.0f}ms, "
              f"{r_with.total_tokens_in}+{r_with.total_tokens_out} tok, "
              f"route={tiers_str}, conflicts={conflicts})")

        results_without.append({
            "id": qid, "category": category,
            "tokens_in": r_without.total_tokens_in,
            "tokens_out": r_without.total_tokens_out,
            "latency_ms": r_without.total_latency_ms,
            "cost": r_without.total_cost,
            "tier": "opus",
            "response_preview": (r_without.response.text or "")[:200],
            "success": r_without.response.success,
        })

        results_sonnet.append({
            "id": qid, "category": category,
            "tokens_in": r_sonnet_raw.tokens_in,
            "tokens_out": r_sonnet_raw.tokens_out,
            "latency_ms": r_sonnet_raw.latency_ms,
            "cost": r_sonnet_raw.cost,
            "tier": "sonnet",
            "response_preview": (r_sonnet_raw.text or "")[:200],
            "success": r_sonnet_raw.success,
        })

        results_with.append({
            "id": qid, "category": category,
            "tokens_in": r_with.total_tokens_in,
            "tokens_out": r_with.total_tokens_out,
            "latency_ms": r_with.total_latency_ms,
            "cost": r_with.total_cost,
            "tier": tiers_str,
            "conflicts": conflicts,
            "context_confidence": r_with.capsule.context_confidence if r_with.capsule else 0,
            "context_items": r_with.capsule.num_items if r_with.capsule else 0,
            "response_preview": (r_with.response.text or "")[:200],
            "success": r_with.response.success,
        })

        print()

    # ── Aggregate results ─────────────────────────────────────────────────
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    def summarize(label, results):
        latencies = [r["latency_ms"] for r in results]
        tokens_in = [r["tokens_in"] for r in results]
        tokens_out = [r["tokens_out"] for r in results]
        costs = [r["cost"] for r in results]
        successes = sum(1 for r in results if r["success"])
        print(f"\n### {label}")
        print(f"  Success:     {successes}/{len(results)} ({successes/len(results):.0%})")
        print(f"  Latency P50: {percentile(latencies, 50):.0f} ms")
        print(f"  Latency P95: {percentile(latencies, 95):.0f} ms")
        print(f"  Tokens in:   {sum(tokens_in):,} total, {statistics.mean(tokens_in):.0f} avg")
        print(f"  Tokens out:  {sum(tokens_out):,} total, {statistics.mean(tokens_out):.0f} avg")
        print(f"  Total cost:  ${sum(costs):.6f}")

    summarize("WITHOUT Lightify (raw Claude Opus)", results_without)
    summarize("ALWAYS SONNET (trivial baseline)", results_sonnet)
    summarize("WITH Lightify (routed)", results_with)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n\n## Per-Query Comparison")
    print()
    print("| ID | Category | Without (tok/ms/$) | With (tok/ms/$) | Tier | MCD | Savings |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    total_saved_cost = 0.0
    for rw, rl in zip(results_without, results_with):
        wo_tok = rw["tokens_in"] + rw["tokens_out"]
        wi_tok = rl["tokens_in"] + rl["tokens_out"]
        wo_ms = rw["latency_ms"]
        wi_ms = rl["latency_ms"]
        wo_cost = rw["cost"]
        wi_cost = rl["cost"]
        savings = wo_cost - wi_cost
        total_saved_cost += savings
        sav_pct = (savings / wo_cost * 100) if wo_cost > 0 else 0

        print(f"| {rw['id']} | {rw['category']} | "
              f"{wo_tok}/{wo_ms:.0f}ms/${wo_cost:.5f} | "
              f"{wi_tok}/{wi_ms:.0f}ms/${wi_cost:.5f} | "
              f"{rl['tier']} | {rl.get('conflicts', 0)} | "
              f"{'saved' if savings > 0 else 'extra'} ${abs(savings):.5f} ({sav_pct:+.0f}%) |")

    wo_total_cost = sum(r["cost"] for r in results_without)
    wi_total_cost = sum(r["cost"] for r in results_with)
    print(f"\n**Total cost: WITHOUT=${wo_total_cost:.6f}, WITH=${wi_total_cost:.6f}, "
          f"Savings=${total_saved_cost:.6f} "
          f"({total_saved_cost/wo_total_cost*100:.1f}%)**" if wo_total_cost > 0
          else "\n**Cost comparison requires token count data from Claude CLI**")

    # ── Tier distribution ─────────────────────────────────────────────────
    tier_counts: dict[str, int] = {}
    for r in results_with:
        first_tier = r["tier"].split("→")[0] if "→" in r["tier"] else r["tier"]
        tier_counts[first_tier] = tier_counts.get(first_tier, 0) + 1
    print(f"\n**Tier distribution (with Lightify):** {tier_counts}")

    # ── Save detailed results ─────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"real_bench_{ts}.json")
    with open(path, "w") as f:
        json.dump({
            "timestamp": ts,
            "queries": len(BENCHMARKS),
            "without_lightify": results_without,
            "with_lightify": results_with,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {path}")

    store.close()


if __name__ == "__main__":
    run_benchmark()
