"""Benchmark runner — runs all Lightify ablation variants and produces metrics.

Variants:
1. Naive RAG (baseline)
2. Caveman-only
3. Hybrid Lightify
4. Full Lightify

Metrics per variant:
- Success rate
- Token usage (input/output)
- Latency (P50/P95/P99)
- Cost
- Tier utilization
- Conflict detection rate
- Cache hit rate
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightify.pipeline import LightifyPipeline, PipelineConfig, Variant
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import Tier
from benches.generate_data import (
    BENCHMARK_QUERIES,
    QUERY_STRINGS,
    seed_memory,
)


def percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def run_variant(variant: Variant, run_id: int = 0) -> dict:
    """Run all benchmark queries against a single variant."""
    # Fresh store for each variant (fair comparison)
    store = MemoryStore(":memory:")
    count = seed_memory(store)

    config = PipelineConfig(variant=variant, top_k=5)
    pipeline = LightifyPipeline(store, config)

    results = []
    for i, (query, profile) in enumerate(zip(QUERY_STRINGS, BENCHMARK_QUERIES)):
        result = pipeline.run(query, profile)
        results.append(result)

    # Second pass: test cache hits
    cache_hits = 0
    for query in QUERY_STRINGS[:3]:  # re-run first 3
        result = pipeline.run(query)
        if result.cache_hit:
            cache_hits += 1

    # Aggregate metrics
    successes = sum(1 for r in results if r.response.success)
    latencies = [r.total_latency_ms for r in results]
    tokens_in = [r.total_tokens_in for r in results]
    tokens_out = [r.total_tokens_out for r in results]
    costs = [r.total_cost for r in results]
    tier_counts = {Tier.SMALL: 0, Tier.MID: 0, Tier.FRONTIER: 0}
    for r in results:
        if r.tiers_attempted:
            tier_counts[r.tiers_attempted[0]] += 1
    escalations = sum(1 for r in results if len(r.tiers_attempted) > 1)
    conflicts = sum(1 for r in results if r.capsule.conflicts)

    # SECR stats (Full variant only)
    secr_rules = 0
    if variant == Variant.FULL:
        secr_rules = pipeline.secr.stats.get("rules_learned", 0)

    metrics = {
        "variant": variant.value,
        "queries": len(results),
        "success_rate": successes / len(results) if results else 0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0,
        "tokens_in_mean": statistics.mean(tokens_in) if tokens_in else 0,
        "tokens_out_mean": statistics.mean(tokens_out) if tokens_out else 0,
        "tokens_in_total": sum(tokens_in),
        "tokens_out_total": sum(tokens_out),
        "cost_total": sum(costs),
        "cost_mean": statistics.mean(costs) if costs else 0,
        "tier1_pct": tier_counts[Tier.SMALL] / len(results) if results else 0,
        "tier2_pct": tier_counts[Tier.MID] / len(results) if results else 0,
        "tier3_pct": tier_counts[Tier.FRONTIER] / len(results) if results else 0,
        "escalation_rate": escalations / len(results) if results else 0,
        "conflict_detected": conflicts,
        "cache_hits": cache_hits,
        "secr_rules_learned": secr_rules,
        "memory_items": store.count(),
    }

    store.close()
    return metrics


def format_table(all_metrics: list[dict]) -> str:
    """Format metrics as a markdown comparison table."""
    cols = [
        ("Variant", "variant", None),
        ("Success", "success_rate", lambda x: f"{x:.1%}"),
        ("P50 (ms)", "latency_p50_ms", lambda x: f"{x:.0f}"),
        ("P95 (ms)", "latency_p95_ms", lambda x: f"{x:.0f}"),
        ("P99 (ms)", "latency_p99_ms", lambda x: f"{x:.0f}"),
        ("Tok In", "tokens_in_total", lambda x: f"{x:,}"),
        ("Tok Out", "tokens_out_total", lambda x: f"{x:,}"),
        ("Cost ($)", "cost_total", lambda x: f"{x:.4f}"),
        ("Tier-1%", "tier1_pct", lambda x: f"{x:.0%}"),
        ("Tier-2%", "tier2_pct", lambda x: f"{x:.0%}"),
        ("Tier-3%", "tier3_pct", lambda x: f"{x:.0%}"),
        ("Escl", "escalation_rate", lambda x: f"{x:.0%}"),
        ("MCD", "conflict_detected", None),
        ("Cache", "cache_hits", None),
        ("SECR", "secr_rules_learned", None),
    ]

    # Header
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    rows = []
    for m in all_metrics:
        row_parts = []
        for _, key, fmt in cols:
            val = m.get(key, "")
            if fmt and val != "":
                row_parts.append(fmt(val))
            else:
                row_parts.append(str(val))
        rows.append("| " + " | ".join(row_parts) + " |")

    return "\n".join([header, sep] + rows)


def main():
    print("=" * 80)
    print("LIGHTIFY ABLATION BENCHMARK")
    print("=" * 80)
    print()

    variants = [Variant.NAIVE_RAG, Variant.CAVEMAN_ONLY, Variant.HYBRID, Variant.FULL]
    all_metrics = []

    for variant in variants:
        print(f"Running {variant.value}...", end=" ", flush=True)
        t0 = time.time()
        metrics = run_variant(variant)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        all_metrics.append(metrics)

    print()
    print("## Ablation Results")
    print()
    print(format_table(all_metrics))
    print()

    # Detailed per-variant output
    for m in all_metrics:
        print(f"\n### {m['variant']}")
        print(f"  Success rate:     {m['success_rate']:.1%}")
        print(f"  Latency P50/P95:  {m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} ms")
        print(f"  Total tokens:     {m['tokens_in_total']:,} in / {m['tokens_out_total']:,} out")
        print(f"  Total cost:       ${m['cost_total']:.4f}")
        print(f"  Tier distribution: T1={m['tier1_pct']:.0%} T2={m['tier2_pct']:.0%} T3={m['tier3_pct']:.0%}")
        print(f"  Escalations:      {m['escalation_rate']:.0%}")
        print(f"  Conflicts found:  {m['conflict_detected']}")
        print(f"  SECR rules:       {m['secr_rules_learned']}")

    # Save to JSON
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_metrics


if __name__ == "__main__":
    main()
