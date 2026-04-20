"""Full Lightify benchmark — Anthropic-style comparison grid.

Columns: Local | --fast | default | --quality | Sonnet | Opus
Rows: 8 task categories
Metrics: Cost, Latency, Quality
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightify.models.claude_cli import invoke_claude
from lightify.models.ollama_local import invoke_ollama
from lightify.pipeline_real import RealLightifyPipeline
from lightify.router import Router
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import Tier

# ── Benchmark queries ─────────────────────────────────────────────────────

BENCHMARKS = [
    {
        "category": "Factual recall",
        "query": "What is the Python GIL?",
        "keywords": ["GIL", "global interpreter lock", "thread", "parallelism", "multiprocessing"],
        "min_quality_len": 30,
    },
    {
        "category": "Code generation",
        "query": "Write a FastAPI GET endpoint that takes an integer ID parameter and returns a JSON item",
        "keywords": ["app", "get", "async", "def", "return", "int", "id"],
        "min_quality_len": 50,
    },
    {
        "category": "Architecture",
        "query": "Design a caching strategy using Redis with PostgreSQL as the persistence layer",
        "keywords": ["redis", "cache", "postgres", "write", "invalidat", "ttl"],
        "min_quality_len": 80,
    },
    {
        "category": "Multi-hop",
        "query": "Compare Python and Rust approaches to memory management and safety",
        "keywords": ["GIL", "ownership", "garbage", "borrow", "compile", "safety"],
        "min_quality_len": 60,
    },
    {
        "category": "Conflict",
        "query": "How many data structures does Redis support?",
        "keywords": ["string", "list", "set", "hash", "sorted", "stream"],
        "min_quality_len": 20,
    },
    {
        "category": "Security",
        "query": "What are the key improvements in TLS 1.3 over TLS 1.2?",
        "keywords": ["handshake", "rtt", "cipher", "forward", "secrecy", "0-rtt"],
        "min_quality_len": 40,
    },
    {
        "category": "Cold knowledge",
        "query": "What is quantum annealing and how does it differ from gate-based quantum computing?",
        "keywords": ["quantum", "anneal", "optimization", "qubit", "gate", "superposition"],
        "min_quality_len": 50,
    },
    {
        "category": "Long-form",
        "query": "Explain the differences between Docker containers and virtual machines in detail",
        "keywords": ["kernel", "hypervisor", "container", "isolation", "image", "overhead", "layer"],
        "min_quality_len": 80,
    },
]


def score_quality(text: str, keywords: list[str], min_len: int) -> int:
    """Score answer quality 0-100 based on keyword coverage and length."""
    if not text or not text.strip():
        return 0
    text_lower = text.lower()

    # Keyword coverage (60% of score)
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    kw_score = (found / len(keywords)) * 60 if keywords else 30

    # Length adequacy (25% of score)
    words = len(text.split())
    if words >= min_len:
        len_score = 25
    elif words >= min_len * 0.5:
        len_score = 15
    elif words > 5:
        len_score = 8
    else:
        len_score = 0

    # Coherence bonus: has structure (15% of score)
    has_structure = any(marker in text for marker in ["\n", "- ", "* ", "1.", "**", "```"])
    struct_score = 15 if has_structure else 5

    return min(100, int(kw_score + len_score + struct_score))


def run_approach(name: str, query: str, pipeline=None, tier=None, mode=None):
    """Run a single approach and return metrics."""
    t0 = time.time()

    if name == "Local":
        r = invoke_ollama(query, timeout_s=30)
        return {"cost": 0.0, "latency_ms": r.latency_ms, "text": r.text,
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "tier": "local"}

    elif name in ("Sonnet", "Opus", "Haiku"):
        tier_map = {"Haiku": Tier.SMALL, "Sonnet": Tier.MID, "Opus": Tier.FRONTIER}
        r = invoke_claude(query, tier_map[name], max_turns=1, timeout_s=120)
        return {"cost": r.cost, "latency_ms": r.latency_ms, "text": r.text,
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "tier": name.lower()}

    else:  # Lightify modes
        if mode == "fast":
            pipeline.router.tau_tier1 = 0.20
            pipeline.router.tau_tier2 = 0.10
        elif mode == "quality":
            pipeline.router.tau_tier1 = 0.90
            pipeline.router.tau_tier2 = 0.70
        else:
            pipeline.router.tau_tier1 = 0.45
            pipeline.router.tau_tier2 = 0.30

        r = pipeline.run_with_lightify(query)
        tiers = "→".join(t.value for t in r.tiers_attempted)
        return {"cost": r.total_cost, "latency_ms": r.total_latency_ms, "text": r.response.text,
                "tokens_in": r.total_tokens_in, "tokens_out": r.total_tokens_out, "tier": tiers}


def main():
    db_path = os.path.expanduser("~/.lightify/memory.db")
    store = MemoryStore(db_path)
    pipeline = RealLightifyPipeline(store)

    approaches = [
        ("Local", None),
        ("LF --fast", "fast"),
        ("Lightify", None),
        ("LF --quality", "quality"),
        ("Sonnet", None),
        ("Opus", None),
    ]

    # Collect all results
    all_results = {}  # [category][approach] = {cost, latency, quality, tier}

    for i, bench in enumerate(BENCHMARKS):
        cat = bench["category"]
        query = bench["query"]
        keywords = bench["keywords"]
        min_len = bench["min_quality_len"]
        all_results[cat] = {}

        print(f"\n[{i+1}/{len(BENCHMARKS)}] {cat}: {query[:60]}...")

        for approach_name, mode in approaches:
            print(f"  {approach_name:<14}", end=" ", flush=True)

            if approach_name == "Local":
                r = run_approach("Local", query)
            elif approach_name in ("Sonnet", "Opus"):
                r = run_approach(approach_name, query)
            else:
                r = run_approach("Lightify", query, pipeline=pipeline, mode=mode)

            quality = score_quality(r["text"], keywords, min_len)
            r["quality"] = quality
            all_results[cat][approach_name] = r

            cost_str = f"${r['cost']:.3f}" if r["cost"] > 0 else "  FREE"
            print(f"{cost_str:>8}  {r['latency_ms']:>7.0f}ms  Q={quality:>3}  [{r['tier']}]")

    store.close()

    # ── Print tables ──────────────────────────────────────────────────────
    cats = [b["category"] for b in BENCHMARKS]
    appr_names = [a[0] for a in approaches]

    def print_table(title, metric, fmt):
        print(f"\n{'='*95}")
        print(f"  {title}")
        print(f"{'='*95}")
        header = f"  {'Category':<16}" + "".join(f"{a:>14}" for a in appr_names)
        print(header)
        print(f"  {'─'*90}")
        for cat in cats:
            row = f"  {cat:<16}"
            for a in appr_names:
                val = all_results[cat][a][metric]
                row += f"{fmt(val):>14}"
            print(row)
        # Averages
        print(f"  {'─'*90}")
        row = f"  {'AVERAGE':<16}"
        for a in appr_names:
            vals = [all_results[cat][a][metric] for cat in cats]
            avg = sum(vals) / len(vals)
            row += f"{fmt(avg):>14}"
        print(row)

    print_table(
        "COST PER QUERY ($)",
        "cost",
        lambda v: f"${v:.4f}" if v > 0 else "FREE"
    )
    print_table(
        "LATENCY (ms)",
        "latency_ms",
        lambda v: f"{v:.0f}ms"
    )
    print_table(
        "ANSWER QUALITY (0-100)",
        "quality",
        lambda v: f"{v:.0f}"
    )

    # ── Save results ──────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"full_bench_{ts}.json")

    # Serialize (strip long text)
    save_data = {}
    for cat in cats:
        save_data[cat] = {}
        for a in appr_names:
            r = all_results[cat][a]
            save_data[cat][a] = {k: v for k, v in r.items() if k != "text"}
            save_data[cat][a]["answer_preview"] = (r.get("text") or "")[:200]
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
