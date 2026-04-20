"""Grid benchmark: 6 approaches × 20 queries — all metrics, no GPT dependency.

Columns: Local Gemma | Haiku | Sonnet | Opus | Lightify | LF --fast
Rows: 20 queries across 6 categories
Metrics: Cost, Latency, Quality (keyword scoring), Tokens

Uses only: Ollama (local) + Claude CLI (haiku/sonnet/opus)
Judge: keyword-match quality scoring (no external LLM judge needed)

Run: /tmp/lightify_venv/bin/python -m benches.run_grid_bench
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
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import Tier
from benches.queries_20 import EVAL_QUERIES


def score_quality(text: str, keywords: list[str]) -> int:
    """Score 0-100 based on keyword coverage + structure + length."""
    if not text or not text.strip():
        return 0
    t = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in t)
    kw_score = (found / len(keywords)) * 50 if keywords else 25
    words = len(text.split())
    len_score = min(25, words / 4)  # up to 25 for 100+ words
    has_structure = any(m in text for m in ["\n", "- ", "* ", "1.", "**", "```"])
    struct_score = 15 if has_structure else 5
    coherent = 10 if words > 10 and not text.startswith("[") else 0
    return min(100, int(kw_score + len_score + struct_score + coherent))


APPROACHES = [
    ("Local Gemma", "local", None),
    ("Haiku", "claude", Tier.SMALL),
    ("Sonnet", "claude", Tier.MID),
    ("Opus", "claude", Tier.FRONTIER),
    ("Lightify", "lightify", None),
    ("LF --fast", "lightify-fast", None),
]


def run_one(approach_type, tier, query, pipeline):
    """Run a single query through one approach."""
    if approach_type == "local":
        r = invoke_ollama(query, timeout_s=60)
        return {"cost": 0.0, "latency_ms": r.latency_ms, "text": r.text or "",
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "tier": "local"}

    elif approach_type == "claude":
        r = invoke_claude(query, tier, max_turns=1, timeout_s=120)
        return {"cost": r.cost, "latency_ms": r.latency_ms, "text": r.text or "",
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "tier": tier.value}

    elif approach_type == "lightify":
        pipeline.router.tau_tier1 = 0.45
        pipeline.router.tau_tier2 = 0.30
        r = pipeline.run_with_lightify(query)
        tiers = "→".join(t.value for t in r.tiers_attempted)
        return {"cost": r.total_cost, "latency_ms": r.total_latency_ms,
                "text": r.response.text or "",
                "tokens_in": r.total_tokens_in, "tokens_out": r.total_tokens_out, "tier": tiers}

    elif approach_type == "lightify-fast":
        pipeline.router.tau_tier1 = 0.20
        pipeline.router.tau_tier2 = 0.10
        r = pipeline.run_with_lightify(query)
        tiers = "→".join(t.value for t in r.tiers_attempted)
        return {"cost": r.total_cost, "latency_ms": r.total_latency_ms,
                "text": r.response.text or "",
                "tokens_in": r.total_tokens_in, "tokens_out": r.total_tokens_out, "tier": tiers}


def main():
    store = MemoryStore(os.path.expanduser("~/.lightify/memory.db"))
    pipeline = RealLightifyPipeline(store)

    appr_names = [a[0] for a in APPROACHES]
    results = {}  # results[query_id][approach_name] = {cost, latency, quality, ...}

    print("=" * 100)
    print("  LIGHTIFY GRID BENCHMARK")
    print(f"  {len(EVAL_QUERIES)} queries × {len(APPROACHES)} approaches = {len(EVAL_QUERIES)*len(APPROACHES)} calls")
    print(f"  Models: Gemma 1B (local) | Claude Haiku | Claude Sonnet | Claude Opus")
    print("=" * 100)

    for i, q in enumerate(EVAL_QUERIES):
        qid = q["id"]
        query = q["query"]
        keywords = q["keywords"]
        category = q["category"]
        results[qid] = {"category": category, "query": query}

        print(f"\n[{i+1}/{len(EVAL_QUERIES)}] {category}: {query[:65]}...")

        for name, atype, tier in APPROACHES:
            print(f"  {name:<14}", end=" ", flush=True)
            try:
                r = run_one(atype, tier, query, pipeline)
                quality = score_quality(r["text"], keywords)
                r["quality"] = quality
                results[qid][name] = r
                cost_str = f"${r['cost']:.3f}" if r["cost"] > 0 else "  FREE"
                print(f"{cost_str:>8}  {r['latency_ms']:>7.0f}ms  Q={quality:>3}  [{r['tier']}]")
            except Exception as e:
                print(f"  ERROR: {e}")
                results[qid][name] = {"cost": 0, "latency_ms": 0, "quality": 0,
                                       "text": "", "tokens_in": 0, "tokens_out": 0, "tier": "err"}

    store.close()

    # ── Aggregate tables ──────────────────────────────────────────────────
    categories = sorted(set(q["category"] for q in EVAL_QUERIES))

    def avg_by_cat(metric):
        """Get average metric per category per approach."""
        table = {}
        for cat in categories:
            table[cat] = {}
            cat_qs = [q["id"] for q in EVAL_QUERIES if q["category"] == cat]
            for name in appr_names:
                vals = [results[qid][name][metric] for qid in cat_qs
                        if name in results[qid] and metric in results[qid][name]]
                table[cat][name] = sum(vals) / len(vals) if vals else 0
        return table

    def print_grid(title, metric, fmt):
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")
        header = f"  {'Category':<16}" + "".join(f"{a:>14}" for a in appr_names)
        print(header)
        print(f"  {'─'*94}")

        table = avg_by_cat(metric)
        for cat in categories:
            row = f"  {cat:<16}"
            for name in appr_names:
                row += f"{fmt(table[cat][name]):>14}"
            print(row)

        # Overall average
        print(f"  {'─'*94}")
        row = f"  {'OVERALL':<16}"
        for name in appr_names:
            all_vals = [results[qid][name][metric]
                       for qid in results if name in results[qid] and metric in results[qid][name]]
            avg = sum(all_vals) / len(all_vals) if all_vals else 0
            row += f"{fmt(avg):>14}"
        print(row)

    print_grid("COST PER QUERY ($)", "cost",
               lambda v: f"${v:.4f}" if v > 0.0001 else "FREE")

    print_grid("LATENCY (ms)", "latency_ms",
               lambda v: f"{v:.0f}")

    print_grid("ANSWER QUALITY (0-100)", "quality",
               lambda v: f"{v:.0f}")

    print_grid("OUTPUT TOKENS", "tokens_out",
               lambda v: f"{v:.0f}")

    # ── Pareto summary ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  PARETO SUMMARY (avg cost vs avg quality)")
    print(f"{'='*100}")
    print(f"  {'Approach':<16} {'Avg Cost':>10} {'Avg Quality':>12} {'Avg Latency':>12} {'Pareto?':>8}")
    print(f"  {'─'*60}")

    points = []
    for name in appr_names:
        all_costs = [results[qid][name]["cost"] for qid in results if name in results[qid]]
        all_quals = [results[qid][name]["quality"] for qid in results if name in results[qid]]
        all_lats = [results[qid][name]["latency_ms"] for qid in results if name in results[qid]]
        avg_c = sum(all_costs) / len(all_costs) if all_costs else 0
        avg_q = sum(all_quals) / len(all_quals) if all_quals else 0
        avg_l = sum(all_lats) / len(all_lats) if all_lats else 0
        points.append((name, avg_c, avg_q, avg_l))

    # Determine Pareto optimal
    for i, (name, c, q, l) in enumerate(points):
        dominated = any(
            c2 <= c and q2 >= q and (c2 < c or q2 > q)
            for j, (_, c2, q2, _) in enumerate(points) if j != i
        )
        pareto = "  ★" if not dominated else ""
        cost_str = f"${c:.4f}" if c > 0.0001 else "FREE"
        print(f"  {name:<16} {cost_str:>10} {q:>10.1f}/100 {l:>10.0f}ms {pareto:>8}")

    # ── Save ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"grid_bench_{ts}.json")

    save = {}
    for qid in results:
        save[qid] = {"category": results[qid]["category"], "query": results[qid]["query"]}
        for name in appr_names:
            if name in results[qid]:
                r = results[qid][name]
                save[qid][name] = {k: v for k, v in r.items() if k != "text"}
                save[qid][name]["answer_preview"] = r.get("text", "")[:150]
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
