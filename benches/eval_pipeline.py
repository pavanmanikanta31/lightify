"""Lightify evaluation pipeline -- oracle-based routing accuracy + LLM-as-judge quality.

Methodology (agreed by 3-person FAANG research panel):

  1. ORACLE CONSTRUCTION
     For each of 20 queries, run through all 3 tiers (local Ollama, Claude Sonnet,
     Claude Opus). Score correctness via keyword matching. The cheapest tier that
     answers correctly is the oracle's "ideal tier."

  2. ROUTING ACCURACY
     Run each query through Lightify, compare selected tier vs oracle:
       RA  = fraction matching cheapest correct tier
       OSR = fraction routed more expensive than necessary
       USR = fraction routed too cheap (got wrong answer)

  3. QUALITY SCORING (LLM-as-judge)
     Claude Sonnet scores each response on:
       Correctness (0-5), Completeness (0-5), Conciseness (0-5)
     Composite Q = 0.50*Correctness + 0.30*Completeness + 0.20*Conciseness

  4. PARETO DATA COLLECTION
     For 6 approaches (Local, LF --fast, LF default, LF --quality, Sonnet, Opus),
     collect (avg_cost, avg_quality) points for Pareto-frontier analysis.

Run with:
    /tmp/lightify_venv/bin/python -m benches.eval_pipeline

Results saved to:
    /tmp/lightify_venv/results/eval_pipeline_<timestamp>.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from lightify.models.claude_cli import invoke_claude
from lightify.models.ollama_local import invoke_ollama, _ollama_available
from lightify.pipeline_real import RealLightifyPipeline
from lightify.storage.sqlite_memory import MemoryStore
from lightify.types import ModelResponse, Tier
from benches.queries_20 import EVAL_QUERIES

# ── Constants ────────────────────────────────────────────────────────────────

# Tier cost ordering (cheapest first). Index = cost rank.
TIER_COST_ORDER: list[str] = ["local", "sonnet", "opus"]

# Minimum keyword-match fraction to consider a response "correct" for oracle
CORRECTNESS_THRESHOLD = 0.50

# LLM-as-judge system prompt
JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI-generated responses. You will be given a
QUERY and a RESPONSE. Score the response on three dimensions, each from 0 to 5:

1. Correctness (0-5): Are the facts accurate? No hallucinations?
   0 = completely wrong / hallucinated
   5 = entirely accurate, no factual errors

2. Completeness (0-5): Does the response cover the key aspects of the query?
   0 = misses all important points
   5 = comprehensively covers the topic

3. Conciseness (0-5): Is it appropriately concise without omitting substance?
   0 = extremely verbose or uselessly terse
   5 = perfectly balanced

Reply with ONLY a JSON object (no markdown fences, no commentary):
{"correctness": <int>, "completeness": <int>, "conciseness": <int>}
"""

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TierResult:
    """Result from running a query through a specific tier."""
    tier_name: str  # "local", "sonnet", "opus"
    response_text: str = ""
    success: bool = False
    cost: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    keyword_score: float = 0.0  # fraction of expected keywords found


@dataclass
class JudgeScores:
    correctness: int = 0
    completeness: int = 0
    conciseness: int = 0

    @property
    def composite(self) -> float:
        """Q = 0.50*C + 0.30*Comp + 0.20*Conc, normalized to 0-5 scale."""
        return 0.50 * self.correctness + 0.30 * self.completeness + 0.20 * self.conciseness


@dataclass
class OracleEntry:
    """Oracle determination for a single query."""
    query_id: str
    tier_results: dict[str, TierResult] = field(default_factory=dict)
    cheapest_correct_tier: str = "opus"  # fallback
    all_correct_tiers: list[str] = field(default_factory=list)


@dataclass
class RoutingResult:
    """Result of Lightify routing vs oracle for a single query."""
    query_id: str
    oracle_tier: str = ""
    lightify_tier: str = ""
    lightify_cost: float = 0.0
    lightify_latency_ms: float = 0.0
    lightify_response: str = ""
    lightify_success: bool = False
    classification: str = ""  # "correct", "over_spend", "under_spend"
    judge_scores: JudgeScores = field(default_factory=JudgeScores)


@dataclass
class ParetoPoint:
    """A single (cost, quality) observation for one approach."""
    approach: str
    avg_cost: float = 0.0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _tier_name_to_enum(name: str) -> Tier:
    return {"local": Tier.SMALL, "sonnet": Tier.MID, "opus": Tier.FRONTIER}[name]


def _tier_enum_to_name(t: Tier) -> str:
    return {Tier.SMALL: "local", Tier.MID: "sonnet", Tier.FRONTIER: "opus"}[t]


def _cost_rank(tier_name: str) -> int:
    """Return cost rank index (0 = cheapest)."""
    return TIER_COST_ORDER.index(tier_name)


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found (case-insensitive) in response text."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ── Phase 1: Oracle Construction ─────────────────────────────────────────────


def build_oracle(queries: list[dict], verbose: bool = True) -> list[OracleEntry]:
    """Run every query through all 3 tiers and determine cheapest correct tier."""
    has_local = _ollama_available()
    oracle: list[OracleEntry] = []

    for i, q in enumerate(queries):
        qid = q["id"]
        query_text = q["query"]
        keywords = q["keywords"]

        if verbose:
            print(f"  [{i+1}/{len(queries)}] Oracle: {qid} -- {query_text[:60]}...")

        entry = OracleEntry(query_id=qid)

        # Tier 1: Local (Ollama)
        if has_local:
            if verbose:
                print(f"    [local]  ", end="", flush=True)
            r_local = invoke_ollama(query_text, timeout_s=30)
            ks = _keyword_score(r_local.text, keywords)
            entry.tier_results["local"] = TierResult(
                tier_name="local",
                response_text=r_local.text,
                success=r_local.success,
                cost=0.0,
                latency_ms=r_local.latency_ms,
                tokens_in=r_local.tokens_in,
                tokens_out=r_local.tokens_out,
                keyword_score=ks,
            )
            if verbose:
                print(f"kw={ks:.2f}, {r_local.latency_ms:.0f}ms")
        else:
            # If no local model, mark it as failed so oracle skips it
            entry.tier_results["local"] = TierResult(
                tier_name="local",
                response_text="(Ollama not available)",
                success=False,
                cost=0.0,
                keyword_score=0.0,
            )
            if verbose:
                print(f"    [local]  skipped (Ollama not running)")

        # Tier 2: Claude Sonnet
        if verbose:
            print(f"    [sonnet] ", end="", flush=True)
        r_sonnet = invoke_claude(query_text, Tier.MID, max_turns=1, timeout_s=120)
        ks = _keyword_score(r_sonnet.text, keywords)
        entry.tier_results["sonnet"] = TierResult(
            tier_name="sonnet",
            response_text=r_sonnet.text,
            success=r_sonnet.success,
            cost=r_sonnet.cost,
            latency_ms=r_sonnet.latency_ms,
            tokens_in=r_sonnet.tokens_in,
            tokens_out=r_sonnet.tokens_out,
            keyword_score=ks,
        )
        if verbose:
            print(f"kw={ks:.2f}, ${r_sonnet.cost:.4f}, {r_sonnet.latency_ms:.0f}ms")

        # Tier 3: Claude Opus
        if verbose:
            print(f"    [opus]   ", end="", flush=True)
        r_opus = invoke_claude(query_text, Tier.FRONTIER, max_turns=1, timeout_s=120)
        ks = _keyword_score(r_opus.text, keywords)
        entry.tier_results["opus"] = TierResult(
            tier_name="opus",
            response_text=r_opus.text,
            success=r_opus.success,
            cost=r_opus.cost,
            latency_ms=r_opus.latency_ms,
            tokens_in=r_opus.tokens_in,
            tokens_out=r_opus.tokens_out,
            keyword_score=ks,
        )
        if verbose:
            print(f"kw={ks:.2f}, ${r_opus.cost:.4f}, {r_opus.latency_ms:.0f}ms")

        # Determine cheapest correct tier
        for tier_name in TIER_COST_ORDER:
            tr = entry.tier_results[tier_name]
            if tr.success and tr.keyword_score >= CORRECTNESS_THRESHOLD:
                entry.all_correct_tiers.append(tier_name)

        if entry.all_correct_tiers:
            entry.cheapest_correct_tier = entry.all_correct_tiers[0]
        else:
            # None passed threshold -- default to opus (most capable)
            entry.cheapest_correct_tier = "opus"

        if verbose:
            print(f"    => oracle tier: {entry.cheapest_correct_tier} "
                  f"(correct: {entry.all_correct_tiers or 'none'})")
            print()

        oracle.append(entry)

    return oracle


# ── Phase 2: Routing Accuracy ────────────────────────────────────────────────


def measure_routing_accuracy(
    queries: list[dict],
    oracle: list[OracleEntry],
    pipeline: RealLightifyPipeline,
    verbose: bool = True,
) -> list[RoutingResult]:
    """Run queries through Lightify default mode, compare tier selection to oracle."""
    oracle_map = {e.query_id: e for e in oracle}
    results: list[RoutingResult] = []

    # Reset router to default thresholds
    pipeline.router.tau_tier1 = 0.45
    pipeline.router.tau_tier2 = 0.30

    for i, q in enumerate(queries):
        qid = q["id"]
        query_text = q["query"]
        keywords = q["keywords"]
        oracle_entry = oracle_map[qid]

        if verbose:
            print(f"  [{i+1}/{len(queries)}] Routing: {qid} -- {query_text[:55]}...")
            print(f"    [lightify] ", end="", flush=True)

        pr = pipeline.run_with_lightify(query_text)

        # The final tier that produced the answer
        final_tier = _tier_enum_to_name(pr.response.tier)
        lightify_ks = _keyword_score(pr.response.text, keywords)

        # Classify routing decision
        oracle_rank = _cost_rank(oracle_entry.cheapest_correct_tier)
        lightify_rank = _cost_rank(final_tier)

        if lightify_rank == oracle_rank:
            classification = "correct"
        elif lightify_rank > oracle_rank:
            classification = "over_spend"
        else:
            # Routed cheaper, but did it get the right answer?
            if lightify_ks >= CORRECTNESS_THRESHOLD:
                classification = "correct"  # cheaper AND correct = bonus
            else:
                classification = "under_spend"

        rr = RoutingResult(
            query_id=qid,
            oracle_tier=oracle_entry.cheapest_correct_tier,
            lightify_tier=final_tier,
            lightify_cost=pr.total_cost,
            lightify_latency_ms=pr.total_latency_ms,
            lightify_response=pr.response.text or "",
            lightify_success=pr.response.success,
            classification=classification,
        )
        results.append(rr)

        if verbose:
            tiers_attempted = "->".join(t.value for t in pr.tiers_attempted)
            print(f"tier={final_tier}, oracle={oracle_entry.cheapest_correct_tier}, "
                  f"class={classification}, kw={lightify_ks:.2f}, "
                  f"route={tiers_attempted}, ${pr.total_cost:.4f}")

    return results


def compute_routing_metrics(results: list[RoutingResult]) -> dict:
    """Compute RA, OSR, USR from routing results."""
    n = len(results)
    if n == 0:
        return {"RA": 0.0, "OSR": 0.0, "USR": 0.0}

    correct = sum(1 for r in results if r.classification == "correct")
    over = sum(1 for r in results if r.classification == "over_spend")
    under = sum(1 for r in results if r.classification == "under_spend")

    return {
        "RA": correct / n,
        "OSR": over / n,
        "USR": under / n,
        "correct_count": correct,
        "over_spend_count": over,
        "under_spend_count": under,
        "total": n,
    }


# ── Phase 3: LLM-as-Judge Quality Scoring ───────────────────────────────────


def judge_response(query: str, response_text: str, verbose: bool = False) -> JudgeScores:
    """Use Claude Sonnet to score a single response on correctness/completeness/conciseness."""
    if not response_text or not response_text.strip():
        return JudgeScores(correctness=0, completeness=0, conciseness=0)

    judge_prompt = (
        f"QUERY:\n{query}\n\n"
        f"RESPONSE:\n{response_text[:3000]}\n\n"
        f"Score this response. Reply with ONLY JSON: "
        f'{{"correctness": <0-5>, "completeness": <0-5>, "conciseness": <0-5>}}'
    )

    r = invoke_claude(
        prompt=judge_prompt,
        tier=Tier.MID,  # Sonnet as judge
        system_prompt=JUDGE_SYSTEM_PROMPT,
        max_turns=1,
        timeout_s=60,
    )

    if not r.success or not r.text:
        if verbose:
            print(f"    [judge] FAILED: {r.text[:100] if r.text else '(empty)'}")
        return JudgeScores(correctness=0, completeness=0, conciseness=0)

    # Parse JSON from judge response (handle markdown fences)
    text = r.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        scores = json.loads(text)
        return JudgeScores(
            correctness=max(0, min(5, int(scores.get("correctness", 0)))),
            completeness=max(0, min(5, int(scores.get("completeness", 0)))),
            conciseness=max(0, min(5, int(scores.get("conciseness", 0)))),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try to extract numbers from text as a fallback
        if verbose:
            print(f"    [judge] JSON parse failed, raw: {text[:120]}")
        return JudgeScores(correctness=0, completeness=0, conciseness=0)


def score_all_routing_results(
    queries: list[dict],
    routing_results: list[RoutingResult],
    verbose: bool = True,
) -> list[RoutingResult]:
    """Run LLM-as-judge on all Lightify routing results."""
    query_map = {q["id"]: q for q in queries}

    for i, rr in enumerate(routing_results):
        q = query_map[rr.query_id]
        if verbose:
            print(f"  [{i+1}/{len(routing_results)}] Judge: {rr.query_id}...", end=" ", flush=True)

        scores = judge_response(q["query"], rr.lightify_response, verbose=verbose)
        rr.judge_scores = scores

        if verbose:
            print(f"C={scores.correctness} Comp={scores.completeness} "
                  f"Conc={scores.conciseness} Q={scores.composite:.2f}")

    return routing_results


# ── Phase 4: Pareto Data Collection ──────────────────────────────────────────


def collect_pareto_data(
    queries: list[dict],
    pipeline: RealLightifyPipeline,
    verbose: bool = True,
) -> list[ParetoPoint]:
    """Collect (avg_cost, avg_quality) for 6 approaches across all queries."""
    approaches = [
        ("Local", "local", None),
        ("LF --fast", "lightify", "fast"),
        ("Lightify", "lightify", None),
        ("LF --quality", "lightify", "quality"),
        ("Sonnet", "direct", "sonnet"),
        ("Opus", "direct", "opus"),
    ]

    pareto_points: list[ParetoPoint] = []

    for approach_name, approach_type, mode in approaches:
        if verbose:
            print(f"\n  === {approach_name} ===")

        costs: list[float] = []
        qualities: list[float] = []
        latencies: list[float] = []

        for i, q in enumerate(queries):
            query_text = q["query"]
            if verbose:
                print(f"    [{i+1}/{len(queries)}] {q['id']}...", end=" ", flush=True)

            cost = 0.0
            response_text = ""
            latency_ms = 0.0

            if approach_type == "local":
                r = invoke_ollama(query_text, timeout_s=30)
                cost = 0.0
                response_text = r.text
                latency_ms = r.latency_ms

            elif approach_type == "direct":
                tier = Tier.MID if mode == "sonnet" else Tier.FRONTIER
                r = invoke_claude(query_text, tier, max_turns=1, timeout_s=120)
                cost = r.cost
                response_text = r.text
                latency_ms = r.latency_ms

            elif approach_type == "lightify":
                # Set router thresholds based on mode
                if mode == "fast":
                    pipeline.router.tau_tier1 = 0.20
                    pipeline.router.tau_tier2 = 0.10
                elif mode == "quality":
                    pipeline.router.tau_tier1 = 0.90
                    pipeline.router.tau_tier2 = 0.70
                else:
                    pipeline.router.tau_tier1 = 0.45
                    pipeline.router.tau_tier2 = 0.30

                pr = pipeline.run_with_lightify(query_text)
                cost = pr.total_cost
                response_text = pr.response.text or ""
                latency_ms = pr.total_latency_ms

            costs.append(cost)
            latencies.append(latency_ms)

            # Judge quality for this response
            scores = judge_response(q["query"], response_text)
            qualities.append(scores.composite)

            if verbose:
                print(f"${cost:.4f}, Q={scores.composite:.2f}, {latency_ms:.0f}ms")

        pp = ParetoPoint(
            approach=approach_name,
            avg_cost=statistics.mean(costs) if costs else 0.0,
            avg_quality=statistics.mean(qualities) if qualities else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            num_queries=len(queries),
        )
        pareto_points.append(pp)

        if verbose:
            print(f"  => {approach_name}: avg_cost=${pp.avg_cost:.4f}, "
                  f"avg_Q={pp.avg_quality:.2f}, avg_lat={pp.avg_latency_ms:.0f}ms")

    return pareto_points


# ── Output Formatting ────────────────────────────────────────────────────────


def print_oracle_table(oracle: list[OracleEntry], queries: list[dict]):
    """Print oracle results as formatted table."""
    query_map = {q["id"]: q for q in queries}
    print()
    print("=" * 100)
    print("  ORACLE RESULTS: Cheapest Correct Tier Per Query")
    print("=" * 100)
    print(f"  {'ID':<12} {'Category':<16} {'Local KW':<10} {'Sonnet KW':<10} "
          f"{'Opus KW':<10} {'Oracle Tier':<12} {'Correct?'}")
    print(f"  {'-'*92}")

    for entry in oracle:
        q = query_map[entry.query_id]
        local_ks = entry.tier_results.get("local", TierResult(tier_name="local")).keyword_score
        sonnet_ks = entry.tier_results.get("sonnet", TierResult(tier_name="sonnet")).keyword_score
        opus_ks = entry.tier_results.get("opus", TierResult(tier_name="opus")).keyword_score
        correct_str = ", ".join(entry.all_correct_tiers) if entry.all_correct_tiers else "NONE"

        print(f"  {entry.query_id:<12} {q['category']:<16} "
              f"{local_ks:<10.2f} {sonnet_ks:<10.2f} {opus_ks:<10.2f} "
              f"{entry.cheapest_correct_tier:<12} {correct_str}")


def print_routing_table(results: list[RoutingResult], queries: list[dict]):
    """Print routing accuracy results as formatted table."""
    query_map = {q["id"]: q for q in queries}
    metrics = compute_routing_metrics(results)

    print()
    print("=" * 110)
    print("  ROUTING ACCURACY: Lightify vs Oracle")
    print("=" * 110)
    print(f"  {'ID':<12} {'Category':<16} {'Oracle':<8} {'Lightify':<10} "
          f"{'Class':<13} {'Cost':<10} {'C':<3} {'Comp':<4} {'Conc':<4} {'Q':<6}")
    print(f"  {'-'*100}")

    for rr in results:
        q = query_map[rr.query_id]
        js = rr.judge_scores
        class_marker = {
            "correct": "  MATCH",
            "over_spend": "  OVER $",
            "under_spend": "  UNDER !",
        }.get(rr.classification, "  ???")

        print(f"  {rr.query_id:<12} {q['category']:<16} "
              f"{rr.oracle_tier:<8} {rr.lightify_tier:<10} "
              f"{class_marker:<13} ${rr.lightify_cost:<9.4f} "
              f"{js.correctness:<3} {js.completeness:<4} {js.conciseness:<4} "
              f"{js.composite:<6.2f}")

    print(f"  {'-'*100}")
    print(f"  Routing Accuracy (RA):  {metrics['RA']:.1%}  "
          f"({metrics['correct_count']}/{metrics['total']})")
    print(f"  Over-Spend Rate (OSR):  {metrics['OSR']:.1%}  "
          f"({metrics['over_spend_count']}/{metrics['total']})")
    print(f"  Under-Spend Rate (USR): {metrics['USR']:.1%}  "
          f"({metrics['under_spend_count']}/{metrics['total']})")


def print_quality_summary(results: list[RoutingResult]):
    """Print quality score summary."""
    print()
    print("=" * 70)
    print("  QUALITY SCORES (LLM-as-Judge, Lightify Default Mode)")
    print("=" * 70)

    correctness_vals = [r.judge_scores.correctness for r in results]
    completeness_vals = [r.judge_scores.completeness for r in results]
    conciseness_vals = [r.judge_scores.conciseness for r in results]
    composite_vals = [r.judge_scores.composite for r in results]

    print(f"  {'Metric':<16} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*56}")

    for name, vals in [
        ("Correctness", correctness_vals),
        ("Completeness", completeness_vals),
        ("Conciseness", conciseness_vals),
        ("Composite Q", composite_vals),
    ]:
        mean_v = statistics.mean(vals) if vals else 0
        median_v = statistics.median(vals) if vals else 0
        stdev_v = statistics.stdev(vals) if len(vals) > 1 else 0
        min_v = min(vals) if vals else 0
        max_v = max(vals) if vals else 0
        print(f"  {name:<16} {mean_v:>8.2f} {median_v:>8.2f} {stdev_v:>8.2f} "
              f"{min_v:>6.1f} {max_v:>6.1f}")


def print_pareto_table(pareto_points: list[ParetoPoint]):
    """Print Pareto data collection table."""
    print()
    print("=" * 80)
    print("  PARETO FRONTIER DATA: Cost vs Quality by Approach")
    print("=" * 80)
    print(f"  {'Approach':<16} {'Avg Cost ($)':>14} {'Avg Quality':>12} "
          f"{'Avg Latency':>14} {'Queries':>8}")
    print(f"  {'-'*68}")

    for pp in pareto_points:
        cost_str = f"${pp.avg_cost:.4f}" if pp.avg_cost > 0 else "FREE"
        print(f"  {pp.approach:<16} {cost_str:>14} {pp.avg_quality:>12.2f} "
              f"{pp.avg_latency_ms:>11.0f}ms {pp.num_queries:>8}")

    # Identify Pareto-optimal points (not dominated by any other point)
    print()
    print("  Pareto-optimal approaches (not dominated on both cost AND quality):")
    for pp in pareto_points:
        dominated = False
        for other in pareto_points:
            if other.approach == pp.approach:
                continue
            if other.avg_cost <= pp.avg_cost and other.avg_quality >= pp.avg_quality:
                if other.avg_cost < pp.avg_cost or other.avg_quality > pp.avg_quality:
                    dominated = True
                    break
        if not dominated:
            print(f"    * {pp.approach}: cost=${pp.avg_cost:.4f}, Q={pp.avg_quality:.2f}")


# ── Results Serialization ────────────────────────────────────────────────────


def serialize_results(
    oracle: list[OracleEntry],
    routing_results: list[RoutingResult],
    routing_metrics: dict,
    pareto_points: list[ParetoPoint],
    elapsed_s: float,
) -> dict:
    """Build JSON-serializable results dictionary."""
    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_queries": len(EVAL_QUERIES),
            "correctness_threshold": CORRECTNESS_THRESHOLD,
            "elapsed_seconds": round(elapsed_s, 1),
        },
        "oracle": [
            {
                "query_id": e.query_id,
                "cheapest_correct_tier": e.cheapest_correct_tier,
                "all_correct_tiers": e.all_correct_tiers,
                "tier_keyword_scores": {
                    name: {
                        "keyword_score": tr.keyword_score,
                        "cost": tr.cost,
                        "latency_ms": round(tr.latency_ms, 1),
                        "success": tr.success,
                    }
                    for name, tr in e.tier_results.items()
                },
            }
            for e in oracle
        ],
        "routing": {
            "metrics": routing_metrics,
            "per_query": [
                {
                    "query_id": rr.query_id,
                    "oracle_tier": rr.oracle_tier,
                    "lightify_tier": rr.lightify_tier,
                    "classification": rr.classification,
                    "cost": rr.lightify_cost,
                    "latency_ms": round(rr.lightify_latency_ms, 1),
                    "judge_scores": {
                        "correctness": rr.judge_scores.correctness,
                        "completeness": rr.judge_scores.completeness,
                        "conciseness": rr.judge_scores.conciseness,
                        "composite": round(rr.judge_scores.composite, 3),
                    },
                    "response_preview": _truncate(rr.lightify_response, 200),
                }
                for rr in routing_results
            ],
        },
        "pareto": [
            {
                "approach": pp.approach,
                "avg_cost": round(pp.avg_cost, 6),
                "avg_quality": round(pp.avg_quality, 3),
                "avg_latency_ms": round(pp.avg_latency_ms, 1),
                "num_queries": pp.num_queries,
            }
            for pp in pareto_points
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()

    print("=" * 80)
    print("  LIGHTIFY EVALUATION PIPELINE")
    print("  Oracle + Routing Accuracy + LLM-as-Judge + Pareto Analysis")
    print("=" * 80)
    print(f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Queries: {len(EVAL_QUERIES)}")
    print(f"  Phases:  Oracle -> Routing -> Judge -> Pareto")
    print()

    # Setup pipeline
    db_path = os.path.join(_PROJECT_ROOT, "lightify_memory.db")
    store = MemoryStore(db_path)
    if store.count() == 0:
        print("[setup] Seeding memory store...")
        from benches.generate_data import seed_memory
        seed_memory(store)
    print(f"[setup] Memory store: {store.count()} items")
    print(f"[setup] Ollama available: {_ollama_available()}")
    print()

    pipeline = RealLightifyPipeline(store)

    # ── Phase 1: Oracle Construction ──────────────────────────────────────
    print("-" * 80)
    print("  PHASE 1: Oracle Construction (all queries x all tiers)")
    print("-" * 80)
    oracle = build_oracle(EVAL_QUERIES, verbose=True)
    print_oracle_table(oracle, EVAL_QUERIES)

    # ── Phase 2: Routing Accuracy ─────────────────────────────────────────
    print()
    print("-" * 80)
    print("  PHASE 2: Routing Accuracy (Lightify default vs Oracle)")
    print("-" * 80)
    routing_results = measure_routing_accuracy(
        EVAL_QUERIES, oracle, pipeline, verbose=True
    )

    # ── Phase 3: LLM-as-Judge ────────────────────────────────────────────
    print()
    print("-" * 80)
    print("  PHASE 3: LLM-as-Judge Quality Scoring (Sonnet judges Lightify responses)")
    print("-" * 80)
    routing_results = score_all_routing_results(
        EVAL_QUERIES, routing_results, verbose=True
    )
    routing_metrics = compute_routing_metrics(routing_results)

    # Print routing + quality tables
    print_routing_table(routing_results, EVAL_QUERIES)
    print_quality_summary(routing_results)

    # ── Phase 4: Pareto Data Collection ──────────────────────────────────
    print()
    print("-" * 80)
    print("  PHASE 4: Pareto Data Collection (6 approaches x 20 queries)")
    print("-" * 80)
    pareto_points = collect_pareto_data(EVAL_QUERIES, pipeline, verbose=True)
    print_pareto_table(pareto_points)

    # ── Save results ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"eval_pipeline_{ts}.json")

    results_json = serialize_results(
        oracle, routing_results, routing_metrics, pareto_points, elapsed
    )
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print()
    print("=" * 80)
    print("  EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Results saved:  {results_path}")
    print(f"  Routing Accuracy: {routing_metrics['RA']:.1%}")
    print(f"  Over-Spend Rate:  {routing_metrics['OSR']:.1%}")
    print(f"  Under-Spend Rate: {routing_metrics['USR']:.1%}")
    avg_q = statistics.mean(
        r.judge_scores.composite for r in routing_results
    ) if routing_results else 0
    print(f"  Avg Quality (Q):  {avg_q:.2f}/5.0")
    print()

    store.close()


if __name__ == "__main__":
    main()
