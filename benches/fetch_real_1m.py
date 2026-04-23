"""Fetch 1M+ genuinely-human-authored real queries from public datasets.

Sources (all public, non-auth'd parquet downloads from HuggingFace):
  - microsoft/ms_marco (v2.1 train) — real Bing search queries
  - sentence-transformers/natural-questions — Google search-query pairs
  - openai/gsm8k — hand-authored math word problems
  - cais/mmlu (all auxiliary_train) — multiple-choice academic questions
  - allenai/WildChat-1M — real LLM chat conversations (first user turn)

Auto-category assignment:
  - MS MARCO, NQ    → short_lookup  (SMALL oracle)
  - GSM8K, MMLU     → reasoning     (MID oracle)
  - WildChat first-turn → filter to single-line queries, label by regex

Output: benches/datasets/real/queries_public_<N>.json
No API calls, no paid compute. Wall-clock: ~5-15 min depending on network.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# ── helper: HF parquet download ──────────────────────────────────────────
HF_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"

def list_parquet_urls(repo: str, config: str, split: str) -> list[str]:
    """Return list of parquet shard URLs for a dataset split."""
    url = HF_API.format(repo=repo, config=config, split=split)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())

def download_parquet(url: str) -> "pq.Table":
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    return pq.read_table(io.BytesIO(data))

# ── category auto-labeler ────────────────────────────────────────────────
CODE = re.compile(r"\b(write|implement|code|function|class|unit test|refactor)\b", re.I)
LARGE_CODE = re.compile(r"\b(entire|full|whole|rewrite|migrate|port)\b.*\b(module|service|system|codebase|app|project)\b", re.I)
BASH = re.compile(r"\b(ls|cat|grep|find|git|kubectl|docker|curl|chmod|awk|sed|tar|rsync|ssh)\b", re.I)
LOOKUP = re.compile(r"^(what|define|who|when|where|which|how many|convert|is|are)\b", re.I)
REASONING = re.compile(r"\b(compare|why|how would you|explain|pros and cons|trade[- ]offs?|design|decide)\b", re.I)

def auto_label(q: str, source: str) -> tuple[str, str]:
    """Return (category, oracle_tier) for a query."""
    q_lower = q.strip().lower()
    # Source-specific priors
    if source in ("gsm8k", "mmlu"):
        return ("reasoning", "MID")
    if source in ("ms_marco", "natural_questions"):
        return ("short_lookup", "SMALL")
    # WildChat or unknown — regex dispatch
    if LARGE_CODE.search(q):
        return ("large_code", "FRONTIER")
    if CODE.search(q):
        return ("code", "MID")
    if BASH.search(q):
        return ("bash_like", "SMALL")
    if REASONING.search(q):
        return ("reasoning", "MID")
    if LOOKUP.match(q_lower):
        return ("short_lookup", "SMALL")
    return ("reasoning", "MID")  # fallback

# ── per-source extractors ────────────────────────────────────────────────
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())

def _valid_query(q: str) -> bool:
    return bool(q) and 5 <= len(q) <= 500 and "\n" not in q[:200]


def extract_ms_marco(max_rows: int) -> list[dict]:
    print("  fetching MS MARCO v2.1 train parquet urls...")
    urls = list_parquet_urls("microsoft/ms_marco", "v2.1", "train")
    out: list[dict] = []
    for i, u in enumerate(urls):
        if len(out) >= max_rows:
            break
        t0 = time.time()
        try:
            t = download_parquet(u)
            queries = t.column("query").to_pylist()
        except Exception as e:
            print(f"    shard {i} err: {e}")
            continue
        for q in queries:
            q = _clean(q)
            if _valid_query(q):
                out.append({
                    "id": f"msmarco-{len(out)+1}",
                    "category": "short_lookup",
                    "query": q,
                    "oracle_tier": "SMALL",
                    "has_contradiction": False,
                    "source": "ms_marco_v2.1_train",
                })
                if len(out) >= max_rows:
                    break
        print(f"    shard {i+1}/{len(urls)}: {len(out)} cumulative  ({time.time()-t0:.1f}s)")
    return out


def extract_natural_questions(max_rows: int) -> list[dict]:
    print("  fetching NQ (sentence-transformers/natural-questions pair train)...")
    urls = list_parquet_urls("sentence-transformers/natural-questions", "pair", "train")
    out: list[dict] = []
    for i, u in enumerate(urls):
        if len(out) >= max_rows: break
        try:
            t = download_parquet(u)
            qs = t.column("query").to_pylist() if "query" in t.column_names else t.column(0).to_pylist()
        except Exception as e:
            print(f"    shard err: {e}"); continue
        for q in qs:
            q = _clean(q)
            if _valid_query(q):
                out.append({
                    "id": f"nq-{len(out)+1}",
                    "category": "short_lookup",
                    "query": q,
                    "oracle_tier": "SMALL",
                    "has_contradiction": False,
                    "source": "natural_questions",
                })
                if len(out) >= max_rows: break
        print(f"    shard {i+1}: {len(out)} cumulative")
    return out


def extract_gsm8k(max_rows: int) -> list[dict]:
    print("  fetching GSM8K main train...")
    urls = list_parquet_urls("openai/gsm8k", "main", "train")
    out: list[dict] = []
    for u in urls:
        try:
            t = download_parquet(u)
            qs = t.column("question").to_pylist()
        except Exception as e:
            print(f"    err: {e}"); continue
        for q in qs:
            q = _clean(q)
            if _valid_query(q):
                out.append({
                    "id": f"gsm8k-{len(out)+1}",
                    "category": "reasoning",
                    "query": q,
                    "oracle_tier": "MID",
                    "has_contradiction": False,
                    "source": "gsm8k",
                })
                if len(out) >= max_rows: break
    return out


def extract_mmlu(max_rows: int) -> list[dict]:
    print("  fetching MMLU all/auxiliary_train...")
    urls = list_parquet_urls("cais/mmlu", "all", "auxiliary_train")
    out: list[dict] = []
    for i, u in enumerate(urls):
        if len(out) >= max_rows: break
        try:
            t = download_parquet(u)
            qs = t.column("question").to_pylist()
        except Exception as e:
            print(f"    err: {e}"); continue
        for q in qs:
            q = _clean(q)
            if _valid_query(q):
                out.append({
                    "id": f"mmlu-{len(out)+1}",
                    "category": "short_lookup",
                    "query": q,
                    "oracle_tier": "SMALL",
                    "has_contradiction": False,
                    "source": "mmlu_aux_train",
                })
                if len(out) >= max_rows: break
        print(f"    shard {i+1}: {len(out)} cumulative")
    return out


def extract_wildchat(max_rows: int) -> list[dict]:
    print("  fetching WildChat-1M default train (first user turn)...")
    urls = list_parquet_urls("allenai/WildChat-1M", "default", "train")
    out: list[dict] = []
    for i, u in enumerate(urls):
        if len(out) >= max_rows: break
        try:
            t = download_parquet(u)
            convs = t.column("conversation").to_pylist()
        except Exception as e:
            print(f"    err: {e}"); continue
        for conv in convs:
            if not conv: continue
            first = conv[0] if isinstance(conv, list) else None
            if not first or first.get("role") != "user": continue
            q = _clean(first.get("content", ""))
            if not _valid_query(q): continue
            cat, oracle = auto_label(q, "wildchat")
            out.append({
                "id": f"wildchat-{len(out)+1}",
                "category": cat,
                "query": q,
                "oracle_tier": oracle,
                "has_contradiction": False,
                "source": "wildchat",
            })
            if len(out) >= max_rows: break
        print(f"    shard {i+1}/{len(urls)}: {len(out)} cumulative")
    return out


def main(target_total: int = 1_000_000):
    t0 = time.time()
    rows: list[dict] = []
    # target mix (rough — will adjust if sources fall short)
    budget = {
        "ms_marco":         400_000,
        "wildchat":         300_000,
        "mmlu":             100_000,
        "natural_questions":100_000,
        "gsm8k":              7_000,
    }
    extractors = [
        ("ms_marco",          extract_ms_marco),
        ("wildchat",          extract_wildchat),
        ("mmlu",              extract_mmlu),
        ("natural_questions", extract_natural_questions),
        ("gsm8k",             extract_gsm8k),
    ]
    for name, fn in extractors:
        print(f"\n=== {name}  (target {budget[name]}) ===")
        try:
            rows.extend(fn(budget[name]))
        except Exception as e:
            print(f"  FAIL {name}: {e}")
        print(f"  cumulative: {len(rows)}  ({time.time()-t0:.0f}s elapsed)")

    # dedup
    seen = set()
    unique = []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            unique.append(r)

    out_path = Path(__file__).parent / "datasets" / "real" / f"queries_public_{len(unique)}.json"
    out_path.write_text(json.dumps(unique, indent=2))
    print(f"\n=== DONE ===")
    print(f"  {len(rows)} gross rows, {len(unique)} unique")
    print(f"  wrote {out_path}")
    from collections import Counter
    print(f"  by source:   {dict(Counter(r['source'] for r in unique))}")
    print(f"  by category: {dict(Counter(r['category'] for r in unique))}")
    print(f"  total time:  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1_000_000)
    args = p.parse_args()
    main(target_total=args.target)
