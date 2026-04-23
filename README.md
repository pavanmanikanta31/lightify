# Lightify: Cost-Aware Adaptive LLM Routing via Retrieval Confidence and Conflict-Aware Escalation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5756--100X-a6ce39?logo=orcid)](https://orcid.org/0009-0000-5756-100X)

**Cost-aware adaptive LLM routing middleware for agent harnesses.**

Modern agent harnesses (Claude Agent SDK, LangGraph, AutoGen, CrewAI, OpenHands, LlamaIndex Agents) leave cost governance, local-first inference, and memory conflict handling entirely to the application developer. Lightify is an open-source middleware layer that provides these as runtime primitives beneath any agent loop.

## How it works

Lightify routes each query to the cheapest model tier that can handle it reliably:

```
User Query
    │
    ▼
SQLite FTS5 Memory  ──►  Context Confidence Φ(C)
    │
    ├─ MCD (conflict detected?)  ──►  escalate to Tier-2/3
    │
    └─ CDDR: Φ(C) ≥ τ₁?  ──►  Tier-1 (local Gemma 3 1B, $0)
             Φ(C) ≥ τ₂?  ──►  Tier-2 (Claude Sonnet, ~$0.019/q)
             else         ──►  Tier-3 (Claude Opus, ~$0.076/q)
```

**CDDR** (Retrieval-Confidence-Driven Routing) selects the tier based on the aggregate confidence of retrieved memory items — not query length or keyword heuristics.

**MCD** (Memory Conflict Detection) scans retrieved items for temporal contradictions (numeric revisions, negations, antonym pairs) and forces escalation when found. MCD also applies a soft confidence penalty (`Φ ← max(0, Φ − 0.1·|conflicts|)`) to push heavily-conflicted contexts from Tier-2 into Tier-3.

**CDPS** (Confidence-Driven Prompt Shaping) adapts prompt verbosity to retrieval reliability.

**CSE** (Context Sufficiency Estimation) blocks Tier-1 routing when retrieved context is too sparse to support a reliable local answer.

**SECR** (Selective Entry Compression and Ranking) compresses memory items before injecting them into the prompt to reduce token cost.

## Results

Evaluated on 1,197,316 real user queries from five public corpora (MS MARCO v2.1, WildChat-1M, Natural Questions, MMLU, GSM8K) — all exact-string unique, none authored by us.

| Policy | Policy-Oracle Agreement | Cost/query | Cost vs Opus-only |
|---|---|---|---|
| Naïve Opus-only baseline *(reference)* | 0.000 | $0.076 | — |
| Hand-coded length gateway | 0.324 [0.323, 0.325] | $0.0196 | −74% |
| Hand-coded keyword router | 0.794 [0.793, 0.795] | $0.0007 | −99% |
| Lightify CDDR (all 1.2M incl. non-English) | 0.913 [0.912, 0.913] | $0.0025 | −96% |
| **Lightify CDDR** (English subset, N=1.04M) | **0.958 [0.957, 0.958]** | **$0.0012** | **−98%** |

> **Reading the table:** Opus-only is the cost baseline. The keyword router is cheapest (−99%) because it routes nearly everything to Tier-1, but its POA drops to 0.794. Lightify achieves 0.958 POA at −98% cost — more accurate than any baseline while still 98% cheaper than Opus-only. Cost and POA must be read together; a cheaper policy that routes incorrectly is not a win.

**Per-source POA breakdown:**

| Source | N | POA |
|---|---|---|
| MS MARCO v2.1 | 808,465 | 1.000 |
| Natural Questions | 99,903 | 1.000 |
| MMLU | 10,287 | 1.000 |
| WildChat-1M (all) | 271,661 | 0.625 |
| GSM8K | 7,000 | 0.607 |
| **Non-trivial subset** (WildChat-Eng + GSM8K) | **~125K** | **0.648 [0.646, 0.651]** |
| Non-English WildChat | 154,096 | 0.604 |

MS MARCO, NQ, and MMLU are unconditionally labeled Tier-1 by the oracle and account for 77% of the corpus — aggregate POA of 0.913 reflects corpus composition as much as policy quality. The **non-trivial subset** (WildChat-English + GSM8K, the only corpora without unconditional Tier-1 labels) at **0.648** is the honest stress-test operating range.

English subset = queries with pure ASCII text (N=1,043,220, 87.1%); non-English = WildChat queries containing non-ASCII characters (N=154,096, 12.9%).

*Policy-Oracle Agreement (POA) measures routing-decision agreement with a category-level oracle, not per-query output correctness. Per-query cost is a tier-cost projection applied to routing decisions, not measured API spend on 1.2M queries; live spend is reported only in the paper's §V.C 20-query pilot.*

**MCD stress test** (200 pairs: 100 contradictions + 100 controls): Precision 0.943 [0.814, 0.984], Recall 0.330 [0.246, 0.427]. TP=33, FN=67, FP=2, TN=98. High-precision within its lexical signal set; worst categories are version_bump (0/10) and team_change (0/5), which require semantic inference beyond lexical overlap.

## Installation

**Prerequisites:** Python 3.10+, [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (Tier-2/3), [Ollama](https://ollama.com/) (Tier-1 local)

```bash
git clone https://github.com/pavanmanikanta31/lightify.git
cd lightify

python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools
pip install -e .

lightify init
```

Or use the setup script:
```bash
chmod +x setup.sh && ./setup.sh
```

**Ollama (Tier-1 local model):**
```bash
brew install ollama
ollama pull gemma3:1b
ollama serve
```

## Usage

```bash
# Route a query to the cheapest viable tier
lightify query "What prevents true thread parallelism in Python?"

# Compare Lightify vs raw Opus baseline side-by-side
lightify query --compare "How does Rust handle memory safety?"

# Mode presets
lightify query --fast "What is a REST API?"        # prefer local
lightify query --cheap "Explain Docker containers"  # avoid API spend
lightify query --quality "Compare microservices vs monolith"  # frontier

# Force a specific tier
lightify query --model haiku "Quick question"
lightify query --model sonnet "Medium complexity"
lightify query --model opus "Complex reasoning"

# Enable per-action routing overlay
lightify query --action-routing "List files in /tmp"
```

**Memory management:**
```bash
lightify memory list
lightify memory add "Python 3.13 removes the GIL" --topic python
lightify memory search "memory safety"
lightify memory seed
```

**Config:**
```bash
lightify config show
lightify config set tier1 gemma3:4b
lightify config reset
```

**Isolated instance (separate memory DB):**
```bash
LIGHTIFY_HOME=~/my-project lightify init
LIGHTIFY_HOME=~/my-project lightify query "..."
```

## Benchmarks

All benchmark datasets and result JSONs are included in this repo.

```bash
# Simulated ablation on synthetic queries (no API calls, fully reproducible)
python benches/run_routing.py                  # uses queries_5000.json by default

# Real-query benchmark (requires Claude CLI + Ollama)
python benches/run_real.py

# MCD stress test (200 contradiction/control pairs)
python benches/run_mcd_stress.py

# LangGraph-style baseline comparison
python benches/run_langgraph_compare.py

# Assemble the 1.2M real-query benchmark from public HuggingFace parquet shards
python benches/fetch_real_1m.py
```

**Seed queries** (240 hand-authored, reproducibility anchor): `benches/datasets/real/queries_real.json`

**Synthetic query generator** (scale with `--scale N`, e.g. `--scale 25` → 5,000 queries):
```bash
python benches/generate_200.py --scale 25
```

## Project structure

```
lightify/
  cli.py               # CLI entry point (init, query, memory, config, status)
  pipeline.py          # Simulated pipeline — reproducible ablation, no API calls
  pipeline_real.py     # Real pipeline — Claude CLI + Ollama
  router.py            # CDDR: confidence-threshold tier selection
  action_router.py     # Per-action regex overlay (--action-routing flag)
  config.py            # Tier model configuration and LIGHTIFY_HOME handling
  types.py             # Shared types: Tier, MemoryItem, ContextCapsule
  context_builder.py   # Retrieve → rank → compress memory items
  confidence.py        # Φ(C) confidence scoring
  conflict.py          # MCD: Memory Conflict Detection (numeric, negation, antonym)
  prompt_shaper.py     # CDPS: Confidence-Driven Prompt Shaping
  sufficiency.py       # CSE: Context Sufficiency Estimation
  compression.py       # SECR: Selective Entry Compression and Ranking
  banner.txt           # ASCII banner shown on init
  models/
    claude_cli.py      # Claude CLI adapter (--max-budget-usd passthrough)
    ollama_local.py    # Ollama local model adapter
    simulated.py       # Deterministic simulation for benchmarks
  storage/
    sqlite_memory.py   # SQLite + FTS5 persistent memory store with trace table

benches/
  datasets/
    real/
      queries_real.json           # 240 seed queries (reproducibility anchor)
      queries_seeds_combined.json # Combined seed set
      queries_mtbench.json        # MT-Bench queries
      queries_nq.json             # Natural Questions sample
      queries_public_1197316.json # Full 1.2M real-query benchmark
      queries_public_788823.json  # Earlier 788K benchmark snapshot
      README.md
    synthetic/
      queries_200.json            # 200 synthetic ablation queries
      queries_2000.json           # 2,000 synthetic queries
      queries_5000.json           # 5,000 synthetic queries (primary ablation set)
      README.md
    mcd/
      contradictions_100.json     # 100 MCD stress-test contradiction pairs
  results_real_1197316.json       # CDDR benchmark results — 1.2M real queries
  results_mcd_stress.json         # MCD stress test results — 200 pairs
  results_queries_200.json        # Routing ablation results — N=200
  results_queries_2000.json       # Routing ablation results — N=2,000
  results_queries_5000.json       # Routing ablation results — N=5,000
  results_langgraph_queries_5000.json  # LangGraph baseline comparison — N=5,000
  results_real_788823.json        # Earlier 788K benchmark results
  fetch_real_1m.py                # Assemble 1.2M benchmark from HuggingFace
  generate_200.py                 # Synthetic query generator (--scale N)
  run_routing.py                  # Simulated ablation runner
  run_real.py                     # Real-query benchmark runner
  run_mcd_stress.py               # MCD stress test runner
  run_langgraph_compare.py        # LangGraph baseline comparison runner
  run_iterations.py               # Multi-iteration ablation runner
  eval_pipeline.py                # Evaluation pipeline utilities
  expand_real.py                  # Paraphrase expansion for real queries

tests/
  test_action_router.py           # Action router unit tests
  test_trace_budget.py            # Trace table and budget passthrough tests
```

## Troubleshooting

**`pip install -e .` fails.** Upgrade pip: `pip install --upgrade pip setuptools` (pre-22.0 pip cannot build from `pyproject.toml`).

**`python3` gives Python 3.9 on macOS.** Use `python3.12 -m venv venv` explicitly (`brew install python@3.12`).

**`lightify query` errors with "Claude CLI not found".** Install from [docs.anthropic.com/claude-code](https://docs.anthropic.com/en/docs/claude-code).

**Ollama not running.** Lightify falls back to API tiers — start with `ollama serve`.

**Multiple Lightify instances interfering.** Use `LIGHTIFY_HOME` to give each instance its own directory and SQLite database.

## Paper

> **Lightify: Cost-Aware Adaptive LLM Routing via Retrieval Confidence and Conflict-Aware Escalation**
> Pavan Maddula [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5756--100X-a6ce39?logo=orcid)](https://orcid.org/0009-0000-5756-100X). *IEEE Access*, 2026.

```bibtex
@article{maddula2026lightify,
  title   = {Lightify: Cost-Aware Adaptive {LLM} Routing via Retrieval Confidence
             and Conflict-Aware Escalation},
  author  = {Maddula, Pavan},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
