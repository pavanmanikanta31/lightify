# Lightify

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5756--100X-a6ce39?logo=orcid)](https://orcid.org/0009-0000-5756-100X)

**Cost-aware LLM orchestration middleware for agent harnesses.**

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

**MCD** (Memory Conflict Detection) scans retrieved items for temporal contradictions (numeric revisions, negations, antonym pairs) and forces escalation when found.

**CDPS** (Confidence-Driven Prompt Shaping) adapts prompt verbosity to retrieval reliability.

## Results

Evaluated on 1,197,316 real user queries from five public corpora (MS MARCO v2.1, WildChat-1M, Natural Questions, MMLU, GSM8K) — all exact-string unique, none authored by us.

| Policy | Policy-Oracle Agreement | Cost/query | Cost vs Opus-only |
|---|---|---|---|
| Naïve Opus-only baseline *(reference)* | 0.000 | $0.076 | — |
| Hand-coded length gateway | 0.324 | $0.0196 | −74% |
| Hand-coded keyword router | 0.794 | $0.0007 | −99% |
| Lightify CDDR (all 1.2M incl. non-English) | 0.913 [0.912, 0.913] | $0.0025 | −96% |
| **Lightify CDDR** (English subset, N=1.04M) | **0.958 [0.957, 0.958]** | **$0.0012** | **−98%** |

> **Reading the table:** Opus-only is the cost baseline (−0%). The keyword router is cheapest (−99%) because it routes nearly everything to Tier-1, but its POA drops to 0.794. Lightify achieves 0.958 POA at −98% cost — more accurate than any baseline while still 98% cheaper than Opus-only. Cost and POA must be read together; a cheaper policy that routes incorrectly is not a win.

Per-source breakdown: MS MARCO / NQ / MMLU = 1.000 (factoid-lookup dominated); WildChat (English) = 0.625; GSM8K = 0.607. Non-English WildChat subset (N=154K) = 0.604.

English subset = queries with pure ASCII text (N=1,043,220); non-English = WildChat queries containing non-ASCII characters (N=154,096).

*Policy-Oracle Agreement (POA) measures routing-decision agreement with a category-level oracle, not per-query output correctness. Per-query cost is a tier-cost projection applied to routing decisions, not measured API spend on 1.2M queries; live spend is reported only in the paper's §V.C 20-query pilot.*

**MCD stress test** (200 pairs: 100 contradictions + 100 controls): Precision 0.943 [0.814, 0.984], Recall 0.330 [0.246, 0.427]. High-precision within its lexical signal set (numeric ratio, 32-entry antonym dict, negation detector).

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
lightify query --fast "What is a REST API?"       # prefer local
lightify query --cheap "Explain Docker containers" # avoid API spend
lightify query --quality "Compare microservices vs monolith"  # frontier

# Force a specific tier
lightify query --model haiku "Quick question"
lightify query --model sonnet "Medium complexity"
lightify query --model opus "Complex reasoning"
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

## Benchmarks

Seed queries (240 hand-authored, used as reproducibility anchor) are in `benches/datasets/real/queries_real.json`.

```bash
# Simulated ablation (no API calls, reproducible)
python benches/run_routing.py

# Real-query benchmark (requires Claude CLI + Ollama)
python benches/run_real.py

# MCD stress test (200 contradiction/control pairs)
python benches/run_mcd_stress.py

# LangGraph-style baseline comparison
python benches/run_langgraph_compare.py
```

Large derived datasets (1.2M queries, result JSONs) are not in this repo. See the paper for the assembly script (`benches/fetch_real_1m.py`) and dataset provenance.

## Project structure

```
lightify/
  cli.py               # CLI (init, query, bench, status, memory, config)
  pipeline.py          # Simulated pipeline (reproducible ablation)
  pipeline_real.py     # Real pipeline (Claude CLI + Ollama)
  router.py            # CDDR routing logic
  action_router.py     # Per-action regex overlay (optional, --action-routing)
  config.py            # Tier model configuration
  types.py             # Shared types (Tier, MemoryItem, ContextCapsule)
  context_builder.py   # Retrieve → rank → compress
  confidence.py        # Confidence scoring
  conflict.py          # MCD: Memory Conflict Detection
  prompt_shaper.py     # CDPS: Confidence-Driven Prompt Shaping
  sufficiency.py       # CSE: Context Sufficiency Estimation
  compression.py       # SECR compression rules
  models/
    claude_cli.py      # Claude CLI adapter
    ollama_local.py    # Ollama local model adapter
    simulated.py       # Deterministic simulation for benchmarks
  storage/
    sqlite_memory.py   # SQLite + FTS5 persistent memory store
benches/
  datasets/real/
    queries_real.json          # 240 seed queries (reproducibility anchor)
  datasets/mcd/
    contradictions_100.json    # 100 MCD stress-test pairs
  datasets/synthetic/
    queries_200.json            # 200 simulated ablation queries
  results_real_1197316.json    # CDDR benchmark results (1.2M queries)
  results_mcd_stress.json      # MCD stress test results (200 pairs)
  run_routing.py
  run_real.py
  run_mcd_stress.py
  run_langgraph_compare.py
  fetch_real_1m.py
  generate_200.py
tests/
```

## Troubleshooting

**`pip install -e .` fails.** Upgrade pip: `pip install --upgrade pip setuptools` (pre-22.0 pip cannot build from `pyproject.toml`).

**`python3` gives Python 3.9 on macOS.** Use `python3.12 -m venv venv` explicitly (`brew install python@3.12`).

**`lightify query` errors with "Claude CLI not found".** Install from [docs.anthropic.com/claude-code](https://docs.anthropic.com/en/docs/claude-code).

**Ollama not running.** Lightify falls back to API tiers — start with `ollama serve`.

## Paper

> **Lightify: Cost-Aware LLM Orchestration via Retrieval-Confidence Routing, Conflict Detection, and Local-First Inference**
> Pavan Maddula [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5756--100X-a6ce39?logo=orcid)](https://orcid.org/0009-0000-5756-100X). *IEEE Access*, 2026 (under review).
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

```bibtex
@article{maddula2026lightify,
  title   = {Lightify: Cost-Aware {LLM} Orchestration via Retrieval-Confidence
             Routing, Conflict Detection, and Local-First Inference},
  author  = {Maddula, Pavan},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review. Preprint: arXiv:XXXX.XXXXX}
}
```

## License

MIT — see [LICENSE](LICENSE).
