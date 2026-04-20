# Lightify

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Knowledge-aware model routing for LLM inference optimization.**

Lightify is the first model routing system that conditions tier selection on the *temporal consistency* of a persistent knowledge base. Instead of asking "Is this question hard?", Lightify asks "Is our knowledge about this topic still consistent?" -- routing to cheap local models when context is reliable and escalating to frontier models when contradictions are detected.

## Key Results

Benchmarks were run in April 2026 against these specific model versions (resolved from Claude CLI aliases at test time):

| Tier | Alias | Model Version |
|------|-------|---------------|
| Tier-1 (local) | `gemma3:1b` | Gemma 3 (1B, Ollama) |
| Tier-2 (mid) | `haiku` / `sonnet` | Claude Haiku 4.5 / Sonnet 4.6 |
| Tier-3 (frontier) | `opus` | Claude Opus 4.6 |

| Approach | Cost | Latency | Tokens Out |
|----------|------|---------|------------|
| Raw Claude Opus 4.6 | $0.076 | 5,425 ms | 272 |
| Always Sonnet 4.6 | $0.078 | 7,449 ms | 344 |
| Always Haiku 4.5 | $0.017 | 4,275 ms | 346 |
| Local Gemma 3 (1B) | $0.000 | 6,741 ms | 1,004 |
| **Lightify** | **$0.000** | **339 ms** | **22** |

On high-confidence queries with stable knowledge, Lightify routes to local models at **zero API cost** and reduces latency by **92%**. Confidence-Driven Prompt Shaping (CDPS) produces focused answers, reducing output tokens by up to **46x**.

## Architecture

Lightify operates as a three-stage pipeline: **Context Retrieval**, **Context Processing**, and **Routing & Inference**.

```
                         LIGHTIFY PIPELINE
  ================================================================

  User Query
       |
       v
  +--------------------+
  | PERSISTENT MEMORY   |  SQLite + FTS5, confidence-scored entries,
  | (Retrieval)         |  content-hash dedup, WAL mode
  +--------------------+
       |
       v
  +--------------------+
  | CONTEXT BUILDER     |  Filter (word-boundary topic match)
  | (Rank + Compress)   |  Rank  (Jaccard + recency + confidence)
  |                     |  Compress (code-block-aware stopword removal)
  +--------------------+
       |
       v
  +--------------------+
  | CSE                 |  Context Sufficiency Estimation
  | S(C,q) = 1 iff     |  Coverage >= sigma AND Phi(C) >= sigma_conf
  +--------------------+
       |
       v
  +--------------------+
  | MCD                 |  Memory Conflict Detection
  | (Negation,          |  Detects temporal contradictions across
  |  Numerical,         |  retrieved items; penalizes confidence
  |  Antonym)           |  and forces tier escalation
  +--------------------+
       |
       v
  +--------------------+
  | CDPS                |  Confidence-Driven Prompt Shaping
  | High  -> concise    |  Adapts prompt template to knowledge
  | Med   -> cite ctx   |  reliability (3 templates)
  | Low   -> meta-cog   |
  +--------------------+
       |
       v
  +--------------------+
  | SECR                |  Self-Evolving Compression Rules
  | (learns shorthands  |  Learns frequent phrases -> abbreviations
  |  from patterns)     |  over time
  +--------------------+
       |
       v
  +--------------------+
  | CDDR ROUTER         |  Confidence-Driven Dynamic Routing
  | R(q,C) = t_j where  |  R(q,C) = t_j where j = min{m: Phi(C) >= tau_m}
  | j = min{...}        |  tau_1 = 0.45, tau_2 = 0.30
  +----+---+---+--------+
       |   |   |
       v   v   v
  +------+ +------+ +----------+
  |Tier 1| |Tier 2| |  Tier 3  |
  |Local | |Sonnet| |  Opus    |
  |SLM   | |(API) | | (API)    |
  |$0    | |$$    | | $$$      |
  +------+ +------+ +----------+
       \     |     /
        v    v    v
       Response
           |
           v  (background)
       Update memory usage stats
       Evolve SECR rules
```

A Mermaid diagram source is available at [`docs/architecture.mmd`](docs/architecture.mmd).

## Five Novel Capabilities

1. **Local zero-cost inference tier** -- On-device models (Ollama) for high-confidence queries at $0.
2. **Retrieval-confidence routing** -- Evaluates context quality *before* generation, not after.
3. **Persistent memory with confidence scoring** -- SQLite + FTS5 store with usage tracking, recency decay, and content-hash deduplication.
4. **Memory Conflict Detection (MCD)** -- Identifies temporal contradictions (negation, numerical, antonym) across stored knowledge and forces escalation to frontier models.
5. **Confidence-Driven Prompt Shaping (CDPS)** -- Adapts generation strategy to knowledge reliability, producing concise answers when confidence is high and meta-cognitive reasoning when it is low.

## Installation

### Prerequisites

- Python 3.10+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (for Tier-2/3 API inference)
- [Ollama](https://ollama.com/) (optional, for Tier-1 local inference)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/pavanmanikanta31/lightify.git
cd lightify

# Create a Python 3.10+ virtual environment
# (macOS system python3 is often 3.9 — use python3.12 explicitly if needed)
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip (required — older pip cannot install editable pyproject.toml projects)
pip install --upgrade pip setuptools

# Install in editable mode
pip install -e .

# First-time setup (verifies dependencies, seeds knowledge base)
lightify init
```

### Setup with script

```bash
chmod +x setup.sh
./setup.sh
```

### Ollama (Local Tier-1)

```bash
# Install Ollama
brew install ollama

# Pull a local model (815MB, fast on Apple Silicon)
ollama pull gemma3:1b

# Start the Ollama server
ollama serve
```

## Usage

### Query through Lightify

```bash
# Route query to cheapest viable model
lightify query "What prevents true thread parallelism in Python?"

# Direct baseline (raw Claude Opus, no Lightify)
lightify query --baseline "What is the Python GIL?"

# Side-by-side comparison: WITH vs WITHOUT Lightify
lightify query --compare "How does Rust handle memory safety?"
```

### Mode Presets

```bash
# Optimize for speed (prefer local model)
lightify query --fast "What is a REST API?"

# Optimize for cost (avoid expensive API calls)
lightify query --cheap "Explain Docker containers"

# Optimize for quality (prefer frontier model)
lightify query --quality "Compare microservices vs monolith architectures"
```

### Force a Specific Model

```bash
lightify query --model haiku "Quick question"
lightify query --model sonnet "Medium complexity"
lightify query --model opus "Complex reasoning task"
```

### Memory Management

```bash
# Browse the knowledge base
lightify memory list

# Add a fact to the knowledge base
lightify memory add "Python 3.13 removes the GIL with free-threading" --topic python

# Search the knowledge base
lightify memory search "memory safety"

# Seed with default knowledge
lightify memory seed
```

### Model Configuration

```bash
# Show current model per tier
lightify config show

# List all available models
lightify config models

# Change Tier-1 to a larger local model
lightify config set tier1 gemma3:4b

# Reset to defaults
lightify config reset
```

### Status

```bash
lightify status
```

## Benchmarks

### Run Benchmarks

```bash
# Simulated ablation study (no API calls needed)
python -m benches.run_bench

# Real API benchmark (requires Claude CLI + Ollama)
python -m benches.run_real_bench

# 20-query grid benchmark across 6 routing strategies
python -m benches.run_grid_bench

# Full evaluation pipeline with oracle + LLM-as-judge
python -m benches.eval_pipeline
```

### Preliminary Results (20-Query Grid)

| Approach | Avg Cost | Avg Latency | Avg Quality |
|----------|----------|-------------|-------------|
| Local Gemma | $0.000 | 7,320 ms | 96 |
| Haiku | $0.023 | 7,113 ms | 92 |
| Sonnet | $0.041 | 12,060 ms | 89 |
| Opus | $0.087 | 11,162 ms | 92 |
| **Lightify** | **$0.037** | **8,019 ms** | **81** |
| LF --fast | $0.028 | 8,028 ms | 84 |

Lightify achieves **57% lower average cost** than always-frontier inference while correctly escalating on all conflict queries. Quality scores reflect keyword-match methodology; LLM-as-judge evaluation is planned for the full study.

### Ablation Study (Simulated)

| Variant | Cost | Tier-2 % | Tier-3 % | Conflicts | SECR Rules |
|---------|------|----------|----------|-----------|------------|
| Naive RAG | $0.084 | 0% | 100% | 0 | 0 |
| Caveman-only | $0.083 | 0% | 100% | 0 | 0 |
| Hybrid Lightify | $0.077 | 71% | 29% | 8 | 0 |
| **Full Lightify** | **$0.075** | **71%** | **29%** | **7** | **135** |

## Project Structure

```
lightify/
  __init__.py           # Package entry point
  cli.py                # CLI interface (init, query, bench, status, memory, config)
  pipeline.py           # Simulated pipeline (for reproducible ablation studies)
  pipeline_real.py      # Real pipeline (Claude CLI + Ollama)
  types.py              # Shared types (Tier, MemoryItem, ContextCapsule, etc.)
  config.py             # Model configuration per tier
  context_builder.py    # Filter, rank (Jaccard), compress
  compression.py        # Code-block-aware compression + SECR engine
  confidence.py         # Confidence scoring with Platt scaling
  conflict.py           # MCD: Memory Conflict Detection
  prompt_shaper.py      # CDPS: Confidence-Driven Prompt Shaping
  router.py             # CDDR: Confidence-Driven Dynamic Routing
  sufficiency.py        # CSE: Context Sufficiency Estimation
  banner.txt            # ASCII art banner
  models/
    simulated.py        # Deterministic model simulation for benchmarks
    claude_cli.py       # Claude CLI adapter (Haiku/Sonnet/Opus)
    ollama_local.py     # Ollama local model adapter (Gemma, Phi, Llama)
  storage/
    sqlite_memory.py    # SQLite + FTS5 persistent memory store
benches/
  run_bench.py          # Simulated ablation benchmark
  run_real_bench.py     # Real API benchmark (8 queries)
  run_full_bench.py     # Extended benchmark
  run_grid_bench.py     # 20-query grid across 6 strategies
  eval_pipeline.py      # Oracle + LLM-as-judge evaluation
  queries_20.py         # 20 evaluation queries (6 categories)
  generate_data.py      # Memory seeding with knowledge items
docs/
  architecture.mmd      # Mermaid diagram source
```

## Troubleshooting

**`pip install -e .` fails with "setup.py not found".**
Upgrade pip: `pip install --upgrade pip setuptools`. Older pip (pre-22.0) cannot build editable installs from `pyproject.toml` alone.

**`python3` gives Python 3.9 on macOS.**
The system `python3` is tied to Xcode and often 3.9. Install a newer version: `brew install python@3.12`, then use `python3.12 -m venv venv` explicitly.

**`lightify query` errors with "Claude CLI not found".**
Install the Claude CLI: see [docs.anthropic.com/claude-code](https://docs.anthropic.com/en/docs/claude-code). Tier-2/3 API inference requires it.

**Ollama isn't running.**
Lightify falls back to API tiers if Ollama is unavailable — but you lose the zero-cost local tier. Start it with `ollama serve`.

**`lightify memory seed` says memory already exists.**
Use `lightify memory clear` first if you want to reseed from scratch.

## Paper

This work is described in:

> **Lightify: Knowledge-Aware Model Routing via Temporal Consistency of Persistent Memory for LLM Inference Optimization**
> Pavan Maddula, 2026.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

### BibTeX

```bibtex
@article{maddula2026lightify,
  title     = {Lightify: Knowledge-Aware Model Routing via Temporal Consistency
               of Persistent Memory for LLM Inference Optimization},
  author    = {Maddula, Pavan},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
