# Lightify Synthetic Routing Benchmark

Three dataset sizes are included, all generated from the same 48 query
templates by `benches/generate_200.py` with `random.Random(42)` for
reproducibility. Running the generator again with the same seed and
`--scale` reproduces each file byte-for-byte.

**All three datasets are 100% exact-string unique.** The combinatorial
capacity of the template × filler space is approximately 80,000 queries,
so generating 5,000 unique samples leaves ~75,000 of headroom for
scaling further.

| File | N | Unique | Size | Generation command |
|---|---|---|---|---|
| `queries_200.json` | 200 | 200 (100%) | 36 KB | `python3 benches/generate_200.py --scale 1` |
| `queries_2000.json` | 2,000 | 2,000 (100%) | 370 KB | `python3 benches/generate_200.py --scale 10` |
| `queries_5000.json` | 5,000 | 5,000 (100%) | 930 KB | `python3 benches/generate_200.py --scale 25` |

## Schema

Each row is a JSON object:

```json
{
  "id": "bash_like-1",
  "category": "bash_like",
  "query": "git status HEAD",
  "oracle_tier": "SMALL",
  "has_contradiction": false
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | `{category}-{1-based index within category}` |
| `category` | string | one of 7 categories (below) |
| `query` | string | the user-facing query, unique across the entire file |
| `oracle_tier` | string | gold tier label: `SMALL` / `MID` / `FRONTIER` |
| `has_contradiction` | bool | true for the `conflict` category only |

## Category composition (identical across scales)

| Category | % | Oracle tier | MCD | Templates | Example |
|---|---|---|---|---|---|
| `bash_like` | 20% | SMALL | no | 24 | `kubectl get pods -n staging` |
| `short_lookup` | 20% | SMALL | no | 15 | `what does OIDC stand for?` |
| `reasoning` | 15% | MID | no | 13 | `why does a cache thrash when memory is low?` |
| `code` | 15% | MID | no | 10 | `write a Rust function that chunks an iterator` |
| `large_code` | 10% | FRONTIER | no | 10 | `rewrite the full auth module in Go` |
| `conflict` | 10% | FRONTIER | yes | 10 | `is v7 production or staging?` |
| `cold_knowledge` | 10% | FRONTIER | no | 19 | `what is the capital of Tuvalu?` |

Template and filler counts total **48 templates** with **3–24 fillers
per slot**; combinatorial capacity per category exceeds the per-category
sample count by 10–400× at every scale.

## Generation procedure

1. For each category, draw queries one at a time by:
   - Pick a template uniformly at random from the category's template set.
   - Fill each `{slot}` with a uniform-random value from the slot's filler list.
   - If the produced string was already seen in this category, retry
     (up to `20 * N` attempts).
2. Assign gold oracle tier from the category's label.
3. Shuffle the full list under the same seed.

Templates and fillers are defined in `benches/generate_200.py`.

## Context-confidence injection (used by the routing benches)

The benches do not perform real retrieval; instead, per-query context
confidence $\phi$ is drawn from a category-specific uniform distribution
calibrated against the 20-query live pilot:

| Category | $\phi$ distribution |
|---|---|
| `bash_like` | $U(0.50, 0.85)$ |
| `short_lookup` | $U(0.50, 0.85)$ |
| `reasoning` | $U(0.30, 0.55)$ |
| `code` | $U(0.30, 0.55)$ |
| `large_code` | $U(0.25, 0.45)$ |
| `conflict` | $U(0.30, 0.55)$ + 1 injected contradiction |
| `cold_knowledge` | $U(0.02, 0.18)$ |

Coverage is derived from $\phi$ with $\pm 0.1$ uniform noise. Conflict
queries inject a synthetic contradiction into the `ContextCapsule` so
that MCD fires.

## Reproducibility

```bash
# regenerate all three datasets
python3 benches/generate_200.py --scale 1    # N=200
python3 benches/generate_200.py --scale 10   # N=2000
python3 benches/generate_200.py --scale 25   # N=5000

# run all benches
python3 benches/run_n200_routing.py --dataset queries_200.json
python3 benches/run_n200_routing.py --dataset queries_2000.json
python3 benches/run_n200_routing.py --dataset queries_5000.json
python3 benches/run_langgraph_compare.py --dataset queries_200.json
python3 benches/run_langgraph_compare.py --dataset queries_2000.json
python3 benches/run_langgraph_compare.py --dataset queries_5000.json
```

All results are written to `benches/results_*.json`.

## Reference results at N=5,000 (seed=42, 100% unique)

| Policy | RA | 95% CI | OSR | USR | $/query |
|---|---|---|---|---|---|
| **Lightify CDDR** | **0.702** | **[0.689, 0.715]** | 0.000 | 0.298 | **$0.0161** |
| Lightify CDDR + action overlay | 0.705 | [0.692, 0.716] | 0.084 | 0.211 | $0.0174 |
| LangGraph keyword-branch | 0.516 | — | 0.003 | 0.480 | $0.0028 |
| LangGraph + length gateway | 0.484 | — | 0.227 | 0.288 | $0.0164 |
| LangGraph default (Opus only) | 0.300 | [0.287, 0.313] | 0.700 | 0.000 | $0.0760 |

### Per-category effect of the per-action overlay

| Category | N | CDDR RA | +action RA | ΔRA |
|---|---|---|---|---|
| bash_like | 1000 | 1.000 | 0.929 | −0.071 |
| short_lookup | 1000 | 1.000 | 0.651 | **−0.349** |
| reasoning | 750 | 0.607 | 0.988 | **+0.381** |
| code | 750 | 0.580 | 1.000 | **+0.420** |
| large_code | 500 | 0.236 | 0.264 | +0.028 |
| conflict | 500 | 0.000 | 0.000 | 0.000 |
| cold_knowledge | 500 | 1.000 | 0.642 | **−0.358** |

The overlay trades 35 pp on lookup/cold categories for 40 pp on
code/reasoning. Net is approximately flat; the large heterogeneity is
the main finding and motivates per-category dispatch.

## Limitations

- **Oracle is category-level**, not per-query ground truth. A live oracle
  (running each tier against each query, scoring correctness) would give
  stronger routing-accuracy claims but cannot be executed at N=5,000
  without large API spend.
- **Context confidence is injected**, so the bench does not exercise the
  retrieval stack. This is deliberate: the routing metric should isolate
  policy error from retrieval error.
- **Routing accuracy does not measure output quality.** The headline
  live-evidence numbers (Section V.A, V.C, V.D of the paper) come from
  the 20-query live pilot; the N=5,000 synthetic benchmark provides
  statistical power on the routing decision in isolation.
- **Conflict category.** Both CDDR and the overlay stop at MID rather
  than FRONTIER for conflict queries because the current MCD override
  escalates to Tier-2, not Tier-3. Upgrading this override is flagged as
  a concrete next step in the paper.
