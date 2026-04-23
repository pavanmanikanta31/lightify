# Legitimate / hand-curated query benchmark

`queries_real.json` is a hand-curated companion to the synthetic
benchmark under `../synthetic/`. Each query was written to match how a
real developer, SRE, or researcher would phrase the question — full
sentences, partial sentences, stack-trace pastes, casual shorthand,
and so on. No templates; no filler substitution.

## Provenance

Queries were authored drawing on the phrasing patterns of five public
sources:

| Source | Category mapped to | Notes |
|---|---|---|
| MT-Bench reference prompts (LMSYS, Apache-2.0) | reasoning, code | open-ended, multi-turn phrasing |
| HumanEval task descriptions (OpenAI) | code, large_code | "implement a function that..." |
| StackOverflow question titles (CC-BY-SA) | bash_like, code | imperative bash/ops questions |
| Linux man-page "DESCRIPTION" questions | bash_like | "what does -X do" |
| MMLU-Pro sample items | short_lookup, cold_knowledge | factual recall |

Nothing is copy-pasted verbatim from any source — each query is
independently authored to match the *style* of real queries in these
corpora. This avoids licensing ambiguity while exercising the router
on realistic phrasing.

## Schema

Identical to `../synthetic/README.md`:

```json
{
  "id": "real-bash-1",
  "category": "bash_like",
  "query": "how do I find the 10 largest files under /var/log?",
  "oracle_tier": "SMALL",
  "has_contradiction": false
}
```

## Running

```bash
python3 benches/run_n200_routing.py --dataset ../real/queries_real.json
python3 benches/run_langgraph_compare.py --dataset ../real/queries_real.json
```

## Reference results

Results are reported in `../../results_real.json` and
`../../results_langgraph_real.json` when the benches are re-run.
