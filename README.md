# jev-bench

An A/B benchmark of an agentic document-retrieval workflow: the **same compiled LangGraph graph**,
run with two interchangeable decision backends — an LLM (Groq, Gemini, or Ollama — swappable) and
**Jev** (TypeSafe AI's "System One" model, which returns typed probabilities instead of generated
text) — to measure how much latency and cost is recoverable by moving classification-style
decisions off an autoregressive LLM.

There is no generation step. The two variants retrieve, screen, and select documents from
[HotpotQA](https://hotpotqa.github.io/)'s distractor-setting context; the thing under test is the
**selected document set**, scored against HotpotQA's own gold `supporting_facts` labels — not an
LLM-judged answer. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full measurement design,
including several non-obvious bugs and vendor-quota surprises hit along the way.

## Why this comparison

Agentic workflows spend a large share of their LLM calls not *generating* anything but *deciding*
things — is this document relevant, is the selected set sufficient, should the system retry. Those
calls pay full autoregressive-LLM prices and latency for what is, functionally, a classification.
Jev is built specifically for that kind of decision. This project tests the claim directly, on one
concrete workflow, rather than taking it on faith.

## Architecture: one graph, no branching

`graph/builder.py::build_graph()` is the **only** graph constructor; every variant goes through it,
and only the `Decider` instance passed in differs. There is no provider branching anywhere in
`graph/` — this is mechanically enforced, not a convention: `tests/test_topology.py` AST-scans the
graph code for `isinstance`/`.name`-comparison patterns and asserts every backend compiles to an
identical node/edge topology.

```
retrieve → screen (fan-out screen_passage) → sufficiency → decide_retry → loop or end
```

Three typed, semantic decisions make up the `Decider` protocol — `screen_passage`,
`grade_sufficiency`, `decide_retry` — deliberately not a generic `judge(prompt)`, so the two
backends can't silently drift into answering different questions.

## Quickstart

```bash
uv sync --all-extras
cp .env.example .env   # fill in GROQ_API_KEY and TYPESAFE_API_KEY (GOOGLE_API_KEY optional)

uv run jev-bench bench --dry-run        # two ScriptedDeciders, no API keys, no network
uv run jev-bench bench --limit 5        # live run, first 5 of 100 eval questions
uv run jev-bench injection-eval --dry-run   # scores the injection-screening guard
uv run jev-bench serve                  # FastAPI+SSE live demo UI at localhost:8000
```

Groq (`qwen/qwen3.8-27b`) is the default LLM backend; Gemini and a local Ollama model are also
implemented and swappable via `deciders/factory.py` — see `docs/METHODOLOGY.md` for why the default
changed twice during development (short version: free-tier quotas and CPU-bound local latency).

## Example result

A real (not simulated) run against live Groq and Jev APIs, on the **10-query pilot split**
(`pilot` in `eval/splits.json` — held out from threshold tuning; a larger 1,000-question `test`
split has since been added for a properly powered evaluation). Jev's thresholds
were tuned on a separate 80-query train split and confirmed on a 10-query val split beforehand; see
`docs/METHODOLOGY.md`'s "Threshold tuning" section for the full process and the exact picked values
(relevance 0.40, injection 0.50). Document ranking is semantic (local embeddings), shared by both
variants:

| | LLM (Groq, qwen3.8-27b) | Jev |
|---|---|---|
| Selection quality vs. gold (F1, 95% bootstrap CI) | **0.677** [0.534, 0.817] | 0.533 [0.380, 0.670] |
| Injection-guard precision/recall | 1.00 / 1.00 | 1.00 / 1.00 |
| Latency (median) | 26.2s | **3.5s** (~7.5x faster) |
| Cost per 1,000 queries | $6.87 | **$0.24** (~28x cheaper) |

Speed and cost are clear: the latency difference is statistically significant (paired Wilcoxon
p=0.0020). Quality is not settled. The paired per-query F1 difference (Groq minus Jev) is +0.144
with a 95% CI of [-0.020, +0.327] (p=0.22; Groq scored higher on 5 queries, Jev on 1, 4 tied). The
point estimate is well past this project's pre-registered 3-point kill switch, but the interval
includes zero, so at n=10 the accuracy cost is likely, not established — and a single query moves
the mean F1 by 0.10. The honest headline is **"much faster and cheaper, with a likely accuracy cost
that this sample size can't confirm,"** not an unqualified win in either direction. Jev's weakness
here is recall (0.55 vs. 0.85). A first attempt at this run completed only 8 of 10 queries (network
errors) and was discarded; the numbers above are the complete second run, recorded per query in
MLflow (`jev-bench bench --track`). See `docs/METHODOLOGY.md` for the tuning methodology and caveats.

A static, shareable version of this result lives at [`web/results.html`](web/results.html) — open
it directly in a browser (no server, no API keys, no network calls) for a presentation-ready summary
of the same numbers above.

## Project layout

- `src/jev_bench/graph/` — the one shared graph (`builder.py`, `nodes.py`, `state.py`)
- `src/jev_bench/deciders/` — `jev.py`, `groq.py`, `gemini.py`, `ollama.py` (live backends),
  `fake.py` (`ScriptedDecider`, used by tests and `--dry-run`), `cassette.py` (record/replay)
- `src/jev_bench/bench/` — `runner.py`, `analysis.py` (paired stats, PR curves), `report.py`,
  `injection_eval.py`
- `src/jev_bench/api/` + `web/` — the live demo server and static frontend, plus the standalone
  `web/results.html` results summary (no server dependency)
- `eval/` — the frozen, seeded HotpotQA slice (`dataset.jsonl`, `DATA_CARD.md` for license/
  provenance) and the synthetic injection-probe set
- `docs/METHODOLOGY.md` — the authoritative record of what's implemented, what's deliberately
  deferred, and why
- `CLAUDE.md` — architecture notes for anyone (human or AI) working on this codebase

## Data & license

This repo's code is yours to use; `eval/dataset.jsonl` is a curated slice of HotpotQA
(**CC BY-SA 4.0**, share-alike) — see `eval/DATA_CARD.md` before redistributing derived label files
under different terms.

## Status

Core graph, both decider families, the bench/analysis/report pipeline, the live demo UI, and the
injection-probe eval are built and tested (`uv run pytest tests/ -v`, zero live-API dependency).
Deliberately not implemented, with reasoning: the Gemini/Jev batching 2×2 (a real architecture
extension, not a quick add) and multi-repetition latency sampling — both detailed in
`docs/METHODOLOGY.md`.
