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
uv run jev-bench bench --limit 5        # live run, first 5 of 45 eval questions
uv run jev-bench injection-eval --dry-run   # scores the injection-screening guard
uv run jev-bench serve                  # FastAPI+SSE live demo UI at localhost:8000
```

Groq (`qwen/qwen3.8-27b`) is the default LLM backend; Gemini and a local Ollama model are also
implemented and swappable via `deciders/factory.py` — see `docs/METHODOLOGY.md` for why the default
changed twice during development (short version: free-tier quotas and CPU-bound local latency).

## Example result

A real (not simulated) run against live Groq and Jev APIs, 34 of 45 eval questions completed before
hitting Groq's daily token quota (`reports/` — gitignored, regenerate with `jev-bench bench`):

| | LLM (Groq, qwen3.8-27b) | Jev |
|---|---|---|
| Selection quality vs. gold (F1) | **0.861** | 0.569 |
| Latency (median) | 31.7s | **1.2s** (~25x faster) |
| Cost per 1,000 queries | $5.89 | **$0.24** (~25x cheaper) |

Latency difference is statistically significant (Wilcoxon p≈0, paired bootstrap 95% CI entirely
positive). The quality gap (29 F1 points) exceeds this project's own pre-registered kill-switch
threshold (3 points) — the honest headline is **"much faster and cheaper, but meaningfully less
accurate at this untuned threshold,"** not an unqualified win. See `docs/METHODOLOGY.md` for the
iso-recall comparison, caveats on this specific run (single pass, no repetitions, threshold not
tuned on a held-out split), and what would need to change before trusting these numbers as final.

## Project layout

- `src/jev_bench/graph/` — the one shared graph (`builder.py`, `nodes.py`, `state.py`)
- `src/jev_bench/deciders/` — `jev.py`, `groq.py`, `gemini.py`, `ollama.py` (live backends),
  `fake.py` (`ScriptedDecider`, used by tests and `--dry-run`), `cassette.py` (record/replay)
- `src/jev_bench/bench/` — `runner.py`, `analysis.py` (paired stats, PR curves), `report.py`,
  `injection_eval.py`
- `src/jev_bench/api/` + `web/` — the live demo server and static frontend
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
