# jev-bench

An A/B benchmark of a CRAG-style agentic document-retrieval workflow: the **same compiled LangGraph
graph**, run with two interchangeable decision backends — an LLM (qwen3.8-27b on Groq by default;
Gemini and Ollama are swappable) and **Jev** (TypeSafe AI's "System One" model, which returns typed
probabilities instead of generated text).

**The question:** how much latency and cost does a CRAG-style workflow save by replacing its LLM
decision nodes with Jev — using *minimal infrastructure* (a CPU-only machine and a small embedding
model) and *minimal tuning* (two thresholds) — on a *hard multi-hop QA benchmark* (HotpotQA), while
keeping good accuracy?

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

## The setup: minimal by design

- **Hard multi-hop benchmark.** HotpotQA distractor setting, dev set — every question is rated
  "hard". Each question comes with 10 Wikipedia paragraphs, of which 2 are gold and both must be
  found; for the "bridge" questions (~71% of this set), the second paragraph is only reachable through an entity
  named in the first (the rest are "comparison" questions).
- **Minimal infrastructure.** A CPU-only machine, no GPU. Retrieval ranks the 10 paragraphs with
  `nomic-embed-text` (137M parameters, 274 MB) run locally through Ollama — no vector database, no
  reranker, no fine-tuning. Jev and the LLM are called as hosted APIs.
- **Minimal tuning.** Only two Jev thresholds are tuned (relevance 0.40, injection 0.50), on 80 train
  + 10 validation questions; the other three (evidence, contradiction, sufficiency) stay at the
  vendor's defaults. Nothing is re-tuned for the 1,000-question test set, and both backends get
  identical prompt wording.

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
backends can't silently drift into answering different questions. The retriever is injected the
same way (`build_graph(decider, retriever)`, no implicit default), so both variants always read the
identical ranking: live runs use the Ollama embedding retriever, while tests and `--dry-run` use a
deterministic scripted one and need no model server.

## Quickstart

```bash
uv sync --all-extras
cp .env.example .env   # fill in GROQ_API_KEY and TYPESAFE_API_KEY (GOOGLE_API_KEY optional)

uv run jev-bench bench --dry-run            # two scripted deciders: no API keys, no network, no Ollama
uv run jev-bench injection-eval --dry-run   # scores the injection-screening guard
```

Live runs also need a local [Ollama](https://ollama.com) server for the embedding retriever:

```bash
ollama pull nomic-embed-text                # one-time, 274 MB

uv run jev-bench bench --limit 5            # live run, first 5 questions
uv run jev-bench bench --split test --limit 20 --track   # Groq vs. Jev head-to-head, logged to MLflow
uv run jev-bench bench --only jev --split test --track   # Jev alone on all 1,000 test questions (~1 h)
uv run jev-bench injection-eval --split test --only jev  # injection guard on the held-out probes
uv run jev-bench tune-thresholds            # dry-run threshold tuning on train/val (--apply writes)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # browse tracked runs
```

Groq (`qwen/qwen3.8-27b`) is the default LLM backend; Gemini and a local Ollama model are also
implemented and swappable via `deciders/factory.py` — see `docs/METHODOLOGY.md` for why the default
changed twice during development (short version: free-tier quotas and CPU-bound local latency).
Groq's free tier allows only ~30 full queries a day, which is why `--only` exists. (`jev-bench
serve`, the older live-race demo UI, is still wired to Gemini + Jev and has not been updated.)

## Example result

The headline evaluation is a **1,000-question held-out test set** (`test` in `eval/splits.json`):
HotpotQA questions that were never used for tuning and are disjoint from the tuning data. Jev's two
thresholds (relevance 0.40, injection 0.50) were tuned on 80 train / 10 val queries and **not
re-tuned** for this set, so it measures how the tuned system generalizes with minimal tuning.
Document ranking is semantic (local embeddings) and shared by both backends. See
`docs/METHODOLOGY.md` for the full process.

**Jev on all 1,000 test queries** (live API, 0 failures):

| | Jev |
|---|---|
| Selection quality vs. gold (F1, 95% bootstrap CI) | **0.719** [0.703, 0.735] |
| Precision / recall | 0.776 / 0.747 |
| F1 by question type | bridge 0.736 (n=710), comparison 0.678 (n=290) |
| Injection guard (200 probes), precision/recall | 1.00 / 1.00 |
| Median latency / cost for all 1,000 queries | 3.5s / $0.23 |

**Groq vs. Jev head-to-head — small sample.** Groq's free tier allows only about 30 queries a day
(measured: ~6k tokens per query against a 200k/day cap), so the head-to-head runs on 19 of the test
queries (20 were run; 1 was skipped on a Groq timeout). Groq was not run on the injection test.

| | LLM (Groq, qwen3.8-27b) | Jev |
|---|---|---|
| Selection quality vs. gold (F1, 95% bootstrap CI) | **0.686** [0.559, 0.809] | 0.640 [0.541, 0.729] |
| Latency (median) | 29.4s | **2.5s** (~12x faster) |
| Cost per 1,000 queries | $5.18 | **$0.21** (~25x cheaper) |

Speed and cost are clear (paired latency Wilcoxon p<0.0001). Quality is not settled: the paired
per-query F1 difference (Groq minus Jev) is +0.046 with a 95% CI of [-0.029, +0.127] (p=0.21; Groq
higher on 6 queries, Jev on 5, 8 tied). The point estimate exceeds this project's pre-registered
3-point kill switch, but the interval includes zero, so Groq's edge on accuracy is possible, not
established — and with only 19 queries it can't be. What *is* tightly measured is Jev on its own:
F1 0.719 ± 0.016. An earlier 10-query pilot (Groq 0.677 vs. Jev 0.533) was too small to conclude
anything and is superseded by the numbers above. Every per-query score is recorded in MLflow
(`jev-bench bench --track`).

A static, shareable version of this result lives at [`web/results.html`](web/results.html) — open
it directly in a browser (no server, no API keys, no network calls) for a presentation-ready summary
of the same numbers above.

## Project layout

- `src/jev_bench/graph/` — the one shared graph (`builder.py`, `nodes.py`, `state.py`)
- `src/jev_bench/deciders/` — `jev.py`, `groq.py`, `gemini.py`, `ollama.py` (live backends),
  `fake.py` (`ScriptedDecider`, used by tests and `--dry-run`), `cassette.py` (record/replay)
- `src/jev_bench/retrieval/` — `store.py` (`OllamaEmbeddingRetriever`, the semantic ranker) and
  `fake.py` (`ScriptedRetriever` for tests and `--dry-run`)
- `src/jev_bench/bench/` — `runner.py`, `analysis.py` (paired stats, PR curves), `report.py`,
  `tuning.py` (train/val threshold tuning), `tracking.py` (MLflow), `injection_eval.py`
- `src/jev_bench/api/` + `web/` — the older live-demo server and frontend, plus the standalone
  `web/results.html` results page (no server dependency)
- `eval/` — the frozen, seeded HotpotQA slice (`dataset.jsonl`: 1,100 questions; `splits.json`:
  80 train / 10 val / 10 pilot / 1,000 test; `DATA_CARD.md` for license/provenance) and the synthetic
  injection-probe set
- `scripts/` — the deterministic data-curation steps (`curate_eval_set.py`, `split_eval_queries.py`,
  `add_test_set.py`, `generate_injection_probes.py`)
- `docs/METHODOLOGY.md` — the authoritative record of what's implemented, what's deliberately
  deferred, and why
- `CLAUDE.md` — architecture notes for anyone (human or AI) working on this codebase

## Data & license

This repo's code is yours to use; `eval/dataset.jsonl` is a curated slice of HotpotQA
(**CC BY-SA 4.0**, share-alike) — see `eval/DATA_CARD.md` before redistributing derived label files
under different terms.

## Status

Core graph, both decider families, the semantic retriever, the bench/analysis/report pipeline with
MLflow tracking, threshold tuning, and the injection-probe eval are built and tested
(`uv run pytest tests/ -v`, zero live-API dependency). Deliberately not implemented, with reasoning:
the Gemini/Jev batching 2×2 (a real architecture extension, not a quick add) and multi-repetition
latency sampling — both detailed in `docs/METHODOLOGY.md`. Open item: the Groq-vs-Jev accuracy
comparison is only 19 queries because of Groq's free-tier limit; closing it needs a paid Groq tier
(about $7 and 7+ hours for all 1,000).
