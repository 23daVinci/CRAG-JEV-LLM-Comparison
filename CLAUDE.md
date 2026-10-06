# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`jev-bench` is an A/B benchmark of an agentic document-retrieval LangGraph workflow: the *same*
compiled graph run with two interchangeable decision backends — an LLM decider (Groq/`qwen/qwen3.8-27b`
by default; Gemini and Ollama also implemented, kept as swappable options) and Jev (TypeSafe AI) — to
measure how much latency and cost is recoverable by moving classification-style decisions off an
autoregressive LLM. There is no generation step; the output under test is the *selected document set*,
scored against HotpotQA's gold `supporting_facts`.

Read `docs/METHODOLOGY.md` before touching anything under `bench/`, `deciders/`, or `graph/` — it is
the authoritative, load-bearing record of what's implemented, what's deliberately deferred, and *why*
(including several non-obvious bugs/workarounds: Gemini's daily quota, Groq's reasoning-token leakage,
retry-layer stacking, etc.). Don't re-derive decisions that document already explains.

## Commands

Package/env manager is `uv`. Always run tools via `uv run ...`, never a bare `python`/`pytest`/etc.

```
uv sync --all-extras                      # install deps (incl. dev group)
uv run ruff check src tests scripts       # lint
uv run ruff format --check src tests scripts
uv run mypy src                           # strict typing, src/ only (not tests/scripts)
uv run pytest tests/ -v                   # full suite (live-marked tests excluded by default)
uv run pytest tests/unit/test_analysis.py -v            # single file
uv run pytest tests/unit/test_analysis.py::test_name -v # single test
```

This is the exact sequence CI (`.github/workflows/ci.yml`) runs, in that order, on a matrix of
Python 3.11/3.13 with `GOOGLE_API_KEY`/`TYPESAFE_API_KEY`/`GROQ_API_KEY` explicitly set to empty
strings as a tripwire — CI has zero keys, zero network, zero cost by design. Tests that need a real
API key must be marked `@pytest.mark.live`; `addopts = "-m 'not live'"` in `pyproject.toml` excludes
them by default. Never make a failing CI test pass by adding a key or weakening that marker exclusion
— mark the test `live` instead.

Dependency changes go through `uv add`/`uv remove`/`uv lock`, never hand-edited pins in
`pyproject.toml` with a hope that `uv.lock` still matches.

Running the benchmark itself:

```
uv run jev-bench bench --dry-run              # two ScriptedDeciders, no API keys needed
uv run jev-bench bench --limit 5              # live run, first 5 queries of eval/dataset.jsonl
uv run jev-bench injection-eval --dry-run     # scores the injection-screening guard
uv run jev-bench bench --split test --track  # tracked run: per-query scores logged to MLflow
uv run jev-bench bench --only jev --split test --track  # one backend only (Groq's free tier
                                              # can't afford 1,000 queries); no paired stats
uv run jev-bench tune-thresholds              # dry-run threshold tuning on train/val (--apply writes)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse tracked runs
uv run jev-bench serve                        # FastAPI+SSE live demo UI (web/), needs real keys
```

For CI/CD, dependency/lockfile, and pre-commit work, use the `devops` subagent — it owns that surface
and knows not to touch benchmark logic.

## Architecture

**One graph, two backends, no branching.** `graph/builder.py::build_graph()` is the single graph
constructor every variant goes through — only the `Decider` instance passed in differs. There is no
provider branching anywhere in `graph/` (`builder.py`, `nodes.py`, `edges.py`); this is mechanically
enforced, not just a convention — `tests/test_topology.py` AST-scans those three files for
`isinstance`/`.name`-comparison patterns and asserts both backends compile to an identical node/edge
topology. If a change needs to special-case a backend inside graph code, that's a sign the abstraction
belongs in `Decider` instead.

**The graph** (`graph/builder.py`, `graph/nodes.py`, `graph/state.py`): `retrieve` (semantic rank via
`OllamaEmbeddingRetriever` + batch, `retrieval/store.py`) → `screen` (fan-out `screen_passage` over
the batch, async) →
`sufficiency` (`grade_sufficiency` over selected docs so far) → `decide_retry` → loops back to
`retrieve` or ends, capped by `GraphConfig.max_attempts`. State is a `TypedDict`
(`RetrievalState`) threaded through every node; `trace: Annotated[list[NodeSpan], operator.add]`
accumulates telemetry spans via LangGraph's reducer. `build_graph(decider, retriever, cfg)` takes
both `decider` and `retriever` as required, explicit arguments — no implicit default for either —
so both variants in a comparison always read the identical ranking. Live runs use
`OllamaEmbeddingRetriever` (needs a local Ollama server with `nomic-embed-text` pulled); tests and
`--dry-run` use `retrieval/fake.py::ScriptedRetriever` (deterministic, no server needed), the same
pattern `deciders/fake.py::ScriptedDecider` already uses for deciders.

**`Decider` protocol** (`deciders/base.py`): three typed semantic questions —
`screen_passage`, `grade_sufficiency`, `decide_retry` — deliberately not a generic `judge(prompt)`,
so the two backends can't silently drift into answering different questions. The graph only ever
reads `.value` off a returned `Decision`; `.metrics`/`.evidence` exist purely for telemetry and must
never be branched on.

**Deciders** (`deciders/`): `gemini.py`, `groq.py`, `ollama.py` (LLM-backed, via LangChain chat
models), `jev.py` (TypeSafe AI, probability-based rather than hard yes/no), `fake.py`
(`ScriptedDecider`, used by topology/unit tests and `--dry-run`), `cassette.py` (record/replay
wrapper at the Decider boundary — not HTTP — so it survives SDK transport changes;
`latency_source ∈ {measured, replayed, simulated}`). `factory.py` constructs configured instances
from `Settings` and applies provider-specific retry/timeout/concurrency workarounds documented in
`docs/METHODOLOGY.md` — read that before changing any decider's model-construction kwargs, several
look redundant but aren't (e.g. `max_retries=0` to avoid two independently-stacking retry layers).

**Bench pipeline** (`bench/`): `runner.py::run_comparison` interleaves A/B per query (not
all-A-then-all-B) against a frozen doc pool, and hashes the pool before/after each pair of runs to
catch a decider mutating shared state. `analysis.py` computes selected-set precision/recall/F1
against gold titles, paired latency statistics (bootstrap CI, Wilcoxon, Hodges-Lehmann — never a bare
mean), and Jev's precision-recall curve for iso-recall comparison against Gemini's single operating
point. `report.py` renders the markdown report and refuses to emit headline latency from a run
containing replayed cassette entries. `injection_eval.py` scores the injection-screening guard
against `eval/injection_probes.jsonl` separately from the main quality comparison.

**Telemetry** (`telemetry/`): `model.py` defines `Decision`/`NodeSpan`; `instrument.py`'s
`NodeSpanRecorder` wraps node execution; `pricing.py`'s `PriceBook` looks up dated price rows from
`config/pricing.yaml` by `run.started_at` (not wall-clock "today") so historical reports stay
reproducible after a price change; `emitter.py` is the SSE event bus used by the live demo server.

**Config** (`config.py`): `pydantic-settings`, one nested `BaseSettings` subclass per
provider/concern (`GeminiSettings`, `GroqSettings`, `OllamaSettings`, `JevSettings`,
`BenchSettings`), loaded from `.env`. Note: a nested `BaseSettings` built via `default_factory` does
*not* inherit the outer `Settings`' `env_file` — each nested class declares its own `env_file=".env"`.

**`api/app.py` + `web/`**: a separate FastAPI+SSE live demo (`jev-bench serve`) that concurrently
streams both variants' graph execution to a static frontend — distinct from and not benchmark-grade
like `jev-bench bench` (which runs sequentially, A then B, for measurement validity).

**Eval data** (`eval/`): `dataset.jsonl` is a frozen, seeded 1,100-question slice of HotpotQA dev-distractor
(80 train / 10 val / 10 pilot / 1,000 test, see `eval/splits.json`; see `eval/DATA_CARD.md` for license — CC BY-SA 4.0, share-alike, do not relicense derived files as
MIT). `injection_probes.jsonl` is synthetic, ground-truth-by-construction. Both are curated by
scripts in `scripts/`, not hand-edited.
