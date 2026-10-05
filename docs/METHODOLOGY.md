# Measurement methodology

This document is the detailed companion to the plan's "Measurement methodology" section. It states,
for each load-bearing decision, what's implemented, what's deliberately deferred, and why — so a
skeptical reader can check the claim against the code rather than against a description of intent.

## The claim this project is built to support (or refute)

*Same graph, same selected documents, N% faster, M% cheaper* — or an honest null result. Two
variants compile from the identical `build_graph()` (see `tests/test_topology.py`); only the
decision backend differs. Everything below exists to make the N% and M% defensible.

## What's implemented

### Controls

- **Frozen retrieval fixture.** Both variants read from the same `record["context"]` for a query;
  `bench/runner.py::run_comparison` hashes the fixture (`retrieval/store.py::hash_doc_pool`) before
  and after both runs and raises if it changed — `tests/unit/test_runner_fixture_integrity.py`
  exercises this with a decider that deliberately mutates the shared fixture.
- **Shared prompt wording.** Both deciders build their questions from the same constants in
  `graph/prompts.py`, not inlined text — `tests/unit/test_prompts_shared.py` asserts this
  mechanically (checks both decider source files reference every required constant).
- **`temperature=0`, `thinking_budget=0`.** Set on `GeminiDecider`'s model. If Gemini ever reports
  nonzero reasoning/thinking tokens on a grader call, `_metrics()` raises rather than silently
  including them in cost — this is the single most likely silent-cost bug the methodology flagged.
- **`include_raw=True` everywhere** so token usage is never estimated — every `CallMetrics` carries
  provider-reported `input_tokens`/`output_tokens` or the call fails loudly.
- **A/B interleaved per query** (`run_comparison` runs A then B per query, not all-A-then-all-B), so
  provider-side drift (rate limiting, time-of-day latency variance) lands on both sides equally.
- **Topology + no-provider-branching**, enforced by `tests/test_topology.py`'s AST scan, not just
  code review.

### Quality parity

- **Selected-set precision/recall/F1 against HotpotQA's gold `supporting_facts`** — a public,
  objective label neither variant's own grader produced (`bench/analysis.py::score_selection`).
- **Jev's curve vs Gemini's point.** Gemini only ever emits a hard yes/no per document; Jev emits a
  probability. Reporting Jev's quality only at the vendor's untuned cookbook threshold would hide
  how it performs elsewhere, so `bench/analysis.py::sweep_thresholds` builds the full
  precision-recall curve over every observed Jev probability, and `precision_at_recall` reports
  Jev's best precision at *Gemini's own recall* — an iso-recall comparison, not two mismatched
  operating points. Rendered in the report as "Passage-relevance operating points."
  - This required extending `RetrievalState`/`QueryResult` with a `passage_evidence` field — the raw
    per-document evidence was computed by every `screen_passage` call but discarded immediately
    after the node returned, with nothing downstream ever persisting it. Without this, the sweep
    analysis existed as tested library code with no real data flowing into it.
- **Retry-attempt distribution**, not just a mean (`_side_summary`'s "retry-attempt distribution"
  line) — an over-strict screener can buy recall with extra retrieve-retry loops, which is a latency
  cost, not a free win, and a bare mean hides that.
- **Trajectory-equality rate** (`bench/report.py::_trajectory_section`) — identical graphs can still
  take different paths; Jev triggering fewer retry loops would make it look cheaper partly because
  it did less work, not because each decision was cheaper. Same-attempt-count rate plus cost
  restricted to that matched subset is a coarse but checked mitigation (attempt-count equality is a
  proxy for "same path," not a guarantee of identical per-attempt document sets).
- **Injection-screening guard**, scored separately (`bench/injection_eval.py`,
  `jev-bench injection-eval`) against `eval/injection_probes.jsonl` — ground truth there is exact by
  construction (we inserted the injection strings ourselves), unlike every other label in this
  project.

### Threshold tuning

The first real 34-query run used Jev's **untuned vendor cookbook thresholds** and found a 29-point
selected-set F1 gap against the LLM backend — exactly the scenario the pre-registered kill switch
above exists for, and exactly why this methodology always called for "thresholds tuned on a dev
split, reported on held-out test" rather than trusting an untuned number. `jev-bench tune-thresholds`
(backed by `bench/tuning.py`) now implements that process:

- **A frozen, stratified split** (`eval/splits.json`, generated once by
  `scripts/split_eval_queries.py`): 60/20/20 train/val/test (27/9/9 of the 45 queries), stratified by
  `type` so bridge:comparison ratios hold per split. `eval/injection_probes.jsonl` records inherit
  their split from their source query's `query_id` — never an independent assignment — so both halves
  of a clean/injected pair always land together.
- **Tuning rule: maximize Jev's own F1 on TRAIN**, not iso-recall-matched to the LLM backend — iso-
  recall (`precision_at_recall`) is the right tool for *reporting* Jev's curve against another
  backend's single operating point, but using it to *pick* Jev's threshold would let the comparison
  target's behavior dictate Jev's own operating point instead of finding Jev's best one.
  `bench/tuning.py::collect_relevance_probabilities` calls `Decider.screen_passage` directly against
  every document in a query's context — **not a graph run** — because a graph's retry loop stops as
  soon as `grade_sufficiency` says "sufficient" (confirmed from the real run: 24/34 queries only
  screened 5 of 10 docs), so graph-collected `passage_evidence` systematically under-samples whichever
  documents a run happened to stop before reaching.
- **VAL confirms, it does not re-search.** The picked threshold is replayed once against VAL via
  `analysis.py::metrics_at_threshold`; if VAL's F1 drops more than 10 points from TRAIN's,
  `tune-thresholds`'s dry-run output warns loudly rather than silently locking the threshold anyway —
  at train/val sizes this small (27/9), that's a real risk, not a formality.
- **TEST is scored exactly once**, read-only, at the locked threshold, via the normal
  `jev-bench bench --split test` / `jev-bench injection-eval --split test` path — the tuning command
  itself never touches the test split. `bench/report.py::_side_summary` now reports a 95% bootstrap CI
  (`analysis.py::bootstrap_mean_ci`) alongside mean selected-set F1 for exactly this reason: **9 test
  queries is small enough that one flipped query moves test F1 by about 11 points**, so a bare point
  estimate on that split is not defensible — state the CI every time, not just when it's flattering.
- **`relevance` and `injection` are tuned this way; `evidence`, `contradiction`, and `sufficiency` are
  not** and stay at the vendor's cookbook defaults (`config/thresholds.yaml` documents this inline).
  No ground truth exists in this project's eval data for "contains direct evidence" or "contradicts a
  premise" as distinct from relevance; a sufficiency label derived from gold-title coverage is a
  plausible future addition but was deliberately scoped out of this tuning pass rather than bundled in
  speculatively.
- **`tune-thresholds` is dry-run by default**, printing the TRAIN sweep summary, the picked threshold,
  and VAL confirmation; it only writes `config/thresholds.yaml` with an explicit `--apply` flag. A
  threshold change is a frozen methodological decision — consistent with this project's
  pre-registration ethos elsewhere — so a human reviews the diff rather than the script silently
  committing it.

### Latency statistics

- **Median + IQR as the per-side headline**, never a bare mean (`analysis.py::summarize_latency`).
- **Paired inference on the per-query difference**, not independent medians — both variants ran
  against the identical fixture per query, so pairing removes per-query difficulty as a confound
  (`analysis.py::paired_latency_comparison`):
  - Bootstrap 95% CI on the median delta (distribution-free).
  - Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) — is the direction of the effect real.
  - Hodges-Lehmann shift estimate (median of all pairwise Walsh averages) — a robust point estimate
    paired with the Wilcoxon test the way a median pairs with a sign test.
- **No generation step**, so summed decision-node time and end-to-end graph wall clock should be
  close; the report doesn't currently compute both separately (see "Not implemented" below), but the
  `NodeSpan`/`trace` data needed to do so is already captured.

### Cost

- **Provider-reported usage only**, never tiktoken estimates (`telemetry/pricing.py`).
- **Jev's output tokens are free** (`$0`/M) — displayed, not hidden, so the freebie is visible.
- **Dated price rows** (`config/pricing.yaml`), looked up by `run.started_at`, not wall-clock "today"
  — a historical report stays reproducible even after a price change.

### Reproducibility

- **Record/replay at the Decider boundary** (`deciders/cassette.py`), not HTTP — survives SDK
  transport changes. `latency_source ∈ {measured, replayed, simulated}`; `report.py` refuses to emit
  headline latency from any run containing `replayed` entries (`ReplayedLatencyError`), though
  provider-exact cost/token figures from replay are still allowed through.
- **Gemini concurrency + retry.** The free tier enforces a hard 15 RPM cap with explicit 429s
  (confirmed: `quotaValue: 15` for `gemini-3.1-flash-lite`) — the graph's 5-way parallel
  `screen_passage` fan-out blows through that in one batch. `GeminiDecider` caps concurrency via a
  semaphore (default 3) and retries on `ModelRateLimitError`/`httpx.RequestError` with exponential
  backoff, so a transient quota hit or connection blip degrades a run instead of crashing it. A flat
  per-call rate limiter (pacing every call to 15/min regardless of actual contention) was tried
  first and measured *slower* for a single query (76.8s vs. 31.6s unthrottled) — reverted in favor
  of the concurrency cap, which only throttles when there's real contention.
- **Groq replaced Ollama as the default variant A.** Ollama (local, CPU-only on this machine) solved
  Gemini's quota problem but introduced a worse one: real HotpotQA-length passages measured
  7-28s/call (a 5-doc `screen_passage` batch alone took 96.4s), projecting to ~1.5-2h for a full
  run — and concurrency didn't help, since Ollama serializes requests internally regardless (5
  concurrent calls measured at 20.2s, 5 sequential at 20.3s — no difference). Groq's LPU-hosted
  inference measured sub-second per call on the same real passages. Model choice within Groq's
  catalog mattered: `openai/gpt-oss-20b` is fast but emits hidden reasoning tokens
  (`output_token_details.reasoning`) even at `reasoning_effort="low"` (the minimum allowed —
  "none"/"minimal" are rejected), reintroducing the same thinking-token fairness problem guarded
  against for Gemini; `qwen/qwen3.8-27b` reports none at all, so it's used instead. Caveat: it's a
  "Preview" catalog model Groq says may be discontinued at short notice, and its OTPM (output-
  tokens-per-minute) budget on this account is a tight 1000 — tight enough that the field's own
  default `max_tokens` (2048) alone exceeded it, rejecting every request outright before checking
  actual usage. Fixed by setting `max_tokens=200` explicitly (structured JSON answers need well
  under 100 in practice) and by catching `groq.RateLimitError` in `GroqDecider._invoke` and parsing
  the server's own suggested wait from its error message, rather than guessing with blind backoff —
  unlike Gemini's *daily* quota, this is a rolling per-minute window, so retrying is worthwhile here.
  Ollama is kept in the codebase (tested, working) as a zero-cost, zero-quota fallback.
- **Gemini daily quota fast-fail.** The free tier also caps at 500 requests/*day*
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), confirmed by exhausting it during repeated
  test/debug runs. Retrying this with the backoff above (capped at 30s) is pointless — the server's
  own suggested retry delay was ~2.5 hours — and doing so anyway is exactly what made two runs look
  hung for 10-40+ minutes before the real cause was diagnosed. `_invoke` now detects
  `"RequestsPerDay"` in a `ModelRateLimitError`'s message and re-raises immediately instead of
  retrying. Two more compounding bugs were found at the same time and are independently worth
  keeping even once a paid key removes the quota problem: `ChatGoogleGenerativeAI`'s own
  `max_retries` defaults to 6 and stacks *independently* on top of `_invoke`'s retry loop (up to 36
  total HTTP attempts for one decision call) — set to `0` in `deciders/factory.py` so retry is
  handled in exactly one place; and no default request `timeout` existed at all, so a stalled
  connection could hang indefinitely rather than raising — set to `60.0`. **Net effect: a paid
  Gemini key is close to mandatory for a full 45-query run** — the free tier's 500 RPD is easy to
  exhaust across just a few partial/debug runs in a single day, independent of the per-call latency
  problem above.

## What's deliberately not implemented, and why

- **The batching 2×2** (Gemini/Jev × per-decision/batched-across-documents). The plan called for
  implementing all four configurations — per-criterion-per-document calls (4 calls/doc) through
  all-documents-in-one-call — and pre-registering the fully-batched pair as the headline. What's
  built instead sits at a middle point: each `screen_passage` call combines all 4 criteria for *one*
  document into one call, for both backends. Extending this to batch *across* documents (one call
  covering all 5 candidates) would require a new `Decider` method and a second graph-node code path,
  not just a parameter — a real architecture extension, not a mechanism this document can wave at.
  **This is the single largest gap between the approved plan and what's built.** If the batching
  story matters for the headline claim, it needs to be built before the real benchmark run, not
  assumed.
- **Gemini price sensitivity for the 2027-01-01 step-up.** The `PriceBook` mechanism supports dated
  rows and would correctly reprice a future run if one existed, but no confirmed future price for
  `gemini-3.1-flash-lite` specifically exists as of this writing (only a general note that some
  Gemini Flash tiers step up 2x on that date) — adding a row would mean fabricating a number we
  don't have evidence for. Mechanism-ready, not data-ready.
- **ECDF/violin/slope-chart visuals, latency-vs-timestamp drift scatter.** Presentation-layer work
  for the final publish step (plan milestone 8), not a measurement mechanism — deferred until
  there's real multi-repetition data worth plotting.
- **n≈30 repetitions per query.** The runner executes each query once per variant per run, not
  repeated. The plan's fuller design (20-40 queries × n=30 reps × 4 configs, 3 discarded warm-ups)
  was scoped down given the confirmed Gemini free-tier latency (~5-12s/call) and 1,000 RPD cap — a
  single full pass over the 45-query set already costs real wall-clock time (30-60 min observed).
  Repeating it 30x isn't practical on the free tier. If a paid key removes that constraint, the
  runner would need a `--reps` flag and the analysis would aggregate per-query before pairing, not
  just concatenate — not built, since it wasn't needed to produce a first real result.
