from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_bench.deciders.base import Decider
from jev_bench.graph.builder import GraphConfig, build_graph
from jev_bench.graph.state import initial_state
from jev_bench.retrieval.store import Retriever, hash_doc_pool
from jev_bench.telemetry.model import NodeSpan

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent.parent / "eval" / "dataset.jsonl"

Record = dict[str, Any]
CompiledGraph = Any


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[Record]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_id: str
    question: str
    gold_titles: frozenset[str]
    selected_titles: frozenset[str]
    attempts: int
    spans: list[NodeSpan]
    passage_evidence: dict[str, dict[str, float]]
    """Raw per-document screen_passage evidence (e.g. Jev's un-thresholded `relevant` probability),
    keyed by doc id — feeds the PR-curve/iso-recall analysis in analysis.py."""

    @property
    def latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    query_id: str
    question: str
    gold_titles: frozenset[str]
    a: QueryResult
    b: QueryResult


@dataclass(frozen=True, slots=True)
class QueryFailure:
    query_id: str
    error: str


async def _run_one(app: CompiledGraph, record: Record) -> QueryResult:
    state = initial_state(record["question"], record["context"])
    final = await app.ainvoke(state)
    return QueryResult(
        query_id=record["id"],
        question=record["question"],
        gold_titles=frozenset(record["gold_titles"]),
        selected_titles=frozenset(final["selected_ids"]),
        attempts=final["attempt"],
        spans=final["trace"],
        passage_evidence=final["passage_evidence"],
    )


async def run_dataset(
    decider: Decider,
    retriever: Retriever,
    records: list[Record] | None = None,
    cfg: GraphConfig | None = None,
) -> list[QueryResult]:
    records = records if records is not None else load_dataset()
    app = build_graph(decider, retriever, cfg=cfg)
    return [await _run_one(app, record) for record in records]


async def run_comparison(
    decider_a: Decider,
    decider_b: Decider,
    retriever: Retriever,
    records: list[Record] | None = None,
    cfg: GraphConfig | None = None,
    on_result: Callable[[ComparisonResult], None] | None = None,
    on_failure: Callable[[QueryFailure], None] | None = None,
) -> tuple[list[ComparisonResult], list[QueryFailure]]:
    """A and B run per-query, interleaved (A then B, then the next query) rather than all-A-then-
    all-B, so provider-side drift (rate limiting, latency variance over time) lands on both sides
    equally instead of concentrating in whichever ran first.

    One query's failure does not sink the whole run — a single transient error (rate limit outlast
    its retries, a DNS blip) used to cost every query's worth of real API spend and ~minutes of
    wall-clock time already spent on the run, with nothing to show for it, since `build_report` only
    assembles output after the full loop finishes. Failures are collected and returned alongside the
    successes rather than silently dropped, so the caller can see exactly what was skipped and why.

    `on_result`/`on_failure` fire as each query finishes, not at the end, so a caller (e.g. the
    MLflow tracker) can persist per-query data incrementally — a crash or a Ctrl-C partway through
    no longer loses the queries that already completed.
    """

    records = records if records is not None else load_dataset()
    app_a = build_graph(decider_a, retriever, cfg=cfg)
    app_b = build_graph(decider_b, retriever, cfg=cfg)
    results: list[ComparisonResult] = []
    failures: list[QueryFailure] = []
    for record in records:
        try:
            fixture_hash = hash_doc_pool(record["context"])
            result_a = await _run_one(app_a, record)
            result_b = await _run_one(app_b, record)
            if hash_doc_pool(record["context"]) != fixture_hash:
                raise RuntimeError(
                    f"retrieval fixture for query {record['id']!r} changed during the run — "
                    "retrieval must never be a variable between variants"
                )
        except Exception as exc:  # noqa: BLE001 — one query's failure must not sink the run
            failure = QueryFailure(query_id=record["id"], error=f"{type(exc).__name__}: {exc}")
            failures.append(failure)
            if on_failure is not None:
                on_failure(failure)
            continue
        comparison = ComparisonResult(
            query_id=record["id"],
            question=record["question"],
            gold_titles=frozenset(record["gold_titles"]),
            a=result_a,
            b=result_b,
        )
        results.append(comparison)
        if on_result is not None:
            on_result(comparison)
    return results, failures
