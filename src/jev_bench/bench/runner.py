from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_bench.deciders.base import Decider
from jev_bench.graph.builder import GraphConfig, build_graph
from jev_bench.graph.state import initial_state
from jev_bench.telemetry.model import NodeSpan

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent.parent / "eval" / "dataset.jsonl"

Record = dict[str, Any]
CompiledGraph = Any


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_id: str
    question: str
    gold_titles: frozenset[str]
    selected_titles: frozenset[str]
    attempts: int
    spans: list[NodeSpan]

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
    )


async def run_dataset(
    decider: Decider, records: list[Record] | None = None, cfg: GraphConfig | None = None
) -> list[QueryResult]:
    records = records if records is not None else load_dataset()
    app = build_graph(decider, cfg=cfg)
    return [await _run_one(app, record) for record in records]


async def run_comparison(
    decider_a: Decider,
    decider_b: Decider,
    records: list[Record] | None = None,
    cfg: GraphConfig | None = None,
) -> list[ComparisonResult]:
    """A and B run per-query, interleaved (A then B, then the next query) rather than all-A-then-
    all-B, so provider-side drift (rate limiting, latency variance over time) lands on both sides
    equally instead of concentrating in whichever ran first."""

    records = records if records is not None else load_dataset()
    app_a = build_graph(decider_a, cfg=cfg)
    app_b = build_graph(decider_b, cfg=cfg)
    results: list[ComparisonResult] = []
    for record in records:
        result_a = await _run_one(app_a, record)
        result_b = await _run_one(app_b, record)
        results.append(
            ComparisonResult(
                query_id=record["id"],
                question=record["question"],
                gold_titles=frozenset(record["gold_titles"]),
                a=result_a,
                b=result_b,
            )
        )
    return results
