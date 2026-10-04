import pytest

from jev_bench.bench.report import ReplayedLatencyError, build_report
from jev_bench.bench.runner import ComparisonResult, QueryResult
from jev_bench.telemetry.model import CallMetrics, LatencySource, NodeSpan


def _span(latency_source: LatencySource) -> NodeSpan:
    call = CallMetrics(
        provider="jev",
        model="jev-latest",
        latency_ms=10.0,
        input_tokens=10,
        output_tokens=0,
        cost_usd=0.0,
        latency_source=latency_source,
    )
    return NodeSpan(
        node="screen_passage", backend="jev", started_at=0.0, finished_at=0.01, calls=[call]
    )


def _result(query_id: str, latency_source: LatencySource) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        question="q",
        gold_titles=frozenset({"a"}),
        selected_titles=frozenset({"a"}),
        attempts=1,
        spans=[_span(latency_source)],
        passage_evidence={"a": {"relevant": 1.0}},
    )


def test_build_report_refuses_replayed_latency() -> None:
    comparison = ComparisonResult(
        query_id="q1",
        question="q",
        gold_titles=frozenset({"a"}),
        a=_result("q1", "measured"),
        b=_result("q1", "replayed"),
    )
    with pytest.raises(ReplayedLatencyError):
        build_report([comparison])


def test_build_report_accepts_measured_latency() -> None:
    comparison = ComparisonResult(
        query_id="q1",
        question="q",
        gold_titles=frozenset({"a"}),
        a=_result("q1", "measured"),
        b=_result("q1", "measured"),
    )
    report = build_report([comparison])
    assert "Benchmark report" in report
