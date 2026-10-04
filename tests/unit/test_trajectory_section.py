from jev_bench.bench.report import build_report
from jev_bench.bench.runner import ComparisonResult, QueryResult


def _result(attempts: int, cost: float) -> QueryResult:
    return QueryResult(
        query_id="q",
        question="q",
        gold_titles=frozenset({"a"}),
        selected_titles=frozenset({"a"}),
        attempts=attempts,
        spans=[],
        passage_evidence={},
    )


def test_matched_and_mismatched_trajectories_are_both_counted() -> None:
    results = [
        ComparisonResult(
            query_id="q1",
            question="q",
            gold_titles=frozenset({"a"}),
            a=_result(1, 0.01),
            b=_result(1, 0.001),
        ),
        ComparisonResult(
            query_id="q2",
            question="q",
            gold_titles=frozenset({"a"}),
            a=_result(2, 0.02),
            b=_result(1, 0.001),
        ),
    ]
    report = build_report(results)
    assert "same retry-attempt count: 1/2 (50.0%)" in report
