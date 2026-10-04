from __future__ import annotations

from jev_bench.bench.analysis import (
    paired_latency_comparison,
    score_selection,
    summarize_latency,
)
from jev_bench.bench.runner import ComparisonResult, QueryResult


class ReplayedLatencyError(RuntimeError):
    pass


def _assert_no_replayed_latency(results: list[ComparisonResult]) -> None:
    """The one rule that protects the headline claim while a real-but-early-access key is in play:
    a cassette is a real past measurement, but conditions drift, so it must never silently stand in
    for a fresh benchmark number. Token/cost figures from replay are still provider-exact and are
    allowed through — only latency is refused."""

    for comparison in results:
        for side in (comparison.a, comparison.b):
            for span in side.spans:
                for call in span.calls:
                    if call.latency_source == "replayed":
                        raise ReplayedLatencyError(
                            f"query {comparison.query_id!r}: span {span.node!r} contains a "
                            f"replayed call (captured_at={call.captured_at}). Refusing to emit "
                            "headline latency stats — rerun in benchmark mode against live APIs."
                        )


def _side_summary(name: str, results: list[QueryResult]) -> str:
    latencies = [r.latency_ms for r in results]
    total_cost = sum(r.cost_usd for r in results)
    metrics = [score_selection(set(r.selected_titles), set(r.gold_titles)) for r in results]
    mean_precision = sum(m.precision for m in metrics) / len(metrics)
    mean_recall = sum(m.recall for m in metrics) / len(metrics)
    mean_f1 = sum(m.f1 for m in metrics) / len(metrics)
    latency = summarize_latency(latencies)
    mean_attempts = sum(r.attempts for r in results) / len(results)

    return (
        f"### {name}\n\n"
        f"- queries: {len(results)}\n"
        f"- selection quality vs gold: precision={mean_precision:.3f} "
        f"recall={mean_recall:.3f} f1={mean_f1:.3f}\n"
        f"- latency: median={latency.median_ms:.1f}ms p90={latency.p90_ms:.1f}ms "
        f"iqr=({latency.iqr_ms[0]:.1f}, {latency.iqr_ms[1]:.1f})ms\n"
        f"- mean retry attempts: {mean_attempts:.2f}\n"
        f"- total cost: ${total_cost:.6f} (${total_cost / len(results) * 1000:.4f} per 1,000 "
        f"queries)\n"
    )


def build_report(results: list[ComparisonResult]) -> str:
    _assert_no_replayed_latency(results)

    a_results = [r.a for r in results]
    b_results = [r.b for r in results]
    a_name = "variant A"
    b_name = "variant B"

    paired = paired_latency_comparison(
        [r.latency_ms for r in a_results], [r.latency_ms for r in b_results]
    )

    lines = [
        "# Benchmark report\n",
        f"{len(results)} queries, interleaved (A then B per query).\n",
        _side_summary(a_name, a_results),
        _side_summary(b_name, b_results),
        "### Paired latency comparison (A minus B, per query)\n\n"
        f"- median delta: {paired.median_delta_ms:.1f}ms\n"
        f"- bootstrap 95% CI: [{paired.ci95_low_ms:.1f}, {paired.ci95_high_ms:.1f}]ms\n",
    ]
    return "\n".join(lines)
