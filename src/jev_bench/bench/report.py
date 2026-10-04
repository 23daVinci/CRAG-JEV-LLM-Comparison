from __future__ import annotations

from collections import Counter

from jev_bench.bench.analysis import (
    gemini_operating_point,
    paired_latency_comparison,
    precision_at_recall,
    score_selection,
    summarize_latency,
    sweep_thresholds,
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
    attempt_counts = Counter(r.attempts for r in results)
    attempt_distribution = ", ".join(
        f"{attempts} attempt(s): {count}" for attempts, count in sorted(attempt_counts.items())
    )

    return (
        f"### {name}\n\n"
        f"- queries: {len(results)}\n"
        f"- selection quality vs gold: precision={mean_precision:.3f} "
        f"recall={mean_recall:.3f} f1={mean_f1:.3f}\n"
        f"- latency: median={latency.median_ms:.1f}ms p90={latency.p90_ms:.1f}ms "
        f"iqr=({latency.iqr_ms[0]:.1f}, {latency.iqr_ms[1]:.1f})ms\n"
        f"- retry-attempt distribution: {attempt_distribution}\n"
        f"- total cost: ${total_cost:.6f} (${total_cost / len(results) * 1000:.4f} per 1,000 "
        f"queries)\n"
    )


def _extract_probability_labels(results: list[QueryResult]) -> tuple[list[float], list[bool]]:
    probabilities: list[float] = []
    labels: list[bool] = []
    for query_result in results:
        for doc_id, evidence in query_result.passage_evidence.items():
            if "relevant" not in evidence:
                continue
            probabilities.append(evidence["relevant"])
            labels.append(doc_id in query_result.gold_titles)
    return probabilities, labels


def _relevance_curve_section(a_results: list[QueryResult], b_results: list[QueryResult]) -> str:
    """Gemini is a point in this space, Jev is a curve — this is the comparison that matters for
    the probability-vs-hard-decision asymmetry, separate from the selected-SET precision/recall in
    `_side_summary` (which also folds in injection-filtering and multi-attempt accumulation)."""

    a_probs, a_labels = _extract_probability_labels(a_results)
    b_probs, b_labels = _extract_probability_labels(b_results)
    if not a_probs or not b_probs:
        return ""

    gemini_point = gemini_operating_point(a_probs, a_labels)
    jev_points = sweep_thresholds(b_probs, b_labels)
    jev_iso_recall = precision_at_recall(jev_points, gemini_point.recall)

    return (
        "### Passage-relevance operating points (screen_passage, per document)\n\n"
        f"- Gemini (hard decision): precision={gemini_point.precision:.3f} "
        f"recall={gemini_point.recall:.3f}\n"
        f"- Jev iso-recall precision (at Gemini's recall {gemini_point.recall:.3f}): "
        f"{jev_iso_recall:.3f}\n"
        f"- Jev operating points swept: {len(jev_points)} (full curve available via "
        "`sweep_thresholds` for plotting)\n"
    )


def _trajectory_section(results: list[ComparisonResult]) -> str:
    """Identical graphs can still take different paths — Jev may trigger fewer rewrite loops, so
    variant B can look cheaper partly because it did less work, not because each decision was
    cheaper. `attempts` equality is a coarse proxy for "took the same path" (same retry count), not
    a guarantee of identical per-attempt document sets — good enough to flag the confound, not
    precise enough to be the headline number itself."""

    same_path = [r for r in results if r.a.attempts == r.b.attempts]
    rate = len(same_path) / len(results) if results else 0.0
    lines = [
        "### Trajectory equality\n\n"
        f"- same retry-attempt count: {len(same_path)}/{len(results)} ({rate:.1%})\n"
    ]
    if same_path:
        a_cost = sum(r.a.cost_usd for r in same_path)
        b_cost = sum(r.b.cost_usd for r in same_path)
        lines.append(
            f"- cost on that matched subset: A=${a_cost:.6f} B=${b_cost:.6f} "
            f"(ratio A/B={a_cost / b_cost:.1f}x)\n"
            if b_cost
            else f"- cost on that matched subset: A=${a_cost:.6f} B=${b_cost:.6f}\n"
        )
    return "".join(lines)


def build_report(results: list[ComparisonResult]) -> str:
    if not results:
        raise ValueError("no successful query results to report on — every query failed")
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
        f"- bootstrap 95% CI: [{paired.ci95_low_ms:.1f}, {paired.ci95_high_ms:.1f}]ms\n"
        f"- Hodges-Lehmann shift estimate: {paired.hodges_lehmann_ms:.1f}ms\n"
        f"- Wilcoxon signed-rank: statistic={paired.wilcoxon_statistic:.1f} "
        f"p={paired.wilcoxon_pvalue:.4f}\n",
        _relevance_curve_section(a_results, b_results),
        _trajectory_section(results),
    ]
    return "\n".join(line for line in lines if line)
