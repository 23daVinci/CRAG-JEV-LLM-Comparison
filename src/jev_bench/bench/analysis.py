from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    """Quality parity against HotpotQA's gold `supporting_facts` titles — not a human or LLM-judge
    label, and not something either variant's own grader produced."""

    precision: float
    recall: float
    f1: float


def score_selection(selected_titles: set[str], gold_titles: set[str]) -> SelectionMetrics:
    precision = (
        len(selected_titles & gold_titles) / len(selected_titles) if selected_titles else 0.0
    )
    recall = len(selected_titles & gold_titles) / len(gold_titles) if gold_titles else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return SelectionMetrics(precision=precision, recall=recall, f1=f1)


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float


def sweep_thresholds(probabilities: list[float], labels: list[bool]) -> list[ThresholdPoint]:
    """Full precision-recall curve over every observed probability as a candidate threshold.

    Gemini is a hard yes/no — it only ever produces one (precision, recall) point. Jev produces a
    probability, so reporting its quality only at the vendor's single untuned cookbook threshold
    would hide how it performs across its whole operating range. This sweep is what lets the
    comparison go beyond "Jev's one point vs Gemini's one point" into "Jev's curve vs Gemini's
    point" — see `precision_at_recall` for the iso-recall comparison that uses it.
    """

    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    total_positive = sum(1 for label in labels if label)
    points: list[ThresholdPoint] = []
    true_positive = 0
    predicted_positive = 0
    n = len(order)
    i = 0
    while i < n:
        threshold = probabilities[order[i]]
        j = i
        while j < n and probabilities[order[j]] == threshold:
            predicted_positive += 1
            if labels[order[j]]:
                true_positive += 1
            j += 1
        precision = true_positive / predicted_positive
        recall = true_positive / total_positive if total_positive else 0.0
        points.append(ThresholdPoint(threshold=threshold, precision=precision, recall=recall))
        i = j
    return points


def gemini_operating_point(probabilities: list[float], labels: list[bool]) -> ThresholdPoint:
    """Gemini's single (precision, recall) point — it only ever produces a hard yes/no, so its
    `evidence["relevant"]` is already 1.0/0.0, not a real probability. Thresholding at 0.5 just
    recovers its original hard decision; this exists so Gemini's point can be computed with the
    same extraction path as Jev's curve, not a separately-written calculation."""

    predicted_positive = sum(1 for p in probabilities if p >= 0.5)
    true_positive = sum(
        1 for p, label in zip(probabilities, labels, strict=True) if p >= 0.5 and label
    )
    total_positive = sum(1 for label in labels if label)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / total_positive if total_positive else 0.0
    return ThresholdPoint(threshold=0.5, precision=precision, recall=recall)


def precision_at_recall(points: list[ThresholdPoint], target_recall: float) -> float:
    """Among thresholds achieving at least `target_recall`, the best precision — this is what gets
    compared against Gemini's point: use Gemini's own recall as `target_recall` and report what
    precision Jev can reach at that same recall, rather than comparing at mismatched operating
    points."""

    candidates = [p.precision for p in points if p.recall >= target_recall]
    return max(candidates) if candidates else 0.0


def metrics_at_threshold(
    probabilities: list[float], labels: list[bool], threshold: float
) -> SelectionMetrics:
    """Precision/recall/F1 at one fixed, already-chosen threshold — used to *confirm* a threshold
    picked elsewhere (e.g. on a train split) against new data (e.g. a val or test split), as opposed
    to `sweep_thresholds`, which searches over every candidate threshold. Confirming must never
    re-search; that would make val/test not actually held out from the choice."""

    predicted_positive = sum(1 for p in probabilities if p >= threshold)
    true_positive = sum(
        1 for p, label in zip(probabilities, labels, strict=True) if p >= threshold and label
    )
    total_positive = sum(1 for label in labels if label)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / total_positive if total_positive else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return SelectionMetrics(precision=precision, recall=recall, f1=f1)


def sweep_fixed_thresholds(
    probabilities: list[float], labels: list[bool], candidates: list[float]
) -> list[ThresholdPoint]:
    """Precision/recall at each of a fixed, caller-supplied candidate list — unlike
    `sweep_thresholds`, whose candidates are whatever probabilities happened to be observed. A fixed
    grid makes tuning reproducible despite live-API jitter in the raw probabilities (a data-driven
    sweep picked 0.360 then 0.370 on two consecutive runs) and keeps one noisy document from
    creating its own "best" candidate."""

    points: list[ThresholdPoint] = []
    for threshold in candidates:
        metrics = metrics_at_threshold(probabilities, labels, threshold)
        points.append(
            ThresholdPoint(threshold=threshold, precision=metrics.precision, recall=metrics.recall)
        )
    return points


def threshold_point_f1(point: ThresholdPoint) -> float:
    if point.precision + point.recall == 0:
        return 0.0
    return 2 * point.precision * point.recall / (point.precision + point.recall)


def pick_best_f1_threshold(points: list[ThresholdPoint]) -> ThresholdPoint:
    """The threshold-tuning selection rule: maximize Jev's own F1 on train, independent of any other
    backend's operating point. (Iso-recall, by contrast, lets a *comparison* target's recall decide
    where on the curve to look — appropriate for reporting, wrong for picking Jev's own threshold.)

    When several adjacent thresholds tie for the best F1 (a plateau — e.g. injection detection,
    where nearly every threshold scores a perfect 1.00), pick the middle of the longest tied run
    rather than either edge: an edge pick sits right next to the cliff where F1 starts to fall, so
    it is the least robust point on the plateau. With an even-length run, the upper-middle point
    is used so selection stays deterministic."""

    if not points:
        raise ValueError("cannot pick a threshold from an empty sweep")

    ordered = sorted(points, key=lambda p: p.threshold)
    best_f1 = max(threshold_point_f1(p) for p in ordered)
    runs: list[list[ThresholdPoint]] = []
    previous_tied = False
    for point in ordered:
        tied = math.isclose(threshold_point_f1(point), best_f1, abs_tol=1e-9)
        if tied and previous_tied:
            runs[-1].append(point)
        elif tied:
            runs.append([point])
        previous_tied = tied
    longest = max(runs, key=len)
    return longest[len(longest) // 2]


def bootstrap_mean_ci(
    values: list[float], n_bootstrap: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI on the mean of `values` (e.g. per-query F1) by resampling queries with
    replacement — the generic version of the resampling already used for latency in
    `paired_latency_comparison`, applied to a quality metric instead. With a handful of test queries
    (this project's test split is 9), a bare mean F1 overstates confidence; always report this
    alongside it, not instead of the paired-comparison machinery above."""

    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.array(
        [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class LatencySummary:
    median_ms: float
    p90_ms: float
    mean_ms: float
    iqr_ms: tuple[float, float]


def summarize_latency(latencies_ms: list[float]) -> LatencySummary:
    if not latencies_ms:
        return LatencySummary(median_ms=0.0, p90_ms=0.0, mean_ms=0.0, iqr_ms=(0.0, 0.0))
    arr = np.asarray(latencies_ms)
    return LatencySummary(
        median_ms=float(np.median(arr)),
        p90_ms=float(np.percentile(arr, 90)),
        mean_ms=float(np.mean(arr)),
        iqr_ms=(float(np.percentile(arr, 25)), float(np.percentile(arr, 75))),
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Headline latency inference belongs on the *paired* per-query difference, not independent
    medians — both variants run against the identical frozen retrieval fixture per query, so the
    pairing removes per-query difficulty as a confound. Three complementary views, deliberately not
    a t-test (latency is never close to normal): a bootstrap CI on the median delta (distribution-
    free, easy to communicate), the Wilcoxon signed-rank test (is the direction of the effect real),
    and the Hodges-Lehmann estimator (a robust point estimate of the shift, paired with the
    Wilcoxon test the way a median pairs with a sign test)."""

    median_delta_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    hodges_lehmann_ms: float


def _hodges_lehmann(deltas: np.ndarray) -> float:
    """Median of all pairwise (Walsh) averages (d_i + d_j) / 2 for i <= j — the estimator the
    Wilcoxon signed-rank test is implicitly testing against zero, so reporting it alongside the
    test gives a robust point estimate rather than just a yes/no on significance."""

    n = len(deltas)
    walsh_averages = [(deltas[i] + deltas[j]) / 2 for i in range(n) for j in range(i, n)]
    return float(np.median(walsh_averages))


def paired_latency_comparison(
    a_ms: list[float], b_ms: list[float], n_bootstrap: int = 10_000, seed: int = 0
) -> PairedComparison:
    if len(a_ms) != len(b_ms):
        raise ValueError("paired comparison requires equal-length, query-aligned samples")
    deltas = np.asarray(a_ms) - np.asarray(b_ms)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_medians = np.array(
        [np.median(rng.choice(deltas, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
    low, high = np.percentile(boot_medians, [2.5, 97.5])

    if np.all(deltas == 0):
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        result = scipy_stats.wilcoxon(deltas, zero_method="wilcox")
        wilcoxon_statistic, wilcoxon_pvalue = float(result.statistic), float(result.pvalue)

    return PairedComparison(
        median_delta_ms=float(statistics.median(deltas)),
        ci95_low_ms=float(low),
        ci95_high_ms=float(high),
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_pvalue=wilcoxon_pvalue,
        hodges_lehmann_ms=_hodges_lehmann(deltas),
    )


@dataclass(frozen=True, slots=True)
class PairedQualityComparison:
    """Per-query F1 difference (A minus B) over the same queries. Both variants ran the identical
    fixture per query, so pairing removes per-query difficulty as a confound — much more powerful
    than eyeballing whether two separately-computed CIs overlap. A claim that B "outperforms" A on
    quality needs this CI to sit entirely on B's side of zero."""

    mean_delta: float
    ci95_low: float
    ci95_high: float
    wilcoxon_pvalue: float
    a_better: int
    b_better: int
    tied: int


def paired_f1_comparison(
    a_f1: list[float], b_f1: list[float], n_bootstrap: int = 10_000, seed: int = 0
) -> PairedQualityComparison:
    if len(a_f1) != len(b_f1):
        raise ValueError("paired comparison requires equal-length, query-aligned samples")
    deltas = np.asarray(a_f1) - np.asarray(b_f1)
    ci_low, ci_high = bootstrap_mean_ci([float(d) for d in deltas], n_bootstrap, seed)
    if np.all(deltas == 0):
        pvalue = 1.0
    else:
        pvalue = float(scipy_stats.wilcoxon(deltas, zero_method="wilcox").pvalue)
    return PairedQualityComparison(
        mean_delta=float(np.mean(deltas)) if len(deltas) else 0.0,
        ci95_low=ci_low,
        ci95_high=ci_high,
        wilcoxon_pvalue=pvalue,
        a_better=int(np.sum(deltas > 0)),
        b_better=int(np.sum(deltas < 0)),
        tied=int(np.sum(deltas == 0)),
    )
