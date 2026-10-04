from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np


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
    pairing removes per-query difficulty as a confound. Median of the per-query deltas plus a
    bootstrap 95% CI, not a t-test (latency is never close to normal)."""

    median_delta_ms: float
    ci95_low_ms: float
    ci95_high_ms: float


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
    return PairedComparison(
        median_delta_ms=float(statistics.median(deltas)),
        ci95_low_ms=float(low),
        ci95_high_ms=float(high),
    )
