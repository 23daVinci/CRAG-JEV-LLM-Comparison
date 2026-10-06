import pytest

from jev_bench.bench.analysis import (
    ThresholdPoint,
    bootstrap_mean_ci,
    metrics_at_threshold,
    pick_best_f1_threshold,
    precision_at_recall,
    sweep_thresholds,
)


def test_perfect_separation_reaches_precision_one_at_full_recall() -> None:
    probabilities = [0.9, 0.8, 0.3, 0.1]
    labels = [True, True, False, False]
    points = sweep_thresholds(probabilities, labels)
    full_recall_points = [p for p in points if p.recall == 1.0]
    assert any(p.precision == 1.0 for p in full_recall_points)


def test_tied_probabilities_are_grouped_into_one_point() -> None:
    probabilities = [0.5, 0.5, 0.5]
    labels = [True, False, True]
    points = sweep_thresholds(probabilities, labels)
    assert len(points) == 1
    assert points[0].precision == 2 / 3
    assert points[0].recall == 1.0


def test_precision_at_recall_picks_best_precision_meeting_the_target() -> None:
    probabilities = [0.9, 0.8, 0.3, 0.1]
    labels = [True, True, False, False]
    points = sweep_thresholds(probabilities, labels)
    assert precision_at_recall(points, target_recall=0.5) == 1.0
    assert precision_at_recall(points, target_recall=1.0) == 1.0


def test_precision_at_recall_returns_zero_when_unreachable() -> None:
    probabilities = [0.9, 0.1]
    labels = [False, False]
    points = sweep_thresholds(probabilities, labels)
    assert precision_at_recall(points, target_recall=0.5) == 0.0


def test_pick_best_f1_threshold_picks_the_perfectly_separating_point() -> None:
    probabilities = [0.9, 0.7, 0.4, 0.1]
    labels = [True, False, True, False]
    points = sweep_thresholds(probabilities, labels)
    best = pick_best_f1_threshold(points)
    # threshold=0.4 catches both positives (0.9, 0.4) and one false positive (0.7):
    # precision=2/3, recall=1.0, f1=0.8 — the max over this sweep.
    assert best.threshold == 0.4
    assert best.recall == 1.0


def test_pick_best_f1_threshold_picks_the_upper_middle_of_an_even_tie() -> None:
    # f1 is symmetric in (precision, recall), so swapping the two values yields identical f1 at a
    # different threshold — a genuine tie that must resolve to the higher, more conservative one.
    points = [
        ThresholdPoint(threshold=0.3, precision=1.0, recall=0.5),
        ThresholdPoint(threshold=0.7, precision=0.5, recall=1.0),
    ]
    best = pick_best_f1_threshold(points)
    assert best.threshold == 0.7


def test_pick_best_f1_threshold_raises_on_empty_sweep() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_best_f1_threshold([])


def test_metrics_at_threshold_matches_sweep_at_the_same_point() -> None:
    probabilities = [0.9, 0.7, 0.4, 0.1]
    labels = [True, False, True, False]
    metrics = metrics_at_threshold(probabilities, labels, threshold=0.4)
    assert metrics.precision == 2 / 3
    assert metrics.recall == 1.0


def test_metrics_at_threshold_handles_no_predicted_positives() -> None:
    metrics = metrics_at_threshold([0.1, 0.2], [True, False], threshold=0.9)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_bootstrap_mean_ci_on_constant_values_is_a_point() -> None:
    low, high = bootstrap_mean_ci([0.8, 0.8, 0.8])
    assert low == pytest.approx(0.8)
    assert high == pytest.approx(0.8)


def test_bootstrap_mean_ci_on_empty_values_is_zero() -> None:
    assert bootstrap_mean_ci([]) == (0.0, 0.0)


def test_pick_best_f1_threshold_picks_the_middle_of_a_plateau() -> None:
    points = [
        ThresholdPoint(threshold=t, precision=1.0, recall=1.0) for t in (0.2, 0.4, 0.6, 0.8)
    ] + [ThresholdPoint(threshold=0.9, precision=1.0, recall=0.5)]
    assert pick_best_f1_threshold(points).threshold == 0.6


def test_pick_best_f1_threshold_uses_the_longest_contiguous_tied_run() -> None:
    perfect = {0.1, 0.5, 0.6, 0.7}
    points = [
        ThresholdPoint(threshold=t, precision=1.0, recall=1.0 if t in perfect else 0.5)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    ]
    assert pick_best_f1_threshold(points).threshold == 0.6
