from jev_bench.bench.analysis import precision_at_recall, sweep_thresholds


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
