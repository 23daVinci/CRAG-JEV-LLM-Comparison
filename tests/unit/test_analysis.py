from jev_bench.bench.analysis import score_selection


def test_perfect_selection() -> None:
    m = score_selection({"a", "b"}, {"a", "b"})
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_empty_selection_has_zero_precision_and_recall() -> None:
    m = score_selection(set(), {"a", "b"})
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_partial_overlap() -> None:
    m = score_selection({"a", "c"}, {"a", "b"})
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5


def test_extra_selections_hurt_precision_not_recall() -> None:
    m = score_selection({"a", "b", "c", "d"}, {"a", "b"})
    assert m.precision == 0.5
    assert m.recall == 1.0
