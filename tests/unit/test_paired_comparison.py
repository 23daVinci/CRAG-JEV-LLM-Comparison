import pytest

from jev_bench.bench.analysis import paired_latency_comparison


def test_identical_series_has_zero_shift_and_high_pvalue() -> None:
    a = [100.0, 200.0, 300.0, 400.0]
    b = [100.0, 200.0, 300.0, 400.0]
    result = paired_latency_comparison(a, b, n_bootstrap=200)
    assert result.median_delta_ms == 0.0
    assert result.hodges_lehmann_ms == 0.0
    assert result.wilcoxon_pvalue == 1.0


def test_consistent_shift_is_detected() -> None:
    a = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]
    b = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    result = paired_latency_comparison(a, b, n_bootstrap=500)
    assert result.median_delta_ms == 50.0
    assert result.hodges_lehmann_ms == 50.0
    assert result.wilcoxon_pvalue < 0.05
    assert result.ci95_low_ms <= 50.0 <= result.ci95_high_ms


def test_mismatched_lengths_raise() -> None:
    import pytest

    with pytest.raises(ValueError, match="equal-length"):
        paired_latency_comparison([1.0, 2.0], [1.0])


def test_paired_f1_comparison_counts_wins_and_ties() -> None:
    from jev_bench.bench.analysis import paired_f1_comparison

    result = paired_f1_comparison([1.0, 0.5, 0.0, 0.4], [0.5, 0.5, 1.0, 0.0])
    assert (result.a_better, result.b_better, result.tied) == (2, 1, 1)
    assert result.mean_delta == pytest.approx((0.5 + 0.0 - 1.0 + 0.4) / 4)
    assert result.ci95_low <= result.mean_delta <= result.ci95_high


def test_paired_f1_comparison_identical_series_is_a_zero_tie() -> None:
    from jev_bench.bench.analysis import paired_f1_comparison

    result = paired_f1_comparison([0.5, 1.0], [0.5, 1.0])
    assert result.mean_delta == 0.0
    assert result.wilcoxon_pvalue == 1.0
    assert result.tied == 2


def test_paired_f1_comparison_rejects_mismatched_lengths() -> None:
    from jev_bench.bench.analysis import paired_f1_comparison

    with pytest.raises(ValueError):
        paired_f1_comparison([1.0], [1.0, 0.5])
