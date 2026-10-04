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
