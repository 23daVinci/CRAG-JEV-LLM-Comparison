from datetime import datetime

from jev_bench.telemetry.pricing import PriceBook


def test_jev_output_is_free() -> None:
    book = PriceBook.load()
    cost = book.cost_usd("jev", "jev-latest", input_tokens=1000, output_tokens=1_000_000, at=_now())
    assert cost == 1000 / 1_000_000 * 0.042


def test_gemini_charges_for_output() -> None:
    book = PriceBook.load()
    cost = book.cost_usd(
        "gemini",
        "gemini-3.1-flash-lite",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        at=_now(),
    )
    assert cost == 0.25 + 1.50


def test_unknown_model_raises() -> None:
    book = PriceBook.load()
    try:
        book.cost_usd("gemini", "not-a-real-model", 1, 1, _now())
    except LookupError:
        return
    raise AssertionError("expected LookupError for an unpriced model")


def _now() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)
