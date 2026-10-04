import pytest

from jev_bench.bench.injection_eval import evaluate_injection_guard
from jev_bench.deciders.base import PassageVerdict
from jev_bench.deciders.fake import ScriptedDecider

PROBES = [
    {"id": "p1", "query": "q", "text": "clean passage one", "is_injection": False},
    {"id": "p2", "query": "q", "text": "clean passage two", "is_injection": False},
    {"id": "p3", "query": "q", "text": "injected passage one", "is_injection": True},
    {"id": "p4", "query": "q", "text": "injected passage two", "is_injection": True},
]


@pytest.mark.asyncio
async def test_perfect_detector_scores_one_on_everything() -> None:
    decider = ScriptedDecider(
        name="perfect",
        screen_fn=lambda q, d: PassageVerdict(
            relevant=True, has_evidence=True, contradicts_query=False, is_injection="injected" in d
        ),
    )
    result = await evaluate_injection_guard(decider, PROBES)
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.false_positives == []
    assert result.false_negatives == []


@pytest.mark.asyncio
async def test_detector_that_never_flags_anything_has_zero_recall() -> None:
    decider = ScriptedDecider(
        name="blind",
        screen_fn=lambda q, d: PassageVerdict(
            relevant=True, has_evidence=True, contradicts_query=False, is_injection=False
        ),
    )
    result = await evaluate_injection_guard(decider, PROBES)
    assert result.accuracy == 0.5
    assert result.recall == 0.0
    assert set(result.false_negatives) == {"p3", "p4"}
    assert result.false_positives == []


@pytest.mark.asyncio
async def test_detector_that_flags_everything_has_zero_precision() -> None:
    decider = ScriptedDecider(
        name="paranoid",
        screen_fn=lambda q, d: PassageVerdict(
            relevant=True, has_evidence=True, contradicts_query=False, is_injection=True
        ),
    )
    result = await evaluate_injection_guard(decider, PROBES)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert set(result.false_positives) == {"p1", "p2"}
