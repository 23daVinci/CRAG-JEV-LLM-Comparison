import pytest

from jev_bench.bench.tuning import (
    filter_probes_by_split,
    filter_records_by_split,
    tune_injection_threshold,
    tune_relevance_threshold,
    tune_target,
)
from jev_bench.deciders.fake import ScriptedDecider


def test_tune_target_picks_train_threshold_and_confirms_on_val_without_resweeping() -> None:
    train_probabilities = [0.9, 0.7, 0.4, 0.1]
    train_labels = [True, False, True, False]
    val_probabilities = [0.95, 0.5, 0.05]
    val_labels = [True, False, False]

    outcome = tune_target(train_probabilities, train_labels, val_probabilities, val_labels)

    # Same sweep as test_threshold_sweep.py's pick_best_f1_threshold case: 0.4 maximizes train F1.
    assert outcome.train_threshold.threshold == 0.4
    assert outcome.train_f1 == pytest.approx(0.8)
    # VAL confirmation reuses that exact threshold (0.4): both 0.95 and 0.5 clear it, but only
    # 0.95 is a true positive, so precision drops to 0.5 while recall stays at 1.0.
    assert outcome.val_metrics.precision == pytest.approx(0.5)
    assert outcome.val_metrics.recall == 1.0
    assert outcome.val_f1_drop == pytest.approx(0.8 - (2 / 3))


def test_tune_target_flags_a_val_f1_drop() -> None:
    train_probabilities = [0.9, 0.8, 0.2, 0.1]
    train_labels = [True, True, False, False]
    # VAL: at threshold 0.8 (perfect on train), only the false positive clears the bar.
    val_probabilities = [0.85, 0.3]
    val_labels = [False, True]

    outcome = tune_target(train_probabilities, train_labels, val_probabilities, val_labels)

    assert outcome.train_f1 == pytest.approx(1.0)
    assert outcome.val_metrics.f1 == pytest.approx(0.0)
    assert outcome.val_f1_drop > 0.10


def _record(record_id: str, title_to_prob: dict[str, float], gold_titles: list[str]) -> dict:
    return {
        "id": record_id,
        "question": f"question for {record_id}",
        "context": {title: f"text for {record_id}/{title}" for title in title_to_prob},
        "gold_titles": gold_titles,
        "_title_to_prob": title_to_prob,
    }


def _decider_for(records: list[dict]) -> ScriptedDecider:
    """Looks up the per-(query, title) probability by matching the doc text back to its title —
    ScriptedDecider.evidence_fn only sees (query, doc text), not the title, so the fixture below
    keys probabilities by doc text directly instead."""

    prob_by_query_and_text: dict[tuple[str, str], float] = {}
    for record in records:
        for title, prob in record["_title_to_prob"].items():
            prob_by_query_and_text[(record["question"], record["context"][title])] = prob
    return ScriptedDecider(
        evidence_fn=lambda query, doc: {"relevant": prob_by_query_and_text[(query, doc)]}
    )


@pytest.mark.asyncio
async def test_tune_relevance_threshold_never_touches_the_test_split() -> None:
    train_record = _record("r-train", {"gold": 0.9, "distractor": 0.1}, ["gold"])
    val_record = _record("r-val", {"gold": 0.95, "distractor": 0.05}, ["gold"])
    # If this leaked into the train sweep it would change the picked threshold — it never should.
    test_record = _record("r-test", {"gold": 0.01, "distractor": 0.99}, ["gold"])
    records = [train_record, val_record, test_record]
    splits = {"r-train": "train", "r-val": "val", "r-test": "test"}
    decider = _decider_for(records)

    outcome = await tune_relevance_threshold(decider, records=records, splits=splits)

    assert outcome.train_threshold.threshold == 0.9
    assert outcome.val_metrics.f1 == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tune_injection_threshold_never_touches_the_test_split() -> None:
    probes = [
        {"id": "p1", "query_id": "q-train", "query": "q", "text": "clean", "is_injection": False},
        {
            "id": "p2",
            "query_id": "q-train",
            "query": "q",
            "text": "injected",
            "is_injection": True,
        },
        {"id": "p3", "query_id": "q-val", "query": "q", "text": "clean2", "is_injection": False},
        {
            "id": "p4",
            "query_id": "q-val",
            "query": "q",
            "text": "injected2",
            "is_injection": True,
        },
        # Inverted probabilities relative to label — would wreck the train sweep if it leaked in.
        {
            "id": "p5",
            "query_id": "q-test",
            "query": "q",
            "text": "clean3",
            "is_injection": False,
        },
        {
            "id": "p6",
            "query_id": "q-test",
            "query": "q",
            "text": "injected3",
            "is_injection": True,
        },
    ]
    splits = {"q-train": "train", "q-val": "val", "q-test": "test"}
    prob_by_text = {
        "clean": 0.1,
        "injected": 0.9,
        "clean2": 0.05,
        "injected2": 0.95,
        "clean3": 0.99,
        "injected3": 0.01,
    }
    decider = ScriptedDecider(evidence_fn=lambda query, doc: {"is_injection": prob_by_text[doc]})

    outcome = await tune_injection_threshold(decider, probes=probes, splits=splits)

    assert outcome.train_threshold.threshold == 0.9
    assert outcome.val_metrics.f1 == pytest.approx(1.0)


def test_filter_records_by_split_keeps_only_matching_ids() -> None:
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    splits = {"a": "train", "b": "val", "c": "test"}
    assert [r["id"] for r in filter_records_by_split(records, splits, "train")] == ["a"]


def test_filter_probes_by_split_keeps_only_matching_query_ids() -> None:
    probes = [{"id": "p1", "query_id": "q1"}, {"id": "p2", "query_id": "q2"}]
    splits = {"q1": "train", "q2": "test"}
    assert [p["id"] for p in filter_probes_by_split(probes, splits, "test")] == ["p2"]
