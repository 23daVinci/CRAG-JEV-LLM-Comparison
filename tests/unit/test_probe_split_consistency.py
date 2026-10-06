"""Consistency checks between the committed eval/splits.json and eval/injection_probes.jsonl —
not a test of split_eval_queries.py's shuffling logic (untested by convention, same as the other
one-off eval/ curation scripts), but a guard that the committed artifacts stay in sync: every
probe's query_id must resolve to a split, and both halves of a clean/injected pair must land
together, since tune_injection_threshold relies on that to keep train/val/test genuinely disjoint.
"""

from jev_bench.bench.injection_eval import load_probes
from jev_bench.bench.runner import load_dataset
from jev_bench.bench.tuning import load_splits


def test_every_query_id_appears_exactly_once_across_the_splits() -> None:
    records = load_dataset()
    splits = load_splits()
    record_ids = {r["id"] for r in records}
    assert set(splits.keys()) == record_ids
    assert set(splits.values()) <= {"train", "val", "pilot", "test"}


def test_every_probe_query_id_resolves_to_a_split() -> None:
    probes = load_probes()
    splits = load_splits()
    for probe in probes:
        assert probe["query_id"] in splits, f"probe {probe['id']!r} has an unknown query_id"


def test_both_members_of_a_clean_injected_pair_share_a_split() -> None:
    probes = load_probes()
    splits = load_splits()
    pairs: dict[str, set[str]] = {}
    for probe in probes:
        pair_id = probe["id"].rsplit("-", 1)[0]  # "probe-003-clean" -> "probe-003"
        pairs.setdefault(pair_id, set()).add(splits[probe["query_id"]])
    for pair_id, pair_splits in pairs.items():
        assert len(pair_splits) == 1, f"pair {pair_id!r} spans multiple splits: {pair_splits}"


def test_test_split_is_disjoint_from_the_tuning_splits_by_id_and_question_text() -> None:
    """The large test set must share nothing with what thresholds were tuned on (train/val) — and
    nothing with the pilot set whose results were already looked at. Ids are unique per split by
    construction (splits.json maps id -> one label), so this also checks the question *text*, which
    would catch the same question re-appearing under a different id."""

    records = load_dataset()
    splits = load_splits()
    tuning_and_pilot = {r["question"] for r in records if splits[r["id"]] != "test"}
    test_questions = [r["question"] for r in records if splits[r["id"]] == "test"]
    assert test_questions, "expected a non-empty test split"
    assert not tuning_and_pilot & set(test_questions)
    assert len(test_questions) == len(set(test_questions))


def test_every_split_has_probes_that_tune_thresholds_can_use() -> None:
    probes = load_probes()
    splits = load_splits()
    split_of_probe = {p["id"]: splits[p["query_id"]] for p in probes}
    for split in ("train", "val", "test"):
        assert any(s == split for s in split_of_probe.values()), f"no injection probes in {split}"
