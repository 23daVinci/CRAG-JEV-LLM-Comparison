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
    assert set(splits.values()) <= {"train", "val", "test"}


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
