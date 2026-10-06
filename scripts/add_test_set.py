"""Append a large, fresh held-out test set (1,000 questions) to eval/dataset.jsonl and label it
`test` in eval/splits.json.

Why: the original 10-query test split was too small to say anything firm (one query moved mean F1 by
0.10, and the paired F1 CI straddled zero). Thresholds are NOT re-tuned for this set — Jev's
relevance/injection thresholds stay at the values tuned on the 80 train / 10 val queries — so this
measures how the tuned system generalizes with minimal tuning, on enough questions to be measured
properly.

Contamination rules, enforced here:
- every new question is absent from the existing dataset, i.e. disjoint from train, val AND the old
  10-query test split (relabelled `pilot` in splits.json: its results were already looked at, so it
  is kept for reproducing the published pilot numbers but is not part of the new test set);
- questions are drawn from the same HotpotQA dev-distractor file under the same documented seed,
  with
  the same bridge:comparison ratio (71:29) as the tuning data, so train and test differ in which
  questions they contain, not in the mix of question types.

Pipeline order (each step is deterministic): curate_eval_set.py -> split_eval_queries.py ->
this script -> generate_injection_probes.py.

Usage:
    uv run python scripts/add_test_set.py tmp_hotpot_dev_distractor_v1.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 20260915  # same documented seed as the other eval/ curation scripts
BASE_N = 100  # train + val + pilot, as written by curate_eval_set.py
N_TEST_BRIDGE = 710
N_TEST_COMPARISON = 290  # 710:290 == 71:29, the ratio used for the tuning data

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "dataset.jsonl"
SPLITS_PATH = REPO_ROOT / "eval" / "splits.json"


def to_record(raw: dict) -> dict:
    context = {title: " ".join(sentences) for title, sentences in raw["context"]}
    gold_titles = sorted({title for title, _ in raw["supporting_facts"]})
    return {
        "id": raw["_id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw["type"],
        "level": raw["level"],
        "context": context,
        "gold_titles": gold_titles,
    }


def main(source_path: str) -> None:
    existing = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(existing) != BASE_N:
        raise ValueError(
            f"expected exactly {BASE_N} base records in {DATASET_PATH}, found {len(existing)} — "
            "re-run curate_eval_set.py first (this script is not meant to be applied twice)"
        )
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    if set(splits) != {r["id"] for r in existing} or "test" in splits.values():
        raise ValueError(
            "eval/splits.json must cover exactly the base records and contain no 'test' labels — "
            "re-run split_eval_queries.py first"
        )
    taken = set(splits)

    raw_records = json.loads(Path(source_path).read_text(encoding="utf-8"))
    bridge = [r for r in raw_records if r["type"] == "bridge"]
    comparison = [r for r in raw_records if r["type"] == "comparison"]
    rng = random.Random(SEED)
    rng.shuffle(bridge)
    rng.shuffle(comparison)

    fresh_bridge = [r for r in bridge if r["_id"] not in taken][:N_TEST_BRIDGE]
    fresh_comparison = [r for r in comparison if r["_id"] not in taken][:N_TEST_COMPARISON]
    if len(fresh_bridge) != N_TEST_BRIDGE or len(fresh_comparison) != N_TEST_COMPARISON:
        raise ValueError("not enough unused questions in the source file for the requested size")

    selected = fresh_bridge + fresh_comparison
    rng.shuffle(selected)
    records = [to_record(r) for r in selected]
    assert not ({r["id"] for r in records} & taken), "test set overlaps train/val/pilot"

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    for record in records:
        splits[record["id"]] = "test"
    SPLITS_PATH.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for label in splits.values():
        counts[label] = counts.get(label, 0) + 1
    print(f"appended {len(records)} test questions to {DATASET_PATH}: splits now {counts}")


if __name__ == "__main__":
    main(sys.argv[1])
