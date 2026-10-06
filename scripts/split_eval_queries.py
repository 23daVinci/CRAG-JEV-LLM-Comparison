"""Deterministic train/val/test split of eval/dataset.jsonl's 100 queries, stratified by `type`
(bridge/comparison) so each split keeps roughly the full set's bridge:comparison ratio.

Built for the Jev threshold-tuning phase (branch `tune-jev-thresholds`, see the plan and
docs/METHODOLOGY.md): train picks a candidate threshold via a precision-recall sweep, val confirms
it without re-sweeping, test is scored exactly once after tuning is locked. `eval/injection_probes
.jsonl` records are assigned to a split by looking up their `query_id` here — never an independent
random assignment — so both members of a clean/injected pair always land in the same split.

This script splits only the 100 BASE queries (the first 100 lines of dataset.jsonl) into train (80),
val (10) and `pilot` (10). The `pilot` split is the original small held-out test set: it was scored
and looked at before the large test set existed, so it is kept for reproducing those published pilot
numbers but is not the headline test set. The large `test` split (1,000 fresh questions) is added by
scripts/add_test_set.py, which runs after this script.

At 10 queries, a bootstrap CI over queries is still required for any pilot-split number; a bare
point estimate is not defensible at this n.

Usage:
    uv run python scripts/split_eval_queries.py

Deterministic (fixed seed, same one used throughout eval/ curation scripts): rerunning reproduces
the same assignment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260915  # same documented seed as curate_eval_set.py / generate_injection_probes.py

# Per-stratum counts chosen to land on an overall 80/10/10 split (80/10/10 of 100) while keeping
# each stratum's own train:val:test ratio close to 80:10:10. Explicit counts are simpler to audit
# than a generic rounding rule that happens to land in the same place.
BRIDGE_COUNTS = {"train": 57, "val": 7, "pilot": 7}  # bridge n=71
COMPARISON_COUNTS = {"train": 23, "val": 3, "pilot": 3}  # comparison n=29

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "dataset.jsonl"
OUTPUT_PATH = REPO_ROOT / "eval" / "splits.json"


BASE_N = 100


def load_dataset() -> list[dict]:
    lines = [line for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[:BASE_N]]


def split_stratum(
    records: list[dict], counts: dict[str, int], rng: random.Random
) -> dict[str, str]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    assignment: dict[str, str] = {}
    i = 0
    for split_name in ("train", "val", "pilot"):
        n = counts[split_name]
        for record in shuffled[i : i + n]:
            assignment[record["id"]] = split_name
        i += n
    return assignment


def main() -> None:
    records = load_dataset()
    bridge = [r for r in records if r["type"] == "bridge"]
    comparison = [r for r in records if r["type"] == "comparison"]

    if len(bridge) != 71 or len(comparison) != 29:
        raise ValueError(
            f"expected 71 bridge + 29 comparison records, found {len(bridge)} + {len(comparison)} "
            "— BRIDGE_COUNTS/COMPARISON_COUNTS need updating if eval/dataset.jsonl changed"
        )

    rng = random.Random(SEED)
    assignment: dict[str, str] = {}
    assignment.update(split_stratum(bridge, BRIDGE_COUNTS, rng))
    assignment.update(split_stratum(comparison, COMPARISON_COUNTS, rng))

    counts = {"train": 0, "val": 0, "pilot": 0}
    for split_name in assignment.values():
        counts[split_name] += 1

    OUTPUT_PATH.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(assignment)} query assignments to {OUTPUT_PATH}: {counts}")


if __name__ == "__main__":
    main()
