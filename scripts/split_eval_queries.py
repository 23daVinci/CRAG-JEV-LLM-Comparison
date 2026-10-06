"""Deterministic train/val/test split of eval/dataset.jsonl's 100 queries, stratified by `type`
(bridge/comparison) so each split keeps roughly the full set's bridge:comparison ratio.

Built for the Jev threshold-tuning phase (branch `tune-jev-thresholds`, see the plan and
docs/METHODOLOGY.md): train picks a candidate threshold via a precision-recall sweep, val confirms
it without re-sweeping, test is scored exactly once after tuning is locked. `eval/injection_probes
.jsonl` records are assigned to a split by looking up their `query_id` here — never an independent
random assignment — so both members of a clean/injected pair always land in the same split.

At 100 queries, an 80/10/10 split still leaves val/test thin (10 queries each) — a bootstrap CI
over queries is still required wherever test-split numbers are reported, a bare point estimate is
not defensible at this n, even though it's roomier than the original 45-query/60-20-20 split.

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
BRIDGE_COUNTS = {"train": 57, "val": 7, "test": 7}  # bridge n=71
COMPARISON_COUNTS = {"train": 23, "val": 3, "test": 3}  # comparison n=29

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "dataset.jsonl"
OUTPUT_PATH = REPO_ROOT / "eval" / "splits.json"


def load_dataset() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_stratum(
    records: list[dict], counts: dict[str, int], rng: random.Random
) -> dict[str, str]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    assignment: dict[str, str] = {}
    i = 0
    for split_name in ("train", "val", "test"):
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

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name in assignment.values():
        counts[split_name] += 1

    OUTPUT_PATH.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(assignment)} query assignments to {OUTPUT_PATH}: {counts}")


if __name__ == "__main__":
    main()
