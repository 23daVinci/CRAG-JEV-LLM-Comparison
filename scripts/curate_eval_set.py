"""One-time curation step (plan milestone 2): slice a frozen, small eval set out of HotpotQA's
dev-distractor file and commit just that slice — not the 46MB source file — into eval/dataset.jsonl.

Usage:
    uv run python scripts/curate_eval_set.py tmp_hotpot_dev_distractor_v1.json

Deterministic (fixed seed): rerunning against the same source file reproduces the same slice.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 20260915  # Jev's launch date, as a fixed, documented seed — not tuned to flatter a result.
# 71:29 preserves the original 45-query slice's 32:13 bridge:comparison ratio, scaled to a 100-
# query total — chosen so an 80/10/10 split lands on clean per-stratum counts (57/7/7 bridge,
# 23/3/3 comparison), see scripts/split_eval_queries.py.
N_BRIDGE = 71
N_COMPARISON = 29

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "eval" / "dataset.jsonl"


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
    raw_records = json.loads(Path(source_path).read_text(encoding="utf-8"))

    bridge = [r for r in raw_records if r["type"] == "bridge"]
    comparison = [r for r in raw_records if r["type"] == "comparison"]

    rng = random.Random(SEED)
    rng.shuffle(bridge)
    rng.shuffle(comparison)

    selected = bridge[:N_BRIDGE] + comparison[:N_COMPARISON]
    rng.shuffle(selected)

    records = [to_record(r) for r in selected]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} records to {OUTPUT_PATH}")
    print("Note: eval/DATA_CARD.md is maintained by hand (covers this script's output plus")
    print("split_eval_queries.py's and generate_injection_probes.py's) — update it manually.")


if __name__ == "__main__":
    main(sys.argv[1])
