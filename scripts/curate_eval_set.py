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
N_BRIDGE = 32
N_COMPARISON = 13

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "eval" / "dataset.jsonl"
DATA_CARD_PATH = REPO_ROOT / "eval" / "DATA_CARD.md"


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

    DATA_CARD_PATH.write_text(
        DATA_CARD_TEMPLATE.format(
            n=len(records), n_bridge=N_BRIDGE, n_comparison=N_COMPARISON, seed=SEED
        ),
        encoding="utf-8",
    )

    print(f"wrote {len(records)} records to {OUTPUT_PATH}")


DATA_CARD_TEMPLATE = """\
# eval/dataset.jsonl — data card

## Source

[HotpotQA](https://hotpotqa.github.io/) dev set, **distractor setting**
(`hotpot_dev_distractor_v1.json`), Yang et al. 2018, Stanford NLP.

License: **CC BY-SA 4.0** (dataset and underlying Wikipedia paragraphs). This slice and anything
derived from it (including `eval/groundedness_labels.jsonl` if added later) inherit that
share-alike obligation — do not relicense derived label files as MIT alongside the rest of the
repo.

## What's in this file

A frozen slice of {n} questions ({n_bridge} bridge + {n_comparison} comparison), selected with a
fixed seed ({seed}) from the full ~7.4k-question dev-distractor set. The full dev set is entirely
"hard" difficulty — HotpotQA's easy/medium questions exist only in the train split, not dev — so
difficulty is not a selection axis here.

Each line is one JSON record:

```json
{{
  "id": "<HotpotQA _id>",
  "question": "...",
  "answer": "...",
  "type": "bridge" | "comparison",
  "level": "hard",
  "context": {{"<Wikipedia title>": "<paragraph text>", ...}},   // all 10 distractor-setting docs
  "gold_titles": ["<title>", ...]                                 // 2 titles, from supporting_facts
}}
```

`gold_titles` is the ground truth this project uses for quality parity: selected-document-set
precision/recall/F1 against `gold_titles`, not a human or LLM-judged label. `answer` is kept for
reference but is not consumed anywhere in this pipeline — there is no generation step (see the
project's plan/README for why).

## Curation method

`scripts/curate_eval_set.py`, run once against a local copy of `hotpot_dev_distractor_v1.json`
(not committed — see `.gitignore`; the official host is
`http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json`, mirrored on Hugging Face
at `RAGLAB/data`). Selection: shuffle bridge and comparison questions separately under a fixed seed,
take the first {n_bridge} and {n_comparison} respectively, interleave. No filtering by content,
length, or difficulty beyond type — the goal is a representative, not a cherry-picked, slice.
"""


if __name__ == "__main__":
    main(sys.argv[1])
