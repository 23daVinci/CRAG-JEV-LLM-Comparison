# eval/dataset.jsonl — data card

## Source

[HotpotQA](https://hotpotqa.github.io/) dev set, **distractor setting**
(`hotpot_dev_distractor_v1.json`), Yang et al. 2018, Stanford NLP.

License: **CC BY-SA 4.0** (dataset and underlying Wikipedia paragraphs). This slice and anything
derived from it (including `eval/groundedness_labels.jsonl` if added later) inherit that
share-alike obligation — do not relicense derived label files as MIT alongside the rest of the
repo.

## What's in this file

A frozen slice of 45 questions (32 bridge + 13 comparison), selected with a
fixed seed (20260915) from the full ~7.4k-question dev-distractor set. The full dev set is entirely
"hard" difficulty — HotpotQA's easy/medium questions exist only in the train split, not dev — so
difficulty is not a selection axis here.

Each line is one JSON record:

```json
{
  "id": "<HotpotQA _id>",
  "question": "...",
  "answer": "...",
  "type": "bridge" | "comparison",
  "level": "hard",
  "context": {"<Wikipedia title>": "<paragraph text>", ...},   // all 10 distractor-setting docs
  "gold_titles": ["<title>", ...]                                 // 2 titles, from supporting_facts
}
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
take the first 32 and 13 respectively, interleave. No filtering by content,
length, or difficulty beyond type — the goal is a representative, not a cherry-picked, slice.
