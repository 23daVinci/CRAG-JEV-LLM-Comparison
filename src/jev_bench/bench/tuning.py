"""Train/val/test threshold tuning for Jev's probability outputs (branch `tune-jev-thresholds`,
see the plan and docs/METHODOLOGY.md).

Deliberately bypasses the graph: `collect_relevance_probabilities` calls `Decider.screen_passage`
directly against every document in a query's context, rather than running `run_dataset`/
`run_comparison`. A graph run's retry loop stops as soon as `grade_sufficiency` says "sufficient",
so graph-collected `passage_evidence` systematically under-samples whichever documents a real run
happened to stop before reaching (confirmed from the real 34-query run: 24/34 queries only screened
5 of 10 docs) — that's a real selection bias, not a neutral shortcut, so tuning must not reuse it.

Train picks a threshold by maximizing Jev's own F1 (not iso-recall against another backend — that
would let the comparison target's behavior dictate Jev's operating point). Val replays that one
fixed threshold with no re-sweeping, purely to confirm it holds up out-of-sample. Test is scored
separately, exactly once, by the caller (the `tune-thresholds` CLI command never touches test).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jev_bench.bench.analysis import (
    SelectionMetrics,
    ThresholdPoint,
    metrics_at_threshold,
    pick_best_f1_threshold,
    sweep_fixed_thresholds,
    threshold_point_f1,
)
from jev_bench.bench.injection_eval import Probe, evaluate_injection_guard, load_probes
from jev_bench.bench.runner import Record, load_dataset
from jev_bench.deciders.base import Decider

DEFAULT_SPLITS_PATH = Path(__file__).parent.parent.parent.parent / "eval" / "splits.json"
DEFAULT_THRESHOLDS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "thresholds.yaml"

Split = str  # "train" | "val" | "test"

# 0.05 steps across the middle, plus 0.98/0.99 because the injection signal separates almost
# perfectly near the top (a coarser grid ending at 0.9 would lose resolution exactly there).
CANDIDATE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)] + [0.98, 0.99]


def load_splits(path: Path = DEFAULT_SPLITS_PATH) -> dict[str, Split]:
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def filter_records_by_split(
    records: list[Record], splits: dict[str, Split], split: Split
) -> list[Record]:
    return [r for r in records if splits.get(r["id"]) == split]


def filter_probes_by_split(
    probes: list[Probe], splits: dict[str, Split], split: Split
) -> list[Probe]:
    """Probes carry a `query_id`, not their own split — they always inherit the split of the
    dataset query they were sampled from, so both halves of a clean/injected pair stay together."""

    return [p for p in probes if splits.get(p["query_id"]) == split]


async def collect_relevance_probabilities(
    decider: Decider, records: list[Record]
) -> tuple[list[float], list[bool]]:
    probabilities: list[float] = []
    labels: list[bool] = []
    for record in records:
        gold = set(record["gold_titles"])
        for title, text in record["context"].items():
            decision = await decider.screen_passage(record["question"], text)
            probabilities.append(decision.evidence["relevant"])
            labels.append(title in gold)
    return probabilities, labels


async def collect_injection_probabilities(
    decider: Decider, probes: list[Probe]
) -> tuple[list[float], list[bool]]:
    result = await evaluate_injection_guard(decider, probes)
    scored = [p for p in probes if p["id"] in result.probe_evidence]
    probabilities = [result.probe_evidence[p["id"]] for p in scored]
    labels = [bool(p["is_injection"]) for p in scored]
    return probabilities, labels


@dataclass(frozen=True, slots=True)
class TuningOutcome:
    train_threshold: ThresholdPoint
    train_f1: float
    train_points: list[ThresholdPoint]
    val_metrics: SelectionMetrics
    val_f1_drop: float
    """train_f1 - val_metrics.f1 — positive means val performed worse than train. Flag (don't
    silently accept) drops over 10 points; at train/val sizes this small, that's a real risk that
    the picked threshold is overfit to train, not a formality."""


def tune_target(
    train_probabilities: list[float],
    train_labels: list[bool],
    val_probabilities: list[float],
    val_labels: list[bool],
) -> TuningOutcome:
    points = sweep_fixed_thresholds(train_probabilities, train_labels, CANDIDATE_THRESHOLDS)
    best = pick_best_f1_threshold(points)
    train_f1 = threshold_point_f1(best)
    val_metrics = metrics_at_threshold(val_probabilities, val_labels, best.threshold)
    return TuningOutcome(
        train_threshold=best,
        train_f1=train_f1,
        train_points=points,
        val_metrics=val_metrics,
        val_f1_drop=train_f1 - val_metrics.f1,
    )


async def tune_relevance_threshold(
    decider: Decider,
    records: list[Record] | None = None,
    splits: dict[str, Split] | None = None,
) -> TuningOutcome:
    records = records if records is not None else load_dataset()
    splits = splits if splits is not None else load_splits()
    train_records = filter_records_by_split(records, splits, "train")
    val_records = filter_records_by_split(records, splits, "val")
    train_probabilities, train_labels = await collect_relevance_probabilities(
        decider, train_records
    )
    val_probabilities, val_labels = await collect_relevance_probabilities(decider, val_records)
    return tune_target(train_probabilities, train_labels, val_probabilities, val_labels)


async def tune_injection_threshold(
    decider: Decider,
    probes: list[Probe] | None = None,
    splits: dict[str, Split] | None = None,
) -> TuningOutcome:
    probes = probes if probes is not None else load_probes()
    splits = splits if splits is not None else load_splits()
    train_probes = filter_probes_by_split(probes, splits, "train")
    val_probes = filter_probes_by_split(probes, splits, "val")
    train_probabilities, train_labels = await collect_injection_probabilities(decider, train_probes)
    val_probabilities, val_labels = await collect_injection_probabilities(decider, val_probes)
    return tune_target(train_probabilities, train_labels, val_probabilities, val_labels)


def apply_threshold_update(
    field_name: str, new_value: float, path: Path = DEFAULT_THRESHOLDS_PATH
) -> None:
    """Edits one `field_name: value` line in place rather than a full yaml.safe_load/dump round
    trip — pyyaml's dumper would drop the header comment block that documents which fields are
    tuned and why, which is exactly the provenance this methodology requires keeping."""

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix = f"{field_name}:"
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{field_name}: {new_value}\n"
            break
    else:
        raise KeyError(f"{field_name!r} not found in {path}")
    path.write_text("".join(lines), encoding="utf-8")
