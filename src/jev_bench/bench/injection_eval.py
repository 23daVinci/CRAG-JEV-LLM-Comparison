from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_bench.deciders.base import Decider

DEFAULT_PROBES_PATH = Path(__file__).parent.parent.parent.parent / "eval" / "injection_probes.jsonl"

Probe = dict[str, Any]


def load_probes(path: Path = DEFAULT_PROBES_PATH) -> list[Probe]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(frozen=True, slots=True)
class InjectionEvalResult:
    accuracy: float
    precision: float
    recall: float
    false_positives: list[str]
    false_negatives: list[str]


async def evaluate_injection_guard(
    decider: Decider, probes: list[Probe] | None = None
) -> InjectionEvalResult:
    probes = probes if probes is not None else load_probes()

    true_positive = false_positive = true_negative = false_negative = 0
    false_positive_ids: list[str] = []
    false_negative_ids: list[str] = []

    for probe in probes:
        decision = await decider.screen_passage(probe["query"], probe["text"])
        predicted = decision.value.is_injection
        actual = probe["is_injection"]
        if predicted and actual:
            true_positive += 1
        elif predicted and not actual:
            false_positive += 1
            false_positive_ids.append(probe["id"])
        elif not predicted and actual:
            false_negative += 1
            false_negative_ids.append(probe["id"])
        else:
            true_negative += 1

    total = len(probes)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    return InjectionEvalResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        false_positives=false_positive_ids,
        false_negatives=false_negative_ids,
    )
