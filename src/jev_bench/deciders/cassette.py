from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from jev_bench.deciders.base import Decider, PassageVerdict, RetryChoice
from jev_bench.telemetry.model import CallMetrics, Decision

CassetteMode = Literal["record", "replay"]


def _hash_key(method: str, backend: str, model: str, **inputs: object) -> str:
    canonical = json.dumps(
        {"method": method, "backend": backend, "model": model, "inputs": inputs}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class CassetteDecider:
    """Intercepts at the Decider boundary, not HTTP — VCR-style HTTP recording bakes in auth
    headers and breaks on SDK transport changes. In "record" mode every call goes through to
    `inner` and the genuine response is appended to the cassette file; in "replay" mode nothing
    touches the network and `latency_source` is overridden to "replayed", so `report.py` can refuse
    to treat a cassette as a fresh benchmark number while still trusting its token/cost figures.
    """

    def __init__(
        self, inner: Decider, cassette_path: Path, mode: CassetteMode, model: str = ""
    ) -> None:
        self._inner = inner
        self._path = cassette_path
        self._mode = mode
        self._model = model or getattr(inner, "name", "")
        self.name = inner.name
        self._entries: dict[str, dict[str, Any]] = {}
        if cassette_path.exists():
            for line in cassette_path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    def _append(self, key: str, method: str, value: object, decision: Decision[Any]) -> None:
        entry = {
            "key": key,
            "method": method,
            "value": value,
            "metrics": asdict(decision.metrics),
            "evidence": decision.evidence,
        }
        self._entries[key] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _replay(self, key: str, method: str, value_fn: Callable[[Any], Any]) -> Decision[Any]:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"no cassette entry for {method} key={key} in {self._path}")
        metrics_dict = dict(entry["metrics"])
        metrics_dict["latency_source"] = "replayed"
        metrics = CallMetrics(**metrics_dict)
        return Decision(value=value_fn(entry["value"]), metrics=metrics, evidence=entry["evidence"])

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]:
        key = _hash_key("screen_passage", self.name, self._model, query=query, doc=doc)
        if self._mode == "replay":
            return self._replay(key, "screen_passage", lambda v: PassageVerdict(**v))
        decision = await self._inner.screen_passage(query, doc)
        self._append(key, "screen_passage", asdict(decision.value), decision)
        return decision

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]:
        key = _hash_key(
            "grade_sufficiency", self.name, self._model, query=query, selected_docs=selected_docs
        )
        if self._mode == "replay":
            return self._replay(key, "grade_sufficiency", bool)
        decision = await self._inner.grade_sufficiency(query, selected_docs)
        self._append(key, "grade_sufficiency", decision.value, decision)
        return decision

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]:
        key = _hash_key(
            "decide_retry",
            self.name,
            self._model,
            query=query,
            attempt=attempt,
            sufficient=sufficient,
        )
        if self._mode == "replay":
            return self._replay(key, "decide_retry", lambda v: v)
        decision = await self._inner.decide_retry(query, attempt, sufficient)
        self._append(key, "decide_retry", decision.value, decision)
        return decision
