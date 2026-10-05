from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jev_bench.deciders.base import PassageVerdict, RetryChoice
from jev_bench.telemetry.model import CallMetrics, Decision

ZERO_METRICS = CallMetrics(
    provider="fake",
    model="scripted",
    latency_ms=0.0,
    input_tokens=0,
    output_tokens=0,
    cost_usd=0.0,
    latency_source="simulated",
)


@dataclass
class ScriptedDecider:
    """A network-free Decider for unit and contract tests. Never feeds a report — `latency_source`
    is pinned to "simulated", which `report.py` refuses to treat as a benchmark number.

    Behavior is overridable per test via the callables below; defaults mark everything relevant
    and sufficient on the first attempt, so a graph wired to a ScriptedDecider terminates quickly.
    """

    name: str = "fake"
    screen_fn: Callable[[str, str], PassageVerdict] = field(
        default=lambda query, doc: PassageVerdict(
            relevant=True, has_evidence=True, contradicts_query=False, is_injection=False
        )
    )
    sufficiency_fn: Callable[[str, list[str]], bool] = field(
        default=lambda query, docs: len(docs) > 0
    )
    retry_fn: Callable[[str, int, bool], RetryChoice] = field(
        default=lambda query, attempt, sufficient: "return" if sufficient else "retry"
    )
    evidence_fn: Callable[[str, str], dict[str, float]] = field(default=lambda query, doc: {})
    """Lets threshold-tuning tests control raw *probabilities* (e.g. `{"relevant": 0.83}`), not just
    the thresholded boolean `screen_fn` returns — unset, behavior is identical to before this field
    existed (`evidence={}`, matching the `Decision` default)."""

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]:
        return Decision(
            value=self.screen_fn(query, doc),
            metrics=ZERO_METRICS,
            evidence=self.evidence_fn(query, doc),
        )

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]:
        return Decision(value=self.sufficiency_fn(query, selected_docs), metrics=ZERO_METRICS)

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]:
        return Decision(value=self.retry_fn(query, attempt, sufficient), metrics=ZERO_METRICS)

    async def aclose(self) -> None:
        pass
