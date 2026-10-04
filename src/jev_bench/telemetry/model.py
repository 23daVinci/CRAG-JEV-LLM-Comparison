from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

LatencySource = Literal["measured", "replayed", "simulated"]


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """What a single decision call cost, regardless of which backend answered it."""

    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_source: LatencySource
    request_id: str | None = None
    captured_at: str | None = None


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Decision(Generic[T]):
    """The only channel between a Decider and the graph: `.value` is read by graph code,
    `.metrics`/`.evidence` are read only by telemetry."""

    value: T
    metrics: CallMetrics
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class NodeSpan:
    node: str
    backend: str
    started_at: float
    finished_at: float
    calls: list[CallMetrics] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


@dataclass(slots=True)
class RunTrace:
    run_id: str
    query: str
    backend: str
    spans: list[NodeSpan] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)
