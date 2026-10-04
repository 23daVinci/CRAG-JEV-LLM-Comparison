from __future__ import annotations

import time
from typing import Any

from jev_bench.telemetry.model import CallMetrics, Decision, NodeSpan


class NodeSpanRecorder:
    """Accumulates CallMetrics from decider calls made inside one graph node, then produces the
    NodeSpan that the node returns under `state["trace"]`. Telemetry flows through LangGraph's own
    `operator.add` reducer on that field, not through ambient contextvars or global state — that's
    what keeps parallel branches from cross-contaminating attribution."""

    def __init__(self, node: str, backend: str) -> None:
        self.node = node
        self.backend = backend
        self.calls: list[CallMetrics] = []
        self._started_at = time.perf_counter()

    def record(self, decision: Decision[Any]) -> None:
        self.calls.append(decision.metrics)

    def finish(self) -> NodeSpan:
        return NodeSpan(
            node=self.node,
            backend=self.backend,
            started_at=self._started_at,
            finished_at=time.perf_counter(),
            calls=self.calls,
        )
