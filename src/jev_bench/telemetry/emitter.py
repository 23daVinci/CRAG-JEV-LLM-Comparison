from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "run_started", "node_finished", "variant_finished", "run_finished", "decision_error"
]
"""Note: never "error" — that name is reserved by the browser's EventSource for connection-level
failures (`source.onerror`), so a server-sent `event: error` would never reach a named listener."""


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    variant: str
    seq: int
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class RunEmitter:
    """One asyncio.Queue per run, fed by the two concurrent variant pumps in api/app.py and
    drained by the SSE endpoint. A plain queue, not a broadcast/pubsub — each run has exactly one
    subscriber (the browser tab that started it)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self._seq = 0

    def emit(self, variant: str, event_type: EventType, payload: dict[str, Any]) -> None:
        self._seq += 1
        self._queue.put_nowait(
            RunEvent(
                run_id=self.run_id, variant=variant, seq=self._seq, type=event_type, payload=payload
            )
        )

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def stream(self) -> Any:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event
