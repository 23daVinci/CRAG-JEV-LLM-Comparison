from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from jev_bench.telemetry.model import Decision


@dataclass(frozen=True, slots=True)
class PassageVerdict:
    """One document's screening verdict. `relevant` is the only field the graph branches on;
    the rest exist for telemetry and for the injection guard."""

    relevant: bool
    has_evidence: bool
    contradicts_query: bool
    is_injection: bool


RetryChoice = Literal["retry", "return"]


class Decider(Protocol):
    """Every method is a typed, semantic question — never a generic `judge(prompt)` — so the two
    backends can't drift into answering different questions. The graph reads only `.value` off
    each Decision; `.metrics`/`.evidence` exist purely for telemetry and are never branched on."""

    name: str

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]: ...

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]: ...

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]: ...

    async def aclose(self) -> None: ...
