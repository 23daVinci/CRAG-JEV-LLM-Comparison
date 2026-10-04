from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, SystemOneResponse, TypeSafeError

from jev_bench.deciders.base import PassageVerdict, RetryChoice
from jev_bench.graph import prompts
from jev_bench.telemetry.model import CallMetrics, Decision
from jev_bench.telemetry.pricing import PriceBook

DEFAULT_THRESHOLDS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "thresholds.yaml"


def _request_id(response: SystemOneResponse) -> str | None:
    try:
        return response.request_id
    except TypeSafeError:
        # The SDK raises if the server omitted the x-typesafe-request-id header — rare, but not
        # worth failing the whole decision over since it's telemetry, not a correctness input.
        return None


@dataclass(frozen=True, slots=True)
class JevThresholds:
    relevance: float
    evidence: float
    injection: float
    contradiction: float
    sufficiency: float

    @classmethod
    def load(cls, path: Path = DEFAULT_THRESHOLDS_PATH) -> JevThresholds:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**raw)


class JevDecider:
    """Every graph decision resolved by Jev. The probability-vs-boolean asymmetry with Gemini is
    resolved entirely in here: thresholds convert Jev's raw probabilities into the same booleans
    the graph reads off either backend, and the raw probability is kept only in `Decision.evidence`
    for telemetry — never exposed to graph code.
    """

    name = "jev"

    def __init__(
        self,
        client: AsyncTypeSafeClient,
        thresholds: JevThresholds,
        price_book: PriceBook,
        model: str = "jev-latest",
    ) -> None:
        self._client = client
        self._thresholds = thresholds
        self._price_book = price_book
        self._model = model

    def _metrics(self, started_at: float, response: SystemOneResponse) -> CallMetrics:
        latency_ms = (time.perf_counter() - started_at) * 1000
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        now = datetime.now(UTC)
        cost = self._price_book.cost_usd("jev", self._model, input_tokens, output_tokens, now)
        return CallMetrics(
            provider="jev",
            model=self._model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_source="measured",
            request_id=_request_id(response),
            captured_at=now.isoformat(),
        )

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]:
        started_at = time.perf_counter()
        response = await self._client.system_one(
            doc,
            {
                "is_relevant": Noul(instructions=prompts.RELEVANCE_INSTRUCTION.format(query=query)),
                "has_evidence": Noul(instructions=prompts.EVIDENCE_INSTRUCTION.format(query=query)),
                "contradicts_query": Noul(
                    instructions=prompts.CONTRADICTION_INSTRUCTION.format(query=query)
                ),
                "is_injection": Noul(instructions=prompts.INJECTION_INSTRUCTION),
            },
            model=self._model,
        )
        t = self._thresholds
        nouls = response.nouls
        relevant_p = nouls["is_relevant"].noul
        evidence_p = nouls["has_evidence"].noul
        contradicts_p = nouls["contradicts_query"].noul
        injection_p = nouls["is_injection"].noul
        is_injection = injection_p >= t.injection
        verdict = PassageVerdict(
            relevant=(relevant_p >= t.relevance) and not is_injection,
            has_evidence=evidence_p >= t.evidence,
            contradicts_query=contradicts_p >= t.contradiction,
            is_injection=is_injection,
        )
        evidence = {
            "relevant": relevant_p,
            "has_evidence": evidence_p,
            "contradicts_query": contradicts_p,
            "is_injection": injection_p,
        }
        metrics = self._metrics(started_at, response)
        return Decision(value=verdict, metrics=metrics, evidence=evidence)

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]:
        started_at = time.perf_counter()
        joined = "\n\n".join(selected_docs)
        response = await self._client.system_one(
            joined,
            {
                "sufficient": Noul(
                    instructions=prompts.SUFFICIENCY_INSTRUCTION.format(query=query)
                ),
            },
            model=self._model,
        )
        probability = response.nouls["sufficient"].noul
        value = probability >= self._thresholds.sufficiency
        metrics = self._metrics(started_at, response)
        return Decision(value=value, metrics=metrics, evidence={"sufficient": probability})

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]:
        started_at = time.perf_counter()
        state = (
            f"question: {query!r}\nattempt so far: {attempt}\ncurrently sufficient: {sufficient}"
        )
        response = await self._client.system_one(
            state,
            {
                "retry_or_return": Choice(
                    instructions=prompts.RETRY_INSTRUCTION,
                    criteria=prompts.RETRY_CRITERIA,
                ),
            },
            model=self._model,
        )
        answer = response.choices["retry_or_return"]
        value: RetryChoice = "retry" if answer.choice == "retry" else "return"
        metrics = self._metrics(started_at, response)
        return Decision(
            value=value,
            metrics=metrics,
            evidence={"confidence": answer.confidence, **answer.probabilities},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
