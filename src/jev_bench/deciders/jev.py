from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    SystemOneResponse,
)

from jev_bench.deciders.base import PassageVerdict, RetryChoice
from jev_bench.graph import prompts
from jev_bench.telemetry.model import CallMetrics, Decision
from jev_bench.telemetry.pricing import PriceBook

DEFAULT_THRESHOLDS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "thresholds.yaml"


def _noul(response: SystemOneResponse, key: str) -> float:
    answer = response.answers[key]
    assert isinstance(answer, NoulAnswer), f"expected a Noul answer for {key!r}, got {answer.type}"
    return answer.noul


def _choice(response: SystemOneResponse, key: str) -> ChoiceAnswer:
    answer = response.answers[key]
    assert isinstance(answer, ChoiceAnswer), (
        f"expected a Choice answer for {key!r}, got {answer.type}"
    )
    return answer


@dataclass(frozen=True, slots=True)
class JevThresholds:
    relevance: float
    evidence: float
    injection: float
    contradiction: float
    sufficiency: float

    @classmethod
    def load(cls, path: Path = DEFAULT_THRESHOLDS_PATH) -> JevThresholds:
        raw = yaml.safe_load(path.read_text())
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

    def _metrics(
        self, started_at: float, usage_input: int | None, usage_output: int | None
    ) -> CallMetrics:
        latency_ms = (time.perf_counter() - started_at) * 1000
        input_tokens = usage_input or 0
        output_tokens = usage_output or 0
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
            # typesafe_sdk's SystemOneResponse does not surface a request id on success; only
            # error responses carry one. Left None rather than faked.
            request_id=None,
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
        relevant_p = _noul(response, "is_relevant")
        evidence_p = _noul(response, "has_evidence")
        contradicts_p = _noul(response, "contradicts_query")
        injection_p = _noul(response, "is_injection")
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
        metrics = self._metrics(
            started_at, response.usage.input_tokens, response.usage.output_tokens
        )
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
        probability = _noul(response, "sufficient")
        value = probability >= self._thresholds.sufficiency
        metrics = self._metrics(
            started_at, response.usage.input_tokens, response.usage.output_tokens
        )
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
        answer = _choice(response, "retry_or_return")
        value: RetryChoice = "retry" if answer.choice == "retry" else "return"
        metrics = self._metrics(
            started_at, response.usage.input_tokens, response.usage.output_tokens
        )
        return Decision(
            value=value,
            metrics=metrics,
            evidence={"confidence": answer.confidence, **answer.probabilities},
        )
