from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from jev_bench.deciders.base import PassageVerdict, RetryChoice
from jev_bench.graph import prompts
from jev_bench.telemetry.model import CallMetrics, Decision
from jev_bench.telemetry.pricing import PriceBook


class _PassageGrade(BaseModel):
    relevant: bool
    has_evidence: bool
    contradicts_query: bool
    is_injection: bool


class _SufficiencyGrade(BaseModel):
    sufficient: bool


class _RetryGrade(BaseModel):
    choice: Literal["retry", "return"]


class GeminiDecider:
    """Every graph decision resolved by a Gemini structured-output call — one call per decision,
    never batched, which is deliberately "what the LangGraph CRAG template actually ships" rather
    than the strongest possible Gemini baseline (see docs/METHODOLOGY.md's batching 2x2)."""

    name = "gemini"

    def __init__(self, model: ChatGoogleGenerativeAI, price_book: PriceBook) -> None:
        self._price_book = price_book
        self._model_name = model.model
        self._screen_chain = model.with_structured_output(_PassageGrade, include_raw=True)
        self._sufficiency_chain = model.with_structured_output(_SufficiencyGrade, include_raw=True)
        self._retry_chain = model.with_structured_output(_RetryGrade, include_raw=True)

    def _metrics(self, started_at: float, raw: AIMessage) -> CallMetrics:
        usage: dict[str, Any] = dict(raw.usage_metadata or {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning", 0)
        if reasoning_tokens:
            raise RuntimeError(
                f"Gemini reported {reasoning_tokens} thinking tokens on a grader call; thinking "
                "must be disabled (thinking_budget=0) so cost/latency stay comparable to Jev."
            )
        latency_ms = (time.perf_counter() - started_at) * 1000
        now = datetime.now(UTC)
        cost = self._price_book.cost_usd(
            "gemini", self._model_name, input_tokens, output_tokens, now
        )
        response_metadata = raw.response_metadata or {}
        return CallMetrics(
            provider="gemini",
            model=self._model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_source="measured",
            request_id=response_metadata.get("id"),
            captured_at=now.isoformat(),
        )

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]:
        started_at = time.perf_counter()
        prompt = (
            f"{prompts.RELEVANCE_INSTRUCTION.format(query=query)}\n"
            f"{prompts.EVIDENCE_INSTRUCTION.format(query=query)}\n"
            f"{prompts.CONTRADICTION_INSTRUCTION.format(query=query)}\n"
            f"{prompts.INJECTION_INSTRUCTION}\n\n"
            f"Passage:\n{doc}"
        )
        result = cast(dict[str, Any], await self._screen_chain.ainvoke(prompt))
        parsed = result["parsed"]
        assert isinstance(parsed, _PassageGrade)
        verdict = PassageVerdict(
            relevant=parsed.relevant and not parsed.is_injection,
            has_evidence=parsed.has_evidence,
            contradicts_query=parsed.contradicts_query,
            is_injection=parsed.is_injection,
        )
        evidence = {
            "relevant": float(parsed.relevant),
            "has_evidence": float(parsed.has_evidence),
            "contradicts_query": float(parsed.contradicts_query),
            "is_injection": float(parsed.is_injection),
        }
        metrics = self._metrics(started_at, result["raw"])
        return Decision(value=verdict, metrics=metrics, evidence=evidence)

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]:
        started_at = time.perf_counter()
        joined = "\n\n".join(selected_docs)
        prompt = f"{prompts.SUFFICIENCY_INSTRUCTION.format(query=query)}\n\nPassages:\n{joined}"
        result = cast(dict[str, Any], await self._sufficiency_chain.ainvoke(prompt))
        parsed = result["parsed"]
        assert isinstance(parsed, _SufficiencyGrade)
        metrics = self._metrics(started_at, result["raw"])
        return Decision(
            value=parsed.sufficient,
            metrics=metrics,
            evidence={"sufficient": float(parsed.sufficient)},
        )

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]:
        started_at = time.perf_counter()
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in prompts.RETRY_CRITERIA.items())
        prompt = (
            f"{prompts.RETRY_INSTRUCTION}\n{criteria_text}\n\n"
            f"question: {query!r}\nattempt so far: {attempt}\ncurrently sufficient: {sufficient}"
        )
        result = cast(dict[str, Any], await self._retry_chain.ainvoke(prompt))
        parsed = result["parsed"]
        assert isinstance(parsed, _RetryGrade)
        metrics = self._metrics(started_at, result["raw"])
        return Decision(value=parsed.choice, metrics=metrics, evidence={})
