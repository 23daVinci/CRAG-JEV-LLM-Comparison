from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any, Literal, cast

import groq
import httpx
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
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


_RETRY_DELAY_RE = re.compile(r"try again in (\d+(?:\.\d+)?)s")


def _suggested_retry_delay(error_message: str) -> float | None:
    match = _RETRY_DELAY_RE.search(error_message)
    if match is None:
        return None
    return float(match.group(1)) + 0.5  # small buffer past the server's own estimate


class GroqDecider:
    """Every graph decision resolved by a Groq structured-output call — one call per decision,
    never batched (see docs/METHODOLOGY.md's batching 2x2). Default variant A backend: Groq's
    LPU-hosted inference measured sub-second per call on real, full-length HotpotQA passages,
    avoiding both Gemini's free-tier daily quota wall and Ollama's CPU-bound latency (~7-28s/call
    on this machine). `qwen/qwen3.8-27b` is used specifically because it reports zero hidden
    reasoning tokens, unlike the catalog's `openai/gpt-oss-*` models — see `config.py`."""

    name = "groq"

    def __init__(
        self,
        model: ChatGroq,
        price_book: PriceBook,
        max_retries: int = 6,
    ) -> None:
        self._model = model
        self._price_book = price_book
        self._model_name = model.model_name
        self._max_retries = max_retries
        self._screen_chain = model.with_structured_output(_PassageGrade, include_raw=True)
        self._sufficiency_chain = model.with_structured_output(_SufficiencyGrade, include_raw=True)
        self._retry_chain = model.with_structured_output(_RetryGrade, include_raw=True)

    async def _invoke(self, chain: Any, prompt: str) -> tuple[dict[str, Any], float]:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            started_at = time.perf_counter()
            try:
                result = cast(dict[str, Any], await chain.ainvoke(prompt))
                return result, time.perf_counter() - started_at
            except groq.RateLimitError as exc:
                # Unlike Gemini's *daily* quota (where retrying is pointless for hours), this is a
                # rolling per-minute output-token budget on a "Preview" catalog model (confirmed:
                # 1000 OTPM on this account) — it's worth waiting out. Groq's error message states
                # the server's own suggested wait ("Please try again in 33.48s"); parse and use it
                # directly rather than guessing with blind exponential backoff.
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(_suggested_retry_delay(str(exc)) or delay)
                delay = min(delay * 2, 20.0)
            except httpx.RequestError:
                # Transient connection issues only — ChatGroq's own max_retries is set to 0 at
                # construction (see factory.py) for the same reason as GeminiDecider: avoid two
                # independent retry layers stacking multiplicatively.
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise AssertionError("unreachable: loop always returns or raises")

    def _metrics(self, duration_seconds: float, raw: AIMessage) -> CallMetrics:
        usage: dict[str, Any] = dict(raw.usage_metadata or {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning", 0)
        if reasoning_tokens:
            raise RuntimeError(
                f"Groq model {self._model_name!r} reported {reasoning_tokens} hidden reasoning "
                "tokens on a grader call — this model family's thinking overhead isn't comparable "
                "to Jev's zero-reasoning decisions. Use a non-reasoning model (qwen3.8-27b, not "
                "gpt-oss-*) rather than relaxing this check."
            )
        latency_ms = duration_seconds * 1000
        now = datetime.now(UTC)
        cost = self._price_book.cost_usd("groq", self._model_name, input_tokens, output_tokens, now)
        return CallMetrics(
            provider="groq",
            model=self._model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_source="measured",
            request_id=None,
            captured_at=now.isoformat(),
        )

    async def screen_passage(self, query: str, doc: str) -> Decision[PassageVerdict]:
        prompt = (
            f"{prompts.RELEVANCE_INSTRUCTION.format(query=query)}\n"
            f"{prompts.EVIDENCE_INSTRUCTION.format(query=query)}\n"
            f"{prompts.CONTRADICTION_INSTRUCTION.format(query=query)}\n"
            f"{prompts.INJECTION_INSTRUCTION}\n\n"
            f"Passage:\n{doc}"
        )
        result, duration = await self._invoke(self._screen_chain, prompt)
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
        metrics = self._metrics(duration, result["raw"])
        return Decision(value=verdict, metrics=metrics, evidence=evidence)

    async def grade_sufficiency(self, query: str, selected_docs: list[str]) -> Decision[bool]:
        joined = "\n\n".join(selected_docs)
        prompt = f"{prompts.SUFFICIENCY_INSTRUCTION.format(query=query)}\n\nPassages:\n{joined}"
        result, duration = await self._invoke(self._sufficiency_chain, prompt)
        parsed = result["parsed"]
        assert isinstance(parsed, _SufficiencyGrade)
        metrics = self._metrics(duration, result["raw"])
        return Decision(
            value=parsed.sufficient,
            metrics=metrics,
            evidence={"sufficient": float(parsed.sufficient)},
        )

    async def decide_retry(
        self, query: str, attempt: int, sufficient: bool
    ) -> Decision[RetryChoice]:
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in prompts.RETRY_CRITERIA.items())
        prompt = (
            f"{prompts.RETRY_INSTRUCTION}\n{criteria_text}\n\n"
            f"question: {query!r}\nattempt so far: {attempt}\ncurrently sufficient: {sufficient}"
        )
        result, duration = await self._invoke(self._retry_chain, prompt)
        parsed = result["parsed"]
        assert isinstance(parsed, _RetryGrade)
        metrics = self._metrics(duration, result["raw"])
        return Decision(value=parsed.choice, metrics=metrics, evidence={})

    async def aclose(self) -> None:
        pass
