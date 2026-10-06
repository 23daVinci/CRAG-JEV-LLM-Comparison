from __future__ import annotations

from pathlib import Path

import httpx2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import SecretStr
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_bench.config import Settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.cassette import CassetteDecider
from jev_bench.deciders.gemini import GeminiDecider
from jev_bench.deciders.groq import GroqDecider
from jev_bench.deciders.jev import JevDecider, JevThresholds
from jev_bench.deciders.ollama import OllamaDecider
from jev_bench.retrieval.store import OllamaEmbeddingRetriever
from jev_bench.telemetry.pricing import PriceBook


def build_gemini_decider(settings: Settings, price_book: PriceBook | None = None) -> GeminiDecider:
    if not settings.gemini.api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    model = ChatGoogleGenerativeAI(
        model=settings.gemini.grader_model,
        google_api_key=settings.gemini.api_key,
        temperature=0,
        thinking_budget=0,
        # No default timeout is set by the SDK — confirmed by a real run hanging for 40+ minutes
        # with zero progress and no exception. An explicit timeout turns a silent hang into a
        # retryable httpx.TimeoutException (a subtype of RequestError, already caught in _invoke).
        timeout=60.0,
        # ChatGoogleGenerativeAI's own default max_retries is 6, stacking independently on top of
        # GeminiDecider._invoke's retry loop — each of *our* retries could trigger up to 6 more at
        # this layer before ever returning control to our code (up to 36 total HTTP attempts × 60s
        # timeout for one decision call). That compounding is what actually produced both hangs,
        # not the timeout alone. Disabling it here leaves retry entirely to _invoke, which has
        # proper visibility (the shared concurrency semaphore, backoff, and both rate-limit and
        # connection-error handling) that this layer doesn't.
        max_retries=0,
    )
    return GeminiDecider(model, price_book or PriceBook.load())


def build_ollama_decider(settings: Settings, price_book: PriceBook | None = None) -> OllamaDecider:
    model = ChatOllama(
        model=settings.ollama.model,
        base_url=settings.ollama.base_url,
        temperature=0,
    )
    return OllamaDecider(model, price_book or PriceBook.load())


def build_groq_decider(settings: Settings, price_book: PriceBook | None = None) -> GroqDecider:
    if not settings.groq.api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = ChatGroq(
        model=settings.groq.model,
        api_key=SecretStr(settings.groq.api_key),
        temperature=0,
        timeout=60.0,
        # See GeminiDecider's identical rationale: avoid two independent retry layers stacking
        # multiplicatively. Retry is handled entirely in GroqDecider._invoke instead.
        max_retries=0,
        # This account's output-tokens-per-minute budget for this "Preview" model is 1000 total —
        # the field's own *default* max_tokens (2048) alone exceeds that, so every request was
        # rejected outright ("Request too large") before even checking actual usage. Our structured
        # JSON answers need well under 100 output tokens in practice; 200 leaves headroom without
        # over-declaring against an already-tight budget.
        max_tokens=200,
    )
    return GroqDecider(model, price_book or PriceBook.load())


def build_jev_decider(settings: Settings, price_book: PriceBook | None = None) -> JevDecider:
    if not settings.jev.api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    client = AsyncTypeSafeClient(
        api_key=settings.jev.api_key,
        model=settings.jev.model,
        # graph/nodes.py's screen node fans out up to GraphConfig.batch_size (5) concurrent
        # screen_passage calls per retrieve-loop iteration (asyncio.gather) — exactly the
        # many-concurrent-requests shape TypeSafe's own SDK docs recommend HTTP/2 for, since it
        # multiplexes requests over one connection instead of queuing them behind HTTP/1.1's
        # per-host connection limit. Requires the `typesafe-sdk[http2]` extra for the `h2` package.
        http_client=httpx2.AsyncClient(http2=True),
        # Explicit rather than relying on the client's implicit default, so a future typesafe-sdk
        # version bump can't silently change retry behavior without it showing up as a diff here —
        # same reproducibility stance as this project's dated price book and frozen eval splits.
        retry=RetryPolicy(),
    )
    return JevDecider(
        client, JevThresholds.load(), price_book or PriceBook.load(), settings.jev.model
    )


def build_ollama_embedding_retriever(settings: Settings) -> OllamaEmbeddingRetriever:
    embeddings = OllamaEmbeddings(
        model=settings.ollama.embedding_model,
        base_url=settings.ollama.base_url,
    )
    return OllamaEmbeddingRetriever(embeddings)


def with_cassette(decider: Decider, cassette_dir: Path, mode: str, model: str = "") -> Decider:
    if mode == "record":
        return CassetteDecider(decider, cassette_dir / f"{decider.name}.jsonl", "record", model)
    if mode == "replay":
        return CassetteDecider(decider, cassette_dir / f"{decider.name}.jsonl", "replay", model)
    return decider
