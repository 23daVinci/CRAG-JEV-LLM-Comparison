from __future__ import annotations

from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from typesafe_sdk import AsyncTypeSafeClient

from jev_bench.config import Settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.cassette import CassetteDecider
from jev_bench.deciders.gemini import GeminiDecider
from jev_bench.deciders.jev import JevDecider, JevThresholds
from jev_bench.telemetry.pricing import PriceBook


def build_gemini_decider(settings: Settings, price_book: PriceBook | None = None) -> GeminiDecider:
    if not settings.gemini.api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    model = ChatGoogleGenerativeAI(
        model=settings.gemini.grader_model,
        google_api_key=settings.gemini.api_key,
        temperature=0,
        thinking_budget=0,
    )
    return GeminiDecider(model, price_book or PriceBook.load())


def build_jev_decider(settings: Settings, price_book: PriceBook | None = None) -> JevDecider:
    if not settings.jev.api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    client = AsyncTypeSafeClient(api_key=settings.jev.api_key, model=settings.jev.model)
    return JevDecider(
        client, JevThresholds.load(), price_book or PriceBook.load(), settings.jev.model
    )


def with_cassette(decider: Decider, cassette_dir: Path, mode: str, model: str = "") -> Decider:
    if mode == "record":
        return CassetteDecider(decider, cassette_dir / f"{decider.name}.jsonl", "record", model)
    if mode == "replay":
        return CassetteDecider(decider, cassette_dir / f"{decider.name}.jsonl", "replay", model)
    return decider
