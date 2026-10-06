from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    # No env_prefix: `api_key`'s alias is the bare, unprefixed env var name. A nested BaseSettings
    # instantiated via `default_factory` does not inherit the outer Settings' env_file — it reads
    # only real process env vars unless told to load the dotenv itself.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    grader_model: str = "gemini-3.1-flash-lite"


class OllamaSettings(BaseSettings):
    """Local inference — no API key, no quota, no rate limit, no per-request cost. Briefly replaced
    Gemini as the default variant A after Gemini's free-tier 500 requests/day quota was exhausted
    mid-project (plan Risk 5), then itself replaced by Groq after real HotpotQA-length passages
    measured ~7-28s/call on this CPU-only machine (no GPU) — a full run would have taken ~1.5-2h.
    Kept in the codebase (tested, working) as a zero-cost, zero-quota fallback option."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OLLAMA__", extra="ignore")

    model: str = "llama3.2:3b"
    base_url: str | None = None
    embedding_model: str = "nomic-embed-text"
    """Used by `OllamaEmbeddingRetriever` for semantic document ranking — a separate, much smaller
    purpose-built embedding model (~274MB) rather than reusing `model` (a full chat model), which
    would be both slower per call and a weaker embedder. Requires `ollama pull nomic-embed-text`
    once; not pulled automatically."""


class GroqSettings(BaseSettings):
    """Default variant A backend: Groq's LPU-hosted inference is fast enough (sub-second per call
    even on full-length real passages, confirmed empirically) to avoid both Gemini's free-tier
    quota wall and Ollama's CPU-bound latency. `qwen/qwen3.8-27b` chosen over the catalog's
    `openai/gpt-oss-*` models specifically because it reports zero hidden reasoning tokens — the
    gpt-oss family emits `output_token_details.reasoning` even at `reasoning_effort="low"` (the
    minimum this API allows; "none"/"minimal" are rejected), which would reintroduce the same
    thinking-token fairness problem `GeminiDecider` guards against. Note: a "Preview" catalog model
    per Groq, who state it may be discontinued at short notice."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    model: str = "qwen/qwen3.8-27b"


class JevSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    model: str = "jev-latest"


class BenchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BENCH__", extra="ignore")

    mode: str = "demo"
    repetitions: int = 5
    batch_size: int = 5
    max_attempts: int = 2


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    jev: JevSettings = Field(default_factory=JevSettings)
    bench: BenchSettings = Field(default_factory=BenchSettings)


def load_settings() -> Settings:
    return Settings()
