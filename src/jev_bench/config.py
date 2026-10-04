from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI__")

    api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    grader_model: str = "gemini-3.1-flash-lite"


class JevSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JEV__")

    api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    model: str = "jev-latest"


class BenchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BENCH__")

    mode: str = "demo"
    repetitions: int = 5
    batch_size: int = 5
    max_attempts: int = 2


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    jev: JevSettings = Field(default_factory=JevSettings)
    bench: BenchSettings = Field(default_factory=BenchSettings)


def load_settings() -> Settings:
    return Settings()
