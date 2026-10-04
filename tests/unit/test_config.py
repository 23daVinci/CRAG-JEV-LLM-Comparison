"""Regression test for a real bug: GeminiSettings/JevSettings are nested BaseSettings built via
`default_factory`, which does NOT inherit the outer Settings' `env_file` — each nested model reads
only real process env vars unless it loads the dotenv itself. Without each nested class's own
`env_file=".env"`, keys placed in a .env file were silently invisible to the app.
"""

from pathlib import Path

import pytest

from jev_bench.config import GeminiSettings, JevSettings


@pytest.fixture
def fake_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # pydantic-settings' source precedence is env vars > .env file — CI deliberately exports
    # GOOGLE_API_KEY="" / TYPESAFE_API_KEY="" as a tripwire (see ci.yml), which otherwise outranks
    # the fake value below and makes this fixture environment-dependent: it passed locally (those
    # vars were simply unset in that shell) but failed in CI for exactly that reason.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "GOOGLE_API_KEY=unit-test-fake-gemini-key\nTYPESAFE_API_KEY=unit-test-fake-jev-key\n"
    )
    monkeypatch.chdir(tmp_path)
    return dotenv


def test_gemini_settings_reads_key_from_dotenv_file(fake_dotenv: Path) -> None:
    settings = GeminiSettings()
    assert settings.api_key == "unit-test-fake-gemini-key"


def test_jev_settings_reads_key_from_dotenv_file(fake_dotenv: Path) -> None:
    settings = JevSettings()
    assert settings.api_key == "unit-test-fake-jev-key"


def test_gemini_settings_defaults_to_none_without_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = GeminiSettings()
    assert settings.api_key is None
