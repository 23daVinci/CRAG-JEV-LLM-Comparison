"""Every LLM-backed decider must build its questions from jev_bench.graph.prompts, not inline its
own wording — otherwise a quality or latency gap between backends could be explained by one of them
being asked an easier or differently-phrased question rather than by the backend itself.
"""

from pathlib import Path

import pytest

REQUIRED_CONSTANTS = [
    "RELEVANCE_INSTRUCTION",
    "EVIDENCE_INSTRUCTION",
    "CONTRADICTION_INSTRUCTION",
    "INJECTION_INSTRUCTION",
    "SUFFICIENCY_INSTRUCTION",
    "RETRY_INSTRUCTION",
    "RETRY_CRITERIA",
]

SRC = Path(__file__).parent.parent.parent / "src" / "jev_bench" / "deciders"

# gemini.py and ollama.py are kept in the codebase (tested, working) but are no longer the default
# variant A — see docs/METHODOLOGY.md on the Gemini quota exhaustion and Ollama's CPU latency. The
# invariant still applies to them in case either is ever re-enabled.
LLM_BACKED_DECIDERS = ["gemini.py", "jev.py", "ollama.py", "groq.py"]


@pytest.mark.parametrize("filename", LLM_BACKED_DECIDERS)
def test_decider_uses_shared_prompts(filename: str) -> None:
    source = (SRC / filename).read_text(encoding="utf-8")
    assert "from jev_bench.graph import prompts" in source
    for name in REQUIRED_CONSTANTS:
        assert f"prompts.{name}" in source, f"{filename} does not reference prompts.{name}"
