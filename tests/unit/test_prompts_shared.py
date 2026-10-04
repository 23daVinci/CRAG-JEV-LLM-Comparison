"""Both deciders must build their questions from jev_bench.graph.prompts, not inline their own
wording — otherwise a quality or latency gap between backends could be explained by one of them
being asked an easier or differently-phrased question rather than by the backend itself.
"""

from pathlib import Path

REQUIRED_CONSTANTS = [
    "RELEVANCE_INSTRUCTION",
    "EVIDENCE_INSTRUCTION",
    "CONTRADICTION_INSTRUCTION",
    "INJECTION_INSTRUCTION",
    "SUFFICIENCY_INSTRUCTION",
    "RETRY_INSTRUCTION",
    "RETRY_CRITERIA",
]

SRC = Path(__file__).parent.parent.parent / "src" / "jev_bench"


def test_gemini_decider_uses_shared_prompts() -> None:
    source = (SRC / "deciders" / "gemini.py").read_text()
    assert "from jev_bench.graph import prompts" in source
    for name in REQUIRED_CONSTANTS:
        assert f"prompts.{name}" in source, f"gemini.py does not reference prompts.{name}"


def test_jev_decider_uses_shared_prompts() -> None:
    source = (SRC / "deciders" / "jev.py").read_text()
    assert "from jev_bench.graph import prompts" in source
    for name in REQUIRED_CONSTANTS:
        assert f"prompts.{name}" in source, f"jev.py does not reference prompts.{name}"
