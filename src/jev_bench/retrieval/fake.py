from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ScriptedRetriever:
    """A network-free `Retriever` for `--dry-run` and the test suite. `OllamaEmbeddingRetriever`
    needs a running local Ollama server with an embedding model pulled — fine for a live benchmark
    run, wrong for graph-wiring/decision-logic tests and a zero-setup dry run, neither of which
    care about ranking quality. Default behavior is deterministic insertion order.
    """

    name: str = "fake"
    rank_fn: Callable[[str, dict[str, str]], list[str]] = field(
        default=lambda query, doc_pool: list(doc_pool)
    )

    async def rank(self, query: str, doc_pool: dict[str, str]) -> list[str]:
        return self.rank_fn(query, doc_pool)
