from __future__ import annotations

import asyncio
import hashlib
import json
import math
from typing import Protocol

import httpx
from langchain_ollama import OllamaEmbeddings


def hash_doc_pool(doc_pool: dict[str, str]) -> str:
    """Canonical hash of a query's retrieval fixture. Used to assert both variants ran against the
    byte-identical document pool — retrieval is never supposed to be a variable under test, so this
    turns "we're pretty sure both sides saw the same docs" into something a test actually checks."""

    canonical = json.dumps(doc_pool, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Retriever(Protocol):
    """Ranks a query's fixed document pool, most-likely-relevant first. `build_graph` takes this
    as a required, explicit argument (no implicit default) for the same reason it requires an
    explicit `Decider`: both variants in a comparison must read the identical ranking, so the
    caller must say which ranking that is rather than one being silently assumed."""

    name: str

    async def rank(self, query: str, doc_pool: dict[str, str]) -> list[str]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class OllamaEmbeddingRetriever:
    """Ranks by cosine similarity between embeddings from a local Ollama model — semantic ranking
    instead of lexical overlap. Both variants in a `run_comparison` call share one instance of
    this class, so the ranking is still a fixed input to the comparison, not a variable: neither
    backend's `Decider` has any influence over it, and `bench/runner.py`'s frozen-fixture hash
    check would catch it if something made that stop being true.

    Requires a running local Ollama server with the embedding model pulled (e.g.
    `ollama pull nomic-embed-text`) — unlike the old TF-IDF ranker, this is not zero-setup, which
    is exactly why `--dry-run` and the test suite use `retrieval.fake.ScriptedRetriever` instead:
    they test graph wiring and decision logic, not ranking quality, and must not require a local
    model server to run.

    Query and document texts get Nomic's recommended task prefixes (`search_query:` /
    `search_document:`) rather than being embedded raw — `nomic-embed-text` is trained to expect
    them, and omitting them measurably hurts ranking quality for this model.
    """

    name = "ollama-embedding"

    def __init__(self, embeddings: OllamaEmbeddings, max_retries: int = 3) -> None:
        self._embeddings = embeddings
        self._max_retries = max_retries

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                return await self._embeddings.aembed_documents(texts)
            except httpx.RequestError:
                # No cloud quota/rate limit here — the only realistic transient failure is the
                # local Ollama server itself hiccuping (e.g. mid-restart), same reasoning as
                # OllamaDecider._invoke.
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise AssertionError("unreachable: loop always returns or raises")

    async def rank(self, query: str, doc_pool: dict[str, str]) -> list[str]:
        doc_ids = list(doc_pool)
        query_vectors, doc_vectors = await asyncio.gather(
            self._embed_with_retry([f"search_query: {query}"]),
            self._embed_with_retry([f"search_document: {doc_pool[doc_id]}" for doc_id in doc_ids]),
        )
        query_vector = query_vectors[0]
        scored = sorted(
            zip(doc_ids, doc_vectors, strict=True),
            key=lambda pair: (-_cosine(query_vector, pair[1]), pair[0]),
        )
        return [doc_id for doc_id, _ in scored]
