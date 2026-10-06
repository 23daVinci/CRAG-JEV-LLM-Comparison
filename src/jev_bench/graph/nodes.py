from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from jev_bench.deciders.base import Decider
from jev_bench.graph.state import RetrievalState
from jev_bench.retrieval.store import Retriever
from jev_bench.telemetry.instrument import NodeSpanRecorder

if TYPE_CHECKING:
    from jev_bench.graph.builder import GraphConfig

Node = Callable[[RetrievalState], Awaitable[dict[str, Any]]]


def make_retrieve_node(retriever: Retriever, cfg: GraphConfig) -> Node:
    async def retrieve(state: RetrievalState) -> dict[str, Any]:
        span = NodeSpanRecorder("retrieve", "shared")
        ranked_ids = state["ranked_ids"] or await retriever.rank(state["query"], state["doc_pool"])
        screened = set(state["screened_ids"])
        remaining = [doc_id for doc_id in ranked_ids if doc_id not in screened]
        batch = remaining[: cfg.batch_size]
        return {
            "ranked_ids": ranked_ids,
            "candidate_ids": batch,
            "trace": [span.finish()],
        }

    return retrieve


def make_screen_node(decider: Decider) -> Node:
    async def screen(state: RetrievalState) -> dict[str, Any]:
        span = NodeSpanRecorder("screen_passage", decider.name)
        candidates = state["candidate_ids"]
        decisions = await asyncio.gather(
            *(
                decider.screen_passage(state["query"], state["doc_pool"][doc_id])
                for doc_id in candidates
            )
        )
        verdicts = dict(state["passage_verdicts"])
        evidence = dict(state["passage_evidence"])
        selected = list(state["selected_ids"])
        for doc_id, decision in zip(candidates, decisions, strict=True):
            span.record(decision)
            verdicts[doc_id] = decision.value
            evidence[doc_id] = decision.evidence
            if decision.value.relevant and not decision.value.is_injection:
                selected.append(doc_id)
        return {
            "passage_verdicts": verdicts,
            "passage_evidence": evidence,
            "selected_ids": selected,
            "screened_ids": [*state["screened_ids"], *candidates],
            "trace": [span.finish()],
        }

    return screen


def make_sufficiency_node(decider: Decider) -> Node:
    async def sufficiency(state: RetrievalState) -> dict[str, Any]:
        span = NodeSpanRecorder("grade_sufficiency", decider.name)
        docs = [state["doc_pool"][doc_id] for doc_id in state["selected_ids"]]
        decision = await decider.grade_sufficiency(state["query"], docs)
        span.record(decision)
        return {"sufficient": decision.value, "trace": [span.finish()]}

    return sufficiency


def make_decide_retry_node(decider: Decider, cfg: GraphConfig) -> Node:
    async def decide_retry(state: RetrievalState) -> dict[str, Any]:
        span = NodeSpanRecorder("decide_retry", decider.name)
        attempt = state["attempt"]
        decision = await decider.decide_retry(state["query"], attempt, state["sufficient"])
        span.record(decision)
        choice = decision.value
        pool_exhausted = len(state["screened_ids"]) >= len(state["doc_pool"])
        if attempt + 1 >= cfg.max_attempts or pool_exhausted:
            choice = "return"
        return {"attempt": attempt + 1, "retry_choice": choice, "trace": [span.finish()]}

    return decide_retry
