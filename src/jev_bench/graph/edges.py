from __future__ import annotations

from jev_bench.deciders.base import RetryChoice
from jev_bench.graph.state import RetrievalState


def route_after_retry(state: RetrievalState) -> RetryChoice:
    """Pure routing, no decider call: `decide_retry_node` already computed and stored the choice
    (including the hard max-attempts cap). Keeping decisions in nodes and edges as plain state
    lookups is what lets every decision produce exactly one telemetry span through the node
    return value — an edge function has no return channel for the state reducer to merge."""

    return state["retry_choice"]
