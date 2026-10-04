from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jev_bench.deciders.base import Decider
from jev_bench.graph import nodes as N
from jev_bench.graph.edges import route_after_retry
from jev_bench.graph.state import RetrievalState
from jev_bench.retrieval.store import Retriever


@dataclass(frozen=True, slots=True)
class GraphConfig:
    batch_size: int = 5
    max_attempts: int = 2


def build_graph(
    decider: Decider,
    retriever: Retriever | None = None,
    cfg: GraphConfig | None = None,
) -> CompiledStateGraph[RetrievalState, None, RetrievalState, RetrievalState]:
    """The one graph builder. Every variant — Gemini-backed, Jev-backed, or a test's
    ScriptedDecider — goes through this exact function; only `decider` changes. There is no
    provider branching anywhere below this line (enforced by tests/test_topology.py and an AST
    scan in tests/unit), so the two benchmark variants are provably the same graph."""

    retriever = retriever or Retriever()
    cfg = cfg or GraphConfig()

    graph: StateGraph[RetrievalState, None, RetrievalState, RetrievalState] = StateGraph(
        RetrievalState
    )
    # mypy cannot resolve StateGraph.add_node's overloads against a plain async Callable here
    # (a generic-stub limitation, not a real type error — the graph runs and the topology tests
    # in tests/test_topology.py pass against the compiled result).
    graph.add_node("retrieve", N.make_retrieve_node(retriever, cfg))  # type: ignore[call-overload]
    graph.add_node("screen", N.make_screen_node(decider))  # type: ignore[call-overload]
    graph.add_node("sufficiency", N.make_sufficiency_node(decider))  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "decide_retry", N.make_decide_retry_node(decider, cfg)
    )

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "screen")
    graph.add_edge("screen", "sufficiency")
    graph.add_edge("sufficiency", "decide_retry")
    graph.add_conditional_edges(
        "decide_retry", route_after_retry, {"retry": "retrieve", "return": END}
    )

    return graph.compile()
