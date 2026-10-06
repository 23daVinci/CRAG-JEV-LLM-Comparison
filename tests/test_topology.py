"""Mechanical enforcement of the plan's one architectural rule: every backend compiles through
`build_graph` to the identical node/edge topology, and the graph code contains no branching on
which backend it was given. If someone special-cases a provider inside graph code, one of these
two tests catches it — this is not meant to be satisfied by discipline alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.graph.builder import build_graph
from jev_bench.retrieval.fake import ScriptedRetriever

GRAPH_SRC_FILES = [
    Path(__file__).parent.parent / "src" / "jev_bench" / "graph" / "builder.py",
    Path(__file__).parent.parent / "src" / "jev_bench" / "graph" / "nodes.py",
    Path(__file__).parent.parent / "src" / "jev_bench" / "graph" / "edges.py",
]


def _topology(decider_name: str) -> tuple[frozenset[str], frozenset[tuple[str, str, object]]]:
    app = build_graph(ScriptedDecider(name=decider_name), ScriptedRetriever())
    g = app.get_graph()
    nodes = frozenset(g.nodes.keys())
    edges = frozenset((e.source, e.target, e.data) for e in g.edges)
    return nodes, edges


def test_both_backends_compile_to_the_same_topology() -> None:
    gemini_like = _topology("gemini")
    jev_like = _topology("jev")
    assert gemini_like == jev_like


def test_topology_matches_committed_snapshot() -> None:
    nodes, edges = _topology("fake")
    assert nodes == frozenset(
        {"__start__", "__end__", "retrieve", "screen", "sufficiency", "decide_retry"}
    )
    assert edges == frozenset(
        {
            ("__start__", "retrieve", None),
            ("retrieve", "screen", None),
            ("screen", "sufficiency", None),
            ("sufficiency", "decide_retry", None),
            ("decide_retry", "__end__", "return"),
            ("decide_retry", "retrieve", "retry"),
        }
    )


class _BranchOnBackendVisitor(ast.NodeVisitor):
    """Flags the specific pattern that would let graph code special-case a backend: comparing
    against `.name` (a Decider's only backend-identifying attribute) or `isinstance`-checking a
    decider. Does not flag `isinstance` in general, since node functions use it for ordinary type
    narrowing unrelated to backend identity.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        if isinstance(node.left, ast.Attribute) and node.left.attr == "name":
            self.violations.append(f"line {node.lineno}: comparison against a `.name` attribute")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            args_src = ast.dump(node)
            if "Decider" in args_src or "decider" in args_src:
                self.violations.append(f"line {node.lineno}: isinstance check on a decider")
        self.generic_visit(node)


def test_graph_code_has_no_provider_branching() -> None:
    violations: list[str] = []
    for path in GRAPH_SRC_FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        visitor = _BranchOnBackendVisitor()
        visitor.visit(tree)
        violations.extend(f"{path.name}:{v}" for v in visitor.violations)
    assert not violations, f"provider branching found in graph code: {violations}"
