from __future__ import annotations

from dataclasses import asdict
from typing import Any

from jev_bench.telemetry.model import NodeSpan


def node_span_to_dict(span: NodeSpan) -> dict[str, Any]:
    return {
        "node": span.node,
        "backend": span.backend,
        "latency_ms": round(span.latency_ms, 1),
        "cost_usd": span.cost_usd,
        "calls": [asdict(c) for c in span.calls],
    }


def summarize_node_update(node_name: str, update: dict[str, Any]) -> str:
    if node_name == "retrieve":
        return f"screening {len(update.get('candidate_ids', []))} documents"
    if node_name == "screen":
        verdicts = update.get("passage_verdicts", {})
        relevant = sum(1 for v in verdicts.values() if v.relevant)
        return f"{relevant}/{len(verdicts)} relevant this batch"
    if node_name == "sufficiency":
        return f"sufficient: {update.get('sufficient')}"
    if node_name == "decide_retry":
        return f"attempt {update.get('attempt')}: {update.get('retry_choice')}"
    return node_name
