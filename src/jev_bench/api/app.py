from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from jev_bench.api.schemas import node_span_to_dict, summarize_node_update
from jev_bench.bench.analysis import score_selection
from jev_bench.bench.runner import Record, load_dataset
from jev_bench.config import load_settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.factory import (
    build_gemini_decider,
    build_jev_decider,
    build_ollama_embedding_retriever,
)
from jev_bench.graph.builder import build_graph
from jev_bench.graph.state import initial_state
from jev_bench.retrieval.store import Retriever
from jev_bench.telemetry.emitter import RunEmitter
from jev_bench.telemetry.pricing import PriceBook

WEB_DIR = Path(__file__).parent.parent.parent.parent / "web"

app = FastAPI(title="jev-bench demo")

_RECORDS: dict[str, Record] = {r["id"]: r for r in load_dataset()}
_RUNS: dict[str, RunEmitter] = {}


class CreateRunRequest(BaseModel):
    query_id: str


class CreateRunResponse(BaseModel):
    run_id: str


@app.get("/api/queries")
def list_queries() -> list[dict[str, str]]:
    return [
        {"id": r["id"], "question": r["question"], "type": r["type"]} for r in _RECORDS.values()
    ]


@app.post("/api/runs")
async def create_run(body: CreateRunRequest) -> CreateRunResponse:
    record = _RECORDS.get(body.query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown query_id {body.query_id!r}")

    run_id = uuid.uuid4().hex
    emitter = RunEmitter(run_id)
    _RUNS[run_id] = emitter
    asyncio.create_task(_execute_run(run_id, record, emitter))
    return CreateRunResponse(run_id=run_id)


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    emitter = _RUNS.get(run_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id {run_id!r}")

    async def wire() -> AsyncIterator[dict[str, str]]:
        async for event in emitter.stream():
            yield {
                "event": event.type,
                "data": json.dumps(
                    {
                        "run_id": event.run_id,
                        "variant": event.variant,
                        "seq": event.seq,
                        **event.payload,
                    }
                ),
            }
        del _RUNS[run_id]

    return EventSourceResponse(wire())


async def _pump_variant(
    decider: Decider, retriever: Retriever, record: Record, variant: str, emitter: RunEmitter
) -> None:
    graph = build_graph(decider, retriever)
    state = initial_state(record["question"], record["context"])
    final_state: dict[str, Any] = dict(state)
    async for mode, chunk in graph.astream(state, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = cast(dict[str, Any], chunk)
            continue
        for node_name, update in cast(dict[str, Any], chunk).items():
            span = update["trace"][0]
            emitter.emit(
                variant,
                "node_finished",
                {**node_span_to_dict(span), "summary": summarize_node_update(node_name, update)},
            )

    selected = set(final_state["selected_ids"])
    gold = set(record["gold_titles"])
    metrics = score_selection(selected, gold)
    emitter.emit(
        variant,
        "variant_finished",
        {
            "selected_titles": sorted(selected),
            "gold_titles": sorted(gold),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "attempts": final_state["attempt"],
        },
    )


async def _execute_run(run_id: str, record: Record, emitter: RunEmitter) -> None:
    settings = load_settings()
    price_book = PriceBook.load()
    gemini_decider = build_gemini_decider(settings, price_book)
    jev_decider = build_jev_decider(settings, price_book)
    # One shared instance for both variants — retrieval must stay a fixed input to the comparison,
    # never something either decider can independently influence.
    retriever = build_ollama_embedding_retriever(settings)

    emitter.emit("gemini", "run_started", {"question": record["question"]})
    emitter.emit("jev", "run_started", {"question": record["question"]})

    try:
        await asyncio.gather(
            _pump_variant(gemini_decider, retriever, record, "gemini", emitter),
            _pump_variant(jev_decider, retriever, record, "jev", emitter),
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
        emitter.emit("gemini", "decision_error", {"message": str(exc)})
        emitter.emit("jev", "decision_error", {"message": str(exc)})
    finally:
        await gemini_decider.aclose()
        await jev_decider.aclose()
        emitter.emit("gemini", "run_finished", {})
        emitter.close()


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
