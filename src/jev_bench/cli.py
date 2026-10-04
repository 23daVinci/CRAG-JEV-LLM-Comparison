from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from jev_bench.bench.injection_eval import evaluate_injection_guard
from jev_bench.bench.report import build_report
from jev_bench.bench.runner import ComparisonResult, QueryFailure, load_dataset, run_comparison
from jev_bench.config import load_settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.factory import build_groq_decider, build_jev_decider, with_cassette
from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.graph.builder import GraphConfig

app = typer.Typer(
    help="A/B benchmark of an agentic document-retrieval workflow: Groq (qwen3.8-27b) vs Jev."
)

DEFAULT_CASSETTE_DIR = Path(__file__).parent.parent.parent / "eval" / "cassettes"


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Run the live demo UI (FastAPI + SSE). Requires GOOGLE_API_KEY and TYPESAFE_API_KEY — this
    is the concurrent, visually-driven demo mode, not the benchmark-grade sequential measurement
    (`jev-bench bench` is the latter)."""

    import uvicorn

    uvicorn.run("jev_bench.api.app:app", host=host, port=port)


@app.command()
def bench(
    limit: int | None = typer.Option(
        None, help="Only run the first N queries from eval/dataset.jsonl."
    ),
    dry_run: bool = typer.Option(
        False, help="Use two ScriptedDeciders instead of live Groq/Jev — no API keys needed."
    ),
    cassette_mode: str = typer.Option(
        "live", help="live | record | replay. record/replay wrap the Jev side only."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Write the markdown report to this path as well as stdout."
    ),
) -> None:
    """Run both variants over the frozen eval set and print a benchmark report."""

    records = load_dataset()
    if limit:
        records = records[:limit]

    decider_a: Decider
    decider_b: Decider
    if dry_run:
        decider_a = ScriptedDecider(name="groq")
        decider_b = ScriptedDecider(name="jev")
    else:
        settings = load_settings()
        decider_a = build_groq_decider(settings)
        jev_decider = build_jev_decider(settings)
        decider_b = with_cassette(
            jev_decider, DEFAULT_CASSETTE_DIR, cassette_mode, model=settings.jev.model
        )

    cfg = GraphConfig()

    async def _run() -> tuple[list[ComparisonResult], list[QueryFailure]]:
        try:
            return await run_comparison(decider_a, decider_b, records=records, cfg=cfg)
        finally:
            await decider_a.aclose()
            await decider_b.aclose()

    results, failures = asyncio.run(_run())
    if failures:
        typer.echo(f"WARNING: {len(failures)}/{len(records)} queries failed and were skipped:")
        for failure in failures:
            typer.echo(f"  {failure.query_id}: {failure.error}")
    if not results:
        typer.echo("No queries succeeded — nothing to report.")
        raise typer.Exit(code=1)
    report = build_report(results)
    typer.echo(report)
    if out:
        out.write_text(report)


@app.command("injection-eval")
def injection_eval(
    dry_run: bool = typer.Option(
        False, help="Use ScriptedDeciders instead of live Groq/Jev — no API keys needed."
    ),
) -> None:
    """Score each backend's injection-screening guard against eval/injection_probes.jsonl — ground
    truth here is exact by construction (we inserted the injection strings ourselves), not human-
    or LLM-judged."""

    decider_a: Decider
    decider_b: Decider
    if dry_run:
        decider_a = ScriptedDecider(name="groq")
        decider_b = ScriptedDecider(name="jev")
    else:
        settings = load_settings()
        decider_a = build_groq_decider(settings)
        decider_b = build_jev_decider(settings)

    async def _run() -> None:
        try:
            for name, decider in (("groq", decider_a), ("jev", decider_b)):
                result = await evaluate_injection_guard(decider)
                typer.echo(
                    f"{name}: accuracy={result.accuracy:.2f} precision={result.precision:.2f} "
                    f"recall={result.recall:.2f}"
                )
                if result.false_positives:
                    typer.echo(f"  false positives: {result.false_positives}")
                if result.false_negatives:
                    typer.echo(f"  false negatives: {result.false_negatives}")
        finally:
            await decider_a.aclose()
            await decider_b.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
