from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from jev_bench.bench.report import build_report
from jev_bench.bench.runner import load_dataset, run_comparison
from jev_bench.config import load_settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.factory import build_gemini_decider, build_jev_decider, with_cassette
from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.graph.builder import GraphConfig

app = typer.Typer(help="A/B benchmark of an agentic document-retrieval workflow: Gemini vs Jev.")

DEFAULT_CASSETTE_DIR = Path(__file__).parent.parent.parent / "eval" / "cassettes"


@app.command()
def serve() -> None:
    """Run the live demo UI (FastAPI + SSE). Not yet implemented."""

    typer.echo("jev-bench serve: not implemented yet.")
    raise typer.Exit(code=1)


@app.command()
def bench(
    limit: int | None = typer.Option(
        None, help="Only run the first N queries from eval/dataset.jsonl."
    ),
    dry_run: bool = typer.Option(
        False, help="Use two ScriptedDeciders instead of live Gemini/Jev — no API keys needed."
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
        decider_a = ScriptedDecider(name="gemini")
        decider_b = ScriptedDecider(name="jev")
    else:
        settings = load_settings()
        decider_a = build_gemini_decider(settings)
        jev_decider = build_jev_decider(settings)
        decider_b = with_cassette(
            jev_decider, DEFAULT_CASSETTE_DIR, cassette_mode, model=settings.jev.model
        )

    cfg = GraphConfig()
    results = asyncio.run(run_comparison(decider_a, decider_b, records=records, cfg=cfg))
    report = build_report(results)
    typer.echo(report)
    if out:
        out.write_text(report)


if __name__ == "__main__":
    app()
