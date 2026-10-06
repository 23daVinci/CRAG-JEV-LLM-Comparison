from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from jev_bench.bench.analysis import threshold_point_f1
from jev_bench.bench.injection_eval import evaluate_injection_guard, load_probes
from jev_bench.bench.report import build_report
from jev_bench.bench.runner import ComparisonResult, QueryFailure, load_dataset, run_comparison
from jev_bench.bench.tuning import (
    TuningOutcome,
    apply_threshold_update,
    filter_probes_by_split,
    filter_records_by_split,
    load_splits,
    tune_injection_threshold,
    tune_relevance_threshold,
)
from jev_bench.config import load_settings
from jev_bench.deciders.base import Decider
from jev_bench.deciders.factory import (
    build_groq_decider,
    build_jev_decider,
    build_ollama_embedding_retriever,
    with_cassette,
)
from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.graph.builder import GraphConfig
from jev_bench.retrieval.fake import ScriptedRetriever
from jev_bench.retrieval.store import Retriever

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
    split: str | None = typer.Option(None, help="Restrict to train|val|test via eval/splits.json."),
) -> None:
    """Run both variants over the frozen eval set and print a benchmark report.

    Live (non-dry-run) ranking needs a local Ollama server with `nomic-embed-text` pulled
    (`ollama pull nomic-embed-text`) — document ranking is semantic (embedding cosine similarity),
    not TF-IDF. `--dry-run` needs none of this; it uses a deterministic scripted ranking instead.
    """

    records = load_dataset()
    if split:
        records = filter_records_by_split(records, load_splits(), split)
    if limit:
        records = records[:limit]

    decider_a: Decider
    decider_b: Decider
    retriever: Retriever
    if dry_run:
        decider_a = ScriptedDecider(name="groq")
        decider_b = ScriptedDecider(name="jev")
        retriever = ScriptedRetriever()
    else:
        settings = load_settings()
        decider_a = build_groq_decider(settings)
        jev_decider = build_jev_decider(settings)
        decider_b = with_cassette(
            jev_decider, DEFAULT_CASSETTE_DIR, cassette_mode, model=settings.jev.model
        )
        retriever = build_ollama_embedding_retriever(settings)

    cfg = GraphConfig()

    async def _run() -> tuple[list[ComparisonResult], list[QueryFailure]]:
        try:
            return await run_comparison(decider_a, decider_b, retriever, records=records, cfg=cfg)
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
    split: str | None = typer.Option(None, help="Restrict to train|val|test via eval/splits.json."),
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

    probes = load_probes()
    if split:
        probes = filter_probes_by_split(probes, load_splits(), split)

    async def _run() -> None:
        try:
            for name, decider in (("groq", decider_a), ("jev", decider_b)):
                result = await evaluate_injection_guard(decider, probes)
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


def _report_outcome(name: str, outcome: TuningOutcome) -> None:
    typer.echo(f"{name}: TRAIN sweep over the fixed candidate grid")
    for point in outcome.train_points:
        typer.echo(
            f"  t={point.threshold:.2f} precision={point.precision:.2f} "
            f"recall={point.recall:.2f} f1={threshold_point_f1(point):.2f}"
        )
    t = outcome.train_threshold
    typer.echo(
        f"{name}: TRAIN best threshold={t.threshold:.3f} precision={t.precision:.2f} "
        f"recall={t.recall:.2f} f1={outcome.train_f1:.2f}"
    )
    v = outcome.val_metrics
    typer.echo(
        f"{name}: VAL   @ threshold={t.threshold:.3f} precision={v.precision:.2f} "
        f"recall={v.recall:.2f} f1={v.f1:.2f}"
    )
    if outcome.val_f1_drop > 0.10:
        typer.echo(
            f"  WARNING: val F1 dropped {outcome.val_f1_drop:.2f} from train — threshold may be "
            "overfit to train, not confirmed out-of-sample"
        )


@app.command("tune-thresholds")
def tune_thresholds(
    target: str = typer.Option("both", help="relevance | injection | both."),
    apply: bool = typer.Option(
        False, "--apply", help="Write config/thresholds.yaml. Default is dry-run (print only)."
    ),
) -> None:
    """Tune Jev's relevance/injection thresholds: sweep TRAIN for the F1-maximizing threshold,
    confirm it on VAL without re-sweeping. Never touches TEST — the final held-out comparison is a
    separate `jev-bench bench --split test` / `jev-bench injection-eval --split test` run. Dry-run
    by default; pass --apply to write the picked threshold(s) into config/thresholds.yaml."""

    targets = ["relevance", "injection"] if target == "both" else [target]
    if any(t not in ("relevance", "injection") for t in targets):
        raise typer.BadParameter(f"target must be relevance|injection|both, got {target!r}")

    settings = load_settings()
    decider = build_jev_decider(settings)

    async def _run() -> None:
        try:
            for t in targets:
                if t == "relevance":
                    outcome = await tune_relevance_threshold(decider)
                else:
                    outcome = await tune_injection_threshold(decider)
                _report_outcome(t, outcome)
                if apply:
                    apply_threshold_update(t, outcome.train_threshold.threshold)
                    typer.echo(
                        f"  wrote {t}={outcome.train_threshold.threshold} to config/thresholds.yaml"
                    )
        finally:
            await decider.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
