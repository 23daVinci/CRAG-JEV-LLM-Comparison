"""MLflow experiment tracking for `jev-bench bench`.

Layout of one tracked comparison (all runs live in one MLflow experiment):

- a **parent run** holding the experiment's metadata (split, thresholds, models, retriever, git
  commit, ...), the paired A-vs-B statistics, the failures, and the markdown report;
- one **child run per variant** holding that variant's per-query metrics (one MLflow `step` per
  query: f1, precision, recall, latency_ms, cost_usd, attempts), a `per_query.json` artifact that
  maps each step back to its `query_id`, and run-level aggregates (mean F1 with bootstrap CI, ...).

Per-query data is written as each query finishes (see `run_comparison`'s `on_result`/`on_failure`),
not at the end, so a crash or connection error partway through never loses the completed queries.

Imported lazily by the CLI: MLflow is a heavy import and is only needed when `--track` is passed.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import RunStatus

from jev_bench.bench.analysis import (
    bootstrap_mean_ci,
    paired_f1_comparison,
    paired_latency_comparison,
    score_selection,
)
from jev_bench.bench.runner import ComparisonResult, QueryFailure, QueryResult

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return str(completed.stdout).strip()


@dataclass(slots=True)
class _QueryRow:
    step: int
    query_id: str
    f1: float
    precision: float
    recall: float
    latency_ms: float
    cost_usd: float
    attempts: int
    selected_titles: list[str]
    gold_titles: list[str]


def _row(step: int, query_id: str, side: QueryResult) -> _QueryRow:
    metrics = score_selection(set(side.selected_titles), set(side.gold_titles))
    return _QueryRow(
        step=step,
        query_id=query_id,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        latency_ms=side.latency_ms,
        cost_usd=side.cost_usd,
        attempts=side.attempts,
        selected_titles=sorted(side.selected_titles),
        gold_titles=sorted(side.gold_titles),
    )


class BenchTracker:
    def __init__(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, Any],
        variant_names: tuple[str, str],
        tracking_uri: str | None = None,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
        self._client = MlflowClient()
        found = self._client.get_experiment_by_name(experiment)
        experiment_id = found.experiment_id if found else self._client.create_experiment(experiment)

        self._parent_id: str = str(
            self._client.create_run(experiment_id, run_name=run_name).info.run_id
        )
        for key, value in params.items():
            self._client.log_param(self._parent_id, key, str(value))

        self._variant_names = variant_names
        self._child_ids = tuple(
            self._client.create_run(
                experiment_id,
                run_name=f"{run_name}/{name}",
                tags={"mlflow.parentRunId": self._parent_id, "variant": name},
            ).info.run_id
            for name in variant_names
        )
        self._rows: tuple[list[_QueryRow], list[_QueryRow]] = ([], [])
        self._failures: list[dict[str, str]] = []

    @property
    def parent_run_id(self) -> str:
        return self._parent_id

    def record(self, result: ComparisonResult) -> None:
        step = len(self._rows[0])
        for run_id, rows, side in zip(
            self._child_ids, self._rows, (result.a, result.b), strict=True
        ):
            row = _row(step, result.query_id, side)
            rows.append(row)
            for key in ("f1", "precision", "recall", "latency_ms", "cost_usd", "attempts"):
                self._client.log_metric(run_id, key, float(getattr(row, key)), step=step)
            self._client.log_dict(run_id, {"queries": [asdict(r) for r in rows]}, "per_query.json")

    def record_failure(self, failure: QueryFailure) -> None:
        self._failures.append({"query_id": failure.query_id, "error": failure.error})
        self._client.log_dict(self._parent_id, {"failures": self._failures}, "failures.json")
        self._client.log_metric(self._parent_id, "n_failed", float(len(self._failures)))

    def finish(self, report: str | None = None, ok: bool = True) -> None:
        """Write aggregates and close every run. `ok=False` marks the runs FAILED (e.g. the process
        is dying on an exception) while keeping everything already logged."""

        status = RunStatus.to_string(RunStatus.FINISHED if ok else RunStatus.FAILED)
        rows_a, rows_b = self._rows
        for run_id, rows in zip(self._child_ids, self._rows, strict=True):
            self._log_aggregates(run_id, rows)
        self._client.log_metric(self._parent_id, "n_completed", float(len(rows_a)))
        self._client.log_metric(self._parent_id, "n_failed", float(len(self._failures)))
        if rows_a:
            f1_cmp = paired_f1_comparison([r.f1 for r in rows_a], [r.f1 for r in rows_b])
            lat_cmp = paired_latency_comparison(
                [r.latency_ms for r in rows_a], [r.latency_ms for r in rows_b]
            )
            for key, value in {
                "paired_f1_mean_delta_a_minus_b": f1_cmp.mean_delta,
                "paired_f1_ci95_low": f1_cmp.ci95_low,
                "paired_f1_ci95_high": f1_cmp.ci95_high,
                "paired_f1_wilcoxon_p": f1_cmp.wilcoxon_pvalue,
                "paired_latency_median_delta_ms": lat_cmp.median_delta_ms,
                "paired_latency_wilcoxon_p": lat_cmp.wilcoxon_pvalue,
            }.items():
                self._client.log_metric(self._parent_id, key, float(value))
        if report is not None:
            self._client.log_text(self._parent_id, report, "report.md")
        for run_id in (*self._child_ids, self._parent_id):
            self._client.set_terminated(run_id, status)

    def _log_aggregates(self, run_id: str, rows: list[_QueryRow]) -> None:
        if not rows:
            return
        f1s = [r.f1 for r in rows]
        ci_low, ci_high = bootstrap_mean_ci(f1s)
        for key, value in {
            "n_queries": float(len(rows)),
            "mean_f1": float(np.mean(f1s)),
            "mean_f1_ci95_low": ci_low,
            "mean_f1_ci95_high": ci_high,
            "mean_precision": float(np.mean([r.precision for r in rows])),
            "mean_recall": float(np.mean([r.recall for r in rows])),
            "median_latency_ms": float(np.median([r.latency_ms for r in rows])),
            "total_cost_usd": float(sum(r.cost_usd for r in rows)),
        }.items():
            self._client.log_metric(run_id, key, value)
