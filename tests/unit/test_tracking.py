import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from jev_bench.bench.runner import QueryFailure, run_comparison
from jev_bench.bench.tracking import BenchTracker
from jev_bench.deciders.base import PassageVerdict
from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.retrieval.fake import ScriptedRetriever


def _record(query_id: str) -> dict:
    return {
        "id": query_id,
        "question": f"question {query_id}",
        "context": {"gold": "the gold passage", "other": "an unrelated passage"},
        "gold_titles": ["gold"],
    }


def _tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BenchTracker:
    monkeypatch.chdir(tmp_path)  # MLflow's default artifact root is ./mlruns
    return BenchTracker(
        experiment="test-exp",
        run_name="unit",
        params={"split": "test", "retriever": "fake"},
        variant_names=("a", "b"),
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )


def _load(client: MlflowClient, run_id: str, artifact: str, tmp_path: Path) -> dict:
    path = client.download_artifacts(run_id, artifact, str(tmp_path / "dl"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_each_query_is_logged_per_variant_with_its_query_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _tracker(tmp_path, monkeypatch)
    # B never marks anything relevant, so A and B get different per-query F1.
    decider_b = ScriptedDecider(
        name="b",
        screen_fn=lambda q, d: PassageVerdict(
            relevant=False, has_evidence=False, contradicts_query=False, is_injection=False
        ),
    )
    results, failures = await run_comparison(
        ScriptedDecider(name="a"),
        decider_b,
        ScriptedRetriever(),
        records=[_record("q1"), _record("q2"), _record("q3")],
        on_result=tracker.record,
        on_failure=tracker.record_failure,
    )
    assert len(results) == 3 and failures == []
    tracker.finish("# report")

    client = MlflowClient()
    children = client.search_runs(
        [client.get_experiment_by_name("test-exp").experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{tracker.parent_run_id}'",
    )
    assert {c.data.tags["variant"] for c in children} == {"a", "b"}
    for child in children:
        history = client.get_metric_history(child.info.run_id, "f1")
        assert [m.step for m in history] == [0, 1, 2]
        rows = _load(client, child.info.run_id, "per_query.json", tmp_path)["queries"]
        assert [r["query_id"] for r in rows] == ["q1", "q2", "q3"]
        assert child.data.metrics["n_queries"] == 3
    by_variant = {c.data.tags["variant"]: c for c in children}
    assert by_variant["a"].data.metrics["mean_f1"] > by_variant["b"].data.metrics["mean_f1"]


@pytest.mark.asyncio
async def test_parent_run_records_params_paired_stats_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _tracker(tmp_path, monkeypatch)
    results, _ = await run_comparison(
        ScriptedDecider(name="a"),
        ScriptedDecider(name="b"),
        ScriptedRetriever(),
        records=[_record("q1"), _record("q2")],
        on_result=tracker.record,
    )
    assert len(results) == 2
    tracker.finish("# the report")

    client = MlflowClient()
    parent = client.get_run(tracker.parent_run_id)
    assert parent.data.params["split"] == "test"
    assert parent.data.metrics["n_completed"] == 2
    assert parent.data.metrics["paired_f1_mean_delta_a_minus_b"] == 0.0
    assert parent.info.status == "FINISHED"
    report = client.download_artifacts(tracker.parent_run_id, "report.md", str(tmp_path / "dl"))
    assert Path(report).read_text(encoding="utf-8") == "# the report"


def test_failures_are_persisted_and_a_failed_finish_keeps_logged_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _tracker(tmp_path, monkeypatch)
    tracker.record_failure(
        QueryFailure(query_id="q9", error="APIConnectionError: Connection error.")
    )
    tracker.finish(ok=False)

    client = MlflowClient()
    parent = client.get_run(tracker.parent_run_id)
    assert parent.info.status == "FAILED"
    assert parent.data.metrics["n_failed"] == 1
    failures = _load(client, tracker.parent_run_id, "failures.json", tmp_path)["failures"]
    assert failures == [{"query_id": "q9", "error": "APIConnectionError: Connection error."}]


@pytest.mark.asyncio
async def test_single_variant_run_is_tracked_without_paired_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jev_bench.bench.runner import run_single

    monkeypatch.chdir(tmp_path)
    tracker = BenchTracker(
        experiment="single-exp",
        run_name="unit-single",
        params={"mode": "single-variant"},
        variant_names=("jev",),
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    results, failures = await run_single(
        ScriptedDecider(name="jev"),
        ScriptedRetriever(),
        records=[_record("q1"), _record("q2"), _record("q3")],
        on_result=tracker.record_single,
        on_failure=tracker.record_failure,
    )
    assert len(results) == 3 and failures == []
    tracker.finish("# single")

    client = MlflowClient()
    parent = client.get_run(tracker.parent_run_id)
    assert parent.data.metrics["n_completed"] == 3
    assert "paired_f1_mean_delta_a_minus_b" not in parent.data.metrics
    children = client.search_runs(
        [client.get_experiment_by_name("single-exp").experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{tracker.parent_run_id}'",
    )
    assert len(children) == 1
    assert [m.step for m in client.get_metric_history(children[0].info.run_id, "f1")] == [0, 1, 2]
