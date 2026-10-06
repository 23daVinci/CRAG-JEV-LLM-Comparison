import pytest

from jev_bench.bench.runner import run_comparison
from jev_bench.deciders.fake import ScriptedDecider
from jev_bench.retrieval.fake import ScriptedRetriever

RECORD = {
    "id": "q1",
    "question": "what is the capital",
    "context": {"doc_a": "paris is the capital of france", "doc_b": "unrelated passage"},
    "gold_titles": ["doc_a"],
}


@pytest.mark.asyncio
async def test_normal_run_does_not_trip_the_fixture_integrity_check() -> None:
    results, failures = await run_comparison(
        ScriptedDecider(name="a"), ScriptedDecider(name="b"), ScriptedRetriever(), records=[RECORD]
    )
    assert len(results) == 1
    assert failures == []


@pytest.mark.asyncio
async def test_mutating_the_shared_fixture_trips_the_integrity_check() -> None:
    record = {**RECORD, "context": dict(RECORD["context"])}

    class MutatingDecider(ScriptedDecider):
        async def screen_passage(self, query: str, doc: str):  # type: ignore[override]
            record["context"]["doc_a"] = "tampered"
            return await super().screen_passage(query, doc)

    # One query's failure is recorded, not raised — a single bad query must not sink an entire
    # 45-query run (this exact bug cost a full real run's worth of API spend before this test
    # existed, see docs/METHODOLOGY.md).
    results, failures = await run_comparison(
        MutatingDecider(name="a"), ScriptedDecider(name="b"), ScriptedRetriever(), records=[record]
    )
    assert results == []
    assert len(failures) == 1
    assert "changed during the run" in failures[0].error


@pytest.mark.asyncio
async def test_one_failing_query_does_not_prevent_others_from_completing() -> None:
    # Each 2-doc query triggers exactly 2 screen_passage calls (one batch, no retry needed), so
    # calls 1-2 belong to q1, 3-4 to q2, 5-6 to q3 — a call-count trigger (rather than checking the
    # doc content, which the mutation itself would corrupt for later queries) deterministically
    # tampers with q2's fixture during q2's own iteration, not anyone else's.
    good_record_1 = {**RECORD, "id": "q1"}
    bad_record = {**RECORD, "id": "q2", "context": dict(RECORD["context"])}
    good_record_2 = {**RECORD, "id": "q3"}
    call_count = [0]

    class MutatingDecider(ScriptedDecider):
        async def screen_passage(self, query: str, doc: str):  # type: ignore[override]
            call_count[0] += 1
            if call_count[0] in (3, 4):
                bad_record["context"]["doc_a"] = "tampered"
            return await super().screen_passage(query, doc)

    results, failures = await run_comparison(
        MutatingDecider(name="a"),
        ScriptedDecider(name="b"),
        ScriptedRetriever(),
        records=[good_record_1, bad_record, good_record_2],
    )
    assert {r.query_id for r in results} == {"q1", "q3"}
    assert [f.query_id for f in failures] == ["q2"]


@pytest.mark.asyncio
async def test_run_single_isolates_a_failing_query_and_fires_callbacks() -> None:
    from jev_bench.bench.runner import run_single

    good = {**RECORD, "id": "q1"}
    bad = {**RECORD, "id": "q2", "context": dict(RECORD["context"])}
    seen: list[str] = []
    failed: list[str] = []

    calls = [0]

    class TamperOnSecondQuery(ScriptedDecider):
        async def screen_passage(self, query: str, doc: str):  # type: ignore[override]
            calls[0] += 1
            if calls[0] in (3, 4):
                bad["context"]["doc_a"] = "tampered"
            return await super().screen_passage(query, doc)

    results, failures = await run_single(
        TamperOnSecondQuery(name="a"),
        ScriptedRetriever(),
        records=[good, bad],
        on_result=lambda r: seen.append(r.query_id),
        on_failure=lambda f: failed.append(f.query_id),
    )
    assert [r.query_id for r in results] == ["q1"]
    assert seen == ["q1"] and failed == ["q2"]
    assert "changed during the run" in failures[0].error
