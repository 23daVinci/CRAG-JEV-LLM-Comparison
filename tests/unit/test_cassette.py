from pathlib import Path

import pytest

from jev_bench.deciders.cassette import CassetteDecider
from jev_bench.deciders.fake import ScriptedDecider


@pytest.mark.asyncio
async def test_record_then_replay_round_trips_value_and_marks_latency_replayed(
    tmp_path: Path,
) -> None:
    cassette_path = tmp_path / "cassette.jsonl"
    scripted = ScriptedDecider(name="fake")

    recorder = CassetteDecider(scripted, cassette_path, mode="record")
    recorded = await recorder.screen_passage("q", "some passage")
    assert recorded.metrics.latency_source == "simulated"

    player = CassetteDecider(scripted, cassette_path, mode="replay")
    replayed = await player.screen_passage("q", "some passage")

    assert replayed.value == recorded.value
    assert replayed.metrics.latency_source == "replayed"
    assert replayed.metrics.input_tokens == recorded.metrics.input_tokens
    assert replayed.metrics.cost_usd == recorded.metrics.cost_usd


@pytest.mark.asyncio
async def test_replay_without_a_recording_raises(tmp_path: Path) -> None:
    cassette_path = tmp_path / "empty.jsonl"
    scripted = ScriptedDecider(name="fake")
    player = CassetteDecider(scripted, cassette_path, mode="replay")
    with pytest.raises(KeyError):
        await player.screen_passage("unseen query", "unseen doc")


@pytest.mark.asyncio
async def test_different_inputs_get_different_cassette_entries(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cassette.jsonl"
    scripted = ScriptedDecider(name="fake")
    recorder = CassetteDecider(scripted, cassette_path, mode="record")

    await recorder.screen_passage("q", "doc a")
    await recorder.screen_passage("q", "doc b")

    player = CassetteDecider(scripted, cassette_path, mode="replay")
    await player.screen_passage("q", "doc a")
    await player.screen_passage("q", "doc b")
