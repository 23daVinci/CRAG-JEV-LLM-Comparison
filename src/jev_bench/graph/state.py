from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from jev_bench.deciders.base import PassageVerdict, RetryChoice
from jev_bench.telemetry.model import NodeSpan


class RetrievalState(TypedDict):
    query: str
    doc_pool: dict[str, str]
    """The frozen, full set of candidate paragraphs for this query (HotpotQA's 10-doc distractor
    context). Identical for both variants — retrieval is never a variable under test."""

    ranked_ids: list[str]
    """Output of the one-time TF-IDF rank; empty until the first `retrieve` node runs."""

    screened_ids: list[str]
    """Ids already passed through `screen_passage`, across all attempts — never re-screened."""

    candidate_ids: list[str]
    """The batch pulled by the current `retrieve` call, about to be screened."""

    selected_ids: list[str]
    """Ids that have passed screening so far; what `grade_sufficiency` judges."""

    passage_verdicts: dict[str, PassageVerdict]
    passage_evidence: dict[str, dict[str, float]]
    """Raw per-document evidence from each `screen_passage` Decision (e.g. Jev's un-thresholded
    `relevant` probability), keyed by doc id. Never read by the graph — `passage_verdicts` is the
    thresholded value the graph acts on — but needed after the fact for the PR-curve/iso-recall
    analysis in bench/analysis.py, which needs Jev's raw probabilities, not just its thresholded
    decisions, to show quality across its whole operating range rather than one fixed point."""

    attempt: int
    sufficient: bool
    retry_choice: RetryChoice
    trace: Annotated[list[NodeSpan], operator.add]


def initial_state(query: str, doc_pool: dict[str, str]) -> RetrievalState:
    return RetrievalState(
        query=query,
        doc_pool=doc_pool,
        ranked_ids=[],
        screened_ids=[],
        candidate_ids=[],
        selected_ids=[],
        passage_verdicts={},
        passage_evidence={},
        attempt=0,
        sufficient=False,
        retry_choice="return",
        trace=[],
    )
