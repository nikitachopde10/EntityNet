"""Entity-set scoring. Deterministic, no LLM required.

For each (retrieved_set, ground_truth_set) we compute:
- precision = |retrieved ∩ truth| / |retrieved|
- recall    = |retrieved ∩ truth| / |truth|
- F1        = 2PR / (P+R)

We also compute *recall@k* for small k, since a retriever that buries the
right answer at rank 50 is much worse than one that surfaces it at rank 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreRow:
    question_id: str
    retriever: str
    precision: float
    recall: float
    f1: float
    recall_at_5: float
    recall_at_10: float
    n_retrieved: int
    n_truth: int


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def score_one(
    question_id: str,
    retriever: str,
    retrieved: list[str],
    truth: list[str],
) -> ScoreRow:
    truth_set = set(truth)
    if not truth_set and not retrieved:
        return ScoreRow(question_id, retriever, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0)
    retrieved_set = set(retrieved)
    hit_set = retrieved_set & truth_set
    precision = _safe_div(len(hit_set), len(retrieved_set))
    recall = _safe_div(len(hit_set), len(truth_set))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    top5_hits = sum(1 for eid in retrieved[:5] if eid in truth_set)
    top10_hits = sum(1 for eid in retrieved[:10] if eid in truth_set)
    return ScoreRow(
        question_id=question_id,
        retriever=retriever,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        recall_at_5=round(_safe_div(top5_hits, len(truth_set)), 3),
        recall_at_10=round(_safe_div(top10_hits, len(truth_set)), 3),
        n_retrieved=len(retrieved_set),
        n_truth=len(truth_set),
    )
