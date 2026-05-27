"""Hybrid retriever: union of vector and graph results, with a small
preference boost for entities surfaced by both retrievers.

Why "union with overlap bonus":
- Vector RAG is great when the answer entity is *explicitly mentioned* in
  some text chunk.
- GraphRAG is great when the answer requires traversal.
- The intersection of the two is the highest-confidence set; everything
  else is appended in the original retriever's rank order.
"""

from __future__ import annotations

import time

from ..config import settings
from ..schemas import RetrievalResult
from .graph_rag import graph_retrieve
from .vector import vector_retrieve


def hybrid_retrieve(question: str, question_id: str = "_adhoc") -> RetrievalResult:
    """Hybrid retrieval = reciprocal rank fusion + precision-preserving cap.

    Strategy:
      1. Always include every entity GraphRAG returns (graph traversal is
         deliberate, so its precision is high — never drop these).
      2. Add a vector entity only if (a) it overlaps with graph, or
         (b) it sits in the top-3 vector ranks. This filters the long
         tail of false positives that flat dense retrieval emits.
      3. Score by reciprocal rank fusion with an overlap bonus.
    """
    t0 = time.perf_counter()
    vec = vector_retrieve(question, question_id=question_id)
    gph = graph_retrieve(question, question_id=question_id)

    graph_set = set(gph.entity_ids)
    vec_top3 = set(vec.entity_ids[:3])

    scores: dict[str, float] = {}
    for rank, eid in enumerate(gph.entity_ids):
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (rank + 1)

    for rank, eid in enumerate(vec.entity_ids):
        if eid in graph_set:
            scores[eid] = scores.get(eid, 0.0) + (1.0 / (rank + 1)) * settings.hybrid_overlap_weight
        elif eid in vec_top3:
            scores[eid] = scores.get(eid, 0.0) + (1.0 / (rank + 1)) * 0.5

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return RetrievalResult(
        retriever="hybrid",
        question_id=question_id,
        entity_ids=sorted_ids,
        evidence=vec.evidence + gph.evidence,
        paths=gph.paths,
        latency_ms=round(latency_ms, 2),
    )
