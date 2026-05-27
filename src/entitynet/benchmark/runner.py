"""Benchmark runner — load JSONL → run all retrievers → write results."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.progress import track

from ..config import BENCHMARK_PATH, BENCHMARK_RESULTS_PATH
from ..retrieve.graph_rag import graph_retrieve
from ..retrieve.hybrid import hybrid_retrieve
from ..retrieve.vector import vector_retrieve
from ..schemas import BenchmarkQuestion
from .metrics import ScoreRow, score_one

console = Console()


RETRIEVERS = {
    "vector": vector_retrieve,
    "graph": graph_retrieve,
    "hybrid": hybrid_retrieve,
}


def load_benchmark(path: Path | None = None) -> list[BenchmarkQuestion]:
    p = path or BENCHMARK_PATH
    questions: list[BenchmarkQuestion] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(BenchmarkQuestion(**json.loads(line)))
    return questions


def run_benchmark(*, write: bool = True) -> dict:
    questions = load_benchmark()
    rows: list[ScoreRow] = []
    timings: dict[str, list[float]] = {k: [] for k in RETRIEVERS}

    for q in track(questions, description="Benchmarking"):
        for name, fn in RETRIEVERS.items():
            try:
                result = fn(q.text, question_id=q.id)
            except Exception as e:
                console.print(f"[red]{name} failed on {q.id}: {e}[/]")
                continue
            timings[name].append(result.latency_ms)
            rows.append(score_one(q.id, name, result.entity_ids, q.ground_truth_ids))

    summary = _summarise(rows, timings, questions)
    if write:
        out_path = BENCHMARK_RESULTS_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        console.print(f"[green]Wrote results to[/] {out_path}")
    return summary


def _summarise(rows: list[ScoreRow], timings: dict[str, list[float]], questions: list[BenchmarkQuestion]) -> dict:
    q_by_id = {q.id: q for q in questions}
    by_retriever: dict[str, list[ScoreRow]] = {}
    for r in rows:
        by_retriever.setdefault(r.retriever, []).append(r)

    overall: dict[str, dict] = {}
    for name, scores in by_retriever.items():
        overall[name] = {
            "f1_mean": round(statistics.mean(s.f1 for s in scores), 3),
            "precision_mean": round(statistics.mean(s.precision for s in scores), 3),
            "recall_mean": round(statistics.mean(s.recall for s in scores), 3),
            "recall_at_5_mean": round(statistics.mean(s.recall_at_5 for s in scores), 3),
            "latency_ms_p50": round(statistics.median(timings.get(name, [0.0]) or [0.0]), 2),
            "latency_ms_p95": round(_p95(timings.get(name, []) or [0.0]), 2),
            "n_questions": len(scores),
        }

    by_category: dict[str, dict[str, dict]] = {}
    categories = sorted({q.category for q in questions})
    for cat in categories:
        by_category[cat] = {}
        for name in by_retriever:
            cat_scores = [
                s for s in by_retriever[name] if q_by_id.get(s.question_id) and q_by_id[s.question_id].category == cat
            ]
            if not cat_scores:
                continue
            by_category[cat][name] = {
                "f1_mean": round(statistics.mean(s.f1 for s in cat_scores), 3),
                "recall_mean": round(statistics.mean(s.recall for s in cat_scores), 3),
                "n": len(cat_scores),
            }

    return {
        "overall": overall,
        "by_category": by_category,
        "per_question": [asdict(r) for r in rows],
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])
