"""Validates the benchmark JSONL file structure."""

import json

from entitynet.config import BENCHMARK_PATH
from entitynet.schemas import BenchmarkQuestion


def test_benchmark_file_exists():
    assert BENCHMARK_PATH.exists()


def test_all_questions_parse():
    lines = [line for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]
    qs = [BenchmarkQuestion(**json.loads(line)) for line in lines]
    assert len(qs) == 50


def test_questions_cover_all_categories():
    lines = [line for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]
    qs = [BenchmarkQuestion(**json.loads(line)) for line in lines]
    cats = {q.category for q in qs}
    assert cats == {
        "direct_sanctions",
        "ownership_chain",
        "hidden_connection",
        "risk_propagation",
        "adverse_media",
    }


def test_all_question_ids_unique():
    lines = [line for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]
    qs = [BenchmarkQuestion(**json.loads(line)) for line in lines]
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids))


def test_all_ground_truth_ids_have_valid_prefix():
    """Every ground-truth ID must start with P / C / S / N."""
    lines = [line for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]
    qs = [BenchmarkQuestion(**json.loads(line)) for line in lines]
    for q in qs:
        for gid in q.ground_truth_ids:
            assert gid[0] in {"P", "C", "S", "N"}, f"{q.id} has bad ground-truth id {gid}"
