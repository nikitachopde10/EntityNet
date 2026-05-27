"""Entity-F1 math tests — protects the headline benchmark numbers."""

from entitynet.benchmark.metrics import score_one


def test_perfect_match():
    s = score_one("Q1", "graph", ["A", "B"], ["A", "B"])
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_no_match():
    s = score_one("Q1", "graph", ["X"], ["A", "B"])
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_partial_match():
    s = score_one("Q1", "graph", ["A", "X", "Y"], ["A", "B"])
    # P = 1/3, R = 1/2, F1 = 2 * (1/3 * 1/2) / (1/3 + 1/2) = (1/3) / (5/6) = 0.4
    assert s.precision == round(1 / 3, 3)
    assert s.recall == 0.5
    assert s.f1 == 0.4


def test_empty_retrieved_empty_truth_is_perfect():
    s = score_one("Q1", "graph", [], [])
    assert s.f1 == 1.0


def test_empty_retrieved_nonempty_truth_is_zero():
    s = score_one("Q1", "graph", [], ["A"])
    assert s.f1 == 0.0


def test_dedup_via_set():
    # Duplicate retrieved entries shouldn't double-count for precision.
    s = score_one("Q1", "graph", ["A", "A", "B"], ["A", "B"])
    assert s.f1 == 1.0


def test_recall_at_k_with_late_hit():
    retrieved = ["X", "Y", "Z", "W", "V", "A"]  # truth 'A' at rank 6
    s = score_one("Q1", "graph", retrieved, ["A"])
    assert s.recall_at_5 == 0.0
    assert s.recall_at_10 == 1.0
