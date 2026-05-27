"""Run the benchmark and print the comparison table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.table import Table

from entitynet.benchmark.runner import run_benchmark

console = Console()


def main() -> None:
    summary = run_benchmark()

    overall = Table(title="Overall benchmark results")
    overall.add_column("Retriever")
    overall.add_column("F1", justify="right")
    overall.add_column("Precision", justify="right")
    overall.add_column("Recall", justify="right")
    overall.add_column("R@5", justify="right")
    overall.add_column("p50 ms", justify="right")
    for name, row in summary["overall"].items():
        overall.add_row(
            name,
            f"{row['f1_mean']:.3f}",
            f"{row['precision_mean']:.3f}",
            f"{row['recall_mean']:.3f}",
            f"{row['recall_at_5_mean']:.3f}",
            f"{row['latency_ms_p50']:.2f}",
        )
    console.print(overall)

    cat = Table(title="F1 by category (the differentiation chart)")
    cat.add_column("Category")
    for name in summary["overall"]:
        cat.add_column(name, justify="right")
    for c, row in summary["by_category"].items():
        cat.add_row(c, *[f"{row.get(n, {}).get('f1_mean', 0):.3f}" for n in summary["overall"]])
    console.print(cat)


if __name__ == "__main__":
    main()
