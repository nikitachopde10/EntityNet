"""End-to-end build script. Run before the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console

from entitynet.graph.builder import build_graph
from entitynet.retrieve.vector import build_index

console = Console()


def main() -> None:
    console.rule("[bold cyan]Building Kuzu graph[/]")
    summary = build_graph(reset=True)
    for k, v in summary.items():
        console.print(f"  {k}: {v}")

    console.rule("[bold cyan]Building ChromaDB vector index[/]")
    n = build_index()
    console.print(f"  documents indexed: {n}")

    console.rule("[bold green]Build complete[/]")
    console.print("Next: [bold]make benchmark[/].")


if __name__ == "__main__":
    main()
