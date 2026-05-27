"""Typer CLI: build, benchmark, ask, report."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .benchmark.runner import run_benchmark
from .graph.builder import build_graph, graph_stats
from .graph.queries import find_entity_by_name
from .retrieve.graph_rag import graph_retrieve
from .retrieve.hybrid import hybrid_retrieve
from .retrieve.risk_report import risk_report
from .retrieve.vector import build_index, vector_retrieve

app = typer.Typer(help="EntityNet — GraphRAG for adverse media & entity risk.")
console = Console()


@app.command()
def build(reset: bool = typer.Option(True, "--reset/--keep", help="Drop and rebuild.")):
    """Build the Kuzu graph + ChromaDB index from sample data."""
    console.print("[bold]Building Kuzu graph...[/]")
    summary = build_graph(reset=reset)
    table = Table(title="Graph build summary")
    table.add_column("Counter")
    table.add_column("Count", justify="right")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)

    console.print("[bold]Building ChromaDB vector index...[/]")
    n = build_index()
    console.print(f"[green]Indexed[/] {n} documents into ChromaDB.")


@app.command()
def stats():
    """Print graph node counts."""
    s = graph_stats()
    table = Table(title="Graph contents")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    for k, v in s.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def benchmark():
    """Run the 50-question benchmark across vector / graph / hybrid."""
    summary = run_benchmark()
    table = Table(title="Benchmark summary")
    table.add_column("Retriever")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("p50 latency (ms)", justify="right")
    for name, row in summary["overall"].items():
        table.add_row(
            name,
            f"{row['f1_mean']:.3f}",
            f"{row['precision_mean']:.3f}",
            f"{row['recall_mean']:.3f}",
            f"{row['recall_at_5_mean']:.3f}",
            f"{row['latency_ms_p50']:.2f}",
        )
    console.print(table)

    cat_table = Table(title="Per-category F1")
    cat_table.add_column("Category")
    for name in summary["overall"]:
        cat_table.add_column(name, justify="right")
    for cat, row in summary["by_category"].items():
        cat_table.add_row(cat, *[f"{row.get(n, {}).get('f1_mean', float('nan')):.3f}" for n in summary["overall"]])
    console.print(cat_table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A risk question, e.g. 'Who owns Delta Energy?'"),
    retriever: str = typer.Option("hybrid", "--retriever", "-r", help="vector | graph | hybrid"),
):
    """Ask a single question and inspect what each retriever returns."""
    fn = {"vector": vector_retrieve, "graph": graph_retrieve, "hybrid": hybrid_retrieve}[retriever]
    result = fn(question)
    console.print(f"[bold]{retriever}[/]  ({result.latency_ms} ms)")
    console.print(f"  entities: {result.entity_ids}")
    if result.paths:
        console.print("  paths:")
        for p in result.paths:
            console.print(f"    {' → '.join(p)}")
    if result.evidence:
        console.print("  evidence:")
        for e in result.evidence[:5]:
            console.print(f"    - {e[:200]}")


@app.command()
def report(entity_id: str = typer.Argument(..., help="Entity ID, e.g. 'C004'.")):
    """Generate a structured risk report for an entity."""
    r = risk_report(entity_id)
    console.print(json.dumps(r.model_dump(), indent=2, default=str))


@app.command()
def search(query: str = typer.Argument(..., help="Partial name to look up.")):
    """Find an entity ID by partial name."""
    hits = find_entity_by_name(query)
    for hit in hits:
        console.print(hit)


if __name__ == "__main__":
    app()
