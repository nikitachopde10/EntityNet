"""Export the benchmark dataset for HuggingFace upload.

Two-step workflow:
1. Produces a clean CSV + Parquet copy under `data/hf_export/`.
2. If `HF_TOKEN` is set, optionally pushes to the Hub.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console

from entitynet.config import BENCHMARK_PATH, DATA_DIR, settings

console = Console()


def main(repo_id: str | None = None) -> None:
    out_dir = DATA_DIR / "hf_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    import pandas as pd

    df = pd.DataFrame(rows)
    df["ground_truth_ids"] = df["ground_truth_ids"].map(lambda x: "|".join(x))
    csv_path = out_dir / "entitynet_benchmark.csv"
    parquet_path = out_dir / "entitynet_benchmark.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    (out_dir / "README.md").write_text(_dataset_card())
    console.print(f"[green]Exported[/] {len(df)} rows to:")
    console.print(f"  {csv_path}")
    console.print(f"  {parquet_path}")
    console.print(f"  {out_dir / 'README.md'}")

    if repo_id and settings.hf_token:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi

            ds = Dataset.from_pandas(df)
            ds.push_to_hub(repo_id, token=settings.hf_token)
            HfApi(token=settings.hf_token).upload_file(
                path_or_fileobj=str(out_dir / "README.md"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            console.print(f"[green]Pushed dataset to:[/] https://huggingface.co/datasets/{repo_id}")
        except ImportError:
            console.print("[yellow]huggingface dependencies not installed.[/] Run: uv pip install -e '.[huggingface]'")
    elif repo_id:
        console.print("[yellow]HF_TOKEN not set in .env — skipping upload.[/]")


def _dataset_card() -> str:
    return """---
license: cc-by-4.0
language: en
tags:
  - graph-rag
  - kyc
  - aml
  - entity-resolution
  - benchmark
size_categories:
  - n<1K
---

# EntityNet Adverse Media & Entity Risk Benchmark

A 50-question benchmark for evaluating retrieval systems on multi-hop
KYC/AML entity-risk questions. Each question has a list of canonical
entity IDs as ground truth, so scoring is deterministic (entity-F1) and
fully reproducible.

The accompanying graph data is synthetic — generated for educational and
benchmarking purposes. No real persons or companies are referenced.

## Categories

| Category | N | Min hops | Description |
|---|---|---|---|
| direct_sanctions  | 10 | 1 | Single-hop sanctions look-ups |
| ownership_chain   | 12 | 1-3 | Multi-hop ownership traversal |
| hidden_connection | 10 | 2-3 | Director-overlap, family-link patterns |
| risk_propagation  | 10 | 1-3 | Indirect risk exposure |
| adverse_media     | 8  | 1 | Entity-to-news mentions |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("YOUR_USER/entitynet-benchmark")
for row in ds["train"]:
    truth = row["ground_truth_ids"].split("|")
    ...
```

## Citation

Cite the project repository: https://github.com/YOUR_USER/entitynet
"""


if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(repo_id)
