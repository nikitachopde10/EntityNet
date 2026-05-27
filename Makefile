.PHONY: install models build benchmark test lint clean export app

# ─── Core workflow ─────────────────────────────────────────────────────
install:
	uv pip install -e ".[dev]"

models:
	uv run python -m spacy download en_core_web_sm

build:
	uv run python scripts/build.py

benchmark:
	uv run python scripts/benchmark.py

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ app/ scripts/

export:
	uv run python scripts/export_hf.py

clean:
	rm -rf data/graph.kuzu data/chroma/ data/benchmark_results.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Optional: experimental dashboard prototype ────────────────────────
# Install dashboard extras first:  uv pip install -e ".[dashboard]"
app:
	uv run streamlit run app/streamlit_app.py
