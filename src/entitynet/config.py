"""Central configuration. Every other module pulls paths/models from here."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_NEWS_DIR = SAMPLE_DIR / "news"
BENCHMARK_PATH = DATA_DIR / "benchmark.jsonl"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    kuzu_path: str = Field(default="data/graph.kuzu", alias="KUZU_PATH")
    chroma_path: str = Field(default="data/chroma", alias="CHROMA_PATH")

    # Embedding model — small, fast, runs on CPU
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBED_MODEL"
    )
    embed_dim: int = 384

    # Optional LLM endpoints — never required
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Retrieval defaults
    vector_top_k: int = 8
    graph_max_hops: int = 3
    hybrid_overlap_weight: float = 1.2  # bonus for entities found by both retrievers

    @property
    def kuzu_abs_path(self) -> Path:
        p = Path(self.kuzu_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def chroma_abs_path(self) -> Path:
        p = Path(self.chroma_path)
        return p if p.is_absolute() else ROOT_DIR / p


settings = Settings()
