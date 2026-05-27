"""Vector-RAG baseline using ChromaDB + sentence-transformers.

Each document in the vector store is one "evidence chunk":
- One per entity (its name + nationality + roles + notes).
- One per sanctions entry (the summary).
- One per news article (title + body, truncated).

A query embeds the question and pulls the top-k by cosine similarity.
The retriever then surfaces the *entity IDs* mentioned in those chunks —
which is exactly what GraphRAG returns, making the two directly comparable
under entity-F1 scoring.

Why ChromaDB:
- Embedded (no server), free, recognisable on resumes.
- Built-in cosine-similarity search.
- The Python API maps cleanly onto the rest of the project.
"""

from __future__ import annotations

import contextlib
import csv
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import SAMPLE_DIR, SAMPLE_NEWS_DIR, settings
from ..schemas import RetrievalResult

COLLECTION_NAME = "entitynet_evidence"


@dataclass
class EvidenceDoc:
    doc_id: str
    text: str
    entity_ids: list[str]
    source: str


@lru_cache(maxsize=1)
def _chroma_client():
    path = settings.chroma_abs_path
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


@lru_cache(maxsize=1)
def _embed_model():
    """Lazily load the sentence-transformer. ~80MB download on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def _embed(texts: list[str]) -> list[list[float]]:
    model = _embed_model()
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return arr.tolist()


def _collection_exists() -> bool:
    client = _chroma_client()
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False


def build_index() -> int:
    """Build (or rebuild) the vector index from the bundled sample data."""
    client = _chroma_client()
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION_NAME)
    coll = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    docs: list[EvidenceDoc] = []
    docs.extend(_load_persons(SAMPLE_DIR / "persons.csv"))
    docs.extend(_load_companies(SAMPLE_DIR / "companies.csv"))
    docs.extend(_load_sanctions(SAMPLE_DIR / "sanctions.csv"))
    docs.extend(_load_news(SAMPLE_NEWS_DIR))

    ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]
    metas = [
        {"source": d.source, "entity_ids": json.dumps(d.entity_ids)} for d in docs
    ]
    embeddings = _embed(texts)
    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    return len(docs)


def _load_persons(path: Path) -> list[EvidenceDoc]:
    out: list[EvidenceDoc] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            text = (
                f"{row['name']} — nationality {row.get('nationality', '?')}, "
                f"roles: {row.get('roles', '')}. {row.get('notes', '')}"
            )
            out.append(
                EvidenceDoc(
                    doc_id=f"person::{row['id']}",
                    text=text,
                    entity_ids=[row["id"]],
                    source="persons.csv",
                )
            )
    return out


def _load_companies(path: Path) -> list[EvidenceDoc]:
    out: list[EvidenceDoc] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            text = (
                f"{row['name']} ({row.get('country', '?')}, {row.get('industry', '?')}). "
                f"{row.get('notes', '')}"
            )
            out.append(
                EvidenceDoc(
                    doc_id=f"company::{row['id']}",
                    text=text,
                    entity_ids=[row["id"]],
                    source="companies.csv",
                )
            )
    return out


def _load_sanctions(path: Path) -> list[EvidenceDoc]:
    out: list[EvidenceDoc] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            text = (
                f"Sanction {row['id']} on list {row['list_name']} "
                f"(program {row.get('program', '')}): {row['summary']}"
            )
            out.append(
                EvidenceDoc(
                    doc_id=f"sanction::{row['id']}",
                    text=text,
                    entity_ids=[row["id"], row["target_entity_id"]],
                    source="sanctions.csv",
                )
            )
    return out


def _load_news(news_dir: Path) -> list[EvidenceDoc]:
    out: list[EvidenceDoc] = []
    for jf in sorted(news_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        text = f"{data['title']}. {data['body']}"
        out.append(
            EvidenceDoc(
                doc_id=f"news::{data['id']}",
                text=text,
                entity_ids=list(data.get("mentioned_entity_ids") or []),
                source=data.get("source", "news"),
            )
        )
    return out


def vector_retrieve(question: str, top_k: int | None = None, question_id: str = "_adhoc") -> RetrievalResult:
    """Run vector retrieval and return entity IDs + evidence."""
    if not _collection_exists():
        raise RuntimeError(
            "Vector index not built. Run `make build` or `python scripts/build.py`."
        )
    k = top_k or settings.vector_top_k
    t0 = time.perf_counter()
    coll = _chroma_client().get_collection(COLLECTION_NAME)
    emb = _embed([question])[0]
    res = coll.query(query_embeddings=[emb], n_results=k, include=["documents", "metadatas", "distances"])
    entity_order: list[str] = []
    evidence: list[str] = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0], strict=False):
        ent_ids = json.loads(meta["entity_ids"])
        for eid in ent_ids:
            if eid not in entity_order:
                entity_order.append(eid)
        evidence.append(doc)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return RetrievalResult(
        retriever="vector",
        question_id=question_id,
        entity_ids=entity_order,
        evidence=evidence,
        paths=[],
        latency_ms=round(latency_ms, 2),
    )
