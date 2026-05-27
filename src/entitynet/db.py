"""Kuzu graph database wrapper.

Why Kuzu (not Neo4j):
- Embedded — single directory on disk, no server to run, no Docker.
- Cypher-like syntax — every senior data engineer recognises it.
- Fast and modern (2024-25 release cadence).
- Apache 2.0 licensed.

`get_conn()` is the only thing the rest of the codebase imports from here.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

import kuzu

from .config import settings


@contextmanager
def get_conn():
    """Yield a Kuzu connection. Use as a context manager."""
    path = settings.kuzu_abs_path
    path.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(path))
    conn = kuzu.Connection(db)
    try:
        yield conn
    finally:
        # Kuzu's Connection has no explicit close; db handle is GC-managed.
        pass


def execute(query: str, params: dict | None = None) -> list[dict]:
    """One-shot query helper. Returns rows as a list of dicts."""
    with get_conn() as conn:
        result = conn.execute(query, parameters=params or {})
        rows: list[dict] = []
        # Kuzu QueryResult is iterable; convert each row to a dict using column names.
        try:
            cols = result.get_column_names()
        except AttributeError:
            cols = []
        while result.has_next():
            row = result.get_next()
            if cols:
                rows.append({c: v for c, v in zip(cols, row, strict=False)})
            else:
                rows.append({"_": row})
        return rows


def reset_db() -> Path:
    """Delete the Kuzu directory entirely. Useful for `make clean`."""
    path = settings.kuzu_abs_path
    if path.exists():
        shutil.rmtree(path)
    return path
