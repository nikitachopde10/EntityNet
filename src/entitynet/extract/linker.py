"""Entity linker: map surface forms to canonical IDs in the graph.

Strategy:
1. Build an alias index of all known entity names from the graph.
2. For each surface form, return the best fuzzy match above a similarity
   threshold (default 85/100 via rapidfuzz token_set_ratio).

This is exactly the pattern used by serious entity-resolution systems —
deterministic, transparent, and inspectable in a notebook. Compared to
embedding-based linking, it's faster, doesn't hallucinate, and gives a
calibratable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz import fuzz, process

from ..db import get_conn


@dataclass(frozen=True)
class EntityAlias:
    entity_id: str
    label: str
    name: str


@lru_cache(maxsize=1)
def load_alias_index() -> list[EntityAlias]:
    """Pull every Person and Company name from the graph as alias index."""
    out: list[EntityAlias] = []
    with get_conn() as conn:
        for kind in ("Person", "Company"):
            rows = conn.execute(f"MATCH (n:{kind}) RETURN n.id, n.name")
            while rows.has_next():
                r = rows.get_next()
                out.append(EntityAlias(entity_id=str(r[0]), label=kind, name=str(r[1])))
    return out


def link_one(surface: str, *, threshold: int = 85) -> EntityAlias | None:
    aliases = load_alias_index()
    names = [a.name for a in aliases]
    if not names:
        return None
    match = process.extractOne(
        surface, names, scorer=fuzz.token_set_ratio, score_cutoff=threshold
    )
    if not match:
        return None
    _, score, idx = match
    return aliases[idx]


def link_text(text: str, *, threshold: int = 85) -> list[EntityAlias]:
    """Return de-duplicated canonical entities mentioned in `text`."""
    from .ner import extract_mentions

    mentions = extract_mentions(text)
    seen: dict[str, EntityAlias] = {}
    for m in mentions:
        hit = link_one(m.surface, threshold=threshold)
        if hit and hit.entity_id not in seen:
            seen[hit.entity_id] = hit
    return list(seen.values())
