"""spaCy-based entity extraction.

Pipeline:
1. Run spaCy NER on text (PERSON, ORG, GPE labels).
2. Use the linker (`extract/linker.py`) to map surface forms to canonical
   entity IDs in the graph.

Why spaCy:
- 100% local, free, fast.
- Industry-standard library — recognised on any data-science resume.
- The small model (`en_core_web_sm`) is ~50MB and runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ExtractedMention:
    surface: str
    label: str
    start: int
    end: int


@lru_cache(maxsize=1)
def _load_spacy():
    """Lazy-load spaCy. Raises a friendly error if the model isn't installed."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError("spaCy is required. Run: uv pip install spacy") from e
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError(
            "spaCy model 'en_core_web_sm' not installed. "
            "Run: uv run python -m spacy download en_core_web_sm"
        ) from e


def extract_mentions(text: str) -> list[ExtractedMention]:
    """Return PERSON / ORG / GPE mentions from `text`."""
    nlp = _load_spacy()
    doc = nlp(text)
    out: list[ExtractedMention] = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE"}:
            out.append(
                ExtractedMention(
                    surface=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
    return out
