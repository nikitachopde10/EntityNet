"""Pydantic schemas — the single contract every layer agrees on.

Design notes:
- Every entity has a stable `id` (e.g. `P001`, `C001`, `S001`). All
  retrievers return entity IDs, and benchmark grading is entity-F1 over
  these IDs — fully deterministic, no LLM judge needed.
- `RelationshipType` is an enum so the graph layer, the retriever layer
  and the eval layer all stay in sync.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ---------- Enums ----------


class EntityType(StrEnum):
    person = "Person"
    company = "Company"
    sanction = "Sanction"
    country = "Country"
    news = "NewsArticle"


class RelationshipType(StrEnum):
    owns = "OWNS"
    director_of = "DIRECTOR_OF"
    beneficial_owner_of = "BENEFICIAL_OWNER_OF"
    subsidiary_of = "SUBSIDIARY_OF"
    business_partner_of = "BUSINESS_PARTNER_OF"
    relative_of = "RELATIVE_OF"
    sanctioned_by = "SANCTIONED_BY"
    mentioned_in = "MENTIONED_IN"
    incorporated_in = "INCORPORATED_IN"
    born_in = "BORN_IN"


class SanctionList(StrEnum):
    ofac_sdn = "OFAC SDN"
    eu_consolidated = "EU Consolidated"
    uk_hmt = "UK HMT"
    un_consolidated = "UN Consolidated"
    us_entity_list = "US Entity List"


class RiskLevel(StrEnum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    severe = "severe"


# ---------- Entity records ----------


class Person(BaseModel):
    id: str
    name: str
    nationality: str | None = None
    date_of_birth: date | None = None
    roles: list[str] = Field(default_factory=list)
    notes: str | None = None


class Company(BaseModel):
    id: str
    name: str
    country: str
    industry: str | None = None
    incorporation_date: date | None = None
    is_shell: bool = False
    notes: str | None = None


class Sanction(BaseModel):
    id: str
    list_name: SanctionList
    target_entity_id: str = Field(description="ID of the Person or Company sanctioned.")
    program: str | None = Field(default=None, description="e.g. 'Russia-EO13662'.")
    issued_date: date | None = None
    summary: str


class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: RelationshipType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    percentage: float | None = Field(
        default=None, ge=0.0, le=100.0, description="For OWNS relationships."
    )
    as_of: date | None = None
    source_doc: str | None = Field(default=None, description="Provenance.")
    notes: str | None = None


class NewsArticle(BaseModel):
    id: str
    title: str
    body: str
    published_date: date
    source: str = Field(description="e.g. 'Reuters', 'FT', 'WSJ'.")
    url: str | None = None
    mentioned_entity_ids: list[str] = Field(
        default_factory=list,
        description="Canonical IDs mentioned. Populated by the entity linker.",
    )
    sentiment: Literal["positive", "neutral", "negative", "very_negative"] = "neutral"
    risk_tags: list[str] = Field(
        default_factory=list,
        description="e.g. 'sanctions', 'fraud', 'aml', 'litigation'.",
    )


# ---------- Retrieval / evaluation ----------


class RetrievalResult(BaseModel):
    retriever: Literal["vector", "graph", "hybrid"]
    question_id: str
    entity_ids: list[str] = Field(description="Canonical IDs returned, in rank order.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Short text snippets supporting each entity.",
    )
    paths: list[list[str]] = Field(
        default_factory=list,
        description="Graph paths (lists of node IDs). Empty for vector-only.",
    )
    latency_ms: float = 0.0


class BenchmarkQuestion(BaseModel):
    id: str
    text: str
    category: Literal[
        "direct_sanctions",
        "ownership_chain",
        "hidden_connection",
        "risk_propagation",
        "adverse_media",
    ]
    hops: int = Field(ge=0, le=5, description="Min hops needed to answer.")
    difficulty: Literal["easy", "medium", "hard"]
    ground_truth_ids: list[str] = Field(
        description="Canonical entity IDs that must appear in a correct answer."
    )


class RiskReport(BaseModel):
    entity_id: str
    entity_name: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    direct_sanctions: list[str] = Field(default_factory=list)
    indirect_sanctions_paths: list[list[str]] = Field(default_factory=list)
    adverse_media: list[str] = Field(default_factory=list)
    high_risk_connections: list[str] = Field(default_factory=list)
    summary: str
