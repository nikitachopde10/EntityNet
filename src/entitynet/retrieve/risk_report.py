"""Structured risk report for a single entity.

Combines several graph traversals into a single deterministic report. The
risk score is a simple weighted sum — fully explainable, no LLM.
"""

from __future__ import annotations

from ..db import get_conn
from ..graph.queries import (
    adverse_media,
    business_partner_exposure,
    direct_sanctions,
    family_links_to_sanctioned,
    ownership_chain_to_sanctioned,
    shared_directors_with_risk,
)
from ..schemas import RiskLevel, RiskReport

WEIGHTS = {
    "direct": 60.0,
    "ownership_chain": 35.0,
    "shared_director": 15.0,
    "family_link": 25.0,
    "business_partner": 20.0,
    "adverse_media": 8.0,
}


def _label_for(entity_id: str) -> str:
    return "Person" if entity_id.startswith("P") else "Company"


def _entity_name(entity_id: str) -> str:
    label = _label_for(entity_id)
    with get_conn() as conn:
        rows = conn.execute(f"MATCH (n:{label} {{id: $eid}}) RETURN n.name", {"eid": entity_id})
        if rows.has_next():
            return str(rows.get_next()[0])
    return entity_id


def _level_from_score(score: float) -> RiskLevel:
    if score >= 80:
        return RiskLevel.severe
    if score >= 50:
        return RiskLevel.high
    if score >= 25:
        return RiskLevel.medium
    if score > 0:
        return RiskLevel.low
    return RiskLevel.none


def risk_report(entity_id: str) -> RiskReport:
    name = _entity_name(entity_id)
    score = 0.0

    direct = direct_sanctions(entity_id)
    if direct:
        score += WEIGHTS["direct"]

    chains: list[list[str]] = []
    if entity_id.startswith("C"):
        chains = ownership_chain_to_sanctioned(entity_id)
        if chains:
            score += WEIGHTS["ownership_chain"]

    shared: list[str] = []
    if entity_id.startswith("C"):
        for other, dir_id, owner_id in shared_directors_with_risk(entity_id):
            shared.append(f"director {dir_id} → {other} (owner: {owner_id})")
        if shared:
            score += WEIGHTS["shared_director"]

    family: list[str] = []
    if entity_id.startswith("P"):
        for rel_id, sid in family_links_to_sanctioned(entity_id):
            family.append(f"{rel_id} (sanction {sid})")
        if family:
            score += WEIGHTS["family_link"]

    partners: list[str] = []
    for pid, sid in business_partner_exposure(entity_id):
        partners.append(f"{pid} (sanction {sid})")
    if partners:
        score += WEIGHTS["business_partner"]

    media = adverse_media(entity_id)
    if media:
        score += WEIGHTS["adverse_media"] * min(3, len(media))  # cap

    score = round(min(score, 100.0), 1)
    high_risk: list[str] = []
    if family:
        high_risk.extend(family)
    if partners:
        high_risk.extend(partners)
    if shared:
        high_risk.extend(shared)

    summary_parts: list[str] = []
    if direct:
        summary_parts.append(f"directly targeted by {len(direct)} sanction(s)")
    if chains:
        summary_parts.append(f"{len(chains)} sanctioned-ownership chain(s)")
    if shared:
        summary_parts.append(f"{len(shared)} sanctioned-linked director overlap(s)")
    if family:
        summary_parts.append(f"{len(family)} sanctioned-relative link(s)")
    if partners:
        summary_parts.append(f"{len(partners)} sanctioned-business-partner link(s)")
    if media:
        summary_parts.append(f"{len(media)} adverse-media mention(s)")
    summary = (
        "; ".join(summary_parts) or "no risk indicators detected against the bundled graph"
    )

    return RiskReport(
        entity_id=entity_id,
        entity_name=name,
        risk_level=_level_from_score(score),
        risk_score=score,
        direct_sanctions=direct,
        indirect_sanctions_paths=chains,
        adverse_media=media,
        high_risk_connections=high_risk,
        summary=summary,
    )
