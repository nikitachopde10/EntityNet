"""GraphRAG retriever — the project's headline component.

Takes a natural-language question, identifies seed entities mentioned in it,
then runs one or more Cypher traversals to surface connected entities. The
specific traversal is picked by a lightweight intent classifier based on
keyword patterns — robust, transparent, no LLM required.

Why this beats vector RAG on the hard categories:
- Multi-hop ownership cannot be answered by similarity search.
- Director-overlap, partner-exposure, family-link patterns are graph
  primitives by definition.
- The retriever can also return the *paths* it traversed, making the
  answer explainable.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..config import settings
from ..extract.linker import EntityAlias, link_one, load_alias_index
from ..graph.queries import (
    adverse_media,
    business_partner_exposure,
    direct_sanctions,
    downstream_holdings,
    family_links_to_sanctioned,
    k_hop_neighborhood,
    ownership_chain_to_sanctioned,
    shared_directors_with_risk,
)
from ..schemas import RetrievalResult

# ---------- Intent detection ----------


PATTERNS = [
    # (intent, keyword regex). Order matters — first match wins.
    ("ownership_chain", r"\b(owner|owners|ownership|controls|controlled|ultimate beneficial|UBO|chain|subsidiary|parent|stake|stakes|holds|holding|holdings)\b"),
    ("shared_directors", r"\b(director|directors|board|share[sd]? a director|interlocking|board members?)\b"),
    ("family_link", r"\b(family|relative|cousin|sibling|brother|sister|wife|husband|spouse|son|daughter|relatives?)\b"),
    ("business_partner", r"\b(business partner|partner|partners|partnership|joint venture|JV|associate|deals? with)\b"),
    ("risk_propagation", r"\b(exposure|exposed|indirect|propagation|risk|risks|risky|connect|connection|connections|path)\b"),
    ("adverse_media", r"\b(allegation|allegations|media|news|reported|reports|adverse|press|article|coverage|allegedly|mention|mentions?)\b"),
    ("direct_sanctions", r"\b(sanction|sanctioned|sanctions|OFAC|SDN|EU list|HMT|UN list|entity list|designated|listed|terror|terrorism|trafficking)\b"),
]


@dataclass
class Intent:
    name: str
    seed_entities: list[EntityAlias]


def detect_intent(question: str) -> Intent:
    text_lower = question.lower()
    matched_intents: list[str] = []
    for intent_name, pattern in PATTERNS:
        if re.search(pattern, text_lower):
            matched_intents.append(intent_name)

    # Find any explicit entity references via the linker.
    seeds: list[EntityAlias] = []
    aliases = load_alias_index()
    for alias in aliases:
        if alias.name.lower() in text_lower or alias.entity_id.lower() in text_lower:
            seeds.append(alias)
    # Fall back to the first NER-resolvable surface form.
    if not seeds:
        # Look for capitalised words that might be names.
        candidates = re.findall(r"[A-Z][A-Za-z0-9&\-]+(?:\s+[A-Z][A-Za-z0-9&\-]+){0,3}", question)
        for cand in candidates:
            hit = link_one(cand, threshold=80)
            if hit:
                seeds.append(hit)

    # Dedup
    seen: set[str] = set()
    unique_seeds: list[EntityAlias] = []
    for s in seeds:
        if s.entity_id not in seen:
            seen.add(s.entity_id)
            unique_seeds.append(s)

    primary = matched_intents[0] if matched_intents else "ownership_chain"
    return Intent(name=primary, seed_entities=unique_seeds)


# ---------- Main retriever ----------


def graph_retrieve(question: str, question_id: str = "_adhoc") -> RetrievalResult:
    t0 = time.perf_counter()
    intent = detect_intent(question)

    entity_ids: list[str] = []
    evidence: list[str] = []
    paths: list[list[str]] = []

    seeds = intent.seed_entities
    if not seeds:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            retriever="graph",
            question_id=question_id,
            entity_ids=[],
            evidence=["[graph] No seed entity found in question."],
            paths=[],
            latency_ms=round(latency_ms, 2),
        )

    def _add(eid: str) -> None:
        if eid and eid not in entity_ids:
            entity_ids.append(eid)

    for seed in seeds:
        _add(seed.entity_id)

        # Direct sanctions always run — cheap and useful in every category.
        for sid in direct_sanctions(seed.entity_id):
            _add(sid)
            evidence.append(f"Direct sanction {sid} targets {seed.name} ({seed.entity_id}).")

        if intent.name in {"ownership_chain", "risk_propagation"} and seed.label == "Company":
            for path in ownership_chain_to_sanctioned(seed.entity_id, max_hops=settings.graph_max_hops):
                paths.append(path)
                for node_id in path:
                    _add(node_id)
                # Also add the sanction nodes targeting the upstream owner.
                if path:
                    for sid in direct_sanctions(path[-1]):
                        _add(sid)
                evidence.append("Ownership chain to sanctioned owner: " + " → ".join(path))
            # Downstream view: what does this entity itself own?
            for path in downstream_holdings(seed.entity_id, max_hops=settings.graph_max_hops):
                paths.append(path)
                for node_id in path:
                    _add(node_id)
                evidence.append("Downstream holding chain: " + " → ".join(path))

        if intent.name in {"ownership_chain", "risk_propagation"} and seed.label == "Person":
            for path in downstream_holdings(seed.entity_id, max_hops=settings.graph_max_hops):
                paths.append(path)
                for node_id in path:
                    _add(node_id)
                evidence.append("Downstream holding chain: " + " → ".join(path))

        if intent.name in {"shared_directors", "risk_propagation", "hidden_connection"} and seed.label == "Company":
            for other, dir_id, owner_id in shared_directors_with_risk(seed.entity_id):
                _add(dir_id)
                _add(other)
                _add(owner_id)
                for sid in direct_sanctions(owner_id):
                    _add(sid)
                paths.append([seed.entity_id, dir_id, other, owner_id])
                evidence.append(
                    f"{seed.entity_id} shares director {dir_id} with {other}, "
                    f"which is owned (≤2 hops) by sanctioned entity {owner_id}."
                )

        if intent.name in {"family_link", "risk_propagation"} and seed.label == "Person":
            for rel_id, sid in family_links_to_sanctioned(seed.entity_id):
                _add(rel_id)
                _add(sid)
                paths.append([seed.entity_id, rel_id, sid])
                evidence.append(
                    f"{seed.entity_id} is related to {rel_id}, who is targeted by sanction {sid}."
                )

        if intent.name in {"business_partner", "risk_propagation"}:
            for partner_id, sid in business_partner_exposure(seed.entity_id):
                _add(partner_id)
                _add(sid)
                paths.append([seed.entity_id, partner_id, sid])
                evidence.append(
                    f"{seed.entity_id} has business-partner exposure to {partner_id} (sanction {sid})."
                )

        if intent.name == "adverse_media":
            for nid in adverse_media(seed.entity_id):
                _add(nid)
                evidence.append(f"Adverse-media article {nid} mentions {seed.entity_id}.")

        if intent.name == "shared_directors":
            # Also surface raw director list for the seed (useful when question
            # asks "who shares the board" without a risk angle).
            for nid in k_hop_neighborhood(seed.entity_id, hops=2):
                if nid.startswith("P") or nid.startswith("C"):
                    _add(nid)

        if intent.name == "direct_sanctions":
            for nid in k_hop_neighborhood(seed.entity_id, hops=1):
                _add(nid)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return RetrievalResult(
        retriever="graph",
        question_id=question_id,
        entity_ids=entity_ids,
        evidence=evidence,
        paths=paths,
        latency_ms=round(latency_ms, 2),
    )
