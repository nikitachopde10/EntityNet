"""Cypher query templates for GraphRAG risk traversals.

These are the queries that vector RAG fundamentally cannot replicate. They
combine pattern matching, multi-hop traversal, and aggregation in a single
shot. Each function returns a list of entity IDs (and optionally paths).

Each query has a docstring explaining the risk scenario it answers — so
when a reader skims the file they immediately see *what kinds of risk
questions* this system can answer.
"""

from __future__ import annotations

from ..db import get_conn

# ---------- Helpers ----------


def _ids_from_rows(rows, key: str = "id") -> list[str]:
    out: list[str] = []
    while rows.has_next():
        r = rows.get_next()
        out.append(str(r[0]))
    return out


# ---------- Direct facts (1-hop) ----------


def find_entity_by_name(name_query: str, limit: int = 10) -> list[tuple[str, str, str]]:
    """Lookup entity by partial name. Returns [(id, label, name), ...]."""
    out: list[tuple[str, str, str]] = []
    with get_conn() as conn:
        rows = conn.execute(
            """
            MATCH (n:Person) WHERE lower(n.name) CONTAINS lower($q)
            RETURN n.id, 'Person', n.name
            UNION
            MATCH (n:Company) WHERE lower(n.name) CONTAINS lower($q)
            RETURN n.id, 'Company', n.name
            LIMIT $lim
            """,
            {"q": name_query, "lim": limit},
        )
        while rows.has_next():
            r = rows.get_next()
            out.append((str(r[0]), str(r[1]), str(r[2])))
    return out


def direct_sanctions(entity_id: str) -> list[str]:
    """Sanctions directly targeting this entity. Returns sanction IDs."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            MATCH (s:Sanction)-[:SANCTIONS_TARGET]->(t)
            WHERE t.id = $eid
            RETURN s.id
            """,
            {"eid": entity_id},
        )
        return _ids_from_rows(rows)


def adverse_media(entity_id: str) -> list[str]:
    """News articles that mention this entity and carry a negative sentiment."""
    label = "Person" if entity_id.startswith("P") else "Company"
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH (e:{label} {{id: $eid}})-[:MENTIONED_IN]->(a:NewsArticle)
            WHERE a.sentiment = 'negative' OR a.sentiment = 'very_negative'
            RETURN a.id
            """,
            {"eid": entity_id},
        )
        return _ids_from_rows(rows)


# ---------- Multi-hop traversals (the GraphRAG sweet spot) ----------


def downstream_holdings(entity_id: str, max_hops: int = 3) -> list[list[str]]:
    """Companies this entity owns directly or indirectly via OWNS chains.

    Useful when the question is about what a sanctioned entity controls
    rather than about who controls a clean entity.
    """
    paths: list[list[str]] = []
    label = "Person" if entity_id.startswith("P") else "Company"
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH p = (start:{label} {{id: $eid}})-[:OWNS*1..{max_hops}]->(owned)
            RETURN nodes(p) AS path_nodes
            """,
            {"eid": entity_id},
        )
        while rows.has_next():
            r = rows.get_next()
            node_list = r[0]
            ids = [n["id"] for n in node_list]
            paths.append(ids)
    return paths


def ownership_chain_to_sanctioned(entity_id: str, max_hops: int = 3) -> list[list[str]]:
    """Return ownership chains from this Company up to any sanctioned ancestor.

    This is exactly the OFAC 50% rule logic: if any chain reaches a sanctioned
    person/company through OWNS edges, the entity is implicated. Vector RAG
    cannot do this because it requires traversal, not similarity.

    Returns a list of paths. Each path is a list of node IDs from the start
    company up to the sanctioned ancestor.
    """
    paths: list[list[str]] = []
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH p = (start:Company {{id: $eid}})<-[:OWNS*1..{max_hops}]-(owner)
            WHERE EXISTS {{
                MATCH (:Sanction)-[:SANCTIONS_TARGET]->(owner)
            }}
            RETURN nodes(p) AS path_nodes
            """,
            {"eid": entity_id},
        )
        while rows.has_next():
            r = rows.get_next()
            node_list = r[0]
            ids = [n["id"] for n in node_list]
            paths.append(ids)
    return paths


def shared_directors_with_risk(entity_id: str, max_hops: int = 2) -> list[tuple[str, str, str]]:
    """For a given company, find other companies that share a director — and
    that director sits on a board that is sanctioned/owned by a sanctioned party.

    Returns [(other_company_id, shared_director_id, risky_company_id), ...].
    """
    out: list[tuple[str, str, str]] = []
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH (target:Company {{id: $eid}})<-[:DIRECTOR_OF]-(dir:Person)-[:DIRECTOR_OF]->(other:Company)
            WHERE other.id <> target.id
            MATCH (other)<-[:OWNS*1..{max_hops}]-(owner)
            WHERE EXISTS {{
                MATCH (:Sanction)-[:SANCTIONS_TARGET]->(owner)
            }}
            RETURN DISTINCT other.id, dir.id, owner.id
            """,
            {"eid": entity_id},
        )
        while rows.has_next():
            r = rows.get_next()
            out.append((str(r[0]), str(r[1]), str(r[2])))
    return out


def family_links_to_sanctioned(person_id: str) -> list[tuple[str, str]]:
    """Find sanctioned relatives of a person. Returns [(relative_id, sanction_id)]."""
    out: list[tuple[str, str]] = []
    with get_conn() as conn:
        rows = conn.execute(
            """
            MATCH (p:Person {id: $pid})-[:RELATIVE_OF]->(rel:Person)
            MATCH (s:Sanction)-[:SANCTIONS_TARGET]->(rel)
            RETURN rel.id, s.id
            UNION
            MATCH (rel:Person)-[:RELATIVE_OF]->(p:Person {id: $pid})
            MATCH (s:Sanction)-[:SANCTIONS_TARGET]->(rel)
            RETURN rel.id, s.id
            """,
            {"pid": person_id},
        )
        while rows.has_next():
            r = rows.get_next()
            out.append((str(r[0]), str(r[1])))
    return out


def business_partner_exposure(entity_id: str) -> list[tuple[str, str]]:
    """Sanctioned business partners (1-hop). Returns [(partner_id, sanction_id)]."""
    label = "Person" if entity_id.startswith("P") else "Company"
    out: list[tuple[str, str]] = []
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH (e:{label} {{id: $eid}})-[:BUSINESS_PARTNER_OF]-(partner)
            MATCH (s:Sanction)-[:SANCTIONS_TARGET]->(partner)
            RETURN partner.id, s.id
            """,
            {"eid": entity_id},
        )
        while rows.has_next():
            r = rows.get_next()
            out.append((str(r[0]), str(r[1])))
    return out


def k_hop_neighborhood(entity_id: str, hops: int = 2) -> list[str]:
    """Return all entity IDs within `hops` hops, of any relationship type.

    Useful for surfacing a focal entity's local neighbourhood (e.g. for
    a subgraph view or as raw context for downstream tooling).
    """
    label = "Person" if entity_id.startswith("P") else "Company"
    out: set[str] = {entity_id}
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            MATCH p = (e:{label} {{id: $eid}})-[*1..{hops}]-(n)
            WHERE n.id IS NOT NULL
            RETURN DISTINCT n.id
            """,
            {"eid": entity_id},
        )
        while rows.has_next():
            r = rows.get_next()
            if r[0] is not None:
                out.add(str(r[0]))
    return list(out)


def subgraph_nodes_and_edges(
    entity_id: str, hops: int = 2
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Return (nodes, edges) for visualisation.

    Implementation: first fetch all entity IDs within `hops` of the focal
    entity, then issue NON-recursive queries to collect direct edges between
    those nodes. Avoids Kuzu's RECURSIVE_REL restrictions.
    """
    ids = set(k_hop_neighborhood(entity_id, hops=hops))
    ids.add(entity_id)
    if not ids:
        return [], []
    id_list = list(ids)

    nodes: list[dict] = []
    with get_conn() as conn:
        for kind in ("Person", "Company"):
            res = conn.execute(
                f"MATCH (n:{kind}) WHERE list_contains($ids, n.id) "
                f"RETURN n.id AS nid, n.name AS nname",
                {"ids": id_list},
            )
            while res.has_next():
                row = res.get_next()
                nodes.append({"id": str(row[0]), "label": kind, "name": str(row[1])})
        for kind in ("Sanction", "NewsArticle"):
            res = conn.execute(
                f"MATCH (n:{kind}) WHERE list_contains($ids, n.id) RETURN n.id AS nid",
                {"ids": id_list},
            )
            while res.has_next():
                row = res.get_next()
                nodes.append({"id": str(row[0]), "label": kind, "name": str(row[0])})

    # Collect edges. One non-recursive query per edge table.
    edges: list[tuple[str, str, str]] = []
    edge_specs = [
        ("OWNS", "(a)-[r:OWNS]->(b)"),
        ("DIRECTOR_OF", "(a)-[r:DIRECTOR_OF]->(b)"),
        ("BUSINESS_PARTNER_OF", "(a)-[r:BUSINESS_PARTNER_OF]->(b)"),
        ("RELATIVE_OF", "(a)-[r:RELATIVE_OF]->(b)"),
        ("SANCTIONS_TARGET", "(a)-[r:SANCTIONS_TARGET]->(b)"),
        ("MENTIONED_IN", "(a)-[r:MENTIONED_IN]->(b)"),
    ]
    with get_conn() as conn:
        for label, pattern in edge_specs:
            res = conn.execute(
                f"MATCH {pattern} "
                f"WHERE list_contains($ids, a.id) AND list_contains($ids, b.id) "
                f"RETURN a.id AS src, b.id AS dst",
                {"ids": id_list},
            )
            while res.has_next():
                row = res.get_next()
                edges.append((str(row[0]), str(row[1]), label))
    return nodes, edges
