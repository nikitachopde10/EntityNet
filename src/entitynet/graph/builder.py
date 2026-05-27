"""Build the Kuzu graph from sample CSVs and news JSONs.

This module is intentionally a single function `build_graph()`. The rest of
the project treats graph construction as a one-shot operation.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import kuzu
from rich.console import Console

from ..config import SAMPLE_DIR, SAMPLE_NEWS_DIR
from ..db import get_conn, reset_db
from .schema import all_ddl

console = Console()


def _parse_date(s: str | None) -> date | None:
    if not s or s.strip() in {"", "NULL", "None"}:
        return None
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


def _apply_schema(conn: kuzu.Connection) -> None:
    for stmt in all_ddl():
        conn.execute(stmt)


def _load_persons(conn: kuzu.Connection, path: Path) -> int:
    n = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            conn.execute(
                """
                CREATE (:Person {
                    id: $id, name: $name, nationality: $nat,
                    date_of_birth: $dob, roles: $roles, notes: $notes
                })
                """,
                {
                    "id": row["id"],
                    "name": row["name"],
                    "nat": row.get("nationality") or "",
                    "dob": _parse_date(row.get("date_of_birth")),
                    "roles": row.get("roles") or "",
                    "notes": row.get("notes") or "",
                },
            )
            n += 1
    return n


def _load_companies(conn: kuzu.Connection, path: Path) -> int:
    n = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            conn.execute(
                """
                CREATE (:Company {
                    id: $id, name: $name, country: $country, industry: $industry,
                    incorporation_date: $inc_date, is_shell: $shell, notes: $notes
                })
                """,
                {
                    "id": row["id"],
                    "name": row["name"],
                    "country": row.get("country") or "",
                    "industry": row.get("industry") or "",
                    "inc_date": _parse_date(row.get("incorporation_date")),
                    "shell": str(row.get("is_shell", "false")).lower() == "true",
                    "notes": row.get("notes") or "",
                },
            )
            n += 1
    return n


def _load_sanctions(conn: kuzu.Connection, path: Path) -> tuple[int, int]:
    """Returns (sanction_nodes_created, sanction_target_edges_created)."""
    n_nodes, n_edges = 0, 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            conn.execute(
                """
                CREATE (:Sanction {
                    id: $id, list_name: $list, target_entity_id: $target,
                    program: $program, issued_date: $issued, summary: $summary
                })
                """,
                {
                    "id": row["id"],
                    "list": row["list_name"],
                    "target": row["target_entity_id"],
                    "program": row.get("program") or "",
                    "issued": _parse_date(row.get("issued_date")),
                    "summary": row.get("summary") or "",
                },
            )
            n_nodes += 1
            target_id = row["target_entity_id"]
            target_label = "Person" if target_id.startswith("P") else "Company"
            conn.execute(
                f"""
                MATCH (s:Sanction {{id: $sid}}), (t:{target_label} {{id: $tid}})
                CREATE (s)-[:SANCTIONS_TARGET]->(t)
                """,
                {"sid": row["id"], "tid": target_id},
            )
            n_edges += 1
    return n_nodes, n_edges


_REL_TYPE_TO_TABLE = {
    "OWNS": "OWNS",
    "DIRECTOR_OF": "DIRECTOR_OF",
    "BUSINESS_PARTNER_OF": "BUSINESS_PARTNER_OF",
    "RELATIVE_OF": "RELATIVE_OF",
}


def _label_for(entity_id: str) -> str:
    if entity_id.startswith("P"):
        return "Person"
    if entity_id.startswith("C"):
        return "Company"
    raise ValueError(f"Unrecognised id prefix: {entity_id}")


def _load_relationships(conn: kuzu.Connection, path: Path) -> int:
    n = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rel_type = row["type"].strip()
            table = _REL_TYPE_TO_TABLE.get(rel_type)
            if not table:
                console.print(f"[yellow]Skipping unknown relationship type: {rel_type}[/]")
                continue

            src_label = _label_for(row["source_id"])
            tgt_label = _label_for(row["target_id"])

            properties = ""
            params: dict = {"sid": row["source_id"], "tid": row["target_id"]}

            if table == "OWNS":
                pct = float(row["percentage"]) if row.get("percentage") else None
                params["pct"] = pct
                params["as_of"] = _parse_date(row.get("as_of"))
                params["src_doc"] = row.get("source_doc") or ""
                properties = " {percentage: $pct, as_of: $as_of, source_doc: $src_doc}"
            elif table == "DIRECTOR_OF":
                params["as_of"] = _parse_date(row.get("as_of"))
                params["src_doc"] = row.get("source_doc") or ""
                properties = " {as_of: $as_of, source_doc: $src_doc}"
            elif table == "BUSINESS_PARTNER_OF":
                params["weight"] = float(row["weight"]) if row.get("weight") else 1.0
                params["as_of"] = _parse_date(row.get("as_of"))
                params["src_doc"] = row.get("source_doc") or ""
                properties = " {weight: $weight, as_of: $as_of, source_doc: $src_doc}"
            elif table == "RELATIVE_OF":
                params["weight"] = float(row["weight"]) if row.get("weight") else 1.0
                params["src_doc"] = row.get("source_doc") or ""
                properties = " {weight: $weight, source_doc: $src_doc}"

            query = (
                f"MATCH (a:{src_label} {{id: $sid}}), (b:{tgt_label} {{id: $tid}}) "
                f"CREATE (a)-[:{table}{properties}]->(b)"
            )
            conn.execute(query, params)
            n += 1
    return n


def _load_news(conn: kuzu.Connection, news_dir: Path) -> tuple[int, int]:
    """Returns (articles_created, mention_edges_created)."""
    n_articles, n_mentions = 0, 0
    for jf in sorted(news_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        conn.execute(
            """
            CREATE (:NewsArticle {
                id: $id, title: $title, body: $body,
                published_date: $pub, source: $src, url: $url,
                sentiment: $sent, risk_tags: $tags
            })
            """,
            {
                "id": data["id"],
                "title": data["title"],
                "body": data["body"],
                "pub": _parse_date(data.get("published_date")),
                "src": data.get("source") or "",
                "url": data.get("url") or "",
                "sent": data.get("sentiment") or "neutral",
                "tags": ";".join(data.get("risk_tags") or []),
            },
        )
        n_articles += 1
        for ent_id in data.get("mentioned_entity_ids") or []:
            label = _label_for(ent_id)
            conn.execute(
                f"MATCH (e:{label} {{id: $eid}}), (a:NewsArticle {{id: $aid}}) "
                f"CREATE (e)-[:MENTIONED_IN]->(a)",
                {"eid": ent_id, "aid": data["id"]},
            )
            n_mentions += 1
    return n_articles, n_mentions


def build_graph(*, reset: bool = True) -> dict[str, int]:
    """Build the full graph from sample data. Returns a summary dict."""
    if reset:
        reset_db()

    summary: dict[str, int] = {}
    with get_conn() as conn:
        _apply_schema(conn)
        summary["persons"] = _load_persons(conn, SAMPLE_DIR / "persons.csv")
        summary["companies"] = _load_companies(conn, SAMPLE_DIR / "companies.csv")
        n_sanctions, n_sanction_edges = _load_sanctions(conn, SAMPLE_DIR / "sanctions.csv")
        summary["sanctions"] = n_sanctions
        summary["sanction_edges"] = n_sanction_edges
        summary["relationships"] = _load_relationships(conn, SAMPLE_DIR / "relationships.csv")
        n_articles, n_mentions = _load_news(conn, SAMPLE_NEWS_DIR)
        summary["news_articles"] = n_articles
        summary["mention_edges"] = n_mentions
    return summary


def graph_stats() -> dict[str, int]:
    """Cheap node-count summary used by tests and downstream tooling."""
    out = {}
    for label in ("Person", "Company", "Sanction", "NewsArticle"):
        with get_conn() as conn:
            rows = conn.execute(f"MATCH (n:{label}) RETURN count(n) AS n")
            row = rows.get_next() if rows.has_next() else [0]
            out[label.lower()] = int(row[0])
    return out
