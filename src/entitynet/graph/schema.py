"""Kuzu DDL — node and relationship tables.

Schema design notes:
- Every entity type gets its own node table. This lets us write efficient
  type-aware queries (e.g. MATCH (p:Person)).
- Relationship tables include a FROM/TO type pair. Kuzu requires explicit
  endpoint types — this is a feature, not a bug; it prevents accidental
  edges between incompatible node types.
- We intentionally use TYPED relationships (OWNS, DIRECTOR_OF, etc.) rather
  than a generic CONNECTED_TO with a `type` property. Typed edges let us
  write tight Cypher patterns like (a)-[:OWNS*1..3]->(b) for ownership
  traversal only.
"""

from __future__ import annotations

NODE_TABLES = [
    """
    CREATE NODE TABLE IF NOT EXISTS Person (
        id            STRING,
        name          STRING,
        nationality   STRING,
        date_of_birth DATE,
        roles         STRING,
        notes         STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Company (
        id                  STRING,
        name                STRING,
        country             STRING,
        industry            STRING,
        incorporation_date  DATE,
        is_shell            BOOLEAN,
        notes               STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Sanction (
        id              STRING,
        list_name       STRING,
        target_entity_id STRING,
        program         STRING,
        issued_date     DATE,
        summary         STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS NewsArticle (
        id              STRING,
        title           STRING,
        body            STRING,
        published_date  DATE,
        source          STRING,
        url             STRING,
        sentiment       STRING,
        risk_tags       STRING,
        PRIMARY KEY (id)
    )
    """,
]

# Relationship tables. Kuzu requires FROM/TO endpoint types.
# We model multi-endpoint relationships (e.g. OWNS can be Person->Company
# OR Company->Company) by registering multiple FROM/TO pairs on one table.
REL_TABLES = [
    "CREATE REL TABLE IF NOT EXISTS OWNS (FROM Person TO Company, FROM Company TO Company, percentage DOUBLE, as_of DATE, source_doc STRING)",
    "CREATE REL TABLE IF NOT EXISTS DIRECTOR_OF (FROM Person TO Company, as_of DATE, source_doc STRING)",
    "CREATE REL TABLE IF NOT EXISTS BUSINESS_PARTNER_OF (FROM Person TO Person, FROM Company TO Company, FROM Person TO Company, FROM Company TO Person, weight DOUBLE, as_of DATE, source_doc STRING)",
    "CREATE REL TABLE IF NOT EXISTS RELATIVE_OF (FROM Person TO Person, weight DOUBLE, source_doc STRING)",
    "CREATE REL TABLE IF NOT EXISTS SANCTIONS_TARGET (FROM Sanction TO Person, FROM Sanction TO Company)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONED_IN (FROM Person TO NewsArticle, FROM Company TO NewsArticle)",
]


def all_ddl() -> list[str]:
    return NODE_TABLES + REL_TABLES
