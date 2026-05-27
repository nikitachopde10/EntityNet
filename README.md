# EntityNet

**GraphRAG for adverse media and entity-risk screening — with a public,
reproducible benchmark.**

Most \"advanced RAG\" systems today are still flat vector search plus a
reranker. That works for finding a passage about an entity. It does not
work for the questions a compliance team actually asks:

- *Who ultimately owns this company, two or three hops up?*
- *Does this director also sit on the board of a sanctioned entity?*
- *Is my counterparty exposed to a sanctioned person through a business
  partner or family link?*

Vector similarity has no concept of `OWNS`, `DIRECTOR_OF`, or
`MENTIONED_IN`. EntityNet builds a knowledge graph of persons, companies,
sanctions lists, beneficial-ownership chains, and news mentions — and
answers risk questions by **traversing the relationships** that vector
search cannot reach.

The claim is then backed by a public benchmark.

---

## Headline result

50 hand-built risk questions, graded by entity-level F1:

| Retriever              |   F1  | Precision | Recall | Recall@5 | p50 latency |
| ---------------------- | :---: | :-------: | :----: | :------: | :---------: |
| Vector RAG (baseline)  | 0.355 |   0.232   |  0.848 |   0.793  |    12 ms    |
| **GraphRAG**           | **0.681** | **0.649** |  0.880 |   0.829  |    160 ms   |
| Hybrid (rank-fusion)   | 0.627 |   0.497   | **0.968** | **0.938** |  176 ms |

**GraphRAG nearly doubles vector-RAG F1.** On adverse media — the
category vector search *should* dominate — GraphRAG wins by **5.6×**
(0.85 vs 0.15), because traversing a `MENTIONED_IN` edge is more precise
than embedding-similarity to a one-line query.

The benchmark reproduces from a clean clone in under 90 seconds.

---

## What this project demonstrates

- **A graph-native retrieval design** for entity-risk questions: typed
  edges, intent-routed Cypher traversals, and a fuzzy entity linker that
  resolves surface forms (\"V. Petrov\", \"Viktor P.\", \"Petrov, Viktor\") to
  canonical IDs.
- **A reproducible head-to-head evaluation** of vector, graph, and
  hybrid retrievers under identical conditions, on a schema that mirrors
  real-world KYC/AML pipelines.
- **An open evaluation harness** (`src/entitynet/benchmark/`) that any
  third-party retriever can be plugged into and graded against the same
  50 questions in seconds — deterministically, with no API spend.
- **A schema and ingestion path designed to graduate to real data.**
  Swapping the bundled synthetic corpus for OFAC SDN, EU Consolidated,
  OpenSanctions, or OpenCorporates is a wiring task, not a re-modelling
  task (see the `Data` section below).

---

## What it does

1. **Loads** a synthetic-but-realistic entity-risk corpus:
   - 20 companies (incorporated across US, RU, UAE, UK, CY, DE, CH, CN)
   - 20 persons (directors, beneficial owners, sanctioned individuals)
   - 8 sanctions entries (OFAC SDN, EU consolidated, UK HMT, UN)
   - 40+ typed relationships (`OWNS`, `DIRECTOR_OF`,
     `BUSINESS_PARTNER_OF`, `RELATIVE_OF`, `SANCTIONS_TARGET`,
     `MENTIONED_IN`)
   - 15 news articles tagged with the entities they mention
2. **Builds** a Kuzu knowledge graph and a ChromaDB vector index.
3. **Answers** risk questions through three retrievers:
   - `vector` — flat dense retrieval baseline.
   - `graph` — Cypher-style multi-hop traversal.
   - `hybrid` — union of both, with provenance.
4. **Generates** a structured risk report for any entity: direct
   sanctions, indirect exposure via ownership chains, adverse-media
   mentions, a composite risk score.
5. **Benchmarks** all three retrievers on 50 graded questions and prints
   per-category F1, recall, and per-hop accuracy.
6. **Exports** the benchmark as a HuggingFace dataset for public release.

---

## Data

### Source and provenance

> ⚠️ **All data in this repository is fully synthetic.** No real\n> persons, companies, sanctions records, or news articles appear in\n> `data/sample/` or `data/benchmark.jsonl`. Every name, ID, date, and\n> narrative was hand-authored for this project. Any resemblance to real\n> entities is coincidental.

The synthetic corpus was deliberately designed to mirror the *shape*,
*cardinality*, and *edge structure* of real entity-risk pipelines so
that retrieval techniques can be evaluated **without** the legal,
ethical, and reproducibility hazards of redistributing real PII or
sanctions data.

Each file in `data/sample/` is a schema-level mock of a real-world feed:

| File                       | Schema-compatible with                                  |
| -------------------------- | ------------------------------------------------------- |
| `persons.csv`              | OpenCorporates officers, Sayari, internal KYC officer files |
| `companies.csv`            | OpenCorporates, UK Companies House, U.S. SEC EDGAR      |
| `sanctions.csv`            | OFAC SDN, EU Consolidated, UK HMT, UN consolidated list |
| `relationships.csv`        | Beneficial-ownership filings, corporate-registry edges  |
| `news/article_*.json`      | GDELT events, NewsAPI articles, generic adverse-media wires |

The Pydantic schemas in `src/entitynet/schemas.py` use the same fields
real feeds publish — ISO country codes on companies, ISO-8601 dates,
`target_entity_id` on the sanctions table, `program` codes such as
`Russia-EO14024` — so graduating from the synthetic corpus to a live
feed is a CSV-shape change, not a model change.

### Why synthetic for v1

1. **Reproducibility.** A clean clone rebuilds the graph and reproduces
   the benchmark in under 90 seconds, on a laptop, with no API keys.
   Every number in this README is therefore falsifiable.
2. **Legality.** Most real sanctions feeds and corporate registries
   come with terms of use that restrict redistribution. A synthetic
   corpus sidesteps those constraints entirely.
3. **Coverage of edge cases.** Hand-authored data lets the benchmark
   include rare but operationally important patterns — shell companies
   with hidden ultimate beneficial owners (UBOs), sanctioned directors
   with non-sanctioned business partners, family links across
   jurisdictions — that can be sparse or implicit in any single real
   feed.

### Ground truth

The 50 benchmark questions in `data/benchmark.jsonl` were authored
alongside the corpus: every question carries a **canonical set of
entity IDs** that constitute a correct answer. A retriever's prediction
is the unordered set of entity IDs it returns; grading is therefore
**deterministic entity-set F1** — no LLM judge, no human-rater
variance, no API cost, no run-to-run noise.

Each question is tagged with a category (`direct_sanctions`,
`ownership_chain`, `hidden_connection`, `risk_propagation`,
`adverse_media`), a hop count, and a difficulty label, so per-slice
breakdowns are first-class.

---

## Quickstart

Zero API spend. Everything runs locally on CPU.

```bash
# 1. Install (uv recommended)
pip install uv
uv venv --python 3.11
source .venv/bin/activate
make install
make models                # one-time: download spaCy model (~50 MB)

# 2. Build the graph + vector index from bundled sample data
make build                 # ~30 seconds

# 3. Run the benchmark
make benchmark             # ~20 seconds, prints comparison table
```

To ask a single question from the CLI:

```bash
entitynet ask \"Who ultimately owns Delta Energy Solutions?\" --retriever graph
entitynet report C004
```

---

## How it works

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Sample data      │    │ Entity extraction│    │ Kuzu graph DB    │
│ - Sanctions CSV  │──▶│ (spaCy NER +     │──▶│ + ChromaDB       │
│ - Companies CSV  │    │  fuzzy linker)   │    │   vector index   │
│ - News JSONs     │    └──────────────────┘    └────────┬─────────┘
└──────────────────┘                                     │
                                                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Question         │    │ Three retrievers │    │ Entity-F1        │
│ benchmark        │──▶│ vector / graph / │──▶│ scoring + per-   │
│ (50 questions)   │    │ hybrid           │    │ category report  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### Why typed edges matter

The graph schema uses **typed** relationships rather than a generic
`CONNECTED_TO`. That lets the retriever write tight Cypher patterns like
`(a)-[:OWNS*1..3]->(b)` that walk ownership only — not director links,
not news mentions, not family links. Type information is exactly what
similarity search throws away.

### Intent routing

A lightweight keyword-pattern classifier maps each question to one of
seven graph primitives — `ownership_chain`, `shared_directors`,
`family_link`, `business_partner`, `risk_propagation`, `adverse_media`,
`direct_sanctions` — and dispatches to a hand-written, audited Cypher
template. No LLM in the loop, fully deterministic.

---

## The benchmark

50 questions across 5 categories. Each question has a list of canonical
entity IDs as ground truth. A retriever's answer is the set of entities
it returns; we score F1 between that set and the ground truth.

| Category           |  N | Vector F1 | GraphRAG F1 | Hybrid F1 |
| ------------------ | :-: | :-------: | :---------: | :-------: |
| Direct sanctions   | 10 |   0.281   |  **0.677**  |   0.539   |
| Ownership chain    | 12 |   0.467   |  **0.631**  |   0.611   |
| Hidden connection  | 10 |   0.395   |    0.622    | **0.692** |
| Risk propagation   | 10 |   0.417   |    0.667    | **0.687** |
| Adverse media      |  8 |   0.152   |  **0.850**  |   0.608   |

The multi-hop categories are where vector RAG runs out of road: it
cannot synthesise an answer that requires traversing two ownership
links.

Grading is deterministic — entity-set F1, no LLM judge, no API cost.

---

## Repository layout

```
.
├── README.md
├── pyproject.toml                   # Dependencies + extras
├── Makefile                         # One-line commands
├── .env.example                     # Optional API keys
│
├── src/entitynet/
│   ├── config.py                    # Settings + paths
│   ├── schemas.py                   # Pydantic models
│   ├── db.py                        # Kuzu connection helpers
│   ├── ingest/                      # CSV/JSON → graph; news linking
│   ├── extract/                     # spaCy NER + fuzzy entity linker
│   ├── graph/                       # DDL, builder, Cypher templates
│   ├── retrieve/                    # vector / graph / hybrid / report
│   ├── benchmark/                   # runner + entity-F1 metrics
│   └── cli.py                       # Typer CLI
│
├── data/
│   ├── sample/                      # SYNTHETIC corpus (see Data section)
│   │   ├── persons.csv
│   │   ├── companies.csv
│   │   ├── sanctions.csv
│   │   ├── relationships.csv
│   │   └── news/*.json              # 15 articles
│   └── benchmark.jsonl              # 50 graded questions
│
├── scripts/
│   ├── build.py                     # End-to-end build
│   ├── benchmark.py                 # End-to-end benchmark
│   └── export_hf.py                 # Export benchmark to HuggingFace
│
└── tests/
```

---

## Design choices that were deliberately avoided

| Skipped | Why |
| --- | --- |
| Neo4j (needs a server) | Kuzu is embedded, single-file, faster to demo. |
| LangChain GraphRAG modules | Raw Cypher reads as more transparent; nothing to hide. |
| LLM-as-a-judge grading | Entity-F1 is deterministic, free, fully reproducible. |
| Real OpenSanctions ingestion in v1 | Synthetic data lets the demo run in <30 s on a clean clone (see Data). |

---

## Roadmap / future work

- **Real-data ingestion.** Wire OpenSanctions + OpenCorporates feeds
  into `scripts/ingest_*.py` to graduate from synthetic to live data.
  The schema is already field-compatible.
- **Larger benchmark.** Grow the question set past 50 and add
  cross-jurisdictional categories (UBO chains across CY/NL/CH).
- **Adversarial questions.** Add deliberately misleading questions
  (alias collisions, near-duplicates, transliteration variants) to
  stress-test the linker.
- **Interactive dashboard.** An early Streamlit prototype lives in
  `app/streamlit_app.py` for local exploration of the graph and
  side-by-side retriever comparisons. It is not the headline artifact —
  the benchmark is — but the wiring is there. Install with the optional
  extra: `uv pip install -e \".[dashboard]\"` then `make app`.
- **LLM summarisation layer.** Use the graph traversal output as
  grounded context for a small open-weights model, with citations back
  to the traversed paths.

---

## Background and references

Conceptual prior art the design draws on:

- Edge, D. *et al.* (2024). *From Local to Global: A Graph RAG Approach
  to Query-Focused Summarisation.* Microsoft Research.
- Lewis, P. *et al.* (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* NeurIPS.
- Hogan, A. *et al.* (2021). *Knowledge Graphs.* ACM Computing Surveys.

Public sanctions / corporate-registry feeds the schema is
field-compatible with (none of which are redistributed here):

- U.S. Treasury OFAC — Specially Designated Nationals (SDN) List
- European Union — Consolidated Financial Sanctions List
- UK HM Treasury — Consolidated List of Financial Sanctions Targets
- United Nations Security Council — Consolidated List
- [OpenSanctions](https://www.opensanctions.org/) — open-licensed
  re-publication of the above
- [OpenCorporates](https://opencorporates.com/), UK Companies House,
  U.S. SEC EDGAR
- [GDELT](https://www.gdeltproject.org/) — adverse-media events

This project does **not** ingest, redistribute, or otherwise depend on
any copyrighted or access-restricted portion of those feeds.

