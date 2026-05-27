"""Sanity checks on the sample CSVs / news files."""

import csv
import json

from entitynet.config import SAMPLE_DIR, SAMPLE_NEWS_DIR


def _read_csv(name: str) -> list[dict]:
    with open(SAMPLE_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def test_persons_csv():
    rows = _read_csv("persons.csv")
    assert len(rows) >= 20
    assert all(r["id"].startswith("P") for r in rows)


def test_companies_csv():
    rows = _read_csv("companies.csv")
    assert len(rows) >= 20
    assert all(r["id"].startswith("C") for r in rows)


def test_sanctions_csv():
    rows = _read_csv("sanctions.csv")
    assert len(rows) >= 10
    persons = {r["id"] for r in _read_csv("persons.csv")}
    companies = {r["id"] for r in _read_csv("companies.csv")}
    for r in rows:
        tgt = r["target_entity_id"]
        assert tgt in persons or tgt in companies, f"Sanction target {tgt} unknown"


def test_relationships_csv():
    rows = _read_csv("relationships.csv")
    assert len(rows) >= 25
    ent_ids = {r["id"] for r in _read_csv("persons.csv")} | {r["id"] for r in _read_csv("companies.csv")}
    for r in rows:
        assert r["source_id"] in ent_ids, f"Missing src {r['source_id']}"
        assert r["target_id"] in ent_ids, f"Missing tgt {r['target_id']}"


def test_news_articles_parse():
    files = list(SAMPLE_NEWS_DIR.glob("*.json"))
    assert len(files) >= 15
    for f in files:
        data = json.loads(f.read_text())
        assert data["id"].startswith("N")
        assert isinstance(data.get("mentioned_entity_ids"), list)
