"""Schema invariants — first line of defense against silent regressions."""

import pytest
from pydantic import ValidationError

from entitynet.schemas import (
    BenchmarkQuestion,
    Company,
    Person,
    RelationshipType,
    Sanction,
    SanctionList,
)


def test_person_required_fields():
    p = Person(id="P001", name="Test")
    assert p.id == "P001"
    assert p.name == "Test"


def test_company_required_fields():
    c = Company(id="C001", name="Acme", country="US")
    assert c.is_shell is False


def test_sanction_uses_enum():
    s = Sanction(
        id="S001",
        list_name=SanctionList.ofac_sdn,
        target_entity_id="P001",
        summary="Test",
    )
    assert s.list_name == SanctionList.ofac_sdn


def test_relationship_enum_completeness():
    expected = {
        "OWNS",
        "DIRECTOR_OF",
        "BENEFICIAL_OWNER_OF",
        "SUBSIDIARY_OF",
        "BUSINESS_PARTNER_OF",
        "RELATIVE_OF",
        "SANCTIONED_BY",
        "MENTIONED_IN",
        "INCORPORATED_IN",
        "BORN_IN",
    }
    assert {r.value for r in RelationshipType} == expected


def test_benchmark_question_validation():
    q = BenchmarkQuestion(
        id="Q001",
        text="Is X sanctioned?",
        category="direct_sanctions",
        hops=1,
        difficulty="easy",
        ground_truth_ids=["P001", "S001"],
    )
    assert q.hops == 1
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            id="Q002",
            text="x",
            category="nonexistent",
            hops=1,
            difficulty="easy",
            ground_truth_ids=[],
        )
