"""Demo API and answer logic with a fake model (no GPU, no download)."""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import create_app
from legal_ft.inference import (
    DISCLAIMER,
    InvalidRequest,
    build_answer,
    confidence_label,
    load_categories,
)

EXCERPT = ("7.1 The Distributor shall not, during the Term, sell any product that competes "
           "with the Products in the Territory.")


class FakeQA:
    model_name = "fake"

    def __init__(self, output='{"present": true, "evidence": ["shall not, during the Term"]}',
                 p_present=0.93):
        self.output, self.p_present = output, p_present
        self.calls = 0

    @property
    def clause_types(self):
        return load_categories()["labels"]

    def ask(self, excerpt, clause_type):
        self.calls += 1
        if clause_type not in self.clause_types:
            raise InvalidRequest(f"Unknown clause type {clause_type!r}.")
        if len(excerpt.split()) > 50:
            raise InvalidRequest("The excerpt is too long.")
        return build_answer(clause_type, excerpt, self.output, self.p_present)


def client_for(fake):
    return TestClient(create_app(lambda: fake))


def test_health_and_clause_types():
    with client_for(FakeQA()) as c:
        assert c.get("/health").json() == {"status": "ok", "model": "fake"}
        types = c.get("/clause-types").json()["clause_types"]
        assert len(types) == 35 and "Non-Compete" in types and "Parties" not in types


def test_ask_returns_grounded_answer_with_disclaimer():
    with client_for(FakeQA()) as c:
        r = c.post("/ask", json={"excerpt": EXCERPT, "clause_type": "Non-Compete"})
    assert r.status_code == 200
    a = r.json()
    assert a["present"] is True and a["confidence"] == pytest.approx(0.93)
    assert a["confidence_label"] == "high"
    assert a["quotes"] == [{"text": "shall not, during the Term", "verified": True}]
    assert a["all_quotes_verified"] is True
    assert a["disclaimer"] == DISCLAIMER
    assert "raw_output" not in a


def test_unverified_quote_is_flagged():
    fake = FakeQA('{"present": true, "evidence": ["shall not hire any employee"]}', 0.8)
    with client_for(fake) as c:
        a = c.post("/ask", json={"excerpt": EXCERPT, "clause_type": "Non-Compete"}).json()
    assert a["quotes"][0]["verified"] is False and a["all_quotes_verified"] is False


def test_unknown_clause_type_is_422_with_message():
    with client_for(FakeQA()) as c:
        r = c.post("/ask", json={"excerpt": EXCERPT, "clause_type": "Parties"})
    assert r.status_code == 422 and "Unknown clause type" in r.json()["detail"]


def test_too_long_excerpt_is_422_not_truncated():
    with client_for(FakeQA()) as c:
        r = c.post("/ask", json={"excerpt": "word " * 60, "clause_type": "Non-Compete"})
    assert r.status_code == 422 and "too long" in r.json()["detail"]


def test_empty_excerpt_rejected_before_model():
    fake = FakeQA()
    with client_for(fake) as c:
        r = c.post("/ask", json={"excerpt": "", "clause_type": "Non-Compete"})
    assert r.status_code == 422 and fake.calls == 0


def test_absent_answer_confidence_is_one_minus_p():
    a = build_answer("Non-Compete", EXCERPT, '{"present": false, "evidence": []}', 0.12)
    assert a.present is False and a.confidence == pytest.approx(0.88)
    assert a.confidence_label == "medium" and a.quotes == []


def test_unparseable_output_falls_back_to_probability():
    a = build_answer("Non-Compete", EXCERPT, "I am not sure.", 0.3)
    assert a.present is None and a.valid_output is False
    assert a.confidence == pytest.approx(0.7)  # decided "absent" from p < 0.5


def test_confidence_labels():
    assert [confidence_label(x) for x in (0.95, 0.9, 0.8, 0.5)] == ["high", "high", "medium", "low"]


def test_packaged_categories_match_data_build():
    from legal_ft.config import REPO_ROOT

    built = REPO_ROOT / "data" / "processed" / "categories.json"
    if not built.exists():
        pytest.skip("processed data not built")
    packaged = load_categories()
    data = json.loads(built.read_text(encoding="utf-8"))
    assert packaged["labels"] == data["labels"]
    assert all(packaged["descriptions"][k] == data["descriptions"][k] for k in data["labels"])
