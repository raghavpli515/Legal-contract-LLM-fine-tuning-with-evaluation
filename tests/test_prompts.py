import json

from legal_ft.config import load_config
from legal_ft.prompts import (
    CUAD_CATEGORIES,
    build_classification_messages,
    build_qa_messages,
    classification_labels,
    qa_target,
)


def test_cuad_has_41_unique_categories():
    assert len(CUAD_CATEGORIES) == 41
    assert len(set(CUAD_CATEGORIES)) == 41


def test_excluded_categories_exist_and_are_removed():
    excluded = load_config("data")["excluded_categories"]
    assert set(excluded) <= set(CUAD_CATEGORIES)
    labels = classification_labels(excluded)
    assert len(labels) == 41 - len(excluded)
    assert not set(labels) & set(excluded)


def test_classification_prompt_lists_every_label():
    labels = classification_labels(["Parties"])
    user = build_classification_messages("Some clause.", labels)[-1]["content"]
    assert all(f"- {label}" in user for label in labels)
    assert "- Parties" not in user


def test_qa_target_is_valid_json_with_present_first():
    target = qa_target(True, 'He said "no competing".')
    assert json.loads(target) == {"present": True, "evidence": 'He said "no competing".'}
    assert target.startswith('{"present"')


def test_qa_target_absent_drops_evidence():
    assert json.loads(qa_target(False, "stray")) == {"present": False, "evidence": None}


def test_qa_messages_contain_excerpt_and_category():
    msgs = build_qa_messages("EXCERPT TEXT", "Non-Compete")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "EXCERPT TEXT" in msgs[1]["content"]
    assert '"Non-Compete"' in msgs[1]["content"]
