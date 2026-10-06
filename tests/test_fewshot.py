"""Few-shot baseline prompt construction (no GPU)."""

import json

import pytest

from legal_ft.eval.fewshot import CROP_CHARS, FewShot, crop_around
from legal_ft.eval.grounding import is_grounded
from legal_ft.prompts import build_classification_messages, build_qa_messages

LABELS = ["Non-Compete", "Audit Rights"]
FILLER = " ".join(f"word{i}" for i in range(400))  # ~3,000 chars


def qa_row(i, category, quote=None, contract="c-train"):
    excerpt = f"{FILLER} {quote} {FILLER}" if quote else FILLER
    return {"id": f"train-qa-{i:03d}", "task": "qa", "category": category, "contract_id": contract,
            "excerpt": excerpt, "present": bool(quote), "evidence": [quote] if quote else []}


def cls_row(i, label):
    return {"id": f"train-cls-{i:03d}", "task": "classification", "contract_id": "c-train",
            "clause": f"A short clause about {label} number {i}.", "target_label": label}


@pytest.fixture
def fewshot():
    rows = []
    for k, cat in enumerate(LABELS):
        rows += [qa_row(10 * k + j, cat, f"the party shall observe {cat} rule {j}") for j in range(3)]
        rows += [qa_row(10 * k + 5 + j, cat) for j in range(2)]
    rows += [cls_row(i, label) for i, label in enumerate(LABELS * 3)]
    rows.append(cls_row(99, "Third Label"))
    return FewShot(rows, LABELS, {c: f"definition of {c}" for c in LABELS})


def test_crop_contains_quote_and_respects_width():
    quote = "the distributor shall not compete"
    excerpt = f"{FILLER} {quote} {FILLER}"
    crop = crop_around(excerpt, quote)
    assert quote in crop and len(crop) <= CROP_CHARS and crop in excerpt
    assert not crop.startswith(" ") and not crop.endswith(" ")


def test_crop_without_quote_takes_the_start():
    crop = crop_around(FILLER, None)
    assert FILLER.startswith(crop) and len(crop) <= CROP_CHARS


def test_short_excerpt_is_not_cropped():
    assert crop_around("short text", "short") == "short text"


def test_qa_demos_are_same_category_present_absent_present(fewshot):
    item = {"task": "qa", "category": "Audit Rights",
            "prompt": build_qa_messages("REAL EXCERPT", "Audit Rights", "definition of Audit Rights")}
    msgs = fewshot.messages_for(item)
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    answers = [json.loads(m["content"]) for m in msgs if m["role"] == "assistant"]
    assert [a["present"] for a in answers] == [True, False, True]
    demo_users = [m["content"] for m in msgs[1:-1] if m["role"] == "user"]
    assert all('"Audit Rights"' in u for u in demo_users)
    assert msgs[-1]["content"] == item["prompt"][-1]["content"]  # the real question is last, intact


def test_demo_quotes_are_verbatim_in_their_cropped_excerpt(fewshot):
    item = {"task": "qa", "category": "Non-Compete",
            "prompt": build_qa_messages("REAL", "Non-Compete")}
    msgs = fewshot.messages_for(item)
    for user, assistant in zip(msgs[1:-1:2], msgs[2:-1:2], strict=True):
        for quote in json.loads(assistant["content"])["evidence"]:
            assert is_grounded(quote, user["content"])


def test_classification_demos_have_distinct_labels(fewshot):
    item = {"task": "classification", "prompt": build_classification_messages("REAL CLAUSE", LABELS)}
    msgs = fewshot.messages_for(item)
    demo_labels = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert len(demo_labels) == 3 and len(set(demo_labels)) == 3
    assert "REAL CLAUSE" in msgs[-1]["content"]


def test_demonstrations_are_deterministic(fewshot):
    item = {"task": "qa", "category": "Non-Compete", "prompt": build_qa_messages("X", "Non-Compete")}
    assert fewshot.messages_for(item) == fewshot.messages_for(item)


def test_real_demonstrations_never_use_eval_contracts():
    from legal_ft.config import REPO_ROOT
    from legal_ft.eval.fewshot import load_fewshot

    data = REPO_ROOT / "data" / "processed"
    if not (data / "train.jsonl").exists():
        pytest.skip("processed data not built")
    fs = load_fewshot(data)
    eval_contracts = set()
    for name in ("eval_qa.jsonl", "eval_classification.jsonl"):
        with (data / name).open(encoding="utf-8") as f:
            eval_contracts |= {json.loads(line)["contract_id"] for line in f}
    assert fs.demo_contracts and not fs.demo_contracts & eval_contracts
