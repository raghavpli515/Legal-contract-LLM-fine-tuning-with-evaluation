"""Eval harness: parsing, grounding, metrics, and an end-to-end oracle check."""

import json
import math

import pytest

from legal_ft.eval.grounding import is_grounded, token_f1
from legal_ft.eval.metrics import (
    auroc,
    brier,
    ece,
    macro_f1,
    score_run,
)
from legal_ft.eval.parse import parse_classification, parse_qa

LABELS = ["Non-Compete", "License Grant", "Non-Transferable License", "Audit Rights"]
EXCERPT = (
    "7.1 Licensee shall not, during the Term, compete with Licensor in the Territory.\n"
    "7.2 Licensor may audit Licensee's books upon thirty (30) days' notice."
)


# ---- parse ----------------------------------------------------------------------

@pytest.mark.parametrize("output", ["Non-Compete", "  non-compete\n", "Clause type: Non-Compete.",
                                    '"Non-Compete"', "**Non-Compete**"])
def test_classification_strict_formats(output):
    assert parse_classification(output, LABELS).label == "Non-Compete"
    assert parse_classification(output, LABELS).valid


def test_classification_verbose_is_scored_but_invalid():
    pred = parse_classification("This clause is best described as Audit Rights, because...", LABELS)
    assert pred.label == "Audit Rights" and not pred.valid


def test_classification_prefers_longest_contained_label():
    pred = parse_classification("It is a Non-Transferable License clause", LABELS)
    assert pred.label == "Non-Transferable License"


def test_classification_garbage():
    assert parse_classification("I cannot determine this.", LABELS).label is None


def test_qa_valid_json():
    p = parse_qa('{"present": true, "evidence": ["a quote", "another"]}')
    assert (p.present, p.evidence, p.valid) == (True, ["a quote", "another"], True)


def test_qa_fenced_json_with_preamble():
    p = parse_qa('Sure! ```json\n{"present": false, "evidence": []}\n```')
    assert (p.present, p.evidence) == (False, [])


def test_qa_string_evidence_accepted_but_invalid():
    p = parse_qa('{"present": true, "evidence": "single quote"}')
    assert p.evidence == ["single quote"] and not p.valid


def test_qa_truncated_json_recovers_decision_only():
    p = parse_qa('{"present": true, "evidence": ["Licensee shall not compe')
    assert p.present is True and p.evidence == [] and not p.valid


def test_qa_absent_drops_evidence():
    assert parse_qa('{"present": false, "evidence": ["x"]}').evidence == []


def test_qa_unparseable():
    assert parse_qa("The contract does not say.").present is None


# ---- grounding -------------------------------------------------------------------

def test_grounded_exact_and_normalized():
    assert is_grounded("Licensee shall not, during the Term, compete", EXCERPT)
    assert is_grounded("LICENSEE  shall not,\nduring the Term, compete", EXCERPT)
    assert is_grounded("Licensor may audit Licensee’s books", EXCERPT)  # curly apostrophe


def test_grounded_ignores_punctuation_slips():
    assert is_grounded("Licensee shall not during the Term compete with Licensor", EXCERPT)
    assert is_grounded("Licensor may audit “Licensee's” books", EXCERPT)


@pytest.mark.parametrize("altered", [
    "Licensor may audit Licensee's books upon sixty (60) days' notice.",  # number changed
    "Licensee shall, during the Term, compete with Licensor",              # negation dropped
    "Licensee shall not, during the Term, compete with Licensr",           # any word changed
    "0) days' notice",                           # "0" is not a whole number in "(30)"
])
def test_altered_quotes_are_not_grounded(altered):
    assert not is_grounded(altered, EXCERPT)


def test_invented_quote_is_not_grounded():
    assert not is_grounded("Licensee shall not solicit any employees of Licensor for two years",
                           EXCERPT)


def test_ellipsis_fragments_all_must_be_grounded():
    assert is_grounded("Licensee shall not ... compete with Licensor", EXCERPT)
    assert not is_grounded("Licensee shall not ... hire any employee of Licensor ever", EXCERPT)


def test_gold_span_ending_mid_word_is_grounded():
    assert is_grounded("Licensor may audit Licensee's bo", EXCERPT)  # CUAD spans can do this


def test_number_edges_must_be_whole():
    excerpt = "payment due within 160 days of invoice"
    assert not is_grounded("60 days of invoice", excerpt)
    assert not is_grounded("payment due within 16", excerpt)
    assert is_grounded("160 days of invoice", excerpt)


def test_empty_quote_not_grounded():
    assert not is_grounded("  ", EXCERPT)


def test_token_f1():
    assert token_f1("the party shall", "The party shall") == 1.0
    assert token_f1("alpha beta", "gamma delta") == 0.0
    assert 0 < token_f1("the party shall not", "the party") < 1


# ---- aggregate metrics -----------------------------------------------------------

def test_ece_perfectly_calibrated_is_zero():
    assert ece([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)


def test_ece_overconfident():
    assert ece([0.9] * 10, [False] * 10) == pytest.approx(0.9)


def test_brier():
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert brier([0.5, 0.5], [True, False]) == 0.25


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.9], [True, False]) == 0.0
    assert auroc([0.5, 0.6], [True, True]) is None


def test_macro_f1():
    assert macro_f1(["a", "a", "b"], ["a", "a", "b"]) == 1.0
    # a: tp1 fn1 -> 2/3; b: tp1 fp1 -> 2/3
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)


# ---- score_run on synthetic items ------------------------------------------------

def qa_item(i, present, evidence, neg_type=None):
    return {"id": f"q{i}", "category": "Non-Compete", "excerpt": EXCERPT, "present": present,
            "evidence": evidence, "neg_type": neg_type}


QUOTE = "Licensee shall not, during the Term, compete with Licensor in the Territory."


def test_score_qa_counts_each_hallucination_type():
    items = [
        qa_item(0, True, [QUOTE]),
        qa_item(1, True, [QUOTE]),
        qa_item(2, False, [], "same_contract"),
        qa_item(3, False, [], "other_contract"),
    ]
    preds = [
        {"id": "q0", "output": json.dumps({"present": True, "evidence": [QUOTE]}), "p_present": 0.9},
        # correct decision, invented quote -> ungrounded
        {"id": "q1", "output": '{"present": true, "evidence": ["Licensee may never sell shoes."]}',
         "p_present": 0.7},
        # clause invented on a hard negative -> fabricated
        {"id": "q2", "output": json.dumps({"present": True, "evidence": [QUOTE]}), "p_present": 0.6},
        {"id": "q3", "output": '{"present": false, "evidence": []}', "p_present": 0.2},
    ]
    m, rows = score_run(items, preds, "qa", n_boot=50)
    assert m["presence_accuracy"] == 0.75
    assert m["fabricated_rate"] == 0.5
    assert m["fabricated_rate_same_contract"] == 1.0 and m["fabricated_rate_other_contract"] == 0.0
    assert m["ungrounded_rate"] == pytest.approx(1 / 3)
    assert m["hallucination_rate"] == 0.5  # q1 and q2
    assert m["missed_rate"] == 0.0
    f1 = {r["id"]: r["evidence_f1"] for r in rows}
    assert f1["q0"] == 1.0 and f1["q1"] < 0.2  # q1 shares only "licensee" with gold
    assert f1["q2"] is None and f1["q3"] is None  # evidence F1 only on true positives
    lo, hi = m["hallucination_rate_ci95"]
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_score_run_requires_every_prediction():
    with pytest.raises(ValueError, match="no prediction"):
        score_run([qa_item(0, True, [QUOTE])], [], "qa")


def test_score_classification_multilabel():
    items = [
        {"id": "c0", "labels": ["Non-Transferable License", "License Grant"],
         "target_label": "Non-Transferable License"},
        {"id": "c1", "labels": ["Audit Rights"], "target_label": "Audit Rights"},
    ]
    preds = [{"id": "c0", "output": "License Grant", "confidence": 0.9},
             {"id": "c1", "output": "gibberish", "confidence": 0.3}]
    m, _ = score_run(items, preds, "classification", labels=LABELS, n_boot=50)
    assert m["accuracy"] == 0.5 and m["format_valid_rate"] == 0.5


# ---- oracle / adversary on the real frozen eval sets -----------------------------

def load_eval(name):
    from legal_ft.config import REPO_ROOT

    path = REPO_ROOT / "data" / "processed" / f"{name}.jsonl"
    if not path.exists():
        pytest.skip("processed data not built")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_gold_answers_score_perfectly():
    """If scoring the gold completions is not perfect, the harness (not a model) is wrong."""
    qa = load_eval("eval_qa")
    preds = [{"id": r["id"], "output": r["completion"][0]["content"],
              "p_present": 1.0 if r["present"] else 0.0} for r in qa]
    m, _ = score_run(qa, preds, "qa", n_boot=20)
    assert m["presence_accuracy"] == 1.0 and m["hallucination_rate"] == 0.0
    assert m["ungrounded_rate"] == 0.0 and m["evidence_f1"] == pytest.approx(1.0)
    assert m["format_valid_rate"] == 1.0 and m["ece"] == pytest.approx(0.0)

    from legal_ft.config import REPO_ROOT

    cls = load_eval("eval_classification")
    labels = json.loads((REPO_ROOT / "data/processed/categories.json").read_text())["labels"]
    preds = [{"id": r["id"], "output": r["completion"][0]["content"], "confidence": 1.0}
             for r in cls]
    m, _ = score_run(cls, preds, "classification", labels=labels, n_boot=20)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["format_valid_rate"] == 1.0


def test_always_present_with_invented_quote_is_caught():
    qa = load_eval("eval_qa")
    fake = '{"present": true, "evidence": ["The parties agree to a perpetual worldwide moratorium."]}'
    m, _ = score_run(qa, [{"id": r["id"], "output": fake, "p_present": 0.99} for r in qa],
                     "qa", n_boot=20)
    assert m["fabricated_rate"] == 1.0 and m["hallucination_rate"] == 1.0
    assert m["ungrounded_rate"] == 1.0
    assert not math.isnan(m["ece"]) and m["ece"] > 0.4
