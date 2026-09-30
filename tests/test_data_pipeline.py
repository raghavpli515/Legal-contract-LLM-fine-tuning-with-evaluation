"""Data pipeline invariants, on synthetic contracts (no download, no tokenizer model)."""

import itertools
import json
import random
import re
from collections import Counter

import pytest

from legal_ft.data.build_sft import (
    QAPositive,
    balanced_sample,
    classification_items,
    free_windows,
    matched_negative,
    qa_positives,
)
from legal_ft.data.cuad import Contract, Span, normalize_whitespace, parse_cuad
from legal_ft.data.split import iterative_split
from legal_ft.data.windows import Tokens, evidence_in_window, positive_window, relation, tile

WINDOW_CFG = {"window_tokens": 20, "max_evidence_chars": 500}
CLS_CFG = {"min_clause_chars": 5, "max_clause_chars": 1000}


def ws_tokens(text: str) -> Tokens:
    """Whitespace 'tokenizer' with char offsets, standing in for the real one."""
    return Tokens([m.span() for m in re.finditer(r"\S+", text)])


def make_contract(cid: str, clauses: dict[str, list[str]], filler_words: int = 30) -> Contract:
    """Filler text with the given clause quotes embedded; spans point at them."""
    parts, spans, pos = [], [], 0

    def add(s: str) -> None:
        nonlocal pos
        parts.append(s)
        pos += len(s)

    for category, quotes in clauses.items():
        for quote in quotes:
            add(" ".join(f"w{i}" for i in range(filler_words)) + " ")
            spans.append(Span(pos, pos + len(quote), category))
            add(quote + " ")
    add(" ".join(f"w{i}" for i in range(filler_words)))
    return Contract(cid, "".join(parts), sorted(spans))


# ---- cuad.py -----------------------------------------------------------------

def test_normalize_whitespace_keeps_span_text_exact():
    raw = "Section 1.\n\n   The   Distributor  shall\tnot compete.\n  End."
    quote = "The   Distributor  shall\tnot compete."
    start = raw.index(quote)
    data = {"data": [{"title": "T", "paragraphs": [{"context": raw, "qas": [{
        "question": 'Highlight the parts (if any) of this contract related to "Non-Compete" '
                    "that should be reviewed by a lawyer. Details: Is there a non-compete?",
        "answers": [{"text": quote, "answer_start": start}],
    }]}]}]}
    (contract,), descriptions = parse_cuad(data)
    assert contract.text == "Section 1.\nThe Distributor shall not compete.\nEnd."
    assert contract.quote(contract.spans[0]) == "The Distributor shall not compete."
    assert descriptions == {"Non-Compete": "Is there a non-compete?"}


def test_normalize_index_map_length():
    text, index_map = normalize_whitespace("a  b\n\nc ")
    assert text == "a b\nc "
    assert len(index_map) == len("a  b\n\nc ") + 1 and index_map[-1] == len(text)


# ---- split.py ----------------------------------------------------------------

def synthetic_labels(n: int = 200, seed: int = 0) -> dict[str, set[str]]:
    rng = random.Random(seed)
    labels = {}
    for i in range(n):
        labs = {c for c in ("common_a", "common_b") if rng.random() < 0.6}
        if i % 20 == 0:
            labs.add("rare")  # 10 items
        labels[f"c{i:03d}"] = labs
    return labels


def test_split_assigns_every_item_exactly_once_with_target_sizes():
    labels = synthetic_labels()
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    split = iterative_split(labels, fractions, seed=1)
    assert set(split) == set(labels)
    sizes = Counter(split.values())
    for name, frac in fractions.items():
        assert abs(sizes[name] - frac * len(labels)) <= 3


def test_split_puts_rare_label_in_every_split():
    labels = synthetic_labels()
    split = iterative_split(labels, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    rare_splits = Counter(split[i] for i, labs in labels.items() if "rare" in labs)
    assert set(rare_splits) == {"train", "val", "test"}


def test_split_is_deterministic():
    labels = synthetic_labels()
    fr = {"train": 0.8, "val": 0.1, "test": 0.1}
    assert iterative_split(labels, fr, 7) == iterative_split(labels, fr, 7)


# ---- windows.py --------------------------------------------------------------

def test_tile_covers_contract_without_overlap():
    contract = make_contract("c", {"X": ["alpha beta"]})
    tokens = ws_tokens(contract.text)
    tiles = tile(tokens, 20)
    assert tiles[0][0] == tokens.starts[0] and tiles[-1][1] == tokens.ends[-1]
    assert all(a[1] < b[0] for a, b in itertools.pairwise(tiles))


@pytest.mark.parametrize("seed", range(20))
def test_positive_window_fully_contains_span(seed):
    contract = make_contract("c", {"X": ["the party shall not compete anywhere"]})
    tokens, span = ws_tokens(contract.text), contract.spans[0]
    window = positive_window(tokens, span, 20, random.Random(seed))
    assert relation(span, window) == "full"
    assert len(re.findall(r"\S+", contract.text[window[0]:window[1]])) <= 20


def test_positive_window_none_when_span_longer_than_window():
    contract = make_contract("c", {"X": [" ".join(["long"] * 25)]})
    tokens = ws_tokens(contract.text)
    assert positive_window(tokens, contract.spans[0], 20, random.Random(0)) is None


def test_evidence_in_window_flags_cut_spans():
    spans = [Span(10, 20, "X"), Span(25, 40, "X")]
    full, cut = evidence_in_window(spans, (5, 30))
    assert full == [spans[0]] and cut


# ---- build_sft.py ------------------------------------------------------------

def test_qa_positive_evidence_is_verbatim_in_window():
    contract = make_contract("c", {"X": ["no competing products", "exclusive rights granted"]},
                             filler_words=3)
    skipped = Counter()
    positives = qa_positives(contract, ws_tokens(contract.text), {"X"}, WINDOW_CFG,
                             random.Random(0), skipped)
    assert positives
    for p in positives:
        excerpt = contract.text[p.window[0]:p.window[1]]
        assert all(q in excerpt for q in p.evidence)


def test_qa_positives_ignore_excluded_categories():
    contract = make_contract("c", {"Parties": ["Acme Corp"], "X": ["no competing"]})
    positives = qa_positives(contract, ws_tokens(contract.text), {"X"}, WINDOW_CFG,
                             random.Random(0), Counter())
    assert {p.category for p in positives} == {"X"}


def test_matched_negative_has_no_span_of_that_category():
    a = make_contract("a", {"X": ["no competing products"], "Y": ["audit rights apply"]})
    b = make_contract("b", {"Y": ["audit rights apply"]})
    contracts = {"a": a, "b": b}
    tiles = {cid: tile(ws_tokens(c.text), 20) for cid, c in contracts.items()}
    pos = QAPositive("a", "X", (0, 1), ["no competing products"])
    for seed in range(20):
        neg = matched_negative(pos, contracts, tiles, ["a", "b"], 0.5, random.Random(seed))
        assert neg["present"] is False and neg["evidence"] == [] and neg["category"] == "X"
        c = contracts[neg["contract_id"]]
        window = tuple(neg["window"])
        assert all(relation(s, window) is None for s in c.spans if s.category == "X")
        assert neg["neg_type"] == ("same_contract" if neg["contract_id"] == "a" else "other_contract")


def test_free_windows_exclude_any_overlap():
    contract = make_contract("a", {"X": ["no competing products"]})
    tiles = tile(ws_tokens(contract.text), 20)
    free = free_windows(contract, tiles, "X")
    assert free and len(free) < len(tiles)


def test_classification_multilabel_target_is_rarest():
    contract = make_contract("a", {"Common": ["license is non-transferable"]})
    span = contract.spans[0]
    contract.spans.append(Span(span.start, span.end, "Rare"))
    rarity = Counter({"Common": 100, "Rare": 3})
    (item,) = classification_items(contract, {"Common", "Rare"}, rarity, CLS_CFG)
    assert item["labels"] == ["Rare", "Common"] and item["target_label"] == "Rare"


def test_balanced_sample_round_robins_groups():
    items = [("big", i) for i in range(100)] + [("small", i) for i in range(5)]
    out = balanced_sample(items, key=lambda x: x[0], n=10, rng=random.Random(0))
    assert Counter(k for k, _ in out) == {"big": 5, "small": 5}


# ---- built files (skipped until `python -m legal_ft.data.build_sft` has run) --

def test_built_files_have_no_contract_leakage():
    from legal_ft.config import REPO_ROOT

    out = REPO_ROOT / "data" / "processed"
    if not (out / "train.jsonl").exists():
        pytest.skip("processed data not built")
    contracts = {}
    for name in ("train", "val", "eval_classification", "eval_qa"):
        with (out / f"{name}.jsonl").open(encoding="utf-8") as f:
            contracts[name] = {json.loads(line)["contract_id"] for line in f}
    evaluation = contracts["eval_classification"] | contracts["eval_qa"]
    assert not contracts["train"] & evaluation
    assert not contracts["train"] & contracts["val"]
    assert not contracts["val"] & evaluation
