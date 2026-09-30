"""Is a quoted evidence string actually in the excerpt?

A quote is grounded iff, after normalization, it occurs in the excerpt as a substring.
Normalization ignores case, whitespace, punctuation and quote/dash styles, so harmless
copy slips (a dropped comma, curly vs straight quotes) pass, but every word and number
must match. A quote may start or end mid-word (CUAD's own gold spans sometimes do), but
a number at either edge must match whole, so "60 days" is not found in "160 days".
Quotes using "..." to skip text are split and every fragment must be grounded.
Deterministic: no LLM judge.

Fuzzy matching was rejected on measurement: at rapidfuzz partial_ratio >= 90, 46/46
gold quotes with one number changed and 46/46 with "shall" -> "shall not" still counted
as grounded (see tests/test_eval.py). The cost of exactness: a model that "fixes" a
typo in the contract is marked ungrounded, a conservative error.
"""

from __future__ import annotations

import re

_TRANSLATE = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", " ": " ",
})
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")
_NON_WORD = re.compile(r"[^0-9a-z]+")
MIN_FRAGMENT_CHARS = 3


def normalize(text: str) -> str:
    """Lowercase alphanumeric words separated by single spaces."""
    return _NON_WORD.sub(" ", text.translate(_TRANSLATE).casefold()).strip()


def _fragment_in(fragment: str, haystack: str) -> bool:
    left = r"(?<![0-9])" if fragment[0].isdigit() else ""
    right = r"(?![0-9])" if fragment[-1].isdigit() else ""
    return re.search(left + re.escape(fragment) + right, haystack) is not None


def is_grounded(quote: str, excerpt: str) -> bool:
    haystack = normalize(excerpt)
    fragments = [f for f in (normalize(p) for p in _ELLIPSIS.split(quote))
                 if len(f) >= MIN_FRAGMENT_CHARS]
    if not fragments:
        return False
    return all(_fragment_in(f, haystack) for f in fragments)


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style bag-of-tokens F1 between predicted and gold evidence."""
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common: dict[str, int] = {}
    for tok in p:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in g:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)
