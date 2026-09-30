"""Parse CUADv1.json (SQuAD format, one context per contract) into Contract objects.

Contract text is whitespace-normalized (CUAD has long runs of spaces/newlines that
waste tokens) and every answer span is remapped onto the normalized text, so
``contract.text[span.start:span.end]`` is always the exact gold quote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")
_CATEGORY = re.compile(r'related to "(.+?)"')


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    category: str


@dataclass
class Contract:
    id: str
    text: str
    spans: list[Span] = field(default_factory=list)

    def categories(self) -> set[str]:
        return {s.category for s in self.spans}

    def quote(self, span: Span) -> str:
        return self.text[span.start : span.end]


def normalize_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs (to "\\n" if the run had a newline, else " ").

    Returns (normalized_text, index_map) where index_map[i] is the position in the
    normalized text of original position i (len(index_map) == len(text) + 1).
    """
    parts: list[str] = []
    index_map: list[int] = []
    out_len = prev = 0
    for m in _WS.finditer(text):
        s, e = m.span()
        seg = text[prev:s]
        parts.append(seg)
        index_map.extend(range(out_len, out_len + len(seg)))
        out_len += len(seg)
        parts.append("\n" if "\n" in m.group() else " ")
        index_map.extend([out_len] * (e - s))
        out_len += 1
        prev = e
    seg = text[prev:]
    parts.append(seg)
    index_map.extend(range(out_len, out_len + len(seg)))
    out_len += len(seg)
    index_map.append(out_len)
    return "".join(parts), index_map


def _remap(start: int, end: int, norm: str, index_map: list[int]) -> tuple[int, int]:
    s, e = index_map[start], index_map[end]
    while s < e and norm[s].isspace():
        s += 1
    while e > s and norm[e - 1].isspace():
        e -= 1
    return s, e


def parse_cuad(raw: dict) -> tuple[list[Contract], dict[str, str]]:
    """Return (contracts, category -> CUAD description)."""
    contracts: list[Contract] = []
    descriptions: dict[str, str] = {}
    for doc in raw["data"]:
        para = doc["paragraphs"][0]
        text, index_map = normalize_whitespace(para["context"])
        spans: set[Span] = set()
        for qa in para["qas"]:
            category = _CATEGORY.search(qa["question"]).group(1)
            descriptions.setdefault(category, qa["question"].split("Details:", 1)[1].strip())
            for ans in qa["answers"]:
                start = ans["answer_start"]
                s, e = _remap(start, start + len(ans["text"]), text, index_map)
                if e > s:
                    spans.add(Span(s, e, category))
        contracts.append(Contract(doc["title"], text, sorted(spans)))
    return contracts, descriptions


def load_contracts(json_path: Path) -> tuple[list[Contract], dict[str, str]]:
    with Path(json_path).open(encoding="utf-8") as f:
        return parse_cuad(json.load(f))
