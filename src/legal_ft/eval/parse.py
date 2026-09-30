"""Parse raw model outputs into structured predictions.

Nothing is silently dropped: every output yields a prediction plus a ``valid`` flag.
Invalid outputs are scored as wrong and reported as the format-valid rate, so a model
cannot look better by failing to answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_PRESENT = re.compile(r'"present"\s*:\s*(true|false)', re.IGNORECASE)
_LABEL_PREFIX = re.compile(r"^(clause type|answer|label)\s*:\s*", re.IGNORECASE)


@dataclass
class ClassificationPred:
    label: str | None
    valid: bool  # output was exactly one label name (after trivial cleanup)


@dataclass
class QAPred:
    present: bool | None
    evidence: list[str] = field(default_factory=list)
    valid: bool = False  # output was well-formed JSON matching the schema


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_classification(output: str, labels: list[str]) -> ClassificationPred:
    """Strict: the cleaned output equals a label. Lenient fallback: the longest label
    name contained in the output (so verbose-but-correct answers are still scored,
    while ``valid`` records that the format was not followed)."""
    cleaned = _LABEL_PREFIX.sub("", output.strip()).strip().strip("\"'`*.").strip()
    by_norm = {_norm_label(lab): lab for lab in labels}
    if _norm_label(cleaned) in by_norm:
        return ClassificationPred(by_norm[_norm_label(cleaned)], True)
    text = _norm_label(output)
    found = [lab for norm, lab in by_norm.items() if norm in text]
    return ClassificationPred(max(found, key=len) if found else None, False)


def _first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_qa(output: str) -> QAPred:
    text = _FENCE.sub("", output.strip())
    obj = _first_json_object(text)
    if obj is not None and isinstance(obj.get("present"), bool):
        ev = obj.get("evidence")
        if ev is None:
            evidence, ok = [], True
        elif isinstance(ev, str):
            evidence, ok = [ev], False  # schema says list
        elif isinstance(ev, list) and all(isinstance(q, str) for q in ev):
            evidence, ok = list(ev), True
        else:
            evidence, ok = [], False
        evidence = [q.strip() for q in evidence if q.strip()]
        return QAPred(obj["present"], evidence if obj["present"] else [], ok)
    m = _PRESENT.search(text)  # truncated/malformed JSON: recover the decision only
    return QAPred(m.group(1).lower() == "true" if m else None, [], False)
