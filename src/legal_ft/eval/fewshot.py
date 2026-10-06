"""Few-shot prompting baseline: worked examples placed in the prompt, no training.

Answers "was fine-tuning needed, or would better prompting do?". Demonstrations come only
from the training split, so no test-contract text can leak into a prompt.

  qa              three demonstrations of the SAME clause type as the question, in the order
                  present / absent / present, so the model sees what that clause looks like
                  and what "not present" looks like
  classification  three fixed demonstrations (clause -> label) with distinct labels

Demonstration excerpts are cropped to ~1,000 characters (~250 tokens) around the quoted
clause: three full 1,200-token excerpts plus the real one would not fit a 6 GB GPU.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from legal_ft.prompts import (
    Message,
    build_classification_messages,
    build_qa_messages,
    classification_target,
    qa_target,
)

CROP_CHARS = 1000
MAX_QUOTE_CHARS = 450
MAX_DEMO_CLAUSE_CHARS = 400
QA_PATTERN = (True, False, True)  # present / absent / present


def crop_around(excerpt: str, quote: str | None, width: int = CROP_CHARS) -> str:
    """A ~``width``-character slice of ``excerpt`` that fully contains ``quote`` (or the
    start of the excerpt if there is no quote), cut at whitespace."""
    if len(excerpt) <= width:
        return excerpt
    if quote:
        idx = excerpt.index(quote)
        start = max(0, min(idx - (width - len(quote)) // 2, len(excerpt) - width))
    else:
        start = 0
    end = min(len(excerpt), start + width)
    if start > 0:  # move forward to a word start, but never past the quote
        nxt = excerpt.find(" ", start)
        if nxt != -1 and (not quote or nxt + 1 <= excerpt.index(quote)):
            start = nxt + 1
    if end < len(excerpt):  # move back to a word end, but never before the quote ends
        prev = excerpt.rfind(" ", start, end)
        if prev != -1 and (not quote or prev >= excerpt.index(quote) + len(quote)):
            end = prev
    return excerpt[start:end]


def _turns(user: Message, answer: str) -> list[Message]:
    return [user, {"role": "assistant", "content": answer}]


class FewShot:
    """Builds deterministic demonstrations from the training pool."""

    def __init__(self, train_rows: list[dict], labels: list[str], descriptions: dict[str, str],
                 n_shots: int = 3, seed: int = 0):
        if n_shots != len(QA_PATTERN):
            raise ValueError(f"only {len(QA_PATTERN)}-shot is implemented")
        self.labels, self.descriptions, self.seed = labels, descriptions, seed
        self.demo_contracts: set[str] = set()
        self._qa: dict[str, list[Message]] = {}
        qa_rows = sorted((r for r in train_rows if r["task"] == "qa"), key=lambda r: r["id"])
        for category in labels:
            self._qa[category] = self._qa_demos(category, qa_rows)
        cls_rows = sorted((r for r in train_rows if r["task"] == "classification"),
                          key=lambda r: r["id"])
        self._classification = self._classification_demos(cls_rows)

    def _qa_demos(self, category: str, qa_rows: list[dict]) -> list[Message]:
        rng = random.Random(f"{self.seed}-{category}")
        positives = [r for r in qa_rows if r["category"] == category and r["present"]
                     and len(r["evidence"]) == 1 and len(r["evidence"][0]) <= MAX_QUOTE_CHARS]
        negatives = [r for r in qa_rows if r["category"] == category and not r["present"]]
        if len(positives) < 2 or not negatives:
            raise ValueError(f"not enough training demonstrations for {category!r}")
        pos, neg = iter(rng.sample(positives, 2)), iter(rng.sample(negatives, 1))
        messages: list[Message] = []
        for want_present in QA_PATTERN:
            row = next(pos) if want_present else next(neg)
            quote = row["evidence"][0] if want_present else None
            crop = crop_around(row["excerpt"], quote)
            user = build_qa_messages(crop, category, self.descriptions.get(category, ""))[1]
            messages += _turns(user, qa_target(want_present, [quote] if quote else []))
            self.demo_contracts.add(row["contract_id"])
        return messages

    def _classification_demos(self, cls_rows: list[dict]) -> list[Message]:
        rng = random.Random(f"{self.seed}-classification")
        short = [r for r in cls_rows if len(r["clause"]) <= MAX_DEMO_CLAUSE_CHARS]
        rng.shuffle(short)
        messages: list[Message] = []
        seen: set[str] = set()
        for row in short:
            if row["target_label"] in seen:
                continue
            seen.add(row["target_label"])
            user = build_classification_messages(row["clause"], self.labels)[1]
            messages += _turns(user, classification_target(row["target_label"]))
            self.demo_contracts.add(row["contract_id"])
            if len(seen) == len(QA_PATTERN):
                break
        return messages

    def messages_for(self, item: dict) -> list[Message]:
        """The item's prompt with demonstrations inserted after the system message."""
        system, *rest = item["prompt"]
        demos = self._qa[item["category"]] if item["task"] == "qa" else self._classification
        return [system, *demos, *rest]


def load_fewshot(data_dir: Path, n_shots: int = 3, seed: int = 0) -> FewShot:
    from legal_ft.inference import load_categories

    with (Path(data_dir) / "train.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    cats = load_categories()
    return FewShot(rows, cats["labels"], cats["descriptions"], n_shots, seed)
