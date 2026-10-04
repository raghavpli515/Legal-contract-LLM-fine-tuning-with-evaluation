"""Demo inference: contract excerpt + clause type -> grounded answer with confidence.

Uses the same prompt builder, greedy decoding and P(present) read-out as the evaluation
(eval/generate.py), so the confidence shown in the demo is the quantity whose calibration
was measured (Q&A ECE 0.026 for the fine-tuned adapter). Each quote is checked against the
excerpt with the evaluation's exact grounding rule.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from importlib import resources

from legal_ft.eval.grounding import is_grounded
from legal_ft.eval.parse import parse_qa
from legal_ft.prompts import build_qa_messages

DEFAULT_ADAPTER = "PimoLee5/qwen2.5-7b-cuad-qlora"
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
# Training excerpts were 1,200 tokens; a little slack, still well inside the 2,048 context.
MAX_EXCERPT_TOKENS = 1300
MAX_NEW_TOKENS = 384
DISCLAIMER = ("Not legal advice. This is a research model: it can miss clauses, mislabel them "
              "or misquote. Always read the contract and consult a qualified lawyer.")


class InvalidRequest(ValueError):
    """Bad input (unknown clause type, empty or over-long excerpt); safe to show the user."""


def load_categories() -> dict:
    """{"labels": [...35 clause types...], "descriptions": {label: CUAD definition}}."""
    text = resources.files("legal_ft").joinpath("categories.json").read_text(encoding="utf-8")
    return json.loads(text)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "low"


@dataclass
class Quote:
    text: str
    verified: bool  # occurs verbatim in the excerpt (case/punctuation/whitespace-insensitive)


@dataclass
class Answer:
    clause_type: str
    present: bool | None          # None if the output could not be parsed
    confidence: float             # probability the stated answer is right
    confidence_label: str
    p_present: float              # P("present": true), the calibrated quantity
    quotes: list[Quote] = field(default_factory=list)
    all_quotes_verified: bool = True
    valid_output: bool = True     # model followed the JSON format
    raw_output: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def build_answer(clause_type: str, excerpt: str, raw_output: str, p_present: float,
                 latency_s: float = 0.0) -> Answer:
    """Turn a raw generation + P(present) into the demo answer. Pure function (tested)."""
    parsed = parse_qa(raw_output)
    present = parsed.present
    decided = present if present is not None else p_present > 0.5
    confidence = p_present if decided else 1 - p_present
    quotes = [Quote(q, is_grounded(q, excerpt)) for q in parsed.evidence]
    return Answer(
        clause_type=clause_type,
        present=present,
        confidence=round(confidence, 4),
        confidence_label=confidence_label(confidence),
        p_present=round(p_present, 4),
        quotes=quotes,
        all_quotes_verified=all(q.verified for q in quotes),
        valid_output=parsed.valid,
        raw_output=raw_output,
        latency_s=round(latency_s, 2),
    )


class ClauseQA:
    """Loads the 4-bit base model + LoRA adapter once and answers one question at a time."""

    def __init__(self, adapter: str | None = DEFAULT_ADAPTER, base_model: str = DEFAULT_BASE,
                 embed_on_cpu: bool = True):
        from legal_ft.eval.generate import single_token_id
        from legal_ft.modeling import load_model

        self.model, self.tokenizer = load_model(base_model, adapter=adapter,
                                                embed_on_cpu=embed_on_cpu)
        self.model_name = f"{base_model} + {adapter}" if adapter else base_model
        self.categories = load_categories()
        self._ids = (single_token_id(self.tokenizer, " true"),
                     single_token_id(self.tokenizer, " false"))
        self._lock = threading.Lock()  # one generation at a time on a single small GPU

    @property
    def clause_types(self) -> list[str]:
        return self.categories["labels"]

    def validate(self, excerpt: str, clause_type: str) -> None:
        if clause_type not in self.categories["descriptions"]:
            raise InvalidRequest(f"Unknown clause type {clause_type!r}.")
        if not excerpt.strip():
            raise InvalidRequest("The excerpt is empty.")
        n = len(self.tokenizer(excerpt, add_special_tokens=False).input_ids)
        if n > MAX_EXCERPT_TOKENS:
            raise InvalidRequest(
                f"The excerpt is {n} tokens; the model was trained on excerpts up to "
                f"~{MAX_EXCERPT_TOKENS}. Paste a shorter section (roughly 3-4 pages of text max).")

    def ask(self, excerpt: str, clause_type: str) -> Answer:
        from legal_ft.eval.generate import chat_text, generate_batch, p_present_batch

        self.validate(excerpt, clause_type)
        messages = build_qa_messages(excerpt, clause_type,
                                     self.categories["descriptions"][clause_type])
        text = chat_text(self.tokenizer, messages)
        with self._lock:
            t0 = time.perf_counter()
            gen = generate_batch(self.model, self.tokenizer, [text], MAX_NEW_TOKENS)[0]
            p = p_present_batch(self.model, self.tokenizer, [text], *self._ids)[0]
            latency = time.perf_counter() - t0
        return build_answer(clause_type, excerpt, gen["output"], p, latency)
