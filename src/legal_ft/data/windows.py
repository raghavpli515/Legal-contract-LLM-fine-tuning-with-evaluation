"""Cut long contracts into fixed-size token windows.

A window is a (char_start, char_end) range of the contract text covering at most
``window_tokens`` tokens. Functions take the contract's token->char offsets (``Tokens``),
so they are tokenizer-agnostic (tests use a whitespace tokenizer; the real build
uses the base model's tokenizer).

For a category, a gold span relates to a window as:
  full    - entirely inside: its text is gold evidence for that window
  partial - cut by a window edge: the (window, category) pair is ambiguous and skipped
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from legal_ft.data.cuad import Span

Window = tuple[int, int]


@dataclass
class Tokens:
    """Token -> (char_start, char_end) offsets for one contract."""

    offsets: list[tuple[int, int]]
    starts: list[int] = field(init=False, repr=False)
    ends: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = [s for s, _ in self.offsets]
        self.ends = [e for _, e in self.offsets]

    def __len__(self) -> int:
        return len(self.offsets)

    @classmethod
    def from_tokenizer(cls, text: str, tokenizer) -> Tokens:
        """``tokenizer`` is a ``tokenizers.Tokenizer``."""
        return cls(list(tokenizer.encode(text, add_special_tokens=False).offsets))

    def chars(self, tok_start: int, tok_end: int) -> Window:
        return self.starts[tok_start], self.ends[tok_end - 1]


def tile(tokens: Tokens, window_tokens: int) -> list[Window]:
    """Non-overlapping windows covering the whole contract."""
    n = len(tokens)
    return [tokens.chars(i, min(i + window_tokens, n)) for i in range(0, n, window_tokens)]


def span_tokens(tokens: Tokens, span: Span) -> tuple[int, int]:
    """[first, last] token indices overlapping the span."""
    first = bisect.bisect_right(tokens.ends, span.start)
    last = bisect.bisect_left(tokens.starts, span.end) - 1
    return first, max(first, last)


def positive_window(
    tokens: Tokens, span: Span, window_tokens: int, rng: random.Random
) -> Window | None:
    """A window fully containing ``span`` at a random position (so the answer is not
    always centred). None if the span alone is longer than the window."""
    first, last = span_tokens(tokens, span)
    if last - first + 1 > window_tokens:
        return None
    lo = max(0, last + 1 - window_tokens)
    hi = min(first, max(0, len(tokens) - window_tokens))
    start = rng.randint(lo, max(lo, hi))
    return tokens.chars(start, min(start + window_tokens, len(tokens)))


def relation(span: Span, window: Window) -> str | None:
    ws, we = window
    if span.end <= ws or span.start >= we:
        return None
    return "full" if ws <= span.start and span.end <= we else "partial"


def evidence_in_window(spans: Iterable[Span], window: Window) -> tuple[list[Span], bool]:
    """(spans fully inside, whether any span is cut by the window edge)."""
    full, cut = [], False
    for span in spans:
        rel = relation(span, window)
        if rel == "full":
            full.append(span)
        elif rel == "partial":
            cut = True
    return full, cut
