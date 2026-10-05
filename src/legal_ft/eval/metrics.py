"""Score predictions against the frozen eval sets.

Prediction rows (written by eval/generate.py), one per eval item, joined on ``id``:
  classification: {"id", "output", "confidence"}  probability of the generated label
  qa:             {"id", "output", "p_present"}   P("present": true), read at the
                                                  "present" value token
Probabilities may be null (e.g. an API baseline); calibration then skips that row.

Metric definitions (README "Evaluation"):
  accuracy            classification: predicted label is any of the gold labels
  macro_f1            per-label F1 averaged over labels with gold support
  fabricated_rate     QA: gold absent, model says present (invented clause)
  ungrounded_rate     QA: model says present and >=1 quote is not in the excerpt
  hallucination_rate  QA: fabricated OR any ungrounded quote, over all items
  missed_rate         QA: gold present, model says absent
  evidence_f1         QA: token F1 of quotes vs gold, on true positives
  ece / brier         calibration of the model's confidence in its own answer
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from legal_ft.eval.grounding import is_grounded, token_f1
from legal_ft.eval.parse import parse_classification, parse_qa

INVALID = "<invalid>"


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ---- per-item scoring ---------------------------------------------------------

def score_classification_item(item: dict, pred: dict, labels: list[str]) -> dict:
    parsed = parse_classification(pred["output"], labels)
    gold = item["labels"]
    correct = parsed.label in gold
    return {
        "id": item["id"],
        "gold": gold,
        "pred": parsed.label or INVALID,
        # multi-label convention: a correct prediction counts as its own gold label;
        # otherwise the item's primary (rarest) gold label is the reference
        "y_true": parsed.label if correct else item["target_label"],
        "correct": correct,
        "valid": parsed.valid,
        "confidence": pred.get("confidence"),
    }


def score_qa_item(item: dict, pred: dict) -> dict:
    parsed = parse_qa(pred["output"])
    gold_present = item["present"]
    said_present = parsed.present is True
    grounded = [is_grounded(q, item["excerpt"]) for q in parsed.evidence]
    ungrounded = said_present and not all(grounded) if grounded else False
    fabricated = said_present and not gold_present
    tp = said_present and gold_present
    p = pred.get("p_present")
    return {
        "id": item["id"],
        "category": item["category"],
        "neg_type": item.get("neg_type"),
        "gold_present": gold_present,
        "pred_present": parsed.present,
        "correct": parsed.present is gold_present,
        "valid": parsed.valid,
        "fabricated": fabricated,
        "ungrounded": ungrounded,
        "hallucinated": fabricated or ungrounded,
        "n_quotes": len(parsed.evidence),
        "n_ungrounded_quotes": grounded.count(False),
        "evidence_f1": token_f1(" ".join(parsed.evidence), " ".join(item["evidence"]))
        if tp else None,
        # confidence in the model's own decision, and whether that decision was right
        "confidence": None if p is None else max(p, 1 - p),
        "conf_correct": None if p is None else (p > 0.5) == gold_present,
    }


# ---- aggregate metrics --------------------------------------------------------

def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(y_true, y_pred, strict=True):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1
    f1s = []
    for label in sorted(set(y_true)):  # fixed order: float sums identical across runs
        denom = 2 * tp[label] + fp[label] + fn[label]
        f1s.append(2 * tp[label] / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    total = 0.0
    for lo, hi, members in reliability_bins(confidences, correct, n_bins):
        if members:
            acc = sum(c for _, c in members) / len(members)
            conf = sum(p for p, _ in members) / len(members)
            total += len(members) / n * abs(acc - conf)
    return total


def reliability_bins(confidences, correct, n_bins: int = 10):
    """[(lo, hi, [(confidence, correct), ...]), ...] for ECE and reliability plots."""
    bins = [(i / n_bins, (i + 1) / n_bins, []) for i in range(n_bins)]
    for p, c in zip(confidences, correct, strict=True):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx][2].append((p, bool(c)))
    return bins


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    return sum((p - float(c)) ** 2 for p, c in zip(confidences, correct, strict=True)) / len(
        confidences
    ) if confidences else float("nan")


def auroc(confidences: Sequence[float], correct: Sequence[bool]) -> float | None:
    """P(confidence of a correct answer > that of a wrong one). None if one class is empty."""
    pos = [p for p, c in zip(confidences, correct, strict=True) if c]
    neg = [p for p, c in zip(confidences, correct, strict=True) if not c]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def calibration(rows: list[dict], correct_key: str, n_bins: int) -> dict:
    scored = [r for r in rows if r["confidence"] is not None]
    conf = [r["confidence"] for r in scored]
    corr = [bool(r[correct_key]) for r in scored]
    return {
        "n_with_confidence": len(scored),
        "ece": ece(conf, corr, n_bins) if scored else None,
        "brier": brier(conf, corr) if scored else None,
        "auroc": auroc(conf, corr),
        "mean_confidence": _mean(conf),
    }


def classification_metrics(rows: list[dict], n_bins: int = 10) -> dict:
    return {
        "n": len(rows),
        "accuracy": _mean([r["correct"] for r in rows]),
        "macro_f1": macro_f1([r["y_true"] for r in rows], [r["pred"] for r in rows]),
        "format_valid_rate": _mean([r["valid"] for r in rows]),
        **calibration(rows, "correct", n_bins),
    }


def qa_metrics(rows: list[dict], n_bins: int = 10) -> dict:
    absent = [r for r in rows if not r["gold_present"]]
    present = [r for r in rows if r["gold_present"]]
    said_present = [r for r in rows if r["pred_present"] is True]
    with_quotes = [r for r in said_present if r["n_quotes"]]
    tps = [r for r in present if r["pred_present"] is True]
    out = {
        "n": len(rows),
        "presence_accuracy": _mean([r["correct"] for r in rows]),
        "hallucination_rate": _mean([r["hallucinated"] for r in rows]),
        "fabricated_rate": _mean([r["fabricated"] for r in absent]),
        "ungrounded_rate": _mean([r["ungrounded"] for r in with_quotes]),
        "missed_rate": _mean([r["pred_present"] is not True for r in present]),
        "present_without_quote_rate": _mean([r["n_quotes"] == 0 for r in said_present]),
        "evidence_f1": _mean([r["evidence_f1"] for r in tps]),
        "format_valid_rate": _mean([r["valid"] for r in rows]),
        **calibration(rows, "conf_correct", n_bins),
    }
    for neg_type in ("same_contract", "other_contract"):
        subset = [r for r in absent if r["neg_type"] == neg_type]
        out[f"fabricated_rate_{neg_type}"] = _mean([r["fabricated"] for r in subset])
    return out


def bootstrap_ci(
    rows: list[dict], aggregate: Callable[[list[dict]], dict], keys: Sequence[str],
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, tuple[float, float] | None]:
    """Percentile CIs for ``keys`` of ``aggregate``, resampling eval items with replacement."""
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        result = aggregate([rows[rng.randrange(len(rows))] for _ in rows])
        for k in keys:
            if result[k] is not None:
                samples[k].append(result[k])
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = (vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals)) - 1])             if vals else None
    return out


# ---- whole run ----------------------------------------------------------------

CI_METRICS = {
    "classification": ("accuracy", "macro_f1"),
    "qa": ("hallucination_rate", "fabricated_rate", "presence_accuracy"),
}


def score_run(
    items: list[dict], preds: list[dict], task: str, labels: list[str] | None = None,
    n_bins: int = 10, n_boot: int = 1000,
) -> tuple[dict, list[dict]]:
    """Return (metrics incl. 95% CIs, per-item rows). Every eval item must have a prediction."""
    by_id = {p["id"]: p for p in preds}
    missing = [it["id"] for it in items if it["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} eval items have no prediction, e.g. {missing[:3]}")
    if task == "classification":
        rows = [score_classification_item(it, by_id[it["id"]], labels) for it in items]
        agg = lambda rs: classification_metrics(rs, n_bins)
    else:
        rows = [score_qa_item(it, by_id[it["id"]]) for it in items]
        agg = lambda rs: qa_metrics(rs, n_bins)
    metrics = agg(rows)
    for name, ci in bootstrap_ci(rows, agg, CI_METRICS[task], n_boot).items():
        metrics[f"{name}_ci95"] = ci
    return metrics, rows
