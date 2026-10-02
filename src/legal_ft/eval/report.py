"""Score prediction runs and write the before/after comparison table.

    python -m legal_ft.eval.report base finetuned

Reads results/predictions/<run>/{classification,qa}.jsonl, writes
results/metrics/<run>.json (metrics) and <run>_rows.jsonl (per-item scores, for error
analysis), and results/comparison.md with one column per run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_ft.config import REPO_ROOT, load_config
from legal_ft.eval.metrics import score_run

# (task, metric key, label, kind) - kind: "pct" or "num"; arrows mark the better direction
TABLE = [
    ("classification", "accuracy", "Clause classification accuracy ↑", "pct"),
    ("classification", "macro_f1", "Clause classification macro-F1 ↑", "num"),
    ("qa", "presence_accuracy", "Q&A presence accuracy ↑", "pct"),
    ("qa", "hallucination_rate", "**Hallucination rate** ↓", "pct"),
    ("qa", "fabricated_rate", "Fabricated-clause rate ↓", "pct"),
    ("qa", "fabricated_rate_same_contract", "  · on hard negatives (same contract) ↓", "pct"),
    ("qa", "fabricated_rate_other_contract", "  · on easy negatives (other contract) ↓", "pct"),
    ("qa", "ungrounded_rate", "Ungrounded-quote rate ↓", "pct"),
    ("qa", "missed_rate", "Missed-clause rate ↓", "pct"),
    ("qa", "evidence_f1", "Evidence token-F1 ↑", "num"),
    ("qa", "ece", "Q&A calibration error (ECE) ↓", "num"),
    ("qa", "brier", "Q&A Brier score ↓", "num"),
    ("qa", "auroc", "Q&A confidence AUROC ↑", "num"),
    ("classification", "ece", "Classification ECE ↓", "num"),
    ("classification", "format_valid_rate", "Classification format-valid ↑", "pct"),
    ("qa", "format_valid_rate", "Q&A format-valid (JSON) ↑", "pct"),
]


def _fmt(value, kind: str, ci=None) -> str:
    if value is None:
        return "–"
    if kind == "pct":
        text = f"{100 * value:.1f}%"
        return text + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")
    return f"{value:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def score(run: str, n_boot: int = 1000) -> dict:
    ecfg, dcfg = load_config("eval"), load_config("data")
    data_dir = REPO_ROOT / dcfg["output_dir"]
    pred_dir = REPO_ROOT / ecfg["outputs"]["predictions_dir"] / run
    labels = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))["labels"]
    n_bins = ecfg["scoring"]["ece_bins"]

    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, all_rows = {}, []
    for task, items_file in (("classification", "eval_classification.jsonl"),
                             ("qa", "eval_qa.jsonl")):
        items = load_jsonl(data_dir / items_file)
        preds = load_jsonl(pred_dir / f"{task}.jsonl")
        metrics, rows = score_run(items, preds, task, labels=labels, n_bins=n_bins, n_boot=n_boot)
        result[task] = metrics
        all_rows += [{"task": task, **r} for r in rows]
    meta_path = pred_dir / "meta.json"
    result["meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    (out_dir / f"{run}.json").write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    with (out_dir / f"{run}_rows.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result


def comparison_table(results: dict[str, dict]) -> str:
    runs = list(results)
    lines = ["| Metric | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for task, key, label, kind in TABLE:
        cells = []
        for run in runs:
            m = results[run][task]
            cells.append(_fmt(m.get(key), kind, m.get(f"{key}_ci95")))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    n = {t: results[runs[0]][t]["n"] for t in ("classification", "qa")}
    lines.append("")
    lines.append(f"n = {n['classification']} classification items, {n['qa']} Q&A items "
                 "(held-out test contracts). Brackets: 95% bootstrap CI.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="prediction run names, e.g. base finetuned")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    results = {run: score(run, args.n_boot) for run in args.runs}
    table = comparison_table(results)
    report_path = REPO_ROOT / load_config("eval")["outputs"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(table + "\n", encoding="utf-8", newline="\n")
    print(table)
    print(f"\n-> {report_path}")


if __name__ == "__main__":
    main()
