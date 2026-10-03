"""Render the model card from the scored results and publish the LoRA adapter to the Hub.

    python scripts/push_adapter.py            # dry run: writes outputs/hub_preview/ only
    python scripts/push_adapter.py --push     # creates the repo (if needed) and uploads

Only adapter files are uploaded (no checkpoints or optimizer state). Every number in the
card comes from results/metrics/*.json and the training log, so it cannot drift from the
README's results table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from string import Template

from legal_ft.config import REPO_ROOT, load_config

ADAPTER_DIR = REPO_ROOT / "outputs" / "qlora" / "adapter"
LOG_HISTORY = REPO_ROOT / "outputs" / "qlora" / "log_history.json"
UPLOAD_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja")
GITHUB_URL = "https://github.com/raghavpli515/Legal-contract-LLM-fine-tuning-with-evaluation"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pct_ci(m: dict, key: str) -> str:
    ci = m.get(f"{key}_ci95")
    return _pct(m[key]) + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")


def card_values() -> dict[str, str]:
    metrics = {run: json.loads((REPO_ROOT / "results" / "metrics" / f"{run}.json").read_text())
               for run in ("base", "finetuned")}
    b, f = metrics["base"], metrics["finetuned"]
    history = json.loads(LOG_HISTORY.read_text())
    losses = [h["loss"] for h in history if "loss" in h]
    evals = [h["eval_loss"] for h in history if "eval_loss" in h]
    summary = next(h for h in history if "train_runtime" in h)
    adapter_cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())

    from safetensors import safe_open

    with safe_open(str(ADAPTER_DIR / "adapter_model.safetensors"), "pt") as st:
        n_params = sum(st.get_slice(k).get_shape()[0] * st.get_slice(k).get_shape()[1]
                       for k in st.keys())  # noqa: SIM118 (safe_open is not a dict)
    total_params = 7_655_986_688  # all params incl. adapter, as reported by PEFT in training
    assert adapter_cfg["r"] == 16 and adapter_cfg["lora_alpha"] == 32

    return {
        "repo_url": GITHUB_URL,
        "hub_repo": load_config("train_qlora")["hub"]["adapter_repo"],
        "cls_acc_base": _pct_ci(b["classification"], "accuracy"),
        "cls_acc_ft": _pct_ci(f["classification"], "accuracy"),
        "cls_f1_base": f"{b['classification']['macro_f1']:.3f}",
        "cls_f1_ft": f"{f['classification']['macro_f1']:.3f}",
        "qa_acc_base": _pct_ci(b["qa"], "presence_accuracy"),
        "qa_acc_ft": _pct_ci(f["qa"], "presence_accuracy"),
        "missed_base": _pct(b["qa"]["missed_rate"]),
        "missed_ft": _pct(f["qa"]["missed_rate"]),
        "ece_base": f"{b['qa']['ece']:.3f}",
        "ece_ft": f"{f['qa']['ece']:.3f}",
        "hall_base": _pct_ci(b["qa"], "hallucination_rate"),
        "hall_ft": _pct_ci(f["qa"], "hallucination_rate"),
        "hall_audit_base": _pct_ci(b["qa"], "hallucination_rate_audited"),
        "hall_audit_ft": _pct_ci(f["qa"], "hallucination_rate_audited"),
        "fab_audit_base": _pct(b["qa"]["fabricated_rate_audited"]),
        "fab_audit_ft": _pct(f["qa"]["fabricated_rate_audited"]),
        "trainable_params": f"{n_params / 1e6:.1f}M",
        "trainable_pct": f"{100 * n_params / total_params:.2f}%",
        "steps": str(summary.get("step") or max(h.get("step", 0) for h in history)),
        "train_hours": f"{summary['train_runtime'] / 3600:.1f}",
        "train_loss_first": f"{losses[0]:.3f}",
        "train_loss_last": f"{losses[-1]:.3f}",
        "eval_losses": " → ".join(f"{x:.3f}" for x in evals),
    }


def render_card() -> str:
    template = Template((REPO_ROOT / "hub" / "MODEL_CARD.template.md").read_text(encoding="utf-8"))
    return template.substitute(card_values())  # raises KeyError on any unfilled $placeholder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="actually create the repo and upload")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    missing = [f for f in UPLOAD_FILES if not (ADAPTER_DIR / f).exists()]
    if missing:
        print(f"FAIL: adapter files missing from {ADAPTER_DIR}: {missing}")
        return 1
    card = render_card()
    preview = REPO_ROOT / "outputs" / "hub_preview"
    shutil.rmtree(preview, ignore_errors=True)
    preview.mkdir(parents=True)
    for name in UPLOAD_FILES:
        shutil.copy2(ADAPTER_DIR / name, preview / name)
    (preview / "README.md").write_text(card, encoding="utf-8", newline="\n")
    size_mb = sum(p.stat().st_size for p in preview.iterdir()) / 1e6
    repo_id = load_config("train_qlora")["hub"]["adapter_repo"]
    print(f"Upload preview: {preview} ({len(UPLOAD_FILES) + 1} files, {size_mb:.0f} MB) -> {repo_id}")

    if not args.push:
        print("Dry run: nothing uploaded. Review outputs/hub_preview/README.md, then rerun with --push.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(folder_path=str(preview), repo_id=repo_id, repo_type="model",
                               commit_message="Upload QLoRA adapter and model card")
    print(f"Pushed: https://huggingface.co/{repo_id} ({commit.oid[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
