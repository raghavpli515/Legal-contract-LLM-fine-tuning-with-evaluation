"""Local dry run of the real training code path: Qwen2.5-0.5B, 20 steps, tiny subset.

Catches data/masking/config/API bugs before spending Kaggle GPU hours, then checks the
saved adapter reloads through legal_ft.modeling and answers a held-out QA prompt.

    python scripts/dry_run_train.py
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time

from legal_ft.config import REPO_ROOT, load_config, load_dotenv

load_dotenv()

import torch

from legal_ft.eval.parse import parse_qa
from legal_ft.modeling import load_model
from legal_ft.train import train

DATA = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "outputs" / "dry_run"


def main() -> int:
    cfg = load_config("train_qlora")
    dry = cfg["dry_run"]
    run_cfg = copy.deepcopy(cfg)
    run_cfg["training"].update(dry["overrides"])

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    adapter = train(run_cfg, dry["model_id"], DATA / "train.jsonl", DATA / "val.jsonl", OUT,
                    n_train=dry["n_train"], n_val=dry["n_val"], max_steps=dry["max_steps"])
    peak = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nTrained {dry['max_steps']} steps in {time.perf_counter() - t0:.0f}s | "
          f"peak GPU {peak:.2f} GB | adapter -> {adapter}")

    history = json.loads((OUT / "log_history.json").read_text(encoding="utf-8"))
    losses = [h["loss"] for h in history if "loss" in h]
    evals = [h["eval_loss"] for h in history if "eval_loss" in h]
    print(f"train loss: first {losses[0]:.3f} -> last {losses[-1]:.3f} | eval loss: {evals}")
    if not all(math.isfinite(x) for x in losses + evals):
        print("FAIL: non-finite loss")
        return 1

    torch.cuda.empty_cache()
    model, tok = load_model(dry["model_id"], adapter=str(adapter))
    with (DATA / "val.jsonl").open(encoding="utf-8") as f:
        item = next(r for r in map(json.loads, f) if r["task"] == "qa")
    text = tok.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    parsed = parse_qa(answer)
    print(f"Reloaded adapter answer (gold present={item['present']}): {answer[:200]!r}")
    print(f"parsed: present={parsed.present} valid_json={parsed.valid}")
    print("PASS" if parsed.present is not None else "WARN: reloaded model output did not parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
