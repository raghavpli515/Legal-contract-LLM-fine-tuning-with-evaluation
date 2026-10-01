"""QLoRA supervised fine-tuning with TRL. One code path for Kaggle (7B, T4) and the
local dry run (0.5B, 3050): only the config section differs.

Notes on the TRL 1.x / transformers 5 API, verified against the installed packages:
- ``SFTConfig.max_length`` defaults to 1024 and truncates with keep_start, which would
  cut the answer off our ~1.5k-token QA examples. It is set explicitly and
  ``check_examples`` fails loudly if any example would still be truncated.
- With a prompt/completion dataset and ``completion_only_loss=True``, TRL masks the
  prompt tokens (labels = -100), so loss is computed on the answer only.
- TRL casts trainable LoRA weights of quantized models to bf16. The T4 has no bf16,
  so they are cast back to fp32 (standard QLoRA: fp32 adapters, fp16 autocast).
- ``warmup_ratio`` became ``warmup_steps`` (a float < 1 is a ratio); ``bf16`` defaults
  to auto-detect, so fp16/bf16 are pinned to match the T4 everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from legal_ft.modeling import bnb_config, load_tokenizer

PROMPT_COLUMNS = ("prompt", "completion")


def load_split(path: Path, n: int | None = None):
    from datasets import Dataset

    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:n] if n else rows
    return Dataset.from_list([{k: r[k] for k in PROMPT_COLUMNS} for r in rows])


def sft_config(cfg: dict, output_dir: Path, max_steps: int | None = None):
    from trl import SFTConfig

    t = cfg["training"]
    return SFTConfig(
        output_dir=str(output_dir),
        max_length=t["max_seq_length"],
        completion_only_loss=t["completion_only_loss"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=max_steps or -1,
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_ratio"],
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=2,
        seed=t["seed"],
        report_to="none",
    )


def lora_config(cfg: dict):
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"], bias=lc["bias"], task_type=lc["task_type"],
    )


def build_trainer(cfg: dict, model_id: str, train_ds, eval_ds, output_dir: Path,
                  max_steps: int | None = None):
    from transformers import AutoModelForCausalLM
    from trl import SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config(cfg["quantization"]),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    trainer = SFTTrainer(
        model=model,
        args=sft_config(cfg, output_dir, max_steps),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=load_tokenizer(model_id),
        peft_config=lora_config(cfg),
    )
    for p in trainer.model.parameters():  # undo TRL's bf16 cast of the adapters (T4)
        if p.requires_grad:
            p.data = p.data.float()
    # The model is pinned to GPU 0. On a multi-GPU machine (Kaggle "T4 x2") Trainer would
    # wrap it in nn.DataParallel, which cannot replicate a 4-bit PEFT model (inputs land on
    # cuda:1, weights stay on cuda:0). Same idiom transformers uses for model-parallel models.
    trainer.args._n_gpu = 1
    return trainer


def check_examples(trainer, n: int = 8) -> dict[str, Any]:
    """Fail loudly unless loss is on the answer only and nothing was truncated."""
    tok = trainer.processing_class
    ds = trainer.train_dataset
    max_len = trainer.args.max_length
    lengths = [len(ids) for ids in ds["input_ids"]]
    if max(lengths) >= max_len:
        raise ValueError(f"{sum(x >= max_len for x in lengths)} examples hit max_length={max_len} "
                         "and were truncated; raise max_seq_length or filter the data")
    samples = []
    for i in range(min(n, len(ds))):
        ids, labels = ds[i]["input_ids"], ds[i]["labels"]
        trained = [t for t, lab in zip(ids, labels, strict=True) if lab != -100]
        first = next(j for j, lab in enumerate(labels) if lab != -100)
        if any(lab != -100 for lab in labels[:first]) or labels[-1] == -100:
            raise ValueError(f"example {i}: unexpected label mask layout")
        samples.append({
            "n_tokens": len(ids),
            "n_trained": len(trained),
            "trained_text": tok.decode(trained),
            "prompt_tail": tok.decode(ids[max(0, first - 12):first]),
        })
    return {"max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths),
            "samples": samples}


def train(cfg: dict, model_id: str, train_path: Path, val_path: Path, output_dir: Path,
          n_train: int | None = None, n_val: int | None = None,
          max_steps: int | None = None) -> Path:
    """Train, save the adapter to ``output_dir / "adapter"`` and the log history to JSON."""
    output_dir = Path(output_dir)
    trainer = build_trainer(cfg, model_id, load_split(train_path, n_train),
                            load_split(val_path, n_val), output_dir, max_steps)
    report = check_examples(trainer)
    print(f"Examples OK: max {report['max_tokens']} tokens, loss on answer only. Sample target:"
          f"\n  ...{report['samples'][0]['prompt_tail']!r} -> {report['samples'][0]['trained_text']!r}")
    trainer.model.print_trainable_parameters()
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    (output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return adapter_dir
