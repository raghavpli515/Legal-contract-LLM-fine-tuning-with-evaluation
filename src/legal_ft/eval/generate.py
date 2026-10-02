"""GPU step: run a model (base, or base + LoRA adapter) over the frozen eval sets and write
one prediction per item. Everything after this (parse, grounding, metrics) is CPU-only.

    python -m legal_ft.eval.generate --run-name base --embed-on-cpu
    python -m legal_ft.eval.generate --run-name finetuned --adapter outputs/qlora/adapter --embed-on-cpu

Writes results/predictions/<run-name>/{classification,qa}.jsonl + meta.json. Rows are
appended as they finish, so an interrupted run resumes where it stopped.

Decoding is explicitly greedy with repetition_penalty=1.0. Qwen2.5 ships a generation config
with sampling (T=0.7, top-p 0.8, top-k 20) and repetition_penalty=1.05; HF applies that
penalty to every token already in the input, including the contract excerpt, so it would
push the model away from verbatim quoting and distort the grounding metric.

Confidence:
  qa              p_present = P(" true") / (P(" true") + P(" false")) after the forced prefix
                  '{"present":' (single tokens in Qwen's vocab). Read the same way for both
                  models regardless of what they go on to generate.
  classification  probability of the generated output sequence (product of greedy token
                  probabilities up to and including end-of-turn).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from legal_ft.config import REPO_ROOT, load_config, load_dotenv

GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
          "repetition_penalty": 1.0}
PRESENT_PREFIX = '{"present":'
TASK_FILES = {"classification": "eval_classification.jsonl", "qa": "eval_qa.jsonl"}


class GreedyLogprobRecorder(LogitsProcessor):
    """Records log P(chosen token) at each greedy step without keeping full-vocab scores
    (output_scores would hold steps x batch x 152k floats). Must be the last processor and
    decoding must be greedy, so the chosen token is the argmax of the scores seen here."""

    def __init__(self):
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.cpu())
        return scores


def chat_text(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode(tokenizer, texts: list[str]):
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to("cuda:0") for k, v in enc.items()}


@torch.inference_mode()
def generate_batch(model, tokenizer, texts: list[str], max_new_tokens: int) -> list[dict]:
    enc = _encode(tokenizer, texts)
    recorder = GreedyLogprobRecorder()
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, logits_processor=LogitsProcessorList([recorder]),
        pad_token_id=tokenizer.pad_token_id, **GREEDY,
    )
    gen = out[:, enc["input_ids"].shape[1]:].cpu()
    logprobs = torch.stack(recorder.steps, dim=1) if recorder.steps else torch.zeros(len(texts), 0)
    eos_ids = set(model.generation_config.eos_token_id or [tokenizer.eos_token_id])
    results = []
    for i in range(len(texts)):
        ids = gen[i].tolist()
        n = next((j + 1 for j, t in enumerate(ids) if t in eos_ids), len(ids))
        seq_logprob = float(logprobs[i, :n].sum())
        results.append({
            "output": tokenizer.decode(ids[:n], skip_special_tokens=True),
            "n_new_tokens": n,
            "hit_max_new_tokens": n == max_new_tokens and ids[n - 1] not in eos_ids,
            "seq_logprob": seq_logprob,
        })
    return results


@torch.inference_mode()
def p_present_batch(model, tokenizer, texts: list[str], true_id: int, false_id: int) -> list[float]:
    enc = _encode(tokenizer, [t + PRESENT_PREFIX for t in texts])
    # left padding: positions must count real tokens only, as generate() does
    position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    logits = model(**enc, position_ids=position_ids, logits_to_keep=1).logits[:, -1, :]
    pair = torch.softmax(logits[:, [true_id, false_id]].float(), dim=-1)
    return pair[:, 0].tolist()


def single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token ({ids}); p_present needs a new prefix")
    return ids[0]


def run_task(model, tokenizer, task: str, items: list[dict], out_path: Path,
             batch_size: int, max_new_tokens: int) -> None:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [it for it in items if it["id"] not in done]
    # similar lengths per batch -> less padding
    todo.sort(key=lambda it: it["n_tokens"])
    true_id, false_id = single_token_id(tokenizer, " true"), single_token_id(tokenizer, " false")
    print(f"[{task}] {len(done)} already done, {len(todo)} to go (batch {batch_size})", flush=True)

    t0 = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for b in range(0, len(todo), batch_size):
            batch = todo[b:b + batch_size]
            texts = [chat_text(tokenizer, it["prompt"]) for it in batch]
            t_batch = time.perf_counter()
            gens = generate_batch(model, tokenizer, texts, max_new_tokens)
            probs = p_present_batch(model, tokenizer, texts, true_id, false_id) if task == "qa" \
                else [None] * len(batch)
            latency = (time.perf_counter() - t_batch) / len(batch)
            for it, g, p in zip(batch, gens, probs, strict=True):
                row = {"id": it["id"], **g, "latency_s": round(latency, 3)}
                if task == "qa":
                    row["p_present"] = p
                else:
                    row["confidence"] = math.exp(g["seq_logprob"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_done = b + len(batch)
            if n_done % (batch_size * 10) < batch_size or n_done == len(todo):
                rate = (time.perf_counter() - t0) / n_done
                print(f"[{task}] {n_done}/{len(todo)} | {rate:.1f}s/item | "
                      f"ETA {rate * (len(todo) - n_done) / 60:.0f} min", flush=True)


def main() -> None:
    load_dotenv()
    from legal_ft.modeling import load_model

    ecfg, tcfg = load_config("eval"), load_config("train_qlora")
    gcfg = ecfg["generation"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="e.g. base, finetuned")
    ap.add_argument("--model", default=tcfg["model_id"])
    ap.add_argument("--adapter", default=None, help="local path or Hub repo id of the LoRA adapter")
    ap.add_argument("--tasks", nargs="+", default=["classification", "qa"], choices=list(TASK_FILES))
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (smoke test)")
    ap.add_argument("--embed-on-cpu", action="store_true", help="6GB GPUs (see modeling.py)")
    args = ap.parse_args()

    data_dir = REPO_ROOT / load_config("data")["output_dir"]
    out_dir = REPO_ROOT / ecfg["outputs"]["predictions_dir"] / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, adapter=args.adapter, embed_on_cpu=args.embed_on_cpu)
    tokenizer.padding_side = "left"

    import bitsandbytes
    import peft
    import transformers
    meta = {
        "run_name": args.run_name, "model": args.model, "adapter": args.adapter,
        "embed_on_cpu": args.embed_on_cpu, "decoding": GREEDY, "limit": args.limit,
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")

    for task in args.tasks:
        with (data_dir / TASK_FILES[task]).open(encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        items = items[: args.limit] if args.limit else items
        run_task(model, tokenizer, task, items, out_dir / f"{task}.jsonl",
                 batch_size=gcfg[f"batch_size_{task}"],
                 max_new_tokens=gcfg[f"max_new_tokens_{task}"])
    print(f"predictions -> {out_dir}")


if __name__ == "__main__":
    main()
