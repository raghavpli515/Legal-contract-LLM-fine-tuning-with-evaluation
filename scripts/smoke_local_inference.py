"""Day-1 check: does the 4-bit base model fit and run on this GPU at demo context length?

Loads the model through legal_ft.modeling (the same path the demo uses), runs one
greedy generation on a synthetic QA prompt of ~--prompt-tokens tokens, and reports
load VRAM, peak VRAM, headroom and decode speed.

    python scripts/smoke_local_inference.py                       # Qwen2.5-7B, ~1.6k-token prompt
    python scripts/smoke_local_inference.py --model Qwen/Qwen2.5-0.5B-Instruct   # plumbing check
"""

from __future__ import annotations

import argparse
import sys
import time

from legal_ft.config import load_dotenv

load_dotenv()  # sets HF_HOME before transformers is imported

import torch

from legal_ft.modeling import load_model
from legal_ft.prompts import build_qa_messages

FILLER_CLAUSE = (
    "The Distributor shall not, during the Term and for a period of two (2) years "
    "thereafter, directly or indirectly manufacture, sell or promote any product that "
    "competes with the Products within the Territory. "
)
MIN_HEADROOM_GB = 0.3


def gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def build_prompt(tokenizer, target_tokens: int) -> str:
    excerpt = FILLER_CLAUSE
    while True:
        messages = build_qa_messages(excerpt, "Non-Compete")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(text).input_ids) >= target_tokens:
            return text
        excerpt += FILLER_CLAUSE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt-tokens", type=int, default=1600)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--embed-on-cpu", action="store_true",
                    help="keep the input embedding table on CPU (saves ~1.1GB VRAM on Qwen2.5-7B)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: torch sees no CUDA GPU (CPU-only torch wheel installed?)")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} | total {gb(total):.2f} GB | "
          f"free before load {gb(free):.2f} GB")

    t0 = time.perf_counter()
    try:
        model, tok = load_model(args.model, embed_on_cpu=args.embed_on_cpu)
    except torch.OutOfMemoryError:
        print("FAIL: OOM while loading weights. Fallback: merged adapter -> GGUF Q4_K_M via llama.cpp.")
        return 1
    load_alloc = torch.cuda.memory_allocated()
    print(f"Loaded {args.model} in {time.perf_counter() - t0:.0f}s | "
          f"weights on GPU: {gb(load_alloc):.2f} GB")

    prompt = build_prompt(tok, args.prompt_tokens)
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    n_in = inputs.input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    except torch.OutOfMemoryError:
        print(f"FAIL: OOM during generation at {n_in} prompt tokens. "
              "Try a shorter --prompt-tokens to find the ceiling.")
        return 1
    elapsed = time.perf_counter() - t0
    n_out = out.shape[1] - n_in

    peak = torch.cuda.max_memory_reserved()
    headroom = free - peak  # vs. memory free at start: the display/other apps hold the rest
    print(f"Prompt tokens: {n_in} | generated: {n_out} | {elapsed:.1f}s "
          f"({n_out / elapsed:.1f} tok/s incl. prefill)")
    print(f"Peak reserved: {gb(peak):.2f} GB | headroom: {gb(headroom):.2f} GB")
    print("Output:", tok.decode(out[0, n_in:], skip_special_tokens=True)[:300])

    if headroom < MIN_HEADROOM_GB * 1024**3:
        print(f"WARN: under {MIN_HEADROOM_GB} GB headroom; the demo may OOM with the display active.")
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
