"""Load the base model in 4-bit NF4 and optionally attach a LoRA adapter.

Shared by training (Kaggle), eval generation, the smoke test and the demo, so every
environment quantizes the model identically.
"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_QUANT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}


# Input embedding on CPU, everything else on GPU 0. Qwen2.5's 152k-token vocab makes
# embed_tokens ~1.1GB in fp16 (bitsandbytes does not quantize it); on CPU it costs only
# a table lookup per token. Used for the 6GB local GPU; Kaggle/T4 keeps all on GPU.
# Loading with this map only keeps the weights off the GPU at load time: accelerate treats
# "cpu" as *offloaded* and would copy the table to the GPU on every forward (+1.1GB peak,
# ~4x slower decode), so load_model() then swaps in a true CpuEmbedding.
EMBED_ON_CPU_DEVICE_MAP = {
    "model.embed_tokens": "cpu",
    "model.layers": 0,
    "model.norm": 0,
    "model.rotary_emb": 0,
    "lm_head": 0,
}


class CpuEmbedding(torch.nn.Module):
    """Embedding lookup executed on CPU; returns activations on the caller's device."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input_ids.cpu(), self.weight)
        return out.to(input_ids.device)


def _move_embedding_to_cpu(model) -> None:
    emb = model.get_input_embeddings()
    hook = getattr(emb, "_hf_hook", None)
    weight = hook.weights_map["weight"] if hook is not None else emb.weight
    model.set_input_embeddings(CpuEmbedding(weight.detach().to("cpu")))


def bnb_config(quant: dict[str, Any] | None = None, cpu_offload: bool = False):
    from transformers import BitsAndBytesConfig

    q = {**DEFAULT_QUANT, **(quant or {})}
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        llm_int8_enable_fp32_cpu_offload=cpu_offload,  # permits non-quantized modules on CPU
    )


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_id: str,
    adapter: str | None = None,
    quant: dict[str, Any] | None = None,
    device_map: Any = None,
    max_memory: dict | None = None,
    embed_on_cpu: bool = False,
):
    """Return (model, tokenizer). ``adapter`` is a local path or Hub repo id.

    ``device_map`` defaults to the whole model on GPU 0, so an out-of-memory
    failure is loud rather than silently offloading layers to CPU.
    ``embed_on_cpu`` applies EMBED_ON_CPU_DEVICE_MAP (for GPUs with ~5GB free).
    """
    from transformers import AutoModelForCausalLM

    if device_map is None:
        device_map = EMBED_ON_CPU_DEVICE_MAP if embed_on_cpu else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config(quant, cpu_offload=embed_on_cpu),
        device_map=device_map,
        max_memory=max_memory,
        dtype=torch.float16,
    )
    if embed_on_cpu:
        _move_embedding_to_cpu(model)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, load_tokenizer(model_id)
