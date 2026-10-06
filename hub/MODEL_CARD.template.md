---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
datasets:
  - theatticusproject/cuad-qa
tags:
  - lora
  - qlora
  - peft
  - legal
  - contracts
  - cuad
---

# Qwen2.5-7B-Instruct · CUAD contract-clause QLoRA adapter

A LoRA adapter for **Qwen2.5-7B-Instruct** that identifies contract clause types and answers
"does this excerpt contain clause X, and what does it say?" with **verbatim quotes** from the
excerpt. Trained with QLoRA on [CUAD](https://www.atticusprojectai.org/cuad) and evaluated
before vs after on held-out contracts for accuracy, hallucination and calibration.

> **Not legal advice.** A research and portfolio model. It can miss clauses, mislabel them,
> or misquote. Always read the contract.

Code, data pipeline and evaluation: $repo_url

Demo video: $video_url

## Results

400 held-out items from 50 CUAD test contracts never seen in training. All runs use the
same 4-bit quantization, greedy decoding and scorer. Brackets are 95% bootstrap CIs.

| Metric | Base (zero-shot) | Base (3-shot) | This adapter |
|---|---|---|---|
| Clause classification accuracy ↑ | $cls_acc_base | $cls_acc_3shot | **$cls_acc_ft** |
| Clause classification macro-F1 ↑ | $cls_f1_base | $cls_f1_3shot | **$cls_f1_ft** |
| Q&A presence accuracy ↑ | $qa_acc_base | $qa_acc_3shot | **$qa_acc_ft** |
| Missed-clause rate ↓ | $missed_base | $missed_3shot | **$missed_ft** |
| Q&A calibration error (ECE) ↓ | $ece_base | $ece_3shot | **$ece_ft** |
| Hallucination rate, after manual audit ↓ | **$hall_audit_base** | **$hall_audit_3shot** | $hall_audit_ft |
| Hallucination rate, raw ↓ | **$hall_base** | $hall_3shot | $hall_ft |

"3-shot" gives the base model three worked examples of the same clause type in the prompt
(no training). It helps Q&A partway and barely moves classification.

**Trade-off.** Fine-tuning fixed recall and calibration: the base model says "not present"
for most clauses that are there, usually with near-certainty. The cost is more
wrong-clause-type answers on contracts without the clause. Those answers still quote real
contract text, so the quote can be checked. Every flagged hallucination in all three runs was
audited by hand; see the repository's `results/audit.json`.

## Intended use

- Locating and quoting candidate clauses in English commercial-contract excerpts (~1,200
  tokens), for a human to review.
- Studying fine-tuning effects on hallucination and calibration.

**Out of scope:** legal advice or decisions, whole-contract review without retrieval,
non-English or non-commercial contracts, and any use where a missed or mislabelled clause
has consequences without human review.

## How to use

The adapter was trained on a fixed prompt format (system prompt, excerpt, question, JSON
answer). Build prompts with `legal_ft.prompts` from the repository, or reproduce that format
exactly; other phrasings will work less well.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id, adapter_id = "Qwen/Qwen2.5-7B-Instruct", "$hub_repo"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(model, adapter_id)

from legal_ft.prompts import build_qa_messages   # pip install from the GitHub repo
messages = build_qa_messages(excerpt, "Non-Compete")
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(0)
out = model.generate(**inputs, max_new_tokens=384, do_sample=False, repetition_penalty=1.0)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
# {"present": true, "evidence": ["...verbatim quote..."]}
```

Use greedy decoding with `repetition_penalty=1.0`. Qwen's default generation config applies
a 1.05 repetition penalty to every token in the input, which discourages verbatim quoting.
On a 6 GB GPU, see the repository's `load_model(..., embed_on_cpu=True)`.

## Training

| | |
|---|---|
| Data | CUAD v1: 4,500 examples from 407 training contracts (1,500 clause classification, 1,500 Q&A with the clause present, 1,500 matched absent examples). Contract-level split, so no test contract text is seen in training. |
| Method | QLoRA: 4-bit NF4 base with double quantization; LoRA r=16, alpha=32, dropout 0.05 on all linear layers ($trainable_params trainable parameters, $trainable_pct of the model) |
| Loss | On answer tokens only |
| Optimizer | Paged AdamW 8-bit, lr 2e-4, cosine schedule, 3% warmup, 1 epoch ($steps steps, effective batch 16), fp16 |
| Hardware | 1× NVIDIA T4 (Kaggle), $train_hours h |
| Loss curve | Train $train_loss_first → $train_loss_last; held-out eval $eval_losses |

## Limitations

- Raises recall at the cost of precision: more answers that label a real passage as the wrong
  clause type ($fab_audit_ft of absent cases after audit vs $fab_audit_base for the base model).
- Excerpt-level: each answer sees ~1,200 tokens, not a whole contract.
- Small evaluation set (200 items per task); rare clause types have 1–7 test items.
- CUAD covers 510 US commercial contracts in English. It does not always annotate every
  occurrence of a clause, which affects both training and evaluation labels.
- Single training run and seed.

## Citation and licence

Adapter: Apache-2.0, following the base model. Training data: CUAD v1 by The Atticus Project,
CC BY 4.0 (Hendrycks et al., 2021, *CUAD: An Expert-Annotated NLP Dataset for Legal Contract
Review*).
