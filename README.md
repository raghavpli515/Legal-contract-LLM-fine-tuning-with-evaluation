# Legal Contract LLM: QLoRA Fine-Tuning with Before/After Evaluation

Fine-tunes **Qwen2.5-7B-Instruct** with **QLoRA** on **CUAD** (510 commercial contracts,
41 expert-annotated clause types) for two tasks:

1. **Clause-type classification**: given a clause, name its type.
2. **Grounded clause Q&A**: given a contract excerpt, say whether a clause type is present
   and quote it verbatim, or answer "not present".

The focus is the evaluation. The base and fine-tuned models are scored on a held-out,
contract-level split for accuracy, **hallucination** (fabricated clauses, ungrounded quotes)
and **calibration**, using identical quantization and decoding.

> **Not legal advice.** This is a research/portfolio project. Outputs may be wrong.

## Results

_Pending: filled in from `results/comparison.md` after milestone 1._

| Metric | Base (zero-shot) | Fine-tuned (QLoRA) |
|---|---|---|
| Classification accuracy | – | – |
| Classification macro-F1 | – | – |
| Fabricated-clause rate ↓ | – | – |
| Ungrounded-quote rate ↓ | – | – |
| Evidence token-F1 | – | – |
| ECE ↓ | – | – |
| Brier score ↓ | – | – |
| Format-valid rate | – | – |

## Repo layout

| Path | Runs on | Purpose |
|---|---|---|
| `src/legal_ft/data/` | local CPU | download, contract-level split, windowing, SFT JSONL |
| `src/legal_ft/prompts.py` | everywhere | single source of prompt templates |
| `src/legal_ft/modeling.py` | GPU | 4-bit NF4 loading + LoRA adapter |
| `notebooks/kaggle_train.ipynb` | Kaggle T4 | thin training wrapper around the package |
| `src/legal_ft/eval/generate.py` | GPU | writes `predictions.jsonl` |
| `src/legal_ft/eval/{parse,grounding,metrics,report}.py` | local CPU | deterministic scoring |
| `app/` | local 3050 | FastAPI + Streamlit demo |
| `configs/` | everywhere | data / training / eval settings |

## Setup (local, Windows)

```powershell
# Python 3.11 venv (matches Kaggle; torch/bitsandbytes wheels are reliable there)
uv venv --python 3.11 .venv
.venv\Scripts\activate
# CUDA torch first, so pip doesn't install the CPU wheel
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[gpu,app,track,dev]"
copy .env.example .env      # set HF_HOME to a drive with ~20GB free
pytest
python scripts/smoke_local_inference.py
```

## Methodology

_To be written: data split, windowing, QLoRA settings and why, eval metric definitions._

## Limitations

_To be written. Will cover: not legal advice; CUAD scope (US commercial contracts, English, 2021);
excerpt-level rather than whole-contract reasoning; metadata categories excluded from classification;
single training run and seed; calibration measured on the CUAD distribution only._
