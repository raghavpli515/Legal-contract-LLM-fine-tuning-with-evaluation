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
python -m legal_ft.data.build_sft          # downloads CUAD (sha256-checked), writes data/processed/
pytest
python scripts/smoke_local_inference.py --embed-on-cpu   # 7B on a 6GB GPU (see Methodology)
```

## Methodology

### Data

CUAD v1: 510 contracts, 13,823 expert-labelled spans, 41 categories. The six metadata
fields (Document Name, Parties, Agreement/Effective/Expiration Date, Renewal Term) are
extraction rather than clause identification and are excluded, leaving **35 clause types**.

- **Contract-level split (407 / 53 / 50).** Every example inherits its contract's split,
  so no contract text appears in both training and evaluation (enforced by a test).
  The split uses iterative stratification over the clause types each contract contains,
  so rare clauses still reach the test set (Source Code Escrow: 13 contracts total, 1 in test).
- **Windows.** The median contract is ~33k characters (roughly 8k tokens; the longest is 338k), too long for the model. Each Q&A example is a
  1,200-token excerpt (measured with the Qwen tokenizer). Positives place a gold clause at a
  random position in the window; windows that cut a gold span in half are dropped as ambiguous.
- **Matched negatives.** Every positive is paired with a `"present": false` example for the
  same clause type: half from elsewhere in the same contract (hard: the clause exists, just
  not in this excerpt), half from a contract without it. The model cannot score well by
  learning which clause types are usually present.
- **Multi-label spans.** 1,210 spans carry two labels (e.g. License Grant + Non-Transferable
  License). Classification scoring accepts any gold label; the training target is the rarer one.
- **Sampling.** Round-robin across clause types, so frequent clauses do not dominate:
  4,500 training examples (1,500 classification + 1,500 Q&A positives + 1,500 negatives).
  Frozen eval sets from the test contracts: 200 classification, 200 Q&A (100/100 present/absent).

### Local inference on a 6GB GPU

Qwen2.5's 152k-token vocabulary makes the input embedding table 1.1 GB in fp16
(bitsandbytes does not quantize it). `load_model(..., embed_on_cpu=True)` runs that lookup
on the CPU, bringing 4-bit weights to 4.16 GB and peak memory to 4.70 GB at a 1.6k-token
prompt (~23 tok/s decode on an RTX 3050 Laptop). A plain `device_map={"...": "cpu"}` is not
enough: accelerate treats it as *offloaded* and copies the table to the GPU on every forward.

### Evaluation

Generation (GPU) writes one prediction per eval item; scoring (CPU) is deterministic,
uses no LLM judge, and is unit-tested. Base and fine-tuned models get the same frozen
items, 4-bit settings and greedy decoding.

| Metric | Definition |
|---|---|
| Accuracy | Classification: predicted label is any of the item's gold labels |
| Macro-F1 | Per-label F1 averaged over the 35 clause types (95% bootstrap CI reported) |
| **Hallucination rate** | Q&A items where the model invents a clause *or* quotes text not in the excerpt |
| Fabricated-clause rate | Gold absent, model says present; split into hard (same contract) and easy negatives |
| Ungrounded-quote rate | Model says present and at least one quote is not in the excerpt |
| Missed-clause rate | Gold present, model says absent |
| Evidence F1 | Token F1 of quoted evidence vs gold, on correctly detected clauses |
| ECE / Brier / AUROC | Calibration of the model's confidence in its own answer (10 equal-width bins) |
| Format-valid rate | Output parsed exactly as instructed; invalid outputs are scored as wrong, never dropped |

**Grounding is exact, not fuzzy.** A quote counts as grounded only if every word and number
occurs in the excerpt (case, whitespace and punctuation ignored). Fuzzy matching was tested
and rejected: at a 90% similarity threshold, 46/46 gold quotes with one number changed and
46/46 with "shall" turned into "shall not" still passed. The exact rule accepts 148/148 gold
quotes and rejects all of those alterations.

**Harness checks.** Scoring the gold answers as predictions must give perfect scores, and a
model that always claims a clause with an invented quote must score 100% hallucination.
Both run as tests against the real eval files.

_To be written: QLoRA settings and why._

## Limitations

_To be written. Will cover: not legal advice; CUAD scope (US commercial contracts, English, 2021);
excerpt-level rather than whole-contract reasoning; metadata categories excluded from classification;
single training run and seed; calibration measured on the CUAD distribution only; Q&A
positives with gold quotes over 1,500 characters are excluded, so very long clauses are
untested; exact grounding marks a model that silently corrects a contract typo as ungrounded._
