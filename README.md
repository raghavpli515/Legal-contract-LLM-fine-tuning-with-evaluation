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

**Adapter on the Hugging Face Hub:** [PimoLee5/qwen2.5-7b-cuad-qlora](https://huggingface.co/PimoLee5/qwen2.5-7b-cuad-qlora) (165 MB LoRA weights; base model loaded separately).

## Results

Qwen2.5-7B-Instruct (4-bit) before and after QLoRA, on 400 held-out items from 50 test
contracts never seen in training. Same quantization, prompts and greedy decoding for both.

| Metric | Base (zero-shot) | Fine-tuned (QLoRA) |
|---|---|---|
| Clause classification accuracy ↑ | 61.5% [55.5, 67.5] | 80.0% [74.5, 85.5] |
| Clause classification macro-F1 ↑ | 0.584 [0.507, 0.636] | 0.795 [0.719, 0.838] |
| Q&A presence accuracy ↑ | 70.5% [64.0, 76.5] | 94.0% [90.5, 97.0] |
| **Hallucination rate** ↓ | 2.0% [0.5, 4.0] | 5.0% [2.5, 8.0] |
|   · after manual audit ↓ | 1.0% [0.0, 2.5] | 3.5% [1.5, 6.5] |
| Fabricated-clause rate ↓ | 2.0% [0.0, 5.1] | 7.0% [2.3, 12.8] |
|   · on hard negatives (same contract) ↓ | 4.1% | 6.1% |
|   · on easy negatives (other contract) ↓ | 0.0% | 7.8% |
|   · after manual audit ↓ | 0.0% [0.0, 0.0] | 5.0% [1.1, 10.0] |
| Ungrounded-quote rate ↓ | 4.7% | 3.0% |
| Missed-clause rate ↓ | 57.0% | 5.0% |
| Evidence token-F1 ↑ | 0.742 | 0.816 |
| Q&A calibration error (ECE) ↓ | 0.283 | 0.026 |
| Q&A Brier score ↓ | 0.286 | 0.055 |
| Q&A confidence AUROC ↑ | 0.818 | 0.823 |
| Classification ECE ↓ | 0.289 | 0.071 |
| Classification format-valid ↑ | 96.0% | 100.0% |
| Q&A format-valid (JSON) ↑ | 99.0% | 99.5% |

n = 200 classification items, 200 Q&A items (held-out test contracts). Brackets: 95% bootstrap CI.

**What changed.** Fine-tuning mostly fixed *recall and calibration*:
- The base model is cautious: it says "not present" for **57%** of clauses that are there,
  usually with near-certainty (51 of its 57 misses had P(present) < 5%). Fine-tuned: **5%**.
- Calibration error drops about **10×** (Q&A ECE 0.283 → 0.026, Brier 0.286 → 0.055), so
  the confidence shown in the demo means something.
- Clause classification: **61.5% → 80.0%** accuracy, macro-F1 0.58 → 0.80, and every
  output is a valid label (96% → 100%).

**What got worse.** The hallucination rate rises from **1.0% to 3.5%** after manual audit
(2.0% → 5.0% raw; the confidence intervals overlap). Having learned to find clauses, the
fine-tuned model sometimes labels a real passage as the wrong clause type, typically on a
surface cue: a "minimum period of 12 months" contract term read as a Minimum Commitment.
Every one of its fabricated-clause answers quotes the contract verbatim, so a reader can
check the quote and see it does not fit; only one answer (0.5%) invents wording.

**Manual audit** ([`results/audit.json`](results/audit.json)). Every flagged hallucination
in both runs (14 items) was checked by hand against CUAD's category definitions:
- 2 "fabricated clauses", flagged identically for both models, are **real clauses CUAD did
  not annotate** (e.g. a second "third party beneficiary" sentence in a recital).
  Hard negatives taken from elsewhere in the same contract carry this label noise.
- 1 "ungrounded quote" is faithful: the model dropped a page number that a PDF page break
  left mid-sentence in CUAD's text.
- The frozen metric is reported unchanged; audited rates are separate rows, and the report
  refuses to compute them unless every flagged item has a verdict.

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
| `hub/`, `scripts/push_adapter.py` | local | model card template (filled from `results/`) and Hub upload |
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

### Training (QLoRA)

TRL `SFTTrainer` on prompt/completion pairs, with loss on the answer tokens only. A Q&A
example is ~1,500 tokens of contract and ~20–300 tokens of answer; training on the whole
sequence would mostly teach the model to reproduce contract text.

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct, 4-bit NF4 + double quantization | Frozen base at ~5 GB fits a free T4; NF4 bins match normally distributed weights |
| LoRA | r=16, alpha=32, dropout 0.05, all linear layers | QLoRA paper: covering every linear layer matters more than rank; alpha/r = 2 |
| Optimizer | paged AdamW 8-bit, lr 2e-4, cosine, 3% warmup, 1 epoch | Standard QLoRA; paging absorbs memory spikes; 1 epoch limits overfitting on 4.5k examples |
| Precision | fp16 autocast, fp32 LoRA weights | The T4 has no bf16 |
| Batch | 1 × 16 gradient accumulation, max length 2,048 | No example is truncated (longest: 1,647 tokens) |

Four library defaults would have broken this silently, and each is pinned and covered by a test
(`tests/test_train_config.py`): TRL's `max_length` defaults to 1024 and would cut answers
off; TRL casts LoRA weights to bf16 for quantized models (cast back to fp32 for the T4);
`bf16` auto-enables on Ampere GPUs, so the local dry run would not match the T4; and an eval
batch of 8 at 2,048 tokens with a 152k vocabulary needs ~10 GB for logits alone.

Before any GPU hours are spent, `scripts/dry_run_train.py` runs the same code path on
Qwen2.5-0.5B for 20 steps locally. It checks that only answer tokens carry loss, that nothing
is truncated, and that the saved adapter reloads and answers in valid JSON.

## Limitations

- **Not legal advice.** A research and portfolio project; outputs can be wrong.
- **Precision/recall trade-off.** Fine-tuning raised recall (95%) at the cost of more
  wrong-clause-type answers (5% of absent cases after audit vs 0% for the base model).
  Training data with more near-miss negatives, or a confidence threshold, would trade some
  of that recall back.
- **Small eval set.** 200 items per task from 50 contracts; rare clause types have 1–7 test
  items, so per-class numbers are noisy (95% bootstrap CIs are reported for headline metrics).
- **Label noise.** CUAD does not always annotate every occurrence of a clause; hard
  negatives inherit those gaps (2 of 14 flagged items in the audit).
- **Excerpt-level, not whole-contract.** Each question sees a ~1,200-token excerpt; the model
  cannot answer "does this contract have X?" across a full agreement without retrieval.
- **Dataset scope.** CUAD: 510 US commercial contracts in English, labelled in 2021.
  Metadata fields (parties, dates) are excluded; Q&A positives whose gold quotes exceed
  1,500 characters were dropped, so very long clauses are untested.
- **One run.** A single training run and seed; no hyperparameter search or rank ablation yet.
- **Exact grounding is strict.** A model that silently fixes a typo, or drops a stray page
  number, is marked ungrounded (seen once in the audit).
