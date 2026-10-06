| Metric | Base (zero-shot) | Base (3-shot) | Fine-tuned (QLoRA) |
|---|---|---|---|
| Clause classification accuracy ↑ | 61.5% [55.5, 67.5] | 63.5% [57.0, 69.5] | 80.0% [74.5, 85.5] |
| Clause classification macro-F1 ↑ | 0.584 [0.507, 0.636] | 0.590 [0.517, 0.657] | 0.795 [0.719, 0.838] |
| Q&A presence accuracy ↑ | 70.5% [64.0, 76.5] | 81.5% [76.0, 86.5] | 94.0% [90.5, 97.0] |
| **Hallucination rate** ↓ | 2.0% [0.5, 4.0] | 4.5% [2.0, 7.5] | 5.0% [2.5, 8.0] |
| **Hallucination rate, after manual audit** ↓ | 1.0% [0.0, 2.5] | 1.0% [0.0, 2.5] | 3.5% [1.5, 6.5] |
| Fabricated-clause rate ↓ | 2.0% [0.0, 5.1] | 3.0% [0.0, 6.9] | 7.0% [2.3, 12.8] |
|   · on hard negatives (same contract) ↓ | 4.1% | 4.1% | 6.1% |
|   · on easy negatives (other contract) ↓ | 0.0% | 2.0% | 7.8% |
| Fabricated-clause rate, after manual audit ↓ | 0.0% [0.0, 0.0] | 1.0% [0.0, 3.3] | 5.0% [1.1, 10.0] |
| Ungrounded-quote rate ↓ | 4.7% | 9.0% | 3.0% |
| Missed-clause rate ↓ | 57.0% | 34.0% | 5.0% |
| Evidence token-F1 ↑ | 0.742 | 0.777 | 0.816 |
| Q&A calibration error (ECE) ↓ | 0.283 | 0.164 | 0.026 |
| Q&A Brier score ↓ | 0.286 | 0.173 | 0.055 |
| Q&A confidence AUROC ↑ | 0.818 | 0.771 | 0.823 |
| Classification ECE ↓ | 0.289 | 0.286 | 0.071 |
| Classification format-valid ↑ | 96.0% | 98.0% | 100.0% |
| Q&A format-valid (JSON) ↑ | 99.0% | 99.0% | 99.5% |

n = 200 classification items, 200 Q&A items (held-out test contracts). Brackets: 95% bootstrap CI.
