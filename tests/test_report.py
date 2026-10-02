"""Report CLI end to end on oracle predictions, plus the generation logprob recorder."""

import json
import shutil

import pytest

from legal_ft.config import REPO_ROOT


@pytest.fixture
def oracle_run():
    data = REPO_ROOT / "data" / "processed"
    if not (data / "eval_qa.jsonl").exists():
        pytest.skip("processed data not built")
    run = "_pytest_oracle"
    pred_dir = REPO_ROOT / "results" / "predictions" / run
    pred_dir.mkdir(parents=True, exist_ok=True)
    for task in ("classification", "qa"):
        with (data / f"eval_{task}.jsonl").open(encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        with (pred_dir / f"{task}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                row = {"id": it["id"], "output": it["completion"][0]["content"]}
                if task == "qa":
                    row["p_present"] = 1.0 if it["present"] else 0.0
                else:
                    row["confidence"] = 1.0
                f.write(json.dumps(row) + "\n")
    yield run
    shutil.rmtree(pred_dir, ignore_errors=True)
    for suffix in (".json", "_rows.jsonl"):
        (REPO_ROOT / "results" / "metrics" / f"{run}{suffix}").unlink(missing_ok=True)


def test_report_on_oracle_predictions(oracle_run):
    from legal_ft.eval.report import comparison_table, score

    result = score(oracle_run, n_boot=20)
    assert result["classification"]["accuracy"] == 1.0
    assert result["qa"]["hallucination_rate"] == 0.0
    table = comparison_table({"oracle": result})
    assert "| Clause classification accuracy ↑ | 100.0%" in table
    assert "| **Hallucination rate** ↓ | 0.0%" in table
    rows = (REPO_ROOT / "results" / "metrics" / f"{oracle_run}_rows.jsonl").read_text().splitlines()
    assert len(rows) == 400


def test_greedy_logprob_recorder_records_chosen_token():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from legal_ft.eval.generate import GreedyLogprobRecorder

    rec = GreedyLogprobRecorder()
    scores = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert rec(None, scores) is scores  # must not alter the scores
    expected = torch.log_softmax(scores, dim=-1).max(dim=-1).values
    assert torch.allclose(rec.steps[0], expected)
