"""Training settings that TRL/transformers would otherwise default silently (no GPU needed)."""

import pytest

pytest.importorskip("trl")

from legal_ft.config import load_config
from legal_ft.train import lora_config, sft_config


@pytest.fixture(scope="module")
def args(tmp_path_factory):
    return sft_config(load_config("train_qlora"), tmp_path_factory.mktemp("out"))


def test_max_length_not_default_1024(args):
    assert args.max_length == 2048  # TRL default 1024 would truncate the answers


def test_loss_on_completion_only(args):
    assert args.completion_only_loss is True


def test_precision_pinned_for_t4(args):
    assert args.fp16 is True and args.bf16 is False


def test_eval_batch_fits_t4(args):
    assert args.per_device_eval_batch_size == 1


def test_warmup_is_a_ratio(args):
    assert args.warmup_steps == pytest.approx(0.03)


def test_lora_targets_all_linear_layers():
    lc = lora_config(load_config("train_qlora"))
    assert (lc.r, lc.lora_alpha, lc.target_modules) == (16, 32, "all-linear")
