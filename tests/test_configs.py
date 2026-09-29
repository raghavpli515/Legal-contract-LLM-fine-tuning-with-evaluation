from legal_ft.config import load_config


def test_split_fractions_sum_to_one():
    split = load_config("data")["split"]
    assert abs(split["train"] + split["val"] + split["test"] - 1.0) < 1e-9


def test_train_config_is_t4_compatible():
    cfg = load_config("train_qlora")
    assert cfg["training"]["bf16"] is False  # T4 has no bf16
    assert cfg["quantization"]["bnb_4bit_compute_dtype"] == "float16"
    assert cfg["training"]["completion_only_loss"] is True


def test_window_fits_training_context():
    window = load_config("data")["windows"]["window_tokens"]
    max_len = load_config("train_qlora")["training"]["max_seq_length"]
    # label list / instructions + answer must fit alongside the excerpt
    assert window + 600 <= max_len
