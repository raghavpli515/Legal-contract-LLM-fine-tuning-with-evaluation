"""CPU-only checks for the 6GB-GPU embedding swap in modeling.py."""

import pytest

torch = pytest.importorskip("torch")

from legal_ft.modeling import CpuEmbedding, _move_embedding_to_cpu


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.hf_device_map = {"model.embed_tokens": "cpu", "model.layers": 0, "lm_head": 0}

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, module):
        self.embed = module


def test_embedding_swap_keeps_weights_and_lookup():
    model = TinyModel()
    ids = torch.tensor([[1, 2, 3]])
    expected = model.embed(ids)
    _move_embedding_to_cpu(model)
    assert isinstance(model.embed, CpuEmbedding)
    assert torch.equal(model.embed(ids), expected)


def test_stale_cpu_entry_removed_from_device_map():
    """A leftover "cpu" entry made PeftModel.from_pretrained re-dispatch the model as
    offloaded and fail with "We need an `offload_dir`"."""
    model = TinyModel()
    _move_embedding_to_cpu(model)
    assert "cpu" not in model.hf_device_map.values()
    assert model.hf_device_map == {"model.layers": 0, "lm_head": 0}
