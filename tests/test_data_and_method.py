from collections.abc import Iterable

import torch

from llm_scratch_lab.data.hf_streaming import PackedTextDataset
from llm_scratch_lab.methods.causal_pretraining import (
    CausalPretrainingConfig,
    CausalPretrainingMethod,
)
from llm_scratch_lab.models.gemma3 import Gemma3Model
from tests.test_gemma3 import tiny_config


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [{"a": 10, "b": 11, "c": 12}[character] for character in text]


def test_packed_dataset_emits_shifted_blocks_and_padded_tail() -> None:
    samples: Iterable[dict[str, object]] = [{"text": "ab"}, {"text": "c"}]
    dataset = PackedTextDataset(
        samples,
        FakeTokenizer(),  # type: ignore[arg-type]
        sequence_length=3,
        drop_last=False,
    )
    blocks = list(dataset)
    assert len(blocks) == 2
    assert blocks[0]["input_ids"].tolist() == [10, 11, 2]
    assert blocks[0]["labels"].tolist() == [11, 2, 12]
    assert blocks[1]["input_ids"].tolist() == [12, 2, 0]
    assert blocks[1]["labels"].tolist() == [2, -100, -100]
    assert blocks[1]["attention_mask"].tolist() == [1, 1, 0]


def test_packed_dataset_state_contains_buffer_and_position() -> None:
    dataset = PackedTextDataset(
        [{"text": "a"}],
        FakeTokenizer(),
        sequence_length=4,
        drop_last=True,  # type: ignore[arg-type]
    )
    assert list(dataset) == []
    state = dataset.state_dict()
    assert state["token_buffer"] == [10, 2]
    assert state["sample_count"] == 1

    restored = PackedTextDataset(
        [],
        FakeTokenizer(),
        sequence_length=4,
        drop_last=True,  # type: ignore[arg-type]
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["token_buffer"] == [10, 2]


def test_causal_pretraining_method_computes_finite_loss() -> None:
    model = Gemma3Model(tiny_config())
    method = CausalPretrainingMethod(CausalPretrainingConfig())
    batch = {
        "input_ids": torch.randint(0, 32, (2, 4)),
        "labels": torch.randint(0, 32, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    output = method.training_step(model, batch)
    assert torch.isfinite(output.loss)
    assert output.item_count == 8
