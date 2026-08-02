import pytest
import torch

from llm_scratch_lab.models.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    RMSNorm,
    apply_rope,
    compute_rope_params,
)


def tiny_config(**updates: object) -> Gemma3Config:
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "hidden_dim": 32,
        "head_dim": 8,
        "n_kv_groups": 1,
        "sliding_window": 4,
        "dtype": "float32",
        "layer_pattern": ["local", "global"],
    }
    values.update(updates)
    return Gemma3Config.model_validate(values)


def test_config_rejects_invalid_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(n_heads=3, n_kv_groups=2)
    with pytest.raises(ValueError, match="even"):
        tiny_config(head_dim=7)


def test_zero_centered_rms_norm_starts_with_unit_scale() -> None:
    layer = RMSNorm(3)
    inputs = torch.tensor([[3.0, 4.0, 0.0]])
    expected = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(layer(inputs), expected)
    assert torch.count_nonzero(layer.weight) == 0


def test_rope_preserves_shape_and_pair_norm() -> None:
    cosine, sine = compute_rope_params(8, 10_000.0, 6)
    inputs = torch.randn(2, 3, 6, 8)
    rotated = apply_rope(inputs, cosine, sine)
    assert rotated.shape == inputs.shape
    torch.testing.assert_close(rotated.square().sum(-1), inputs.square().sum(-1))


def test_model_forward_masks_and_weight_tying() -> None:
    model = Gemma3Model(tiny_config())
    input_ids = torch.randint(0, 32, (2, 6))
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    output = model(input_ids, attention_mask)
    assert output.logits.shape == (2, 6, 32)
    assert model.output_head.weight is model.token_embedding.weight

    global_mask, local_mask = model.create_masks(6, torch.device("cpu"))
    assert bool(global_mask[0, 0, 0, 1])
    assert not bool(global_mask[0, 0, 1, 0])
    assert bool(local_mask[0, 0, 4, 0])


def test_attention_gate_is_an_explicit_variant() -> None:
    base = Gemma3Model(tiny_config(attention_gate="none"))
    gated = Gemma3Model(tiny_config(attention_gate="elementwise"))
    assert base.blocks[0].attention.gate_proj is None
    assert gated.blocks[0].attention.gate_proj is not None
    assert torch.count_nonzero(gated.blocks[0].input_norm.weight) == 0


def test_generate_validates_arguments_and_restores_mode() -> None:
    model = Gemma3Model(tiny_config())
    model.train()
    prompt = torch.tensor([[1, 4]])
    generated = model.generate(
        prompt,
        max_new_tokens=2,
        eos_id=2,
        temperature=0,
        top_k=0,
        top_p=1.0,
    )
    assert generated.shape[1] <= 4
    assert model.training
    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, max_new_tokens=1, eos_id=2, top_p=0)
