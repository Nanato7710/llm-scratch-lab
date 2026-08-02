from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_scratch_lab.core.contracts import ModelOutput
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry

AttentionType = Literal["local", "global"]
AttentionGate = Literal["none", "elementwise"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class Gemma3Config(BaseModel):
    """Configuration for the educational, text-only Gemma 3-style decoder."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(gt=0)
    context_length: int = Field(gt=0)
    emb_dim: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    n_kv_groups: int = Field(default=1, gt=0)
    sliding_window: int = Field(default=512, gt=0)
    rope_local_base: float = Field(default=10_000.0, gt=0)
    rope_global_base: float = Field(default=1_000_000.0, gt=0)
    query_pre_attn_scalar: int | None = Field(default=None, gt=0)
    qk_norm: bool = True
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    layer_pattern: list[AttentionType] = Field(min_length=1)
    attention_gate: AttentionGate = "none"

    @model_validator(mode="after")
    def validate_architecture(self) -> Gemma3Config:
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError("n_heads must be divisible by n_kv_groups")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError("n_layers must be divisible by the layer_pattern length")
        if self.sliding_window > self.context_length:
            raise ValueError("sliding_window must not exceed context_length")
        return self

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    @property
    def layer_types(self) -> list[AttentionType]:
        repeats = self.n_layers // len(self.layer_pattern)
        return self.layer_pattern * repeats


class FeedForward(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        dtype = config.torch_dtype
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, bias=False, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(inputs), approximate="tanh")
        return self.down_proj(gate * self.up_proj(inputs))


class RMSNorm(nn.Module):
    """Gemma-style zero-centered RMSNorm using ``1 + weight``."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        values = inputs.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


def compute_rope_params(
    head_dim: int,
    theta_base: float,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inverse_frequencies = 1.0 / (theta_base ** (dimensions / head_dim))
    positions = torch.arange(context_length, dtype=torch.float32)
    angles = positions[:, None] * inverse_frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    sequence_length = inputs.shape[-2]
    head_dim = inputs.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    first, second = inputs.split(head_dim // 2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cosine = cosine[:sequence_length][None, None, :, :]
    sine = sine[:sequence_length][None, None, :, :]
    return (inputs * cosine + rotated * sine).to(inputs.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        dtype = config.torch_dtype
        self.num_heads = config.n_heads
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups
        self.head_dim = config.head_dim
        self.output_size = config.n_heads * config.head_dim
        self.scaling = (config.query_pre_attn_scalar or config.head_dim) ** -0.5

        self.query_proj = nn.Linear(config.emb_dim, self.output_size, bias=False, dtype=dtype)
        self.key_proj = nn.Linear(
            config.emb_dim,
            config.n_kv_groups * config.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.value_proj = nn.Linear(
            config.emb_dim,
            config.n_kv_groups * config.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.gate_proj = (
            nn.Linear(config.emb_dim, self.output_size, bias=False, dtype=dtype)
            if config.attention_gate == "elementwise"
            else None
        )
        self.output_proj = nn.Linear(self.output_size, config.emb_dim, bias=False, dtype=dtype)
        self.query_norm = RMSNorm(config.head_dim) if config.qk_norm else None
        self.key_norm = RMSNorm(config.head_dim) if config.qk_norm else None

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        queries = self.query_proj(inputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        keys = self.key_proj(inputs).view(
            batch_size, sequence_length, self.num_kv_groups, self.head_dim
        )
        values = self.value_proj(inputs).view(
            batch_size, sequence_length, self.num_kv_groups, self.head_dim
        )
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.query_norm is not None:
            queries = self.query_norm(queries)
        if self.key_norm is not None:
            keys = self.key_norm(keys)

        queries = apply_rope(queries, cosine, sine)
        keys = apply_rope(keys, cosine, sine)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        scores = (queries * self.scaling) @ keys.transpose(-2, -1)
        weights = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
        context = (
            (weights @ values)
            .transpose(1, 2)
            .reshape(batch_size, sequence_length, self.output_size)
        )
        if self.gate_proj is not None:
            context = context * torch.sigmoid(self.gate_proj(inputs))
        return self.output_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, config: Gemma3Config, attention_type: AttentionType) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.input_norm = RMSNorm(config.emb_dim)
        self.post_attention_norm = RMSNorm(config.emb_dim)
        self.pre_feed_forward_norm = RMSNorm(config.emb_dim)
        self.post_feed_forward_norm = RMSNorm(config.emb_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        global_mask: torch.Tensor,
        local_mask: torch.Tensor,
        global_rope: tuple[torch.Tensor, torch.Tensor],
        local_rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mask = local_mask if self.attention_type == "local" else global_mask
        cosine, sine = local_rope if self.attention_type == "local" else global_rope
        attention = self.attention(self.input_norm(inputs), mask, cosine, sine)
        hidden = inputs + self.post_attention_norm(attention)
        feed_forward = self.feed_forward(self.pre_feed_forward_norm(hidden))
        return hidden + self.post_feed_forward_norm(feed_forward)


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.emb_dim, dtype=config.torch_dtype
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, attention_type) for attention_type in config.layer_types
        )
        self.final_norm = RMSNorm(config.emb_dim)
        self.output_head = nn.Linear(
            config.emb_dim, config.vocab_size, bias=False, dtype=config.torch_dtype
        )
        self.output_head.weight = self.token_embedding.weight

        local_cosine, local_sine = compute_rope_params(
            config.head_dim, config.rope_local_base, config.context_length
        )
        global_cosine, global_sine = compute_rope_params(
            config.head_dim, config.rope_global_base, config.context_length
        )
        self.register_buffer("local_cosine", local_cosine, persistent=False)
        self.register_buffer("local_sine", local_sine, persistent=False)
        self.register_buffer("global_cosine", global_cosine, persistent=False)
        self.register_buffer("global_sine", global_sine, persistent=False)

    def create_masks(
        self,
        sequence_length: int,
        device: torch.device,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_length > self.config.context_length:
            raise ValueError("sequence length exceeds configured context_length")
        ones = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
        global_mask = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.config.sliding_window).T
        local_mask = global_mask | far_past
        global_mask = global_mask[None, None, :, :]
        local_mask = local_mask[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.shape[-1] != sequence_length:
                raise ValueError("attention_mask must match input sequence length")
            padding_mask = ~attention_mask.to(device=device, dtype=torch.bool)
            padding_mask = padding_mask[:, None, None, :]
            global_mask = global_mask | padding_mask
            local_mask = local_mask | padding_mask
        return global_mask, local_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        global_mask, local_mask = self.create_masks(
            sequence_length, input_ids.device, attention_mask
        )
        hidden = self.token_embedding(input_ids) * (self.config.emb_dim**0.5)
        for block in self.blocks:
            hidden = block(
                hidden,
                global_mask,
                local_mask,
                (self.global_cosine, self.global_sine),
                (self.local_cosine, self.local_sine),
            )
        logits = self.output_head(self.final_norm(hidden).to(self.config.torch_dtype))
        return ModelOutput(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, non-empty sequence)")
        if not 0 <= eos_id < self.config.vocab_size:
            raise ValueError("eos_id must be within the vocabulary")
        if max_new_tokens < 0 or top_k < 0:
            raise ValueError("max_new_tokens and top_k must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")

        was_training = self.training
        self.eval()
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        try:
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    conditioned = input_ids[:, -self.config.context_length :]
                    logits = self(conditioned).logits[:, -1].float()
                    if repetition_penalty > 1.0:
                        for row_index, row in enumerate(input_ids):
                            token_ids = torch.unique(row)
                            seen_logits = logits[row_index, token_ids]
                            logits[row_index, token_ids] = torch.where(
                                seen_logits < 0,
                                seen_logits * repetition_penalty,
                                seen_logits / repetition_penalty,
                            )
                    if temperature <= 0:
                        next_tokens = logits.argmax(dim=-1, keepdim=True)
                    else:
                        logits /= temperature
                        if 0 < top_k < logits.shape[-1]:
                            cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                            logits = logits.masked_fill(logits < cutoff, -torch.inf)
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
                            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                            remove = cumulative > top_p
                            remove[:, 1:] = remove[:, :-1].clone()
                            remove[:, 0] = False
                            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                            logits = torch.full_like(logits, -torch.inf)
                            logits.scatter_(1, sorted_indices, sorted_logits)
                        next_tokens = torch.multinomial(logits.softmax(dim=-1), 1)
                    next_tokens = torch.where(
                        finished[:, None], torch.full_like(next_tokens, eos_id), next_tokens
                    )
                    input_ids = torch.cat((input_ids, next_tokens), dim=1)
                    finished |= next_tokens[:, 0] == eos_id
                    if bool(finished.all()):
                        break
        finally:
            self.train(was_training)
        return input_ids


def _build_model(config: BaseModel, context: BuildContext) -> Gemma3Model:
    del context
    if not isinstance(config, Gemma3Config):
        raise TypeError("Expected Gemma3Config")
    return Gemma3Model(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "model",
        "gemma3",
        Gemma3Config,
        _build_model,
        description="Educational text-only Gemma 3-style decoder",
    )


__all__ = [
    "Gemma3Config",
    "Gemma3Model",
    "GroupedQueryAttention",
    "RMSNorm",
    "apply_rope",
    "compute_rope_params",
]
