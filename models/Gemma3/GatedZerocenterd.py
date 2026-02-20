# 下記のリンク先のコードを元に，Gemma3のベースとなるクラスを実装する．
# https://github.com/rasbt/LLMs-from-scratch/blob/7b1f740f74cbeb9e1c1c24ee19ab6e1729209240/ch05/12_gemma3/standalone-gemma3.ipynb

from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer

import torch
import torch.nn as nn



attention_types = Literal["sliding_attention", "full_attention"]

def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")

def _str_to_dtype(s: str) -> torch.dtype:
    s = s.replace("torch.", "")
    if not hasattr(torch, s):
        raise ValueError(f"Unknown torch dtype: {s}")
    return getattr(torch, s)

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    head_dim: int
    qk_norm: bool = True
    n_kv_groups: int = 1
    rope_local_base: float = 10_000.0
    rope_base: float = 1_000_000.0
    sliding_window: int = 512
    dtype: torch.dtype = torch.bfloat16
    query_pre_attn_scalar: int = 128
    layer_types: list[attention_types]

    @field_validator("dtype", mode="before")
    @classmethod
    def _parse_dtype(cls, v):
        if isinstance(v, torch.dtype):
            return v
        if isinstance(v, str):
            return _str_to_dtype(v)
        raise TypeError(f"dtype must be torch.dtype or str, got {type(v)}")

    @field_serializer("dtype", when_used="json")
    def _serialize_dtype(self, v: torch.dtype):
        return _dtype_to_str(v)


class FeedForward(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False, dtype=cfg.dtype)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.activation(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Zerocenterdのためにzerosからonesに変更
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale)
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(input_dtype)


def compute_rope_params(head_dim: int, theta_base: float, context_length: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # 今回のRoPEの実装は入力の前半を実部，後半を虚部とみなして実装されている．

    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # RoFormerの論文における3.2.2節の式(15)と式(16)の間にあるθの計算
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:(head_dim//2)].float() / head_dim)) # (head_dim // 2)

    # 角度mθ_iの計算
    positions = torch.arange(context_length, dtype=dtype) # (context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # (context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=-1) # (context_length, head_dim)

    cos = torch.cos(angles) # (context_length, head_dim)
    sin = torch.sin(angles) # (context_length, head_dim)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, num_heads, seq_length, head_dim)
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # RoPEの実装は入力の前半を実部，後半を虚部とみなして実装されている．そのため，xを実部と虚部に分割する．
    re = x[..., :head_dim // 2] # (batch_size, num_heads, seq_length, head_dim // 2)
    im = x[..., head_dim // 2:] # (batch_size, num_heads, seq_length, head_dim // 2)

    # 回転行列の作成．RoFormerの論文における3.2.2節の式(15)．また，その適用
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_length,)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_length,)
    # [re * cos - im * sin, re * sin + im * cos]に相当する計算．RoFormerの論文における3.2.1節の式(13)の適用．
    # rotated = torch.cat((re * cos - im * sin, re * sin + im * cos), dim=-1) # (batch_size, num_heads, seq_length, head_dim)
    rotated = torch.cat((-im, re), dim=-1)
    x_rotated = x * cos + rotated * sin
    return x_rotated.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in: int, num_heads: int, num_kv_groups: int, head_dim: int|None=None, qk_norm: bool=True,
        query_pre_attn_scalar: int|None=None, dtype: torch.dtype|None=None
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must be divisible by num_heads if head_dim is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # Gated Attentionのゲート用の線形層．
        self.W_gate = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm: RMSNorm|None = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm|None = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # 次元の呪い対策
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5
        else:
            self.scaling = head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape

        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        # Gateの計算
        gate: torch.Tensor = torch.sigmoid(self.W_gate(x))

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2) # (batch_size, num_heads, seq_length, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2) # (batch_size, num_kv_groups, seq_length, head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2) # (batch_size, num_kv_groups, seq_length, head_dim)

        if isinstance(self.q_norm, RMSNorm):
            queries = self.q_norm(queries)
        if isinstance(self.k_norm, RMSNorm):
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1) # (batch_size, num_heads, seq_length, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) # (batch_size, num_heads, seq_length, head_dim)

        queries = queries * self.scaling

        attn_scores = queries @ keys.transpose(2, 3) # (batch_size, num_heads, seq_length, seq_length)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values) # (batch_size, num_heads, seq_length, head_dim)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.d_out) # (batch_size, seq_length, d_out)
        gated_context = context * gate
        out = self.out_proj(gated_context) # (batch_size, seq_length, d_in)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, attn_type: attention_types) -> None:
        super().__init__()
        self.attn_type = attn_type
        self.attn = GroupedQueryAttention(
            d_in=cfg.emb_dim,
            num_heads=cfg.n_heads,
            num_kv_groups=cfg.n_kv_groups,
            head_dim=cfg.head_dim,
            qk_norm=cfg.qk_norm,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
            dtype=cfg.dtype
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg.emb_dim)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim)
        self.pre_feedforward_layernorm = RMSNorm(cfg.emb_dim)
        self.post_feedforward_layernorm = RMSNorm(cfg.emb_dim)

    def forward(
        self, x: torch.Tensor, mask_global: torch.BoolTensor, mask_local: torch.BoolTensor,
        cos_global: torch.Tensor, sin_global: torch.Tensor, cos_local: torch.Tensor, sin_local: torch.Tensor) -> torch.Tensor:

        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.attn(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x


class Gemma3(nn.Module):
    cos_local: torch.Tensor
    sin_local: torch.Tensor
    cos_global: torch.Tensor
    sin_global: torch.Tensor
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.layer_types is not None and len(cfg.layer_types) == cfg.n_layers, "layer_types must be a list of attention types with length equal to n_layers"

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type) for attn_type in cfg.layer_types
        ])

        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        self.out_head.weight = self.token_embedding.weight
        self.cfg = cfg

        cos_local, sin_local = compute_rope_params(cfg.head_dim, cfg.rope_local_base, cfg.context_length, torch.float32)
        cos_global, sin_global = compute_rope_params(cfg.head_dim, cfg.rope_base, cfg.context_length, torch.float32)
        self.register_buffer("cos_local", cos_local, False)
        self.register_buffer("sin_local", sin_local, False)
        self.register_buffer("cos_global", cos_global, False)
        self.register_buffer("sin_global", sin_global, False)

    def _create_masks(self, seq_len: int, device: torch.device) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # full attentionのマスクは上三角行列．これにより，位置iのトークンは位置j > iのトークンに注意を払うことができない．
        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        #
        # torch.triuは上三角行列を返す関数．diagonal=1とすることで，対角線の1つ上から上三角行列を作成する．
        mask_global = torch.triu(ones, diagonal=1) # (seq_len, seq_len)

        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg.sliding_window).T

        # mask_globalとfar_pastの論理和を取ることで，位置iのトークンは位置j > iのトークンに注意を払うことができないか，位置j < iのトークンのうちiからsliding_window以内のトークンにのみ注意を払うことができるマスクを作成する．
        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past

        return mask_global, mask_local

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape

        # embeddingとheadが共有されているため，入力のスケーリングを行う．
        x: torch.Tensor = self.token_embedding(input_ids) * (self.cfg.emb_dim ** 0.5) # (batch_size, seq_len, emb_dim)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x, mask_global, mask_local,
                self.cos_global, self.sin_global, self.cos_local, self.sin_local
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.dtype)) # (batch_size, seq_len, vocab_size)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int, temperature: float = 1.0, top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.2) -> torch.Tensor:
        # max_new_tokensが0以下なら何も生成せずそのまま返す．
        if max_new_tokens <= 0:
            return input_ids

        # input_idsは(batch_size, seq_len)を想定．
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch_size, seq_len)")

        # 一般的なrepetition penaltyは1.0以上で使う．
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")

        # 推論中はevalモードにし，最後に元のモードへ戻す．
        was_training = self.training
        self.eval()

        # バッチ内の各系列がEOSに到達したかどうかを管理するフラグ．
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 文脈長上限を超えないよう，直近context_lengthトークンのみを条件に使う．
                idx_cond = input_ids[:, -self.cfg.context_length:]
                logits = self(idx_cond)
                # 次トークン位置のlogitsだけを取り出す．
                next_token_logits = logits[:, -1, :].float()

                if repetition_penalty > 1.0:
                    # 既出トークンのlogitを抑制して同一トークンの連発を減らす．
                    for batch_idx in range(input_ids.size(0)):
                        seen_token_ids = torch.unique(input_ids[batch_idx])
                        seen_logits = next_token_logits[batch_idx, seen_token_ids]
                        penalized_logits = torch.where(
                            seen_logits < 0,
                            seen_logits * repetition_penalty,
                            seen_logits / repetition_penalty,
                        )
                        next_token_logits[batch_idx, seen_token_ids] = penalized_logits

                # temperature <= 0 は貪欲法(argmax)で決定する．
                if temperature <= 0:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    # temperatureで分布の鋭さを調整する．
                    next_token_logits = next_token_logits / temperature

                    # top-k: 上位k個以外の候補を無効化する．
                    if top_k > 0 and top_k < next_token_logits.size(-1):
                        topk_vals, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                        cutoff = topk_vals[:, -1].unsqueeze(-1)
                        next_token_logits = torch.where(
                            next_token_logits < cutoff,
                            torch.full_like(next_token_logits, -torch.inf),
                            next_token_logits,
                        )

                    # top-p (nucleus): 累積確率がpを超える低順位候補を無効化する．
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        sorted_mask = cumulative_probs > top_p
                        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                        sorted_mask[:, 0] = False

                        sorted_logits = sorted_logits.masked_fill(sorted_mask, -torch.inf)
                        filtered_logits = torch.full_like(next_token_logits, -torch.inf)
                        filtered_logits.scatter_(1, sorted_indices, sorted_logits)
                        next_token_logits = filtered_logits

                    # 制約後の分布からサンプリングして次トークンを決める．
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)

                # 既にEOS済みの系列は以降EOSを連結し続ける．
                next_tokens = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_tokens, eos_id),
                    next_tokens,
                )

                # 1トークンずつ自己回帰的に入力へ連結していく．
                input_ids = torch.cat((input_ids, next_tokens), dim=1)
                finished = finished | (next_tokens.squeeze(-1) == eos_id)

                # バッチ内の全系列がEOSに達したら早期終了する．
                if torch.all(finished):
                    break

        # 呼び出し前がtrainモードだった場合のみ復元する．
        if was_training:
            self.train()

        return input_ids

