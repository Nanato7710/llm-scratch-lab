# %%
from models.Gemma3.Base import Gemma3, Config
import torch

# %%
cfg = Config(
    vocab_size=32_000,
    context_length=32_768,
    emb_dim=640,
    n_heads=4,
    n_layers=8,
    hidden_dim=2048,
    head_dim=256,
    qk_norm=True,
    n_kv_groups=2,
    rope_local_base=10_000.0,
    rope_base=1_000_000.0,
    sliding_window=512,
    layer_types=[
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    dtype=torch.bfloat16,
    query_pre_attn_scalar=256
)
model = Gemma3(cfg)

# %%
model.generate(
    input_ids=torch.randint(1, 32_000, (2, 5)),
    max_new_tokens=10,
    eos_id=0,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=10
)

# %%
def count_parameters(model, is_print=False):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_print:
        print(f"Trainable parameters: {total_params / 1e6:.2f}M ({total_params / 1e9:.2f}B)")
    return total_params

# %%
count_parameters(model, is_print=True)


