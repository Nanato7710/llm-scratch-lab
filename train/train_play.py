#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()
#%%
import time
from models.Gemma3.Base import Gemma3, Config
from dataset.DataLoaders import load_default_tokenizer
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from schedulefree import RAdamScheduleFree
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
import wandb
from my_utils import count_parameters
# %%
LR = 0.01
BETA1 = 0.99
WEIGHT_DECAY = 0.0
SUM_SAMPLES = 1_048_576 # 1BTに近い2の冪数．fineweb2の1sampleが平均540tokensだったから．
BATCH_SIZE = 16
ACCUMULATE_STEPS = 16
TOTAL_STEPS = SUM_SAMPLES // BATCH_SIZE
CHECKPOINT_INTERVAL = 16 * ACCUMULATE_STEPS
SEQ_LENGTH = 1_024
RANDOM_SEED = 42
TIME_STR = time.strftime("%Y%m%d-%H%M%S")
CHECKPOINT_PATH = f"checkpoints/gemma3_play/{TIME_STR}"
TENSORBOARD_ROOT = "logs/gemma3_play"
TENSORBOARD_LOG_DIR = f"{TENSORBOARD_ROOT}/{TIME_STR}"
#%%

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

tokenizer = load_default_tokenizer()

with open("train/menhera.txt", "r", encoding="utf-8") as f:
    menhera_text = f.read()

with open("train/oji.txt", "r", encoding="utf-8") as f:
    oji_text = f.read()

class CustomOptimizer():
    def __init__(self, model: torch.nn.Module, muon_lr: float=0.02, radam_schedulefree_lr: float=0.004, betas=(0.99, 0.999), weight_decay=0.01):
        muon_params: list[torch.nn.Parameter] = []
        radam_schedulefree_params: list[torch.nn.Parameter] = []
        for _, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                radam_schedulefree_params.append(param)
        self.muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.radam_schedulefree = RAdamScheduleFree(radam_schedulefree_params, lr=radam_schedulefree_lr, betas=betas, weight_decay=weight_decay)

    def zero_grad(self):
        self.muon.zero_grad()
        self.radam_schedulefree.zero_grad()

    def train(self):
        self.radam_schedulefree.train()

    def eval(self):
        self.radam_schedulefree.eval()

    def step(self):
        self.muon.step()
        self.radam_schedulefree.step()

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "radam_schedulefree": self.radam_schedulefree.state_dict()
        }

vocab_size: int = tokenizer.vocab_size

cfg = Config(
    vocab_size=vocab_size,
    context_length=SEQ_LENGTH,
    emb_dim=512,
    n_heads=8,
    n_layers=24,
    hidden_dim=1_024,
    head_dim=64,
    qk_norm=True,
    n_kv_groups=2,
    rope_local_base=10_000.0,
    rope_base=1_000_000.0,
    sliding_window=512,
    dtype=torch.bfloat16,
    query_pre_attn_scalar=128,
    layer_types=[
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]*6
)

model = Gemma3(cfg).to(device)
count_parameters(model, is_print=True)
# %%
from datasets import load_dataset
# %%
train_ds = load_dataset("epfml/FineWeb2-HQ", "jpn_Jpan", split="train", streaming=True)
train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col != "text"])
test_ds = load_dataset("globis-university/aozorabunko-clean", split="train", streaming=True)
test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col != "text"])
# %%
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=SEQ_LENGTH, padding=False)
train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])
# %%
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collator)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collator)
# %%
def save_checkpoint(model: Gemma3, cfg: Config, optimizer: RAdamScheduleFree, step: int, checkpoint_dir: str = "checkpoints", is_best: bool = False):
    checkpoint_path = Path(checkpoint_dir) / "best" if is_best else Path(checkpoint_dir) / "normal"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    raw_model: Gemma3 = model._orig_mod if hasattr(model, "_orig_mod") else model
    state: dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    optim_state: dict = optimizer.state_dict()
    torch.save({
        "model": state,
        "optimizer": optim_state,
        "step": step
    }, checkpoint_path/"models.pt")
    with open(checkpoint_path/"config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=4))
# %%
model = torch.compile(model)
# optimizer = RAdamScheduleFree(model.parameters(), lr=LR, betas=(BETA1, 0.999), weight_decay=WEIGHT_DECAY)
optimizer = CustomOptimizer(model, radam_schedulefree_lr=LR, betas=(BETA1, 0.999))
# %%
save_cfg_dict = cfg.model_dump()
save_cfg_dict["LR"] = LR
save_cfg_dict["BETA1"] = BETA1
save_cfg_dict["WEIGHT_DECAY"] = WEIGHT_DECAY
run = wandb.init(project="gemma3_play", config=save_cfg_dict, name=TIME_STR+f"_lr{LR}_beta1_{BETA1}_weight_decay_{WEIGHT_DECAY}", sync_tensorboard=True)
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
# %%
def train_one_step(model: Gemma3, batch: dict[str, torch.Tensor], optimizer: RAdamScheduleFree, now_step: int, grad_accumulate_steps: int=1)-> tuple[float, float]:
    model.train()
    optimizer.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"][:,1:].to(device)

    if now_step % grad_accumulate_steps == 0:
        optimizer.zero_grad()
    outputs = model(input_ids=input_ids)[:,:-1]
    loss = cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
    (loss/grad_accumulate_steps).backward()
    if (now_step+1) % grad_accumulate_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduled_lr = optimizer.radam_schedulefree.param_groups[0]["scheduled_lr"]
    return loss.item(), scheduled_lr


def test(model: Gemma3, dataloader: DataLoader, optimizer: RAdamScheduleFree) -> float:
    model.eval()
    optimizer.eval()
    total_nll = 0.0
    total_tokens = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"][:,1:].to(device)

            outputs = model(input_ids=input_ids)[:,:-1]
            nll = cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += (labels != -100).sum().item()  # -100はignore_index
            count += 1
            if count >= 30:
                break

    if total_tokens > 0:
        log_ppl = total_nll / total_tokens
    else:
        log_ppl = -100
    return log_ppl

def const_eval(model: Gemma3, optimizer: RAdamScheduleFree, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    optimizer.eval()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"][:,1:].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)[:,:-1]
        nll = cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1), reduction="sum")
        total_nll = nll.item()
        total_tokens = (labels != -100).sum().item()

    if total_tokens > 0:
        log_ppl = total_nll / total_tokens
    else:
        log_ppl = -100
    return log_ppl

def generate_sample(model: Gemma3, optimizer: RAdamScheduleFree, tokenizer, prompt: str, max_new_tokens: int = 100):
    model.eval()
    optimizer.eval()
    input_ids = tokenizer(tokenizer.eos_token+prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
# %%
step = 0
const_eval_batch = next(iter(test_loader))
best_val_log_ppl = float('inf')
for batch in train_loader:
    train_loss, lr = train_one_step(model, batch, optimizer, now_step=step, grad_accumulate_steps=ACCUMULATE_STEPS)
    writer.add_scalar("Loss/Train", train_loss, step)
    writer.add_scalar("Learning Rate", lr, step)
    run.log({"Loss/Train": train_loss, "Learning Rate": lr}, step=step)

    if (step+1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(model, cfg, optimizer, step, checkpoint_dir=CHECKPOINT_PATH, is_best=False)
        val_log_ppl = test(model, test_loader, optimizer)
        const_eval_log_ppl = const_eval(model, optimizer, const_eval_batch)
        if val_log_ppl < best_val_log_ppl:
            best_val_log_ppl = val_log_ppl
            save_checkpoint(model, cfg, optimizer, step, checkpoint_dir=CHECKPOINT_PATH, is_best=True)
        writer.add_scalar("Loss/Validation", val_log_ppl, step)
        writer.add_scalar("Loss/ConstEval", const_eval_log_ppl, step)
        run.log({"Loss/Validation": val_log_ppl, "Loss/ConstEval": const_eval_log_ppl}, step=step)
        menhera_sample = generate_sample(model, optimizer, tokenizer, prompt=menhera_text)
        oji_sample = generate_sample(model, optimizer, tokenizer, prompt=oji_text)
        writer.add_text("Sample/Menhera", menhera_sample, step)
        writer.add_text("Sample/Oji", oji_sample, step)
        run.log({"Sample/Menhera": wandb.Html(menhera_sample), "Sample/Oji": wandb.Html(oji_sample)}, step=step)
    step += 1
run.finish()
