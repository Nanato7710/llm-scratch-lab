from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from schedulefree import RAdamScheduleFree


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, is_print: bool = False) -> int:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_print:
        print(f"Trainable parameters: {total_params / 1e6:.2f}M ({total_params / 1e9:.2f}B)")
    return total_params


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


class CustomOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        radam_schedulefree_lr: float = 0.004,
        betas: tuple[float, float] = (0.99, 0.999),
        weight_decay: float = 0.01,
    ) -> None:
        muon_params: list[torch.nn.Parameter] = []
        radam_schedulefree_params: list[torch.nn.Parameter] = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                radam_schedulefree_params.append(param)

        self.muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.radam_schedulefree = RAdamScheduleFree(
            radam_schedulefree_params,
            lr=radam_schedulefree_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def zero_grad(self) -> None:
        self.muon.zero_grad()
        self.radam_schedulefree.zero_grad()

    def train(self) -> None:
        self.radam_schedulefree.train()

    def eval(self) -> None:
        self.radam_schedulefree.eval()

    def step(self) -> None:
        self.muon.step()
        self.radam_schedulefree.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "radam_schedulefree": self.radam_schedulefree.state_dict(),
        }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    step: int,
    checkpoint_dir: str | Path = "checkpoints",
    cfg: Any | None = None,
    is_best: bool = False,
) -> None:
    checkpoint_path = Path(checkpoint_dir) / ("best" if is_best else "normal")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_compiled_model(model)
    state = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    torch.save(
        {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path / "models.pt",
    )

    if cfg is not None:
        (checkpoint_path / "config.json").write_text(
            cfg.model_dump_json(indent=4),
            encoding="utf-8",
        )


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    epoch: int,
    checkpoint_dir: str | Path = "checkpoints",
) -> None:
    checkpoint_path = Path(checkpoint_dir) / f"gemma3_epoch_{epoch}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_compiled_model(model)
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in raw_model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )
