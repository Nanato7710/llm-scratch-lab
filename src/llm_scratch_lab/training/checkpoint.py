from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path: str | Path, *, map_location: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "latest.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
