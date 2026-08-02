from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    item_count: int = 0


class TrainingMethod(Protocol):
    def training_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput: ...

    def evaluation_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput: ...


class OptimizerAdapter(Protocol):
    def zero_grad(self) -> None: ...

    def step(self) -> None: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class Tracker(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...

    def log_text(self, name: str, text: str, step: int) -> None: ...

    def close(self) -> None: ...
