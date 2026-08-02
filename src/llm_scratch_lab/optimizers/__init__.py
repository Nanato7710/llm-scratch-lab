from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from schedulefree import RAdamScheduleFree

from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class MuonRAdamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    muon_lr: float = Field(default=0.02, gt=0)
    radam_lr: float = Field(default=0.004, gt=0)
    beta1: float = Field(default=0.99, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    weight_decay: float = Field(default=0.01, ge=0)


class CombinedOptimizer:
    def __init__(self, model: torch.nn.Module, config: MuonRAdamConfig) -> None:
        matrix_parameters: list[torch.nn.Parameter] = []
        other_parameters: list[torch.nn.Parameter] = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            target = matrix_parameters if parameter.ndim >= 2 else other_parameters
            target.append(parameter)
        if not matrix_parameters or not other_parameters:
            raise ValueError("Combined optimizer requires matrix and non-matrix parameters")
        self.muon = torch.optim.Muon(
            matrix_parameters,
            lr=config.muon_lr,
            weight_decay=config.weight_decay,
        )
        self.radam = RAdamScheduleFree(
            other_parameters,
            lr=config.radam_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    def zero_grad(self) -> None:
        self.muon.zero_grad()
        self.radam.zero_grad()

    def step(self) -> None:
        self.muon.step()
        self.radam.step()

    def train(self) -> None:
        self.radam.train()

    def eval(self) -> None:
        self.radam.eval()

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "radam": self.radam.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.radam.load_state_dict(state_dict["radam"])

    def metrics(self) -> dict[str, float]:
        group = self.radam.param_groups[0]
        learning_rate = group.get("scheduled_lr", group["lr"])
        return {"learning_rate": float(learning_rate)}


class MuonRAdamBuilder:
    def __init__(self, config: MuonRAdamConfig) -> None:
        self.config = config

    def build(self, model: torch.nn.Module) -> CombinedOptimizer:
        return CombinedOptimizer(model, self.config)


def _build_optimizer(config: BaseModel, context: BuildContext) -> MuonRAdamBuilder:
    del context
    if not isinstance(config, MuonRAdamConfig):
        raise TypeError("Expected MuonRAdamConfig")
    return MuonRAdamBuilder(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "optimizer",
        "muon_radam_schedulefree",
        MuonRAdamConfig,
        _build_optimizer,
        description="Muon for matrices and schedule-free RAdam for other parameters",
    )


__all__ = ["CombinedOptimizer", "MuonRAdamBuilder", "MuonRAdamConfig"]
