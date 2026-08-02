from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_scratch_lab.core.registry import BuildContext, ComponentSpec


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    seed: int = 42
    max_updates: int = Field(default=1_000, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    evaluation_interval: int = Field(default=100, gt=0)
    checkpoint_interval: int = Field(default=100, gt=0)
    evaluation_batches: int = Field(default=30, gt=0)
    compile: bool = False
    num_workers: int = Field(default=0, ge=0)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backends: list[Literal["tensorboard", "wandb"]] = Field(default_factory=list)
    project: str = "llm-scratch-lab"

    @field_validator("backends")
    @classmethod
    def unique_backends(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("tracking.backends must not contain duplicates")
        return value


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "../../outputs"
    experiment_name: str = Field(min_length=1)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ComponentSpec
    data: ComponentSpec
    method: ComponentSpec
    optimizer: ComponentSpec
    runtime: RuntimeConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, BuildContext]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    return ExperimentConfig.model_validate(raw), BuildContext(config_dir=config_path.parent)
