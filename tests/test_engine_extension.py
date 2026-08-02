from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader

from llm_scratch_lab.core.config import ExperimentConfig
from llm_scratch_lab.core.contracts import ModelOutput
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry
from llm_scratch_lab.methods.causal_pretraining import register as register_method
from llm_scratch_lab.training import run_experiment


class EmptyConfig(BaseModel):
    pass


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> ModelOutput:
        del attention_mask
        return ModelOutput(self.output(self.embedding(input_ids)))


class TinyDataBundle:
    def __init__(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        self.train_loader = DataLoader([batch] * 4, batch_size=None)
        self.eval_loader_factory = lambda: DataLoader([batch], batch_size=None)
        self.position = 0

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.position = int(state["position"])


class TinyDataModule:
    def build(self, *, num_workers: int) -> TinyDataBundle:
        assert num_workers == 0
        return TinyDataBundle()


class OptimizerAdapter:
    def __init__(self, model: torch.nn.Module) -> None:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)


class OptimizerBuilder:
    def build(self, model: torch.nn.Module) -> OptimizerAdapter:
        return OptimizerAdapter(model)


def _factory(value: object):
    def build(config: BaseModel, context: BuildContext) -> object:
        del config, context
        return value() if isinstance(value, type) else value

    return build


def test_new_components_run_through_the_shared_engine(tmp_path: Path) -> None:
    registry = ComponentRegistry()
    registry.register("model", "tiny", EmptyConfig, _factory(TinyModel), description="test model")
    registry.register(
        "data", "memory", EmptyConfig, _factory(TinyDataModule), description="test data"
    )
    registry.register(
        "optimizer",
        "sgd",
        EmptyConfig,
        _factory(OptimizerBuilder),
        description="test optimizer",
    )
    register_method(registry)

    experiment = ExperimentConfig.model_validate(
        {
            "model": {"name": "tiny"},
            "data": {"name": "memory"},
            "method": {"name": "causal_pretraining"},
            "optimizer": {"name": "sgd"},
            "runtime": {
                "device": "cpu",
                "max_updates": 2,
                "gradient_accumulation_steps": 2,
                "evaluation_interval": 1,
                "checkpoint_interval": 1,
                "evaluation_batches": 1,
            },
            "output": {"root": str(tmp_path), "experiment_name": "extension"},
        }
    )
    run_dir = run_experiment(experiment, BuildContext(config_dir=tmp_path), registry)
    assert (run_dir / "config.json").is_file()
    checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
    assert checkpoint["update"] == 2
    assert checkpoint["experiment"]["model"]["name"] == "tiny"
