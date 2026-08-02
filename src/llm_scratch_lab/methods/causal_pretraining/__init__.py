from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from llm_scratch_lab.core.contracts import Batch, ModelOutput, StepOutput
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class CausalPretrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ignore_index: int = -100


class CausalPretrainingMethod:
    required_batch_keys = frozenset({"input_ids", "labels"})

    def __init__(self, config: CausalPretrainingConfig) -> None:
        self.config = config

    def _step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        missing = self.required_batch_keys - batch.keys()
        if missing:
            raise ValueError(f"Batch is missing required keys: {sorted(missing)}")
        result = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        if not isinstance(result, ModelOutput):
            raise TypeError("Causal pretraining models must return ModelOutput")
        labels = batch["labels"]
        valid_tokens = int((labels != self.config.ignore_index).sum().item())
        if valid_tokens == 0:
            raise ValueError("Batch contains no valid target tokens")
        loss = F.cross_entropy(
            result.logits.reshape(-1, result.logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.config.ignore_index,
        )
        return StepOutput(loss=loss, metrics={"nll": float(loss.detach())}, item_count=valid_tokens)

    def training_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        return self._step(model, batch)

    def evaluation_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        return self._step(model, batch)


def _build_method(config: BaseModel, context: BuildContext) -> CausalPretrainingMethod:
    del context
    if not isinstance(config, CausalPretrainingConfig):
        raise TypeError("Expected CausalPretrainingConfig")
    return CausalPretrainingMethod(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "method",
        "causal_pretraining",
        CausalPretrainingConfig,
        _build_method,
        description="Next-token prediction with token-weighted NLL evaluation",
    )


__all__ = ["CausalPretrainingConfig", "CausalPretrainingMethod"]
