from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from llm_scratch_lab.core.config import ExperimentConfig
from llm_scratch_lab.core.contracts import Batch, OptimizerAdapter, TrainingMethod
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry
from llm_scratch_lab.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    unwrap_model,
)
from llm_scratch_lab.training.trackers import create_tracker

LOGGER = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return {name: value.to(device) for name, value in batch.items()}


def evaluate(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    method: TrainingMethod,
    optimizer: OptimizerAdapter,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    optimizer.eval()
    total_nll = 0.0
    total_items = 0
    try:
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                if index >= max_batches:
                    break
                output = method.evaluation_step(model, move_batch(batch, device))
                total_nll += float(output.loss) * output.item_count
                total_items += output.item_count
    finally:
        optimizer.train()
        model.train()
    if total_items == 0:
        raise RuntimeError("Evaluation dataset produced no valid target tokens")
    mean_nll = total_nll / total_items
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = math.inf
    return {"eval/nll": mean_nll, "eval/perplexity": perplexity}


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: OptimizerAdapter,
    data_bundle: Any,
    experiment: ExperimentConfig,
    update: int,
    best_nll: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": experiment.model_dump(mode="json"),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "data": data_bundle.state_dict(),
        "rng": capture_rng_state(),
        "update": update,
        "best_nll": best_nll,
    }


def _validate_checkpoint_components(
    checkpoint: dict[str, Any], experiment: ExperimentConfig
) -> None:
    saved = checkpoint.get("experiment", {})
    for category in ("model", "data", "method", "optimizer"):
        saved_name = saved.get(category, {}).get("name")
        current_name = getattr(experiment, category).name
        if saved_name != current_name:
            raise ValueError(
                f"Checkpoint {category}={saved_name!r} does not match config {current_name!r}"
            )


def run_experiment(
    experiment: ExperimentConfig,
    context: BuildContext,
    registry: ComponentRegistry,
    *,
    resume: str | Path | None = None,
) -> Path:
    runtime = experiment.runtime
    random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.seed)
    device = resolve_device(runtime.device)

    model = registry.build("model", experiment.model, context)
    data_module = registry.build("data", experiment.data, context)
    method = registry.build("method", experiment.method, context)
    optimizer_builder = registry.build("optimizer", experiment.optimizer, context)

    provided_batch_keys = getattr(data_module, "batch_keys", None)
    required_batch_keys = getattr(method, "required_batch_keys", None)
    if (
        provided_batch_keys is not None
        and required_batch_keys is not None
        and not required_batch_keys <= provided_batch_keys
    ):
        missing = sorted(required_batch_keys - provided_batch_keys)
        raise ValueError(f"Data component cannot provide method batch keys: {missing}")

    model = model.to(device)
    tokenizer = getattr(data_module, "tokenizer", None)
    model_config = getattr(model, "config", None)
    if (
        tokenizer is not None
        and model_config is not None
        and tokenizer.vocab_size != model_config.vocab_size
    ):
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) does not match "
            f"model vocabulary ({model_config.vocab_size})"
        )
    if runtime.compile:
        model = torch.compile(model)
    optimizer = optimizer_builder.build(model)
    data_bundle = data_module.build(num_workers=runtime.num_workers)

    output_root = context.resolve_path(experiment.output.root)
    if resume is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / experiment.output.experiment_name / run_name
    else:
        resume_path = Path(resume).expanduser().resolve()
        run_dir = (
            resume_path.parent.parent
            if resume_path.parent.name == "checkpoints"
            else resume_path.parent
        )
        run_name = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = experiment.model_dump(mode="json")
    (run_dir / "config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tracker = create_tracker(
        experiment.tracking.backends,
        run_dir=run_dir,
        project=experiment.tracking.project,
        run_name=run_name,
        config=resolved_config,
    )

    update = 0
    best_nll = math.inf
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        _validate_checkpoint_components(checkpoint, experiment)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_bundle.load_state_dict(checkpoint.get("data", {}))
        restore_rng_state(checkpoint["rng"])
        update = int(checkpoint["update"])
        best_nll = float(checkpoint["best_nll"])
        if runtime.num_workers != 0:
            LOGGER.warning("Exact streaming resume requires runtime.num_workers=0")

    LOGGER.info("Training on %s from optimizer update %d", device, update)
    model.train()
    optimizer.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_microbatches = 0

    try:
        for batch in data_bundle.train_loader:
            output = method.training_step(model, move_batch(batch, device))
            (output.loss / runtime.gradient_accumulation_steps).backward()
            accumulated_loss += float(output.loss.detach())
            accumulated_microbatches += 1
            if accumulated_microbatches < runtime.gradient_accumulation_steps:
                continue

            if runtime.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            update += 1

            train_metrics = {"train/nll": accumulated_loss / accumulated_microbatches}
            metrics_method = getattr(optimizer, "metrics", None)
            if callable(metrics_method):
                train_metrics.update(
                    {f"train/{name}": value for name, value in metrics_method().items()}
                )
            tracker.log(train_metrics, update)
            accumulated_loss = 0.0
            accumulated_microbatches = 0

            if update % runtime.evaluation_interval == 0:
                eval_metrics = evaluate(
                    model,
                    data_bundle.eval_loader_factory(),
                    method,
                    optimizer,
                    device=device,
                    max_batches=runtime.evaluation_batches,
                )
                tracker.log(eval_metrics, update)
                if eval_metrics["eval/nll"] < best_nll:
                    best_nll = eval_metrics["eval/nll"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            data_bundle=data_bundle,
                            experiment=experiment,
                            update=update,
                            best_nll=best_nll,
                        ),
                    )

            if update % runtime.checkpoint_interval == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / "latest.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        data_bundle=data_bundle,
                        experiment=experiment,
                        update=update,
                        best_nll=best_nll,
                    ),
                )

            if update >= runtime.max_updates:
                break
        else:
            raise RuntimeError(
                f"Training data ended at update {update}, before max_updates={runtime.max_updates}"
            )

        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                data_bundle=data_bundle,
                experiment=experiment,
                update=update,
                best_nll=best_nll,
            ),
        )
    finally:
        tracker.close()
    return run_dir
