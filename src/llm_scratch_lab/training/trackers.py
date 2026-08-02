from __future__ import annotations

from pathlib import Path
from typing import Any


class NoOpTracker:
    def log(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def log_text(self, name: str, text: str, step: int) -> None:
        del name, text, step

    def close(self) -> None:
        pass


class TensorBoardTracker:
    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard tracking requires: uv sync --extra tracking") from exc
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(name, text, step)

    def close(self) -> None:
        self.writer.close()


class WandBTracker:
    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        try:
            import wandb
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("W&B tracking requires: uv sync --extra tracking") from exc
        load_dotenv()
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.run.log({name: self.wandb.Html(text)}, step=step)

    def close(self) -> None:
        self.run.finish()


class CompositeTracker:
    def __init__(self, trackers: list[Any]) -> None:
        self.trackers = trackers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for tracker in self.trackers:
            tracker.log(metrics, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        for tracker in self.trackers:
            tracker.log_text(name, text, step)

    def close(self) -> None:
        for tracker in reversed(self.trackers):
            tracker.close()


def create_tracker(
    backends: list[str],
    *,
    run_dir: Path,
    project: str,
    run_name: str,
    config: dict[str, Any],
) -> NoOpTracker | CompositeTracker:
    trackers: list[Any] = []
    for backend in backends:
        if backend == "tensorboard":
            trackers.append(TensorBoardTracker(run_dir / "tensorboard"))
        elif backend == "wandb":
            trackers.append(WandBTracker(project=project, run_name=run_name, config=config))
        else:
            raise ValueError(f"Unknown tracker backend: {backend}")
    return CompositeTracker(trackers) if trackers else NoOpTracker()
