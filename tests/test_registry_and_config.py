from pathlib import Path

import pytest
from pydantic import BaseModel

from llm_scratch_lab.core.components import create_default_registry
from llm_scratch_lab.core.config import load_experiment_config
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry, ComponentSpec


class EmptyConfig(BaseModel):
    pass


def build_empty(config: BaseModel, context: BuildContext) -> tuple[BaseModel, Path]:
    return config, context.config_dir


def test_registry_rejects_duplicate_and_unknown_components(tmp_path: Path) -> None:
    registry = ComponentRegistry()
    registry.register("model", "tiny", EmptyConfig, build_empty, description="tiny")
    with pytest.raises(ValueError, match="Duplicate model"):
        registry.register("model", "tiny", EmptyConfig, build_empty, description="tiny")
    with pytest.raises(ValueError, match="Unknown model"):
        registry.get("model", "missing")

    built_config, built_path = registry.build(
        "model", ComponentSpec(name="tiny"), BuildContext(tmp_path)
    )
    assert isinstance(built_config, EmptyConfig)
    assert built_path == tmp_path


@pytest.mark.parametrize("filename", ["gemma3_base.toml", "gemma3_gated.toml"])
def test_example_experiment_configs_are_valid(filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config, _ = load_experiment_config(root / "configs" / "experiments" / filename)
    registry = create_default_registry()
    for kind in ("model", "data", "method", "optimizer"):
        registry.validate_config(kind, getattr(config, kind))


def test_build_context_resolves_relative_paths(tmp_path: Path) -> None:
    context = BuildContext(tmp_path / "configs")
    assert context.resolve_path("../artifact") == (tmp_path / "artifact").resolve()
