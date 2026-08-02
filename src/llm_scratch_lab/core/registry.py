from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ComponentKind = Literal["model", "data", "method", "optimizer"]


class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    config_dir: Path

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.config_dir / candidate).resolve()


ComponentFactory = Callable[[BaseModel, BuildContext], Any]


@dataclass(frozen=True)
class RegistryEntry:
    config_type: type[BaseModel]
    factory: ComponentFactory
    description: str


class ComponentRegistry:
    def __init__(self) -> None:
        self._entries: dict[ComponentKind, dict[str, RegistryEntry]] = {
            "model": {},
            "data": {},
            "method": {},
            "optimizer": {},
        }

    def register(
        self,
        kind: ComponentKind,
        name: str,
        config_type: type[BaseModel],
        factory: ComponentFactory,
        *,
        description: str,
    ) -> None:
        if not name:
            raise ValueError("Component name must not be empty.")
        if name in self._entries[kind]:
            raise ValueError(f"Duplicate {kind} component: {name}")
        self._entries[kind][name] = RegistryEntry(config_type, factory, description)

    def get(self, kind: ComponentKind, name: str) -> RegistryEntry:
        try:
            return self._entries[kind][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries[kind])) or "(none)"
            raise ValueError(f"Unknown {kind} component {name!r}. Available: {available}") from exc

    def build(self, kind: ComponentKind, spec: ComponentSpec, context: BuildContext) -> Any:
        entry = self.get(kind, spec.name)
        config = entry.config_type.model_validate(spec.config)
        return entry.factory(config, context)

    def validate_config(self, kind: ComponentKind, spec: ComponentSpec) -> BaseModel:
        entry = self.get(kind, spec.name)
        return entry.config_type.model_validate(spec.config)

    def list(
        self, kind: ComponentKind | None = None
    ) -> dict[ComponentKind, dict[str, RegistryEntry]]:
        kinds = [kind] if kind is not None else list(self._entries)
        return {current: dict(self._entries[current]) for current in kinds}
