from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from datasets import IterableDataset as HuggingFaceIterableDataset
from datasets import interleave_datasets, load_dataset
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llm_scratch_lab.core.contracts import Batch
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    name: str | None = None
    split: str = "train"
    text_column: str = "text"
    weight: float = Field(default=1.0, gt=0)


class HFStreamingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_path: str
    train_sources: list[SourceConfig] = Field(min_length=1)
    eval_sources: list[SourceConfig] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    shuffle_train: bool = True
    shuffle_seed: int = 42
    shuffle_buffer_size: int = Field(default=10_000, gt=0)
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "all_exhausted"
    add_eos_between_samples: bool = True
    drop_last_train: bool = True
    max_train_samples: int | None = Field(default=None, gt=0)
    max_eval_samples: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_weights(self) -> HFStreamingConfig:
        for label, sources in (
            ("train_sources", self.train_sources),
            ("eval_sources", self.eval_sources),
        ):
            if len(sources) > 1 and not math.isclose(
                sum(source.weight for source in sources), 1.0, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"{label} weights must sum to 1.0")
        return self


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("A fast tokenizer is required")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad, EOS, or unknown token")
    return tokenizer


class PackedTextDataset(IterableDataset[Batch]):
    """Pack a stream of texts into next-token prediction blocks."""

    def __init__(
        self,
        dataset: Iterable[Mapping[str, object]],
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int,
        *,
        text_column: str = "text",
        add_eos_between_samples: bool = True,
        max_samples: int | None = None,
        drop_last: bool = True,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.text_column = text_column
        self.add_eos_between_samples = add_eos_between_samples
        self.max_samples = max_samples
        self.drop_last = drop_last
        self.ignore_index = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._token_buffer: list[int] = []
        self._sample_count = 0
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must not be None")

    def _worker_dataset(self) -> Iterable[Mapping[str, object]]:
        worker = get_worker_info()
        if worker is None:
            return self.dataset
        shard = getattr(self.dataset, "shard", None)
        if callable(shard):
            try:
                return shard(num_shards=worker.num_workers, index=worker.id, contiguous=True)
            except TypeError:
                return shard(worker.num_workers, worker.id)
        return (
            sample
            for index, sample in enumerate(self.dataset)
            if index % worker.num_workers == worker.id
        )

    def _emit(self, *, finalize: bool) -> Iterator[Batch]:
        block_size = self.sequence_length + 1
        while len(self._token_buffer) >= block_size:
            inputs = torch.tensor(self._token_buffer[: self.sequence_length], dtype=torch.long)
            labels = torch.tensor(self._token_buffer[1:block_size], dtype=torch.long)
            del self._token_buffer[: self.sequence_length]
            yield {
                "input_ids": inputs,
                "labels": labels,
                "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
            }

        if finalize and not self.drop_last and len(self._token_buffer) > 1:
            original_length = len(self._token_buffer)
            padded = self._token_buffer + [self.pad_token_id] * (block_size - original_length)
            inputs = torch.tensor(padded[: self.sequence_length], dtype=torch.long)
            labels = torch.tensor(padded[1:block_size], dtype=torch.long)
            valid_targets = original_length - 1
            labels[valid_targets:] = self.ignore_index
            attention_mask = torch.zeros(self.sequence_length, dtype=torch.long)
            attention_mask[: min(original_length, self.sequence_length)] = 1
            self._token_buffer.clear()
            yield {
                "input_ids": inputs,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    def __iter__(self) -> Iterator[Batch]:
        yield from self._emit(finalize=False)
        for sample in self._worker_dataset():
            if self.max_samples is not None and self._sample_count >= self.max_samples:
                break
            text = sample.get(self.text_column)
            if not isinstance(text, str) or not text:
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            self._sample_count += 1
            self._token_buffer.extend(token_ids)
            if self.add_eos_between_samples and self.eos_token_id is not None:
                self._token_buffer.append(self.eos_token_id)
            yield from self._emit(finalize=False)
        yield from self._emit(finalize=True)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "token_buffer": list(self._token_buffer),
            "sample_count": self._sample_count,
        }
        dataset_state = getattr(self.dataset, "state_dict", None)
        if callable(dataset_state):
            state["dataset"] = dataset_state()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._token_buffer = list(state.get("token_buffer", []))
        self._sample_count = int(state.get("sample_count", 0))
        dataset_state = state.get("dataset")
        load_dataset_state = getattr(self.dataset, "load_state_dict", None)
        if dataset_state is not None and callable(load_dataset_state):
            load_dataset_state(dataset_state)


@dataclass
class DataBundle:
    train_loader: DataLoader[Batch]
    eval_loader_factory: Callable[[], DataLoader[Batch]]
    train_dataset: PackedTextDataset

    def state_dict(self) -> dict[str, Any]:
        return self.train_dataset.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.train_dataset.load_state_dict(state)


class HFStreamingDataModule:
    batch_keys = frozenset({"input_ids", "labels", "attention_mask"})

    def __init__(self, config: HFStreamingConfig, context: BuildContext) -> None:
        self.config = config
        tokenizer_path = context.resolve_path(config.tokenizer_path)
        self.tokenizer = load_tokenizer(str(tokenizer_path))

    def _load_sources(
        self,
        sources: list[SourceConfig],
        *,
        shuffle: bool,
    ) -> tuple[HuggingFaceIterableDataset, str]:
        datasets: list[HuggingFaceIterableDataset] = []
        for source in sources:
            dataset = load_dataset(
                source.path,
                name=source.name,
                split=source.split,
                streaming=True,
            )
            if source.text_column not in dataset.column_names:
                raise ValueError(
                    f"{source.path} does not contain column {source.text_column!r}; "
                    f"available: {dataset.column_names}"
                )
            if source.text_column != "text":
                dataset = dataset.rename_column(source.text_column, "text")
            dataset = dataset.select_columns(["text"])
            datasets.append(dataset)
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = interleave_datasets(
                datasets,
                probabilities=[source.weight for source in sources],
                seed=self.config.shuffle_seed,
                stopping_strategy=self.config.stopping_strategy,
            )
        if shuffle:
            combined = combined.shuffle(
                seed=self.config.shuffle_seed,
                buffer_size=self.config.shuffle_buffer_size,
            )
        return combined, "text"

    def build(self, *, num_workers: int = 0) -> DataBundle:
        train_raw, train_column = self._load_sources(
            self.config.train_sources, shuffle=self.config.shuffle_train
        )
        train_dataset = PackedTextDataset(
            train_raw,
            self.tokenizer,
            self.config.sequence_length,
            text_column=train_column,
            add_eos_between_samples=self.config.add_eos_between_samples,
            max_samples=self.config.max_train_samples,
            drop_last=self.config.drop_last_train,
        )
        loader_options: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "num_workers": num_workers,
        }

        def build_eval_loader() -> DataLoader[Batch]:
            eval_raw, eval_column = self._load_sources(self.config.eval_sources, shuffle=False)
            eval_dataset = PackedTextDataset(
                eval_raw,
                self.tokenizer,
                self.config.sequence_length,
                text_column=eval_column,
                add_eos_between_samples=self.config.add_eos_between_samples,
                max_samples=self.config.max_eval_samples,
                drop_last=False,
            )
            return DataLoader(eval_dataset, **loader_options)

        return DataBundle(
            train_loader=DataLoader(train_dataset, **loader_options),
            eval_loader_factory=build_eval_loader,
            train_dataset=train_dataset,
        )


def _build_data(config: BaseModel, context: BuildContext) -> HFStreamingDataModule:
    if not isinstance(config, HFStreamingConfig):
        raise TypeError("Expected HFStreamingConfig")
    return HFStreamingDataModule(config, context)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "data",
        "hf_streaming",
        HFStreamingConfig,
        _build_data,
        description="Streaming Hugging Face datasets with causal-LM token packing",
    )


__all__ = [
    "DataBundle",
    "HFStreamingConfig",
    "HFStreamingDataModule",
    "PackedTextDataset",
    "SourceConfig",
    "load_tokenizer",
]
