from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dataset.PretrainDataset import (
    Genre,
    MixPreset,
    Split,
    StoppingStrategy,
    get_dataset,
    get_recommended_mix,
)


def load_default_tokenizer(
    tokenizer_path: str | Path | None = None,
) -> PreTrainedTokenizerFast:
    """
    Load the project tokenizer and normalize pad token handling.
    """
    if tokenizer_path is None:
        tokenizer_path = (
            Path(__file__).resolve().parents[1] / "tokenizer" / "hf_tokenizer_32k"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("A fast tokenizer is required for pretraining data loading.")

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token_id is None and tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide pad/eos/unk token id.")

    return tokenizer


class StreamingTextDataset(IterableDataset):
    """
    Streaming causal-LM dataset that packs text into fixed-length token blocks.
    """

    def __init__(
        self,
        dataset: Iterable[Mapping[str, object]],
        tokenizer: PreTrainedTokenizerFast,
        seq_length: int,
        text_key: str = "text",
        add_eos_between_samples: bool = True,
        max_samples: int | None = None,
        drop_last: bool = True,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if seq_length <= 0:
            raise ValueError("seq_length must be a positive integer.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be None or a positive integer.")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_key = text_key
        self.add_eos_between_samples = add_eos_between_samples
        self.max_samples = max_samples
        self.drop_last = drop_last
        self.ignore_index = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must not be None.")

    def _split_for_worker(self) -> Iterable[Mapping[str, object]]:
        worker = get_worker_info()
        if worker is None:
            return self.dataset

        if hasattr(self.dataset, "shard"):
            try:
                return self.dataset.shard(
                    num_shards=worker.num_workers,
                    index=worker.id,
                    contiguous=True,
                )
            except TypeError:
                return self.dataset.shard(worker.num_workers, worker.id)

        return (
            sample
            for idx, sample in enumerate(self.dataset)
            if idx % worker.num_workers == worker.id
        )

    def _emit_from_buffer(
        self, token_buffer: list[int], finalize: bool
    ) -> Iterator[dict[str, torch.Tensor]]:
        block_plus_one = self.seq_length + 1
        while len(token_buffer) >= block_plus_one:
            input_ids = torch.tensor(token_buffer[: self.seq_length], dtype=torch.long)
            labels = torch.tensor(token_buffer[1:block_plus_one], dtype=torch.long)
            attention_mask = torch.ones(self.seq_length, dtype=torch.long)
            del token_buffer[: self.seq_length]

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        if finalize and not self.drop_last and len(token_buffer) > 1:
            padded = token_buffer + [self.pad_token_id] * (block_plus_one - len(token_buffer))
            input_ids = torch.tensor(padded[: self.seq_length], dtype=torch.long)
            labels = torch.tensor(padded[1:block_plus_one], dtype=torch.long)
            attention_mask = torch.zeros(self.seq_length, dtype=torch.long)

            valid_inputs = min(len(token_buffer), self.seq_length)
            valid_targets = max(0, len(token_buffer) - 1)
            attention_mask[:valid_inputs] = 1
            if valid_targets < self.seq_length:
                labels[valid_targets:] = self.ignore_index

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        stream = self._split_for_worker()
        token_buffer: list[int] = []
        sample_count = 0

        for sample in stream:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            text = sample.get(self.text_key)
            if not isinstance(text, str):
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            sample_count += 1
            token_buffer.extend(token_ids)
            if self.add_eos_between_samples and self.eos_token_id is not None:
                token_buffer.append(self.eos_token_id)

            yield from self._emit_from_buffer(token_buffer, finalize=False)

        yield from self._emit_from_buffer(token_buffer, finalize=True)


def get_pretrain_dataloader(
    batch_size: int,
    seq_length: int,
    tokenizer: PreTrainedTokenizerFast | None = None,
    tokenizer_path: str | Path | None = None,
    genre: Genre | Mapping[Genre, float] | None = None,
    mix_preset: MixPreset = "balanced",
    split: Split = "train",
    shuffle: bool = True,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    stopping_strategy: StoppingStrategy = "all_exhausted",
    add_eos_between_samples: bool = True,
    max_samples: int | None = None,
    drop_last_sample: bool = True,
    ignore_index: int = -100,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    drop_last_batch: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for LLM pretraining.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if prefetch_factor is not None and prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer.")

    if tokenizer is None:
        tokenizer = load_default_tokenizer(tokenizer_path)

    if genre is None:
        genre = get_recommended_mix(mix_preset)

    raw_dataset = get_dataset(
        genre=genre,
        split=split,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_buffer_size=shuffle_buffer_size,
        stopping_strategy=stopping_strategy,
    )

    lm_dataset = StreamingTextDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        seq_length=seq_length,
        add_eos_between_samples=add_eos_between_samples,
        max_samples=max_samples,
        drop_last=drop_last_sample,
        ignore_index=ignore_index,
    )

    dataloader_kwargs: dict[str, object] = {
        "dataset": lm_dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last_batch,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = (
            True if persistent_workers is None else persistent_workers
        )
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)


__all__ = [
    "StreamingTextDataset",
    "get_pretrain_dataloader",
    "load_default_tokenizer",
]
