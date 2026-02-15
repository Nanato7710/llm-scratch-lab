from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
import time
from typing import Literal

from datasets import IterableDataset, interleave_datasets, load_dataset


Genre = Literal["literature", "wiki", "textbook", "research", "web", "cc100"]
Split = Literal["train", "validation", "test"]
StoppingStrategy = Literal[
    "all_exhausted",
    "first_exhausted",
    "all_exhausted_without_replacement",
]
MixPreset = Literal["small", "balanced", "web_heavy"]


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None = None
    text_column: str = "text"
    available_splits: tuple[str, ...] = ("train",)


DATASET_SPECS: dict[Genre, DatasetSpec] = {
    "literature": DatasetSpec(path="globis-university/aozorabunko-clean"),
    "wiki": DatasetSpec(path="hpprc/jawiki-slim"),
    "textbook": DatasetSpec(path="hpprc/jawiki-books-paragraphs"),
    "research": DatasetSpec(path="kunishou/J-ResearchCorpus"),
    "web": DatasetSpec(path="hpprc/llmjp-warp-html"),
    "cc100": DatasetSpec(path="singletongue/cc100-documents", name="ja"),
}

RECOMMENDED_MIXES: dict[MixPreset, dict[Genre, float]] = {
    "small": {
        "wiki": 0.45,
        "textbook": 0.2,
        "web": 0.2,
        "literature": 0.1,
        "research": 0.05,
    },
    "balanced": {
        "wiki": 0.35,
        "web": 0.3,
        "cc100": 0.2,
        "textbook": 0.1,
        "literature": 0.03,
        "research": 0.02,
    },
    "web_heavy": {
        "web": 0.4,
        "cc100": 0.35,
        "wiki": 0.15,
        "textbook": 0.07,
        "literature": 0.02,
        "research": 0.01,
    },
}

VALID_STOPPING_STRATEGIES: set[str] = {
    "all_exhausted",
    "first_exhausted",
    "all_exhausted_without_replacement",
}


def get_recommended_mix(preset: MixPreset = "balanced") -> dict[Genre, float]:
    if preset not in RECOMMENDED_MIXES:
        raise ValueError(
            f"未知のプリセットです: {preset}. "
            f"利用可能: {', '.join(sorted(RECOMMENDED_MIXES.keys()))}"
        )
    return dict(RECOMMENDED_MIXES[preset])


def _resolve_split(spec: DatasetSpec, requested_split: Split) -> str:
    if requested_split in spec.available_splits:
        return requested_split

    available = ", ".join(spec.available_splits)
    raise ValueError(
        f"{spec.path} は split={requested_split!r} をサポートしていません。"
        f"利用可能な split: {available}"
    )


def _load_single_genre_dataset(genre: Genre, split: Split) -> IterableDataset:
    spec = DATASET_SPECS[genre]
    last_error: Exception | None = None

    # ネットワーク一時障害を吸収するため、軽いリトライを行う。
    for attempt in range(1, 4):
        try:
            ds = load_dataset(
                spec.path,
                name=spec.name,
                split=_resolve_split(spec, split),
                streaming=True,
            )
            if spec.text_column not in ds.column_names:
                raise ValueError(
                    f"{spec.path} に text 列がありません。"
                    f"取得した列: {ds.column_names}"
                )
            return ds.select_columns([spec.text_column])
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(float(attempt))

    raise RuntimeError(
        f"{spec.path} の読み込みに失敗しました。"
        "一時的なネットワーク障害の可能性があります。"
    ) from last_error


def _validate_mix(genre_mix: Mapping[Genre, float]) -> tuple[list[Genre], list[float]]:
    if not genre_mix:
        raise ValueError("genre の辞書が空です。少なくとも1つのジャンルを指定してください。")

    unknown = [key for key in genre_mix.keys() if key not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"未知のジャンルが含まれています: {unknown}")

    genres = list(genre_mix.keys())
    weights = list(genre_mix.values())
    if any(weight <= 0 for weight in weights):
        raise ValueError("各重みは 0 より大きい必要があります。")

    if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("重みの合計は 1.0 である必要があります。")

    return genres, weights


def get_dataset(
    genre: Genre | Mapping[Genre, float] = "literature",
    split: Split = "train",
    shuffle: bool = True,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    stopping_strategy: StoppingStrategy = "first_exhausted",
) -> IterableDataset:
    """
    単一ジャンルまたは重み付きミックスで、ストリーミング可能な日本語データセットを返します。

    `split='validation'` / `split='test'` は、対象データセット側に split がない場合はエラーになります。
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split は 'train', 'validation', 'test' のいずれかで指定してください。")

    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size は正の整数である必要があります。")

    if isinstance(genre, str):
        if genre not in DATASET_SPECS:
            raise ValueError(f"未知のジャンルです: {genre}")
        ds = _load_single_genre_dataset(genre, split)
    elif isinstance(genre, Mapping):
        if stopping_strategy not in VALID_STOPPING_STRATEGIES:
            raise ValueError(
                "stopping_strategy は "
                "'all_exhausted', 'first_exhausted', "
                "'all_exhausted_without_replacement' のいずれかで指定してください。"
            )

        genres, weights = _validate_mix(genre)
        datasets = [_load_single_genre_dataset(g, split) for g in genres]
        ds = interleave_datasets(
            datasets,
            probabilities=weights,
            seed=shuffle_seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        raise TypeError("genre は文字列または重み付き辞書で指定してください。")

    if shuffle:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return ds


def print_dataset_samples(
    ds: IterableDataset, sample_count: int = 3, prefix: str = "Sample"
) -> None:
    iterator = iter(ds)
    try:
        for i in range(sample_count):
            try:
                sample = next(iterator)
            except StopIteration:
                break
            print(f"{prefix} {i + 1}: {sample['text'][:100]}...")
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    # 単一ジャンルの取得例
    literature_ds = get_dataset(genre="literature", split="train")
    print_dataset_samples(literature_ds, sample_count=3, prefix="Literature sample")

    # mixed の streaming + interleave は環境依存で終了時にハングすることがあるため opt-in にする。
    if os.getenv("RUN_MIXED_EXAMPLE", "0") == "1":
        mixed_genre = get_recommended_mix("small")
        mixed_ds = get_dataset(genre=mixed_genre, split="train")
        print_dataset_samples(mixed_ds, sample_count=3, prefix="Mixed sample")
    else:
        print("Mixed sample はスキップしました。実行する場合は RUN_MIXED_EXAMPLE=1 を指定してください。")

    print("データセットの取得と表示が完了しました。")
