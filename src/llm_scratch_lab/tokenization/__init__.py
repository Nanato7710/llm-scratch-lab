from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from pathlib import Path
from typing import Literal

import sentencepiece as spm
from datasets import interleave_datasets, load_dataset
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from llm_scratch_lab.data.hf_streaming import SourceConfig

LOGGER = logging.getLogger(__name__)


class TokenizerPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceConfig] = Field(min_length=1)
    artifact_dir: str
    corpus_path: str = "corpus.txt"
    vocab_size: int = Field(default=32_768, gt=0)
    max_sentences: int = Field(default=10_000_000, gt=0)
    max_file_size_bytes: int = Field(default=20 * 1024**3, gt=0)
    shuffle_seed: int = 42
    shuffle_buffer_size: int = Field(default=10_000, gt=0)
    character_coverage: float = Field(default=0.9995, gt=0, le=1)
    normalization_rule_name: str = "nmt_nfkc"
    byte_fallback: bool = True
    user_defined_symbols: list[str] = Field(
        default_factory=lambda: ["<mask>", "<|im_start|>", "<|im_end|>"]
    )

    @model_validator(mode="after")
    def validate_source_weights(self) -> TokenizerPipelineConfig:
        if len(self.sources) > 1:
            total = sum(source.weight for source in self.sources)
            if abs(total - 1.0) > 1e-9:
                raise ValueError("source weights must sum to 1.0")
        if len(self.user_defined_symbols) != len(set(self.user_defined_symbols)):
            raise ValueError("user_defined_symbols must not contain duplicates")
        reserved = {"<unk>", "<s>", "</s>", "<pad>"}
        if reserved.intersection(self.user_defined_symbols):
            raise ValueError("user_defined_symbols must not repeat reserved tokens")
        return self


def load_tokenizer_pipeline_config(
    path: str | Path,
) -> tuple[TokenizerPipelineConfig, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as file:
        config = TokenizerPipelineConfig.model_validate(tomllib.load(file))
    return config, config_path.parent


def _resolve(base_dir: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def make_corpus(config: TokenizerPipelineConfig, base_dir: Path) -> Path:
    datasets = []
    for source in config.sources:
        dataset = load_dataset(
            source.path,
            name=source.name,
            split=source.split,
            streaming=True,
        )
        if source.text_column not in dataset.column_names:
            raise ValueError(f"{source.path} has no {source.text_column!r} column")
        if source.text_column != "text":
            dataset = dataset.rename_column(source.text_column, "text")
        datasets.append(dataset.select_columns(["text"]))
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = interleave_datasets(
            datasets,
            probabilities=[source.weight for source in config.sources],
            seed=config.shuffle_seed,
            stopping_strategy="all_exhausted",
        )
    combined = combined.shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)

    output_path = _resolve(base_dir, config.corpus_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    sentences_written = 0
    with output_path.open("w", encoding="utf-8") as output:
        for sample in combined:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if not text:
                continue
            encoded = (text + "\n").encode("utf-8")
            if bytes_written + len(encoded) > config.max_file_size_bytes:
                break
            output.write(encoded.decode("utf-8"))
            bytes_written += len(encoded)
            sentences_written += 1
            if sentences_written >= config.max_sentences:
                break
    if sentences_written == 0:
        raise RuntimeError("Tokenizer corpus contains no usable sentences")
    LOGGER.info("Wrote %d sentences to %s", sentences_written, output_path)
    return output_path


def train_sentencepiece(config: TokenizerPipelineConfig, base_dir: Path) -> Path:
    corpus_path = _resolve(base_dir, config.corpus_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    artifact_dir = _resolve(base_dir, config.artifact_dir)
    sentencepiece_dir = artifact_dir / "sentencepiece"
    sentencepiece_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = sentencepiece_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=config.vocab_size,
        model_type="unigram",
        character_coverage=config.character_coverage,
        normalization_rule_name=config.normalization_rule_name,
        byte_fallback=config.byte_fallback,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        user_defined_symbols=config.user_defined_symbols,
    )
    return model_prefix.with_suffix(".model")


def _backend_from_sentencepiece(model_path: Path) -> Tokenizer:
    from sentencepiece import sentencepiece_model_pb2

    model_proto = sentencepiece_model_pb2.ModelProto()
    model_proto.ParseFromString(model_path.read_bytes())
    if model_proto.trainer_spec.model_type != 1:
        raise ValueError("Only SentencePiece Unigram models are supported")
    vocabulary = [(piece.piece, piece.score) for piece in model_proto.pieces]
    backend = Tokenizer(
        Unigram(
            vocabulary,
            model_proto.trainer_spec.unk_id,
            model_proto.trainer_spec.byte_fallback,
        )
    )
    charsmap = model_proto.normalizer_spec.precompiled_charsmap
    normalize_steps = []
    if charsmap:
        normalize_steps.append(normalizers.Precompiled(charsmap))
    normalize_steps.append(normalizers.Replace(Regex(" {2,}"), " "))
    backend.normalizer = normalizers.Sequence(normalize_steps)
    backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    backend.decoder = decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Metaspace(replacement="▁", prepend_scheme="always"),
        ]
    )
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    return backend


def export_huggingface(config: TokenizerPipelineConfig, base_dir: Path) -> Path:
    artifact_dir = _resolve(base_dir, config.artifact_dir)
    model_path = artifact_dir / "sentencepiece" / "tokenizer.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
    output_dir = artifact_dir / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend_from_sentencepiece(model_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.save_pretrained(output_dir)
    write_manifest(config, artifact_dir)
    return output_dir


def write_manifest(config: TokenizerPipelineConfig, artifact_dir: Path) -> None:
    files = sorted(path for path in artifact_dir.rglob("*") if path.is_file())
    checksums = {
        str(path.relative_to(artifact_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.name != "manifest.json"
    }
    processor = spm.SentencePieceProcessor(
        model_file=str(artifact_dir / "sentencepiece" / "tokenizer.model")
    )
    known_special_tokens = {
        "unk": "<unk>",
        "bos": "<s>",
        "eos": "</s>",
        "pad": "<pad>",
        "mask": "<mask>",
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
    }
    special_token_ids = {
        name: processor.piece_to_id(token)
        for name, token in known_special_tokens.items()
        if processor.piece_to_id(token) >= 0
    }
    manifest = {
        "format_version": 1,
        "vocab_size": config.vocab_size,
        "special_token_ids": special_token_ids,
        "sources": [source.model_dump(mode="json") for source in config.sources],
        "sha256": checksums,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(
    mode: Literal["corpus", "train", "export", "all"],
    config: TokenizerPipelineConfig,
    base_dir: Path,
) -> Path:
    if mode == "corpus":
        return make_corpus(config, base_dir)
    if mode == "train":
        return train_sentencepiece(config, base_dir)
    if mode == "export":
        return export_huggingface(config, base_dir)
    make_corpus(config, base_dir)
    train_sentencepiece(config, base_dir)
    return export_huggingface(config, base_dir)


__all__ = [
    "TokenizerPipelineConfig",
    "export_huggingface",
    "load_tokenizer_pipeline_config",
    "make_corpus",
    "run_pipeline",
    "train_sentencepiece",
]
