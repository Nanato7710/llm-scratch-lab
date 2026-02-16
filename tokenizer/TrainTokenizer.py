"""
Tokenizer の作成・変換・読み込みを行うモジュール。

主な流れ:
1. コーパスを作る: make_corpus()
2. SentencePiece を学習する: train_tokenizer()
3. Hugging Face tokenizer(JSON)に変換する: convert_to_hf_tokenizer()
4. FastTokenizer として設定・保存する: load_hf_tokenizer()
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import sentencepiece as spm
from tokenizers import decoders
from tokenizers.implementations import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing


LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logging() -> None:
    """
    外部ライブラリの INFO ログを抑制し、このモジュールのログのみ表示する。
    """
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)

    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "datasets",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "fsspec",
        "fsspec.asyn",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # datasets / huggingface_hub の進捗バーを抑制する。
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    try:
        import datasets

        if hasattr(datasets, "disable_progress_bars"):
            datasets.disable_progress_bars()
        elif hasattr(datasets, "disable_progress_bar"):
            datasets.disable_progress_bar()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def _get_dataset_module() -> tuple[Any, Any]:
    """
    dataset.PretrainDataset から get_dataset / get_recommended_mix を取り出す。
    スクリプト直実行時の import パス差分も吸収する。
    """
    try:
        from dataset.PretrainDataset import get_dataset, get_recommended_mix

        return get_dataset, get_recommended_mix
    except ImportError:
        project_root = Path(__file__).resolve().parents[1]
        import sys

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from dataset.PretrainDataset import get_dataset, get_recommended_mix

        return get_dataset, get_recommended_mix


def make_corpus(
    dataset_config: dict[str, float] | None = None,
    max_sentences: int = 1_000_000,
    max_file_size: int = 4 * 1024 * 1024 * 1024,  # 4GB
    output_file: str = "corpus.txt",
    shuffle_seed: int = 42,
) -> str:
    """
    ストリーミングデータセットから学習用コーパスを作成する。
    """
    if max_sentences <= 0:
        raise ValueError(f"max_sentences は正の整数である必要があります: {max_sentences}")
    if max_file_size <= 0:
        raise ValueError(f"max_file_size は正の整数である必要があります: {max_file_size}")

    get_dataset, get_recommended_mix = _get_dataset_module()
    if dataset_config is None:
        dataset_config = get_recommended_mix("balanced")

    ds = get_dataset(
        genre=dataset_config,
        split="train",
        shuffle=True,
        shuffle_seed=shuffle_seed,
        stopping_strategy="all_exhausted",
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("コーパスを作成します: %s", output_path)

    iterator = iter(ds)
    try:
        with output_path.open("w", encoding="utf-8") as fp:
            for i, sample in enumerate(iterator):
                text = sample["text"]
                fp.write(text.replace("。", "。\n") + "\n")

                if i % 10_000 == 0:
                    size_gb = output_path.stat().st_size / (1024**3)
                    print(
                        f"\r書き込み中: {i:,}文 / ファイルサイズ {size_gb:.2f} GB",
                        end="",
                        flush=True,
                    )

                if i + 1 >= max_sentences:
                    break

                if i % 1_000 == 0 and output_path.stat().st_size >= max_file_size:
                    LOGGER.warning(
                        "ファイルサイズが上限 %.2fGB を超えたため終了します",
                        max_file_size / (1024**3),
                    )
                    break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    print()
    LOGGER.info("コーパスを保存しました: %s", output_path)
    return str(output_path)


def train_tokenizer(
    vocab_size: int = 12_000,
    input_corpus: str = "corpus.txt",
    output_model_prefix: str | None = None,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
    special_tokens: list[str] | None = None,
    normalization_rule_name: str = "nmt_nfkc",
    normalization_rule_tsv: str | None = None,
) -> str:
    """
    SentencePiece tokenizer を学習する。
    """
    if vocab_size < 100:
        raise ValueError(f"vocab_size は 100 以上で指定してください: {vocab_size}")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError(
            f"character_coverage は 0.0 より大きく 1.0 以下で指定してください: {character_coverage}"
        )
    if model_type not in {"unigram", "bpe", "word", "char"}:
        raise ValueError(f"不正な model_type です: {model_type}")
    if not normalization_rule_name.strip():
        raise ValueError("normalization_rule_name は空文字を指定できません。")

    corpus_path = Path(input_corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"コーパスが見つかりません: {input_corpus}")
    if normalization_rule_tsv is not None and not Path(normalization_rule_tsv).exists():
        raise FileNotFoundError(
            f"normalization_rule_tsv が見つかりません: {normalization_rule_tsv}"
        )

    if output_model_prefix is None:
        output_model_prefix = f"tokenizer_{vocab_size // 1000}k"

    if special_tokens is None:
        special_tokens = ["<pad>", "<mask>", "<|im_start|>", "<|im_end|>"]

    LOGGER.info(
        "SentencePiece を学習します (vocab_size=%s, model_type=%s, normalization=%s)",
        vocab_size,
        model_type,
        normalization_rule_name,
    )

    train_kwargs = {
        "input": str(corpus_path),
        "model_prefix": output_model_prefix,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": model_type,
        "byte_fallback": True,
        "num_threads": max(1, os.cpu_count() or 1),
        "shuffle_input_sentence": True,
        "pad_id": 3,
        "pad_piece": "<pad>",
        "user_defined_symbols": special_tokens,
        "normalization_rule_name": normalization_rule_name,
        # sentencepiece の C++ INFO ログを抑える。
        "minloglevel": 2,
    }
    if normalization_rule_tsv is not None:
        train_kwargs["normalization_rule_tsv"] = normalization_rule_tsv

    try:
        spm.SentencePieceTrainer.Train(**train_kwargs)
    except OSError as exc:
        if 'unknown field name "minloglevel"' in str(exc):
            train_kwargs.pop("minloglevel", None)
            spm.SentencePieceTrainer.Train(**train_kwargs)
        else:
            raise

    LOGGER.info("学習済みモデルを出力しました: %s.model", output_model_prefix)
    return output_model_prefix


def convert_to_hf_tokenizer(
    spm_model_path: str = "tokenizer_12k.model",
    json_output_path: str | None = None,
) -> str:
    """
    SentencePiece(.model) を tokenizers JSON 形式へ変換する。
    """
    model_path = Path(spm_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece モデルが見つかりません: {spm_model_path}")

    if json_output_path is None:
        json_output_path = str(model_path.with_suffix(".json"))

    LOGGER.info("HF tokenizer(JSON)へ変換します: %s", model_path)
    tokenizer = SentencePieceUnigramTokenizer.from_spm(str(model_path))
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), tokenizer.decoder])
    tokenizer.save(json_output_path)
    LOGGER.info("変換結果を保存しました: %s", json_output_path)
    return json_output_path


def load_hf_tokenizer(
    tokenizer_json: str = "tokenizer_12k.json",
    output_dir: str = "./hf_tokenizer_12k",
):
    """
    PreTrainedTokenizerFast として読み込み、special token 設定を付与して保存する。
    """
    tokenizer_path = Path(tokenizer_json)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer JSON が見つかりません: {tokenizer_json}")

    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "load_hf_tokenizer を使うには transformers が必要です。"
            " `uv add transformers` を実行してください。"
        ) from exc

    LOGGER.info("PreTrainedTokenizerFast を構成します: %s", tokenizer_path)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"
    tok.pad_token = "<pad>"
    tok.mask_token = "<mask>"

    bos_id = tok.convert_tokens_to_ids(tok.bos_token)
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)
    if bos_id is None or eos_id is None:
        raise RuntimeError("<s> または </s> の token id を取得できませんでした。")

    tok.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{tok.bos_token} $A {tok.eos_token}",
        pair=f"{tok.bos_token} $A {tok.eos_token} $B:1 {tok.eos_token}:1",
        special_tokens=[(tok.bos_token, bos_id), (tok.eos_token, eos_id)],
    )

    tok.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(output_dir)
    LOGGER.info("Tokenizer を保存しました: %s", output_dir)
    return tok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenizer 作成ユーティリティ")
    parser.add_argument(
        "--mode",
        choices=["corpus", "train", "convert", "load", "all"],
        default="train",
        help="実行モード",
    )
    parser.add_argument("--vocab_size", type=int, default=12_000, help="語彙数")
    parser.add_argument("--input_corpus", default="corpus.txt", help="入力コーパス")
    parser.add_argument("--output_model_prefix", default=None, help="SPM モデル出力 prefix")
    parser.add_argument("--spm_model", default="tokenizer_12k.model", help="SPM モデルパス")
    parser.add_argument("--tokenizer_json", default="tokenizer_12k.json", help="tokenizer JSON")
    parser.add_argument("--output_dir", default="./hf_tokenizer_12k", help="HF 保存先")
    parser.add_argument("--max_sentences", type=int, default=1_000_000, help="最大文数")
    parser.add_argument("--max_file_size", type=int, default=4 * 1024 * 1024 * 1024, help="最大バイト")
    parser.add_argument(
        "--normalization_rule_name",
        default="nmt_nfkc",
        help="SentencePiece の正規化ルール名 (例: nmt_nfkc, nfkc, nfkc_cf, identity)",
    )
    parser.add_argument(
        "--normalization_rule_tsv",
        default=None,
        help="SentencePiece カスタム正規化TSVのパス",
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    if args.mode == "corpus":
        make_corpus(
            max_sentences=args.max_sentences,
            max_file_size=args.max_file_size,
            output_file=args.input_corpus,
        )
        return

    if args.mode == "train":
        train_tokenizer(
            vocab_size=args.vocab_size,
            input_corpus=args.input_corpus,
            output_model_prefix=args.output_model_prefix,
            normalization_rule_name=args.normalization_rule_name,
            normalization_rule_tsv=args.normalization_rule_tsv,
        )
        return

    if args.mode == "convert":
        convert_to_hf_tokenizer(spm_model_path=args.spm_model, json_output_path=args.tokenizer_json)
        return

    if args.mode == "load":
        load_hf_tokenizer(tokenizer_json=args.tokenizer_json, output_dir=args.output_dir)
        return

    make_corpus(
        max_sentences=args.max_sentences,
        max_file_size=args.max_file_size,
        output_file=args.input_corpus,
    )
    model_prefix = train_tokenizer(
        vocab_size=args.vocab_size,
        input_corpus=args.input_corpus,
        output_model_prefix=args.output_model_prefix,
        normalization_rule_name=args.normalization_rule_name,
        normalization_rule_tsv=args.normalization_rule_tsv,
    )
    tokenizer_json = convert_to_hf_tokenizer(spm_model_path=f"{model_prefix}.model")
    load_hf_tokenizer(tokenizer_json=tokenizer_json, output_dir=f"./hf_{model_prefix}")


if __name__ == "__main__":
    main()
