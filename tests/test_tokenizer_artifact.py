import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llm_scratch_lab.tokenization import _backend_from_sentencepiece


def test_tokenizer_manifest_matches_files() -> None:
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizers" / "ja_32k"
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vocab_size"] == 32_768
    assert manifest["special_token_ids"] == {
        "unk": 0,
        "bos": 1,
        "eos": 2,
        "pad": 3,
        "mask": 4,
        "im_start": 5,
        "im_end": 6,
    }
    for relative_path, expected in manifest["sha256"].items():
        actual = hashlib.sha256((artifact_dir / relative_path).read_bytes()).hexdigest()
        assert actual == expected


def test_huggingface_tokenizer_has_expected_vocabulary() -> None:
    tokenizer_json = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "tokenizers"
        / "ja_32k"
        / "huggingface"
        / "tokenizer.json"
    )
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocabulary = payload["model"]["vocab"]
    assert len(vocabulary) == 32_768
    assert [piece for piece, _ in vocabulary[:5]] == [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "<mask>",
    ]


def test_sentencepiece_export_matches_tracked_huggingface_tokenizer() -> None:
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizers" / "ja_32k"
    tracked = AutoTokenizer.from_pretrained(artifact_dir / "huggingface", use_fast=True)
    converted = PreTrainedTokenizerFast(
        tokenizer_object=_backend_from_sentencepiece(
            artifact_dir / "sentencepiece" / "tokenizer.model"
        ),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    for text in ("こんにちは、世界。", "LLMを一から学ぶ", "emoji🙂と未知文字𰻞"):
        expected = tracked.encode(text, add_special_tokens=True)
        actual = converted.encode(text, add_special_tokens=True)
        assert actual == expected
        assert converted.decode(actual) == tracked.decode(expected)
