# Japanese 32k tokenizer

このディレクトリには、現在の事前学習設定が使用する日本語Unigram tokenizerを置く。

- `sentencepiece/` は再変換可能なSentencePiece原本である。
- `huggingface/` は学習コードが直接読み込む成果物である。
- 語彙数は32,768で、`<unk>`、`<s>`、`</s>`、`<pad>`、`<mask>`のIDは順に0〜4である。
- 既存成果物の正確なコーパスsnapshotと生成日時は過去に記録されていない。現在の推奨生成条件は `configs/tokenizer/ja_32k.toml` を参照する。

再生成すると既存モデルとの語彙互換性が失われるため、生成後は`manifest.json`のSHA-256を更新し、新規実験として扱う。
