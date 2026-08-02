# llm-scratch-lab

`llm-scratch-lab`は、言語モデルの構造と学習処理をPyTorchで追いながら実験するためのリポジトリです。
現在は、テキスト専用のGemma 3風decoderと日本語データによる事前学習を実装しています。
公式Gemma 3のcheckpointとの互換性はありません。

## セットアップ

Python 3.13と[uv](https://docs.astral.sh/uv/)を使用します。

```bash
uv sync
```

TensorBoardまたはWeights & Biasesを使う場合だけ、tracking extraを追加します。

```bash
uv sync --extra tracking
```

Linuxでは、`pyproject.toml`に定義したPyTorch CUDA 12.6用indexを使用します。
macOSでは通常のPyTorch packageが選ばれ、利用可能ならMPSを使います。

## ドキュメント

初めてこのリポジトリを変更する場合は、次の順で参照してください。

[視覚的なWeb版](docs/web/index.html)では、三つの文書を図解、検索、コードコピー付きの一画面で読めます。
`docs/web/index.html`を直接開くか、リポジトリ直下で次のコマンドを実行して`http://localhost:8000/docs/web/`を開きます。

```bash
uv run python -m http.server 8000
```

1. [プロジェクトの構成](docs/architecture.md)では、設定から学習結果までの流れと各packageの責務を説明します。
2. [コンポーネントの追加手順](docs/adding-components.md)では、モデル、データ、学習方法、オプティマイザの実装テンプレートとテスト方法を示します。
3. [実験の設定と実行](docs/experiment-guide.md)では、TOML、CLI上書き、出力、checkpointからの再開を説明します。

## 実験の実行

base modelの事前学習は次のコマンドで始めます。

```bash
uv run llm-lab train --config configs/experiments/gemma3_base.toml
```

elementwise attention gateを加えた実験は、別の設定ファイルを指定します。

```bash
uv run llm-lab train --config configs/experiments/gemma3_gated.toml
```

設定の一部はCLIから上書きできます。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --device cuda \
  --max-updates 1000 \
  --tracker tensorboard
```

trackerは既定で無効です。
有効にしたtrackerのimportと初期化は、学習を明示的に実行した後にだけ行われます。

## コンポーネントの組み合わせ

実験設定は、モデル、データ、学習方式、optimizerを独立した**コンポーネント**として選びます。
各コンポーネントは固有のPydantic設定を持ち、学習開始前に検証されます。

```toml
[model]
name = "gemma3"

[model.config]
attention_gate = "none"

[data]
name = "hf_streaming"

[method]
name = "causal_pretraining"

[optimizer]
name = "muon_radam_schedulefree"
```

登録済みコンポーネントと設定schemaはCLIから確認できます。

```bash
uv run llm-lab components list
uv run llm-lab components describe model gemma3
```

共通training engineは、勾配蓄積、評価、checkpoint、trackerだけを管理します。
lossの定義とbatchの意味はtraining methodが担当するため、モデルやデータの実装へ学習ループを複製する必要はありません。

## 新しいコンポーネントの追加

新しい実装は、設定型、本体、factory、Registryへの明示的な登録を一つの単位として追加します。
種類ごとの契約とコード例は[コンポーネントの追加手順](docs/adding-components.md)を参照してください。
最低限の受入条件は、小型モデルとメモリ上のデータで共通Engineを通し、一つ以上のoptimizer updateとcheckpoint保存を完了できることです。

## モデル実装

現在の`gemma3`コンポーネントは、Q/K/V射影、Grouped-Query Attention、RoPE、global/local causal mask、RMSNorm、feed-forward networkを明示的に実装しています。
処理を追いやすくするため、Scaled Dot Product AttentionやKV cacheへの置換は行っていません。

RMSNormは重みを0で初期化し、forward時に`1 + weight`を掛けます。
`attention_gate = "elementwise"`では、attention headを結合した後、出力射影の前にsigmoid gateを掛けます。
このgateはGemma 3本体の仕様ではなく、独立した実験条件です。

モデル実装は、次の資料を参照しています。

- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Gemma 3 From Scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/12_gemma3)
- [Gated Attention for Large Language Models](https://github.com/qiuzh20/gated_attention)

## データとtokenizer

`hf_streaming`コンポーネントは、Hugging Face Hub上のデータをstreamingで読み、複数sourceを重み付きで混合できます。
テキストはsample境界へEOSを挿入した後、固定長のnext-token prediction blockへ詰めます。

現在のtokenizerは`artifacts/tokenizers/ja_32k/`でGit管理しています。
SentencePiece原本、Hugging Face実行用ファイル、checksumを分けて保存しています。
既存成果物の正確なコーパスsnapshotは記録されていないため、再生成したtokenizerを既存モデルと互換だとは扱えません。

tokenizerの推奨生成条件はTOMLに置いています。

```bash
uv run llm-lab tokenizer all --config configs/tokenizer/ja_32k.toml
```

コーパスと再生成物は大きくなるため、出力先と空き容量を確認してから実行してください。

## 出力と再開

各実験の出力は`outputs/<experiment>/<run-id>/`に保存され、Gitでは追跡しません。
ディレクトリには解決済み設定、tracker log、`latest.pt`が入ります。
評価を実行した場合は、評価値が最良だった`best.pt`も保存します。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --resume outputs/gemma3-base/<run-id>/checkpoints/latest.pt
```

checkpointはモデル、optimizer、乱数、optimizer update数、データストリーム位置、packing中のtokenを保存します。
`num_workers = 0`かつshuffleなしなら、次のbatchから再開する動作をテストできます。
Hugging Face streaming datasetでshuffle bufferを使う場合、再開後にbufferが再充填されるためsample順序は完全には一致しません。

## 検証

通常の検証はネットワークやGPUを必要としません。

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

## Git運用

GitHub Flowを採用します。
ブランチ名は`type/short-description`とします。
`type`には`feature`、`fix`、`refactor`、`docs`、`test`、`perf`、`chore`を使用します。
