# 実験の設定と実行

この文書は、実験TOMLの読み方、CLIによる実行、出力、checkpointからの再開を説明します。
実装の責務は[プロジェクトの構成](architecture.md)を、新しい実装の接続方法は[コンポーネントの追加手順](adding-components.md)を参照してください。

## 実行環境

このプロジェクトはPython 3.13とuvを使用します。

```bash
uv sync
```

TensorBoardまたはWeights & Biasesを使う場合は、tracking用の依存関係も同期します。

```bash
uv sync --extra tracking
```

Linuxでは`pyproject.toml`に定義したCUDA 12.6用PyTorch indexを使用します。
macOSでは通常のPyTorch packageを使用し、`device = "auto"`ならCUDA、MPS、CPUの順で利用可能なdeviceを選びます。

## 実験TOMLの構造

実験TOMLは、四種類のコンポーネント、runtime、tracking、outputで構成します。

```toml
[model]
name = "gemma3"

[model.config]
vocab_size = 32768
context_length = 1024
# モデル固有の設定を続ける

[data]
name = "hf_streaming"

[data.config]
tokenizer_path = "../../artifacts/tokenizers/ja_32k/huggingface"
batch_size = 16
sequence_length = 1024

[[data.config.train_sources]]
path = "epfml/FineWeb2-HQ"
name = "jpn_Jpan"
split = "train"
text_column = "text"

[[data.config.eval_sources]]
path = "globis-university/aozorabunko-clean"
split = "train"
text_column = "text"

[method]
name = "causal_pretraining"

[method.config]
ignore_index = -100

[optimizer]
name = "muon_radam_schedulefree"

[optimizer.config]
muon_lr = 0.02
radam_lr = 0.005

[runtime]
device = "auto"
seed = 42
max_updates = 8192
gradient_accumulation_steps = 8
max_grad_norm = 1.0
evaluation_interval = 16
checkpoint_interval = 16
evaluation_batches = 30
compile = false
num_workers = 0

[tracking]
backends = []
project = "gemma3-play"

[output]
root = "../../outputs"
experiment_name = "gemma3-base"
```

`model.config`、`data.config`、`method.config`、`optimizer.config`のschemaは、選択した登録名によって決まります。
登録済みの名前とschemaはCLIで確認できます。

```bash
uv run llm-lab components list
uv run llm-lab components describe model gemma3
```

未知の項目は無視されず、Pydanticのvalidation errorになります。
設定名の入力ミスを学習前に検出できるため、共通設定と各コンポーネント設定のどちらも`extra="forbid"`です。

## 相対パスの基準

実験TOML内のローカル相対パスは、コマンドを実行したディレクトリではなくTOMLが置かれたディレクトリを基準に解決します。
たとえば`configs/experiments/gemma3_base.toml`の`../../artifacts`は、リポジトリ直下の`artifacts`を指します。

現在この規則を使う主な項目は、`data.config.tokenizer_path`と`output.root`です。
Hugging Face Datasetの`path`はHub上の識別子なので、ローカル相対パスとしては解決しません。

## 実験の開始

Base modelの事前学習は次のコマンドで始めます。

```bash
uv run llm-lab train --config configs/experiments/gemma3_base.toml
```

Elementwise attention gateを使う実験は、別の設定ファイルを指定します。

```bash
uv run llm-lab train --config configs/experiments/gemma3_gated.toml
```

現在CLIから上書きできるTOML項目は、`runtime.device`、`runtime.max_updates`、`tracking.backends`です。
CLIで指定した値はTOMLより優先されます。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --device cuda \
  --max-updates 1000 \
  --tracker tensorboard
```

複数のtrackerを使う場合は`--tracker`を繰り返します。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --tracker tensorboard \
  --tracker wandb
```

`--resume`は設定値の上書きではなく、読み込むcheckpointの指定です。

## Runtime設定

| 項目 | 意味 |
| --- | --- |
| `device` | `auto`またはPyTorchが解釈できるdevice名です。 |
| `seed` | Python、PyTorch、利用可能なCUDA deviceへ設定する乱数seedです。 |
| `max_updates` | 完了した時点で学習を終了するoptimizer update数です。 |
| `gradient_accumulation_steps` | 一度のoptimizer updateへ蓄積するmicrobatch数です。 |
| `max_grad_norm` | 勾配normの上限で、`null`ならclipしません。 |
| `evaluation_interval` | 何updateごとに評価するかを指定します。 |
| `checkpoint_interval` | 何updateごとに`latest.pt`を保存するかを指定します。 |
| `evaluation_batches` | 一度の評価で処理する最大batch数です。 |
| `compile` | `true`ならOptimizerを作る前に`torch.compile()`を適用します。 |
| `num_workers` | DataLoaderへ渡すworker数です。 |

**microbatch**はDataLoaderから一度取得するbatchです。
**optimizer update**は、指定数のmicrobatchでbackwardした後に`optimizer.step()`を一度実行する単位です。

`evaluation_interval`、`checkpoint_interval`、`max_updates`はいずれもmicrobatch数ではなくoptimizer update数です。
学習データが`max_updates`より前に終了した場合、Engineは未完了の実験として例外を送出します。

## 評価値

現在の共通評価関数は、学習方法が返したlossを`item_count`で重み付けして平均します。
因果言語モデルの事前学習では、有効なtarget token数を`item_count`としてNLLを集計します。

記録する評価値は`eval/nll`と`eval/perplexity`です。
Perplexityは平均NLLの指数で、指数計算がoverflowした場合は無限大になります。
評価対象の有効tokenが一つもない場合は、数値を記録せずに失敗します。

学習時の`train/nll`は、勾配蓄積に使ったmicrobatch lossの単純平均です。
評価時のtoken数による重み付き平均とは集計方法が異なる点に注意してください。

## Tracker

`tracking.backends = []`ではNo-op Trackerを使い、外部tracking packageをimportしません。
`tensorboard`を選ぶとrun directory内の`tensorboard/`へ記録します。
`wandb`を選ぶと`tracking.project`とrun directory名を使ってrunを初期化します。

Trackerの作成に失敗した場合は、`uv sync --extra tracking`を実行したか確認します。
Weights & Biasesの認証や通信設定はW&B側の環境設定に従います。

## 出力ディレクトリ

新規実行の出力先は次の形です。

```text
<output.root>/
└── <experiment_name>/
    └── <YYYYMMDD-HHMMSS>/
        ├── config.json
        ├── checkpoints/
        │   ├── latest.pt
        │   └── best.pt
        └── tensorboard/
```

`config.json`はCLI上書き後の解決済み設定です。
`latest.pt`は設定した間隔と正常終了時に保存されます。
`best.pt`は評価を実行し、NLLがそれまでの最良値を更新した場合だけ作られます。
`tensorboard/`も、そのbackendを選んだ場合だけ作られます。

`outputs/`は再生成可能な実験出力の置き場であり、Gitでは追跡しません。
共有が必要な小さな固定成果物は、来歴とchecksumを付けて`artifacts/`へ移します。

## Checkpointからの再開

`--resume`にはcheckpoint fileまたは`checkpoints` directoryを指定できます。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --resume outputs/gemma3-base/<run-id>/checkpoints/latest.pt
```

`checkpoints` directoryを指定した場合は、その直下の`latest.pt`を読みます。
Run directory自体ではなく、その中の`checkpoints`を指定してください。

再開時は新しいrun directoryを作らず、checkpointが属するrun directoryの`config.json`とcheckpointを更新します。
Engineはモデル、データ、学習方法、オプティマイザの登録名が現在のTOMLと一致することを確認します。

Checkpointにはモデル、オプティマイザ、データストリーム、乱数、update数、最良NLLが保存されます。
同じ登録名でもモデル形状やOptimizerのパラメータ構成を変えるとstate dictを読み込めないため、通常は固有設定も変更しません。

ストリーミングデータの位置をmain processから正確に保存するには`runtime.num_workers = 0`を使います。
Hugging Face streaming datasetでshuffle bufferを有効にすると、再開後にbufferを再充填するためsample順序は完全には一致しません。
次のbatchまで一致させる検証では、`num_workers = 0`かつshuffleなしにします。

## Tokenizer成果物

現在の学習設定は`artifacts/tokenizers/ja_32k/huggingface/`を参照します。
SentencePiece原本、Hugging Face形式、語彙情報とchecksumは`artifacts/tokenizers/ja_32k/`に分けて保存しています。

既存tokenizerの正確なコーパスsnapshotと生成日時は記録されていません。
そのため、同じ推奨条件で再生成しても既存tokenizerや既存モデルと互換だとは扱えません。

現在の推奨生成条件は`configs/tokenizer/ja_32k.toml`にあります。

```bash
uv run llm-lab tokenizer all --config configs/tokenizer/ja_32k.toml
```

`all`はコーパス構築、SentencePiece学習、Hugging Face形式への変換を順に実行します。
個別に実行する場合のmodeは`corpus`、`train`、`export`です。

コーパスと途中成果物は大きくなるため、`corpus_path`、`artifact_dir`、空き容量を確認してから実行します。
Tokenizerを更新した場合は`manifest.json`のSHA-256と来歴を更新し、既存モデルとは別の実験として扱います。

## 実験前の検証

通常の自動テストはネットワークやGPUを必要としません。

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

これらの検査は、Hugging Face Hubから実データを取得する長時間学習の完走を保証しません。
実学習の前には、対象環境で小さい`max_updates`を指定し、データ取得、device、tracker、checkpoint保存を確認します。

```bash
uv run llm-lab train \
  --config configs/experiments/gemma3_base.toml \
  --max-updates 1
```

この確認も実データとモデルを読み込むため、ネットワーク、メモリ、選択したdeviceが必要です。
