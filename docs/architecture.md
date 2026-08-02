# プロジェクトの構成

`llm-scratch-lab`は、言語モデルの構造と学習処理をPyTorchで追跡できる形に保ちながら、実験条件を交換できるようにした学習基盤です。
現在はテキスト専用のGemma 3風decoderと、日本語テキストを使う因果言語モデルの事前学習を実装しています。

この文書は、設定ファイルから学習結果が作られるまでの流れと、各ディレクトリの責務を説明します。
新しい実装を追加する場合は、[コンポーネントの追加手順](adding-components.md)も参照してください。

## 実験を構成する四つのコンポーネント

実験は、役割の異なる四つの**コンポーネント**をTOMLで選択します。
コンポーネントは、名前、Pydantic設定型、factory、説明文をRegistryへ登録した単位です。

- **モデル**：入力Tensorからlogitsを計算する`torch.nn.Module`です。
- **データ**：学習用DataLoader、評価用DataLoader factory、再開可能な状態を構築します。
- **学習方法**：batchの解釈、lossの計算、評価値の集計単位を定めます。
- **オプティマイザ**：モデルを受け取って、学習と評価の状態切り替えに対応するOptimizer Adapterを構築します。

Trackerはこの四種類には含まれません。
Trackerは学習基盤側のProtocolであり、`tracking.backends`からTensorBoardまたはWeights & Biasesを選びます。

## 設定から学習結果までの流れ

実験は次の順で組み立てられます。

```text
実験TOML
  ↓ load_experiment_config
ExperimentConfig と BuildContext
  ↓ create_default_registry
四種類のコンポーネントを検証して構築
  ↓ run_experiment
互換性検査 → DataLoader構築 → 学習と評価
  ↓
config.json、trackerの記録、checkpoint
```

`load_experiment_config()`はTOMLを読み、共通設定を`ExperimentConfig`として検証します。
この時点の各コンポーネント固有設定は`dict`であり、RegistryがコンポーネントのPydantic設定型を使って検証します。

**BuildContext**は、読み込んだ実験TOMLが置かれているディレクトリを保持します。
コンポーネントは`BuildContext.resolve_path()`を使うことで、設定内の相対パスをカレントディレクトリではなく実験TOMLから解決できます。
現在はtokenizerのパスと出力先がこの仕組みを使います。
Hugging Face Datasetの`path`はローカルパスではなくHub上の識別子なので、`resolve_path()`の対象ではありません。

`create_default_registry()`は、各コンポーネントpackageの`register()`を明示的に呼びます。
新しいmoduleを作成しただけではRegistryに追加されません。

`run_experiment()`は四種類のコンポーネントを構築した後、データが提供するbatch keyと学習方法が要求するbatch keyを比較します。
データが`batch_keys`を、学習方法が`required_batch_keys`を公開している場合、要求を満たさない組み合わせはDataLoaderを作る前に失敗します。

データコンポーネントが`tokenizer`を持ち、モデルが`config.vocab_size`を持つ場合は、両者の語彙数も比較します。
この検査により、異なる語彙で作ったtokenizerとモデルを誤って組み合わせる問題を学習前に検出できます。

互換性検査を通過すると、Engineはモデルをdeviceへ移動し、必要なら`torch.compile()`を適用します。
その後にOptimizer Builderへモデルを渡し、データコンポーネントからData Bundleを作ります。

## 各packageの責務

| 場所 | 責務 |
| --- | --- |
| `core/` | 設定、Registry、コンポーネント間で共有するProtocolとデータ型を定義します。 |
| `models/` | モデル設定と`torch.nn.Module`を実装します。 |
| `data/` | 生データの読み込み、tokenize、packing、DataLoader、再開状態を実装します。 |
| `methods/` | batchからlossと評価集計値を計算します。 |
| `optimizers/` | パラメータの分類、Optimizerの生成、状態保存を実装します。 |
| `training/` | コンポーネントを組み合わせ、勾配蓄積、評価、checkpoint、trackerを制御します。 |
| `tokenization/` | tokenizer用コーパスの構築、SentencePiece学習、Hugging Face形式への変換を行います。 |
| `configs/` | 実験条件とtokenizer生成条件をTOMLで管理します。 |
| `artifacts/` | 実験から参照するtokenizerなどの小さな固定成果物と来歴を管理します。 |
| `tests/` | コンポーネント単体と、共通Engineでの組み合わせを検証します。 |

コンポーネント固有の処理は各packageに置き、`training/`へ個別モデルやデータセットの分岐を追加しない構成です。
新しい組み合わせに必要な共通機能が見つかった場合だけ、`core/`または`training/`の契約を拡張します。

## コンポーネント間の契約

**ComponentSpec**は、TOMLの`name`と`config`を保持します。
Registryは`name`から登録情報を選び、`config`を登録済みのPydantic型で検証してからfactoryへ渡します。

**factory**は、検証済みの`BaseModel`と`BuildContext`を受け取るcallableです。
Factoryの戻り値はコンポーネントごとに異なりますが、共通Engineが期待する操作を提供する必要があります。

モデルの`forward()`は`ModelOutput`を返します。
現在の`ModelOutput`は`logits: torch.Tensor`だけを持ちます。

データコンポーネントの`build(num_workers=...)`は、学習用の`train_loader`と評価用の`eval_loader_factory`を持つbundleを返します。
Bundleはcheckpointに含める状態を`state_dict()`で返し、`load_state_dict()`で復元します。

学習方法の`training_step()`と`evaluation_step()`は`StepOutput`を返します。
`StepOutput.loss`は逆伝播または評価集計に使う平均lossで、`item_count`はその平均値の分母に使った要素数です。
因果言語モデルの事前学習では、`item_count`は`ignore_index`ではないtarget token数です。

Optimizer Adapterは`zero_grad()`、`step()`、`train()`、`eval()`、`state_dict()`、`load_state_dict()`を提供します。
任意の`metrics()`を実装すると、Engineは返された値へ`train/`を付けて記録します。

## 学習ループが管理する状態

**optimizer update**は、`optimizer.step()`を一度実行した単位です。
**microbatch**は、DataLoaderから一度取得してforwardとbackwardを実行したbatchです。

Engineは`gradient_accumulation_steps`個のmicrobatchについてlossを割ってbackwardし、その後に勾配clipとoptimizer updateを実行します。
評価間隔、checkpoint間隔、`max_updates`はすべてoptimizer update数で判定します。

評価では、各batchのlossへ`item_count`を掛けて合計し、全item数で割ってNLLを計算します。
有効なitemが一つもない場合は、誤った評価値を返さずに失敗します。

Engineが現在記録する学習値は、蓄積したmicrobatchの平均lossである`train/nll`と、Optimizer Adapterが任意に返すmetricsです。
`StepOutput.metrics`は共通契約に含まれますが、現在のEngineはその値をtrackerへ転送しません。

## 出力とcheckpoint

新規実行では、`<output.root>/<experiment_name>/<実行日時>/`をrun directoryとして作ります。
解決済みの設定は`config.json`へ保存され、checkpointは`checkpoints/`へ保存されます。

`latest.pt`は設定した間隔と学習終了時に更新されます。
評価NLLがそれまでの最良値を下回った場合は、`best.pt`も更新されます。

Checkpointには次の状態が入ります。

- format version
- 解決済みの実験設定
- モデルのstate dict
- オプティマイザのstate dict
- データbundleのstate dict
- PythonとPyTorchの乱数状態
- 完了済みoptimizer update数
- 最良の評価NLL

再開時にEngineが事前照合するのは、モデル、データ、学習方法、オプティマイザの登録名です。
同じ登録名でもモデル形状などを変えるとstate dictを読み込めないため、通常はコンポーネント固有設定も維持します。

## 拡張時に維持する境界

将来SFTや選好学習を追加する場合は、batchの意味とlossを新しい学習方法へ置きます。
会話データの読み込みやmask作成が必要な場合は、それを新しいデータコンポーネントへ置きます。
モデルの`forward()`で必要な引数が増える場合は、学習方法との呼び出し契約も同時に定義します。

このリポジトリ内のRegistryは、導入済みコンポーネントを明示的に列挙する仕組みです。
外部packageを実行時に探索するplugin機構ではありません。
