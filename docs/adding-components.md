# コンポーネントの追加手順

この文書は、新しいモデル、データ、学習方法、オプティマイザを既存の共通Engineへ接続する手順を示します。
最初に[プロジェクトの構成](architecture.md)を読むと、各実装を置く理由を確認できます。

## 追加作業の全体像

四種類のコンポーネントは、次の手順で追加します。

1. コンポーネントの種類とTOMLで使う登録名を決めます。
2. Pydanticでコンポーネント固有の設定型を定義します。
3. 共通Engineの契約に従う本体を実装します。
4. 検証済み設定と`BuildContext`を受け取るfactoryを実装します。
5. Packageの`register()`でRegistryへ登録します。
6. `create_default_registry()`から新しい`register()`を呼びます。
7. 実験TOMLと、小型データでEngineまで通すテストを追加します。

登録名はTOMLとcheckpointに保存される識別子です。
既存名の意味を変更せず、新しい挙動には新しい名前または設定値を使います。

## 設定型とfactory

各設定型は、意図しないTOML項目を受理しないように`extra="forbid"`を指定します。
値の範囲や項目間の制約はPydanticの`Field`とvalidatorで検証します。

```python
from pydantic import BaseModel, ConfigDict, Field


class TinyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(default=64, gt=0)
```

Factoryの型は`Callable[[BaseModel, BuildContext], Any]`です。
Registryが設定型を検証してからfactoryを呼ぶため、factoryでは型を確認して具体的な実装へ渡します。

```python
def _build_component(config: BaseModel, context: BuildContext) -> object:
    if not isinstance(config, TinyConfig):
        raise TypeError("Expected TinyConfig")
    return TinyComponent(config, context)
```

ローカルファイルを参照する設定は、factoryまたは構築されるobjectで`context.resolve_path()`を使います。
設定ファイルの場所に依存しない値だけを使う場合は、`del context`として未使用であることを明示できます。

## モデルの追加

モデルは`src/llm_scratch_lab/models/<name>/`へ追加します。
`forward()`は`ModelOutput`を返し、`logits`の末尾次元を語彙数にします。

```python
from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field

from llm_scratch_lab.core.contracts import ModelOutput
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class TinyModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)


class TinyModel(torch.nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.output = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        del attention_mask
        return ModelOutput(logits=self.output(self.embedding(input_ids)))


def _build_model(config: BaseModel, context: BuildContext) -> TinyModel:
    del context
    if not isinstance(config, TinyModelConfig):
        raise TypeError("Expected TinyModelConfig")
    return TinyModel(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "model",
        "tiny_model",
        TinyModelConfig,
        _build_model,
        description="Small model used as an extension example",
    )
```

現在の`causal_pretraining`は`input_ids`と任意の`attention_mask`をkeyword引数で渡します。
別の引数を必要とするモデルを追加する場合は、そのモデルを呼び出す学習方法も追加します。

データコンポーネントがtokenizerを公開する場合、Engineは`model.config.vocab_size`と`tokenizer.vocab_size`を比較します。
この検査を利用するモデルは、構築後の`config`から`vocab_size`を参照できるようにします。

## データコンポーネントの追加

データコンポーネントは`src/llm_scratch_lab/data/<name>/`へ追加します。
Data Moduleは`batch_keys`を公開し、`build(num_workers=...)`でData Bundleを返します。

Data Bundleは学習用DataLoader、評価用DataLoaderを毎回作るfactory、保存可能な学習データ状態を提供します。
評価用DataLoaderをfactoryにする理由は、評価のたびに先頭から新しいiteratorを作るためです。

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from llm_scratch_lab.core.contracts import Batch
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class MemoryDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class MemoryDataBundle:
    train_loader: DataLoader[Batch]
    eval_loader_factory: Callable[[], DataLoader[Batch]]
    position: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.position = int(state.get("position", 0))


class MemoryDataModule:
    batch_keys = frozenset({"input_ids", "labels", "attention_mask"})

    def __init__(self, config: MemoryDataConfig) -> None:
        self.config = config

    def build(self, *, num_workers: int = 0) -> MemoryDataBundle:
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        train_loader = DataLoader([batch] * 4, batch_size=None, num_workers=num_workers)

        def build_eval_loader() -> DataLoader[Batch]:
            return DataLoader([batch], batch_size=None, num_workers=num_workers)

        return MemoryDataBundle(train_loader, build_eval_loader)


def _build_data(config: BaseModel, context: BuildContext) -> MemoryDataModule:
    del context
    if not isinstance(config, MemoryDataConfig):
        raise TypeError("Expected MemoryDataConfig")
    return MemoryDataModule(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "data",
        "memory",
        MemoryDataConfig,
        _build_data,
        description="In-memory batches used as an extension example",
    )
```

`batch_keys`は、DataLoaderが返し得るすべてのkeyを表します。
学習方法が要求するkeyを含めることで、Engineが組み合わせを学習開始前に検査できます。

再開可能なdatasetを実装する場合は、消費位置だけでなくpacking途中のbufferなど、次のbatchを決める状態も保存します。
複数workerがdatasetのcopyを進める構成ではmain processから各copyの正確な位置を取得できないため、現在のEngineで正確に再開する実験は`num_workers=0`を使います。

## 学習方法の追加

学習方法は`src/llm_scratch_lab/methods/<name>/`へ追加します。
学習方法はモデルとbatchを受け取り、`training_step()`と`evaluation_step()`から`StepOutput`を返します。

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from llm_scratch_lab.core.contracts import Batch, ModelOutput, StepOutput
from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassificationMethod:
    required_batch_keys = frozenset({"input_ids", "labels"})

    def _step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        missing = self.required_batch_keys - batch.keys()
        if missing:
            raise ValueError(f"Batch is missing required keys: {sorted(missing)}")
        result = model(input_ids=batch["input_ids"])
        if not isinstance(result, ModelOutput):
            raise TypeError("Classification models must return ModelOutput")
        loss = F.cross_entropy(result.logits, batch["labels"])
        return StepOutput(
            loss=loss,
            metrics={"loss": float(loss.detach())},
            item_count=batch["labels"].numel(),
        )

    def training_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        return self._step(model, batch)

    def evaluation_step(self, model: torch.nn.Module, batch: Batch) -> StepOutput:
        return self._step(model, batch)


def _build_method(config: BaseModel, context: BuildContext) -> ClassificationMethod:
    del config, context
    return ClassificationMethod()


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "method",
        "classification",
        ClassificationConfig,
        _build_method,
        description="Classification loss example",
    )
```

`loss`はbackward可能なscalar Tensorにします。
`item_count`は評価lossを重み付き平均する分母なので、評価対象のsample数またはtoken数を返します。
評価対象がないbatchは`item_count=0`で通さず、その原因がわかる例外にします。

現在の評価関数は`StepOutput.loss`をNLLとして集計し、`eval/nll`と`eval/perplexity`を記録します。
異なる評価指標が必要な学習方法を追加する場合は、Engine側の評価集約契約も拡張する必要があります。

## オプティマイザの追加

オプティマイザは`src/llm_scratch_lab/optimizers/`へ追加します。
RegistryのfactoryはOptimizer本体ではなく、モデルを受け取るBuilderを返します。
この二段階構築により、モデルをdeviceへ移動し、必要ならcompileした後でパラメータをOptimizerへ渡せます。

```python
from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from llm_scratch_lab.core.registry import BuildContext, ComponentRegistry


class SGDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=0.01, gt=0)


class SGDAdapter:
    def __init__(self, model: torch.nn.Module, config: SGDConfig) -> None:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def metrics(self) -> dict[str, float]:
        return {"learning_rate": float(self.optimizer.param_groups[0]["lr"])}


class SGDBuilder:
    def __init__(self, config: SGDConfig) -> None:
        self.config = config

    def build(self, model: torch.nn.Module) -> SGDAdapter:
        return SGDAdapter(model, self.config)


def _build_optimizer(config: BaseModel, context: BuildContext) -> SGDBuilder:
    del context
    if not isinstance(config, SGDConfig):
        raise TypeError("Expected SGDConfig")
    return SGDBuilder(config)


def register(registry: ComponentRegistry) -> None:
    registry.register(
        "optimizer",
        "sgd",
        SGDConfig,
        _build_optimizer,
        description="SGD extension example",
    )
```

Schedule-free optimizerのように学習時と評価時のパラメータ表現が異なる実装は、`train()`と`eval()`で状態を切り替えます。
切り替えが不要なOptimizerでも、共通Engineから呼び出せる空のmethodを用意します。
`metrics()`は任意であり、実装した場合は`dict[str, float]`を返します。

## デフォルトRegistryへの接続

Package内に`register()`を実装した後、`create_default_registry()`でimportして呼びます。

```python
def create_default_registry() -> ComponentRegistry:
    from llm_scratch_lab.models.tiny_model import register as register_tiny_model

    registry = ComponentRegistry()
    register_tiny_model(registry)
    return registry
```

実際の関数には既存コンポーネントの登録も残します。
登録名が重複すると`Duplicate <kind> component`で失敗するため、既存名を置き換える用途には使えません。

登録後はCLIで一覧とPydantic schemaを確認します。

```bash
uv run llm-lab components list
uv run llm-lab components list --kind model
uv run llm-lab components describe model tiny_model
```

## 実験TOMLへの追加

コンポーネントの登録名を`name`へ、固有設定を`config`へ記述します。

```toml
[model]
name = "tiny_model"

[model.config]
vocab_size = 128
hidden_size = 64
```

設定型にない項目、必須項目の不足、値の範囲違反は学習開始前のPydantic検証で失敗します。
同じモデルの実験条件だけを変える場合は設定ファイルを分け、実装を複製しません。

## テストの追加

コンポーネント単体のテストに加えて、共通Engineで小さな実験を完了するテストを追加します。
既存の`tests/test_engine_extension.py`は、メモリ上のbatchと小型モデルで四種類のコンポーネントを組み合わせる例です。

統合テストは少なくとも次を確認します。

- CPUだけで一つ以上のoptimizer updateを完了できます。
- `config.json`と`checkpoints/latest.pt`が作られます。
- checkpointのupdate数と登録名が実験設定に一致します。
- 勾配蓄積を使う場合は、指定したmicrobatch数ごとに一度だけupdateします。
- 評価を使う場合は、空ではない評価DataLoaderを毎回新しく作れます。

モデルのshape、設定validator、datasetの状態保存など、コンポーネント固有の境界条件は別の単体テストにします。
ネットワークやGPUを必要とする実データ学習は通常の単体テストへ含めません。

## 代表的な失敗

| 症状 | 原因と確認箇所 |
| --- | --- |
| `Unknown ... component` | `register()`が`create_default_registry()`から呼ばれていないか、TOMLの登録名が異なります。 |
| `Duplicate ... component` | 同じ種類と名前を二度登録しています。 |
| Pydanticのvalidation error | 必須設定の不足、未知の設定項目、値の範囲違反があります。 |
| `Data component cannot provide method batch keys` | データの`batch_keys`が学習方法の`required_batch_keys`を満たしていません。 |
| `Batch is missing required keys` | 宣言上は互換でも、実際のDataLoaderが必要なkeyを返していません。 |
| `Tokenizer vocabulary ... does not match model vocabulary` | tokenizerとモデルの語彙数が異なります。 |
| `... models must return ModelOutput` | モデルがTensorや別の型を直接返しています。 |
| `Evaluation dataset produced no valid target tokens` | 評価DataLoaderが空か、全targetが無効です。 |
| `Checkpoint ... does not match config` | checkpointと現在のTOMLでコンポーネントの登録名が異なります。 |
| state dictの読み込み失敗 | 同じ登録名のままモデル形状やOptimizer構成を変更しています。 |

## 完了条件

次の検査がすべて成功すれば、コンポーネントを通常の実験へ追加できます。

```bash
uv run llm-lab components list
uv run llm-lab components describe <kind> <name>
uv run pytest tests/test_engine_extension.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

`<kind>`は`model`、`data`、`method`、`optimizer`のいずれかに置き換えます。
最後に、新しい実験TOMLの差分へ登録名、設定値、tokenizerやデータの来歴が残っていることを確認します。
