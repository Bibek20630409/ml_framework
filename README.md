# ML Framework

Production-grade training + serving framework for **tabular**, **image**, and mixed
data. Supports binary classification, multi-class classification, and regression
across neural networks (PyTorch Lightning) and gradient-boosted trees (XGBoost,
LightGBM, CatBoost) — driven end-to-end by a single validated YAML config and a
`mlf` CLI.

```
lr finder → HPO (Optuna) → train → serve (FastAPI)
```

## Install

**The base install carries no ML runtime.** Pick the model families you need:

```bash
pip install -e ".[lightning]"          # MLP / CNN (PyTorch Lightning)
pip install -e ".[gbdt]"               # XGBoost / LightGBM / CatBoost
pip install -e ".[gbdt,serve]"         # serve a tree model — no torch, ~500 MB lighter
pip install -e ".[dev]"                # everything, for running the test suite
mlf --help
```

Nothing is hidden by an install you skipped: `mlf models` lists **every** plugin
with an availability column, and selecting one you have not installed fails
immediately with the command that fixes it —

```
model 'xgboost' requires xgboost>=2.0. Install it with: pip install 'ml-framework[gbdt]'
```

Extras: `lightning`, `gbdt`, `image`, `serve`, `hpo`, `parquet`, `diagnostics`,
`logging`, `mlops`, `security`, `monitoring`, `dev`.

> **torch + torchvision are a pair.** Every torchvision release pins one exact
> torch patch, so install `[lightning,image]` together and from one index.

## Quickstart

1. **Write one YAML config** (copy `configs/example_tabular.yaml`). Set `task`,
   `data.path`, `data.target`. That's the only file you edit per project.

2. **Learning rate** — find a good LR, paste into `fit.params.lr`:
   ```bash
   mlf lr --config configs/example_tabular.yaml
   ```

3. **HPO** — search architecture, paste the printed `model.params.*` / `fit.params.*`:
   ```bash
   mlf hpo --config configs/example_tabular.yaml
   ```

4. **Train** — final run, writes the artifact bundle:
   ```bash
   mlf train --config configs/example_tabular.yaml
   # override anything inline:
   mlf train -c configs/example_tabular.yaml --set fit.budget.max_epochs=5 --set fit.params.lr=3e-4
   ```

5. **Serve**:
   ```bash
   mlf serve --artifacts outputs --port 8000
   curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
        -d '{"instances": [[0.1, 0.2, 0.3, 0.4]]}'
   ```

### Config schema v2

The config is organized by *ownership*: fixed blocks for what the framework owns
(`task`, `runtime`, `data`, `fit`, `tune`, `logging`) and free-form `params`
sub-dicts for what a plugin owns.

| block | holds | validated by |
|---|---|---|
| `runtime` | seed, output_dir, workers, accelerator, precision | the schema |
| `data` | kind, path, target, `split.*` | the schema |
| `data.params` | source-specific knobs (imbalance, img_size…) | the data source |
| `model.params` | architecture knobs (hidden_dims, backbone…) | the model plugin |
| `fit` | budget, patience, batch_size | the schema |
| `fit.params` | loop knobs (lr, weight_decay, LR schedule, clipping) | the backend |

`params` blocks are free-form in core and **strict** in the plugin — each ships a
frozen `extra="forbid"` schema, so a typo is still an error at config-load time,
while a third-party plugin never has to edit `config/schema.py`.

Coming from a v1 config, convert it mechanically:

```bash
mlf migrate-config -i configs/old.yaml -o configs/new.yaml
```

It refuses to drop a key it cannot map, and validates the result before writing.

## Project structure

```
src/ml_framework/
├── config/              schema.py (v2 ExperimentConfig) · migrate.py (v1 → v2)
├── core/                framework internals — you rarely touch these
│   ├── types.py         Task/DataKind vocabulary, Requirement, Capabilities
│   ├── protocols.py     Estimator · TrainingBackend · Preprocessor · Splitter
│   ├── plugins.py       PluginRegistry, ModelSpec/BackendSpec/SourceSpec
│   ├── task.py          TaskSpec table: metric, direction, monitor, postprocess
│   ├── bundle.py        artifact bundle v2 + manifest.json
│   ├── lit_model.py     BaseModel: steps, metrics, loss, optimizer
│   ├── evaluate.py      report.txt · predictions.csv · confusion_matrix.txt
│   ├── inference.py     Inferencer.from_artifacts(dir)
│   └── registry.py      MODELS / BACKENDS / SOURCES
├── backends/            one per fit-loop shape — lightning.py owns pl.Trainer
├── plugins/             mlp.py (tabular), cnn.py (image transfer learning)
├── data/                sources · preprocess · splitters · lightning_adapter
├── pipeline/            train · hpo · lr_finder
├── serving/api.py       FastAPI: /health /predict /predict_proba
├── utils/               logging, seed, platform-aware workers
└── cli.py               `mlf` entry point
configs/                 example_tabular.yaml · example_image.yaml
tests/                   unit · integration · serving · backends · data
```

## Extending (scalability)

Add a new model without touching the pipeline. A plugin is a params schema, a
network, and a spec:

```python
from pydantic import BaseModel as PydanticModel
from ml_framework.core import BaseModel, ModelSpec, register_model, register_model_spec

class MyNetParams(PydanticModel):
    model_config = {"frozen": True, "extra": "forbid"}
    width: int = 64

@register_model("my_net")
class MyNet(BaseModel):
    @classmethod
    def params_model(cls): return MyNetParams
    def build_network(self):
        ...   # uses self.input_dim, self.output_dim, self.params.width

def build(ctx):
    return MyNet(input_dim=ctx.input_dim, output_dim=ctx.output_dim, task=ctx.task,
                 params=ctx.params, optim=ctx.optim, class_weights=ctx.class_weights)

register_model_spec(ModelSpec(
    name="my_net", backend="lightning", build=build, params_model=MyNetParams,
    tasks=frozenset({"binary", "multiclass"}), data_kinds=frozenset({"tabular"}),
))
```

Then set `model.name: my_net` in the YAML. The spec is what makes the model
introspectable: its tasks, data kinds, optional-dependency requirements and search
space are all declared rather than discovered by branching somewhere else. Keep
heavy imports **inside** `build_network()` so the module stays importable — and the
plugin listable — on an install without its extra. Data sources register the same
way with `SourceSpec`.

## What runs automatically

- Train/val/test split (holdout for large data, KFold-derived for small; **KFold for
  regression** to avoid stratification crashes). Set `data.split.time_col` or
  `group_col` and `strategy: auto` switches to a temporal or grouped split — the two
  leakage modes a shuffled split hides
- Scaling (`StandardScaler`, fit on train only), persisted for inference
- Imbalance handling — one explicit strategy: `smote | class_weights | none`
- Gradient clipping, early stopping, best-checkpointing, LR scheduling
- Metrics + CSV/WandB logging; final report, predictions, confusion matrix
- Self-contained artifact bundle v2: `manifest.json` (the only file a loader must
  understand) · `config.json` · `model/` · `preprocessor/` · `metrics.json` ·
  `report.txt` · `predictions.csv`. The v1 root files (`model.ckpt`, `scaler.pkl`,
  `metadata.json`) are still written for the current loader

## Task reference

| task | loss | metrics |
|---|---|---|
| `binary` | `BCEWithLogitsLoss` (scalar `pos_weight`) | acc, F1 |
| `multiclass` | `CrossEntropyLoss` (class weights) | acc, macro-F1 |
| `regression` | `MSELoss` | MAE, RMSE |

## Backends

A **backend** owns the fit loop; a **model** knows only its architecture. Roughly
15 models map onto 3 loop shapes, so there are 3 backends rather than 15 `fit()`
implementations — and adding CatBoost was ~40 lines, not a new backend.

| backend | shape | models |
|---|---|---|
| `lightning` | epoch loop + validation callbacks | `mlp`, `cnn` |
| `gbdt` | one-shot `fit(X, y, eval_set=…)` + native early stopping | `xgboost`, `lightgbm`, `catboost` |
| `forecast` | fit-per-series, predict-by-horizon | *(time-series work)* |

Switching families is a config edit — compare `configs/example_tabular.yaml` with
`configs/example_gbdt.yaml`: same task, same data, same blocks, different
`model.name`. Nothing about the orchestration, the bundle or the serving contract
changes because the fit loop did.

Capability flags on each model do real work rather than describing intent. A tree
declares `needs_scaling: false`, so the preprocessor skips `StandardScaler` —
scaling buys a tree nothing and turns an interpretable split ("age > 41") into an
opaque one ("age > 0.34"). It declares `supports_sample_weight: true`, so
`imbalance_strategy: auto` gives it per-row weights instead of SMOTE, which
interpolates synthetic neighbours an axis-aligned splitter uses poorly.

**A GBDT bundle serves without torch installed.** `core/inference.py` dispatches
on `manifest.model.backend`, so nothing in the serving path imports a checkpoint
loader; `docker build --target serve-gbdt` is roughly 500 MB lighter than the
deep-learning image, with a cold start to match.

## Production MLOps

Beyond training + serving, the framework ships a full, self-hostable MLOps stack —
**MLflow** (tracking + model registry), **DVC** (data versioning), **Apache Spark**
(preprocessing) orchestrated by **Apache Airflow**, deployed on **Kubernetes**
(autoscaled FastAPI, KServe optional), and monitored with **Prometheus + Grafana**.

```bash
pip install -e ".[dev,serve,mlops]"
docker compose up --build      # api + mlflow + prometheus + grafana + minio
```

See **[docs/MLOPS.md](docs/MLOPS.md)** for the architecture and per-component guide.

## Development

```bash
make install   # editable install with dev extras
make lint      # ruff + black --check
make type      # mypy
make test      # pytest
make cov       # pytest with coverage
```

CI runs lint + type-check + tests on Python 3.10–3.12. Licensed MIT.
