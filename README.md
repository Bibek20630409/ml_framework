# ML Framework

Production-grade training + serving framework for **tabular**, **image** and
**time-series** data. Supports binary/multi-class classification, regression and
forecasting across neural networks (PyTorch Lightning), gradient-boosted trees
(XGBoost, LightGBM, CatBoost) and statistical forecasters (Prophet, ARIMA,
seasonal-naive) — driven end-to-end by a single validated YAML config and a
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

3. **Train** — **tunes by default**, then fits the winner and writes the bundle:
   ```bash
   mlf train --config configs/example_tabular.yaml
   # override anything inline:
   mlf train -c configs/example_tabular.yaml --set fit.budget.max_epochs=5 --set fit.params.lr=3e-4
   ```

4. **Control the search** — turning it up is as easy as turning it off:
   ```bash
   mlf train -c configs/example_gbdt.yaml --no-tune                    # skip it
   mlf train -c configs/example_gbdt.yaml --tune-budget 10m            # spend longer
   mlf train -c configs/example_gbdt.yaml --tune-trials 100
   mlf tune  -c configs/example_gbdt.yaml --emit-config configs/tuned.yaml   # search only
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
├── plugins/             mlp · cnn · gbdt/ (xgboost…) · ts/ (naive, arima, prophet, lstm)
├── data/                sources · preprocess · splitters · lightning_adapter
├── pipeline/            train · tune · lr_finder
├── serving/api.py       FastAPI: /health /predict /predict_proba
├── utils/               logging, seed, platform-aware workers
└── cli.py               `mlf` entry point
configs/                 example_tabular · example_gbdt · example_image · example_timeseries
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
- Scaling (`StandardScaler`, fit on train only), persisted for inference. Image
  augmentation applies to **training only** — validation and test images go
  through the eval pipeline, so the score measures the model rather than the
  augmentation
- Imbalance handling — one explicit strategy: `smote | class_weights | none`
- Gradient clipping, early stopping, best-checkpointing, LR scheduling
- Metrics + CSV/WandB logging; final report, predictions, confusion matrix
- Self-contained artifact bundle v2: `manifest.json` (the only file a loader must
  understand) · `config.json` · `model/` · `preprocessor/` · `metrics.json` ·
  `report.txt` · `predictions.csv` · `hpo.json` (and `cv.json` when cross-validating)

## Task reference

| task | loss | metrics |
|---|---|---|
| `binary` | `BCEWithLogitsLoss` (scalar `pos_weight`) | acc, F1, ROC-AUC |
| `multiclass` | `CrossEntropyLoss` (class weights) | acc, macro-F1, ROC-AUC |
| `regression` | `MSELoss` | MAE, RMSE, R² |
| `forecasting` | `MSELoss` (windowed) / per-model | **MASE**, sMAPE, MAE, RMSE |

## Backends

A **backend** owns the fit loop; a **model** knows only its architecture. Roughly
15 models map onto 3 loop shapes, so there are 3 backends rather than 15 `fit()`
implementations — and adding CatBoost was ~40 lines, not a new backend.

| backend | shape | models |
|---|---|---|
| `lightning` | epoch loop + validation callbacks | `mlp`, `cnn`, `ts.lstm` |
| `gbdt` | one-shot `fit(X, y, eval_set=…)` + native early stopping | `xgboost`, `lightgbm`, `catboost` |
| `forecast` | fit-per-series, no X/y, predict-by-horizon | `ts.naive`, `ts.arima`, `ts.prophet` |

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

## Tuning

`mlf train` searches before it fits. The budget is **per backend**, because a
boosting trial costs seconds and a neural trial costs minutes — one uniform trial
count would either waste the cheap case or make the expensive one feel broken:

| backend | trials | wall clock | per-trial cap |
|---|---|---|---|
| `gbdt` | 30 | 300 s | — |
| `lightning` | 10 | 900 s | 25 epochs |

The space is assembled, not hardcoded: `model.search_space | backend.search_space()
| tune.overrides`. A plugin declares its tree shape or architecture; the backend
declares the loop's knobs (`lr`, `batch_size`, `learning_rate`, `n_estimators`)
once for everything that rides it. Keys are dotted config paths, so **applying a
trial is exactly `config.with_overrides(values)`**.

Narrow a space from YAML without touching code:

```yaml
tune:
  max_trials: 50
  metric: null            # null → the task's primary metric
  refit: best             # best = retrain at full budget | reuse = keep the trial model
  overrides:
    model.params.max_depth: {type: int, low: 3, high: 8}
    fit.params.learning_rate: {type: float, low: 0.05, high: 0.2, log: true}
    fit.params.subsample: 0.9        # a bare value pins it
```

## Deep-learning knobs

All of these resolve in one place and say what they did:

```yaml
runtime:
  precision: bf16-mixed    # 32 | 16-mixed | bf16-mixed  (16/bf16 also accepted)
  strategy: ddp            # auto | ddp | ddp_spawn
fit:
  params:
    optimizer: adamw       # adam | adamw | sgd
    scheduler: cosine      # plateau | cosine | step | none
    accumulate_grad_batches: 4
```

Asking for mixed precision on a backend that cannot do it — or fp16 on a CPU,
where there is no gradient scaler — **downgrades with a WARNING** rather than
silently training in fp32. That is the failure you otherwise discover months
later from a wall-clock number that never improved.

```bash
mlf train -c configs/example_tabular.yaml --resume        # from model/last.ckpt
mlf train -c configs/example_tabular.yaml --folds 5       # cross-validate first
```

`--resume` continues the epoch counter, optimizer and scheduler — not just the
weights. `last.ckpt` is what it reads; the manifest still points at the *best*
checkpoint, because a loader wants the best weights and only a resuming trainer
wants the last state.

**Cross-validation is an orchestration mode, not a Lightning feature.** It drives
the splitter and the same protocol calls every backend implements, so `--folds 5`
works for XGBoost too. Each fold re-fits its own scaler and imbalance correction —
sharing one across folds would leak every fold's test set into every other fold's
preprocessing. `cv.json` records the per-fold scores as well as the mean, because
a mean of 0.85 across 0.84/0.86 and across 0.70/1.00 are the same number and
completely different results. The CV estimate sits *beside* the holdout score
rather than replacing it.

It adapts to the data kind rather than assuming one shape:

| kind | folds are | why |
|---|---|---|
| tabular | stratified k-fold | the default |
| timeseries | **rolling origin** | shuffled folds would leak the future |
| image | k-fold over the **train directory** | `params.test_dir` is a decision made on disk; CV does not override it |
| text | *not yet* | there is no text source to partition |

For images that means "test" inside CV is a held-out slice of `data.path`, while
the final bundle's `test_acc` still comes from `test_dir` — two numbers answering
different questions. `cv_test_source` in the bundle records which.

The result is **applied, not printed**: the winner lands in `bundle/config.json`,
the full record (winner, ranges searched, every trial) in `bundle/hpo.json` and
`manifest.hpo`, and `--emit-config` writes a YAML you can commit. A search that
does not run — tuning off, an empty space, optuna not installed — still writes
`hpo.json` saying which, because a missing file is indistinguishable from an old
bundle.

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

## Forecasting

```bash
mlf train --config configs/example_timeseries.yaml
```

**Shuffling a time series is refused, not warned about.** `strategy: random` on
`kind: timeseries` raises at config-load time:

```
data.split.strategy: random on kind: timeseries shuffles the future into
training and reports a score that is not an estimate of anything.
```

That is the most damaging silent failure in this domain — nothing crashes, the
score simply comes back *better*. Warning and proceeding is the conventional
choice and the wrong one, so the escape hatch
(`data.split.allow_temporal_leakage: true`) costs typing the word.

Cross-validation follows the same rule: `folds: 3` on a series runs
**rolling-origin** validation, where each fold's training data precedes its test
window, not shuffled k-fold.

| model | backend | needs |
|---|---|---|
| `ts.naive` | forecast | nothing — the baseline MASE is measured against |
| `ts.arima` | forecast | `[timeseries]`; explicit (p,d,q), prediction intervals |
| `ts.prophet` | forecast | `[timeseries]`; interpretable trend + seasonality |
| `ts.lstm` | **lightning** | `[lightning]`; recurrent, over sliding windows |

That last row is the backend split earning its keep: an LSTM forecaster trains in
mini-batches over epochs exactly as an MLP does, so it rides the loop that already
exists. One source serves both — it emits the raw ordered values for Prophet and
sliding windows for the LSTM, chosen by what each model declares it `accepts`.

Forecasts are served **by horizon**, not by rows, because `predict(X)` is a lying
signature for a model that continues from where it was fitted:

```bash
curl -X POST localhost:8000/predict -d '{"horizon": 7}'
# {"forecast": [...], "lower": [...], "upper": [...]}
```

> **On MASE.** It scales the error by the series' average step change, so it is
> comparable across series. It is **not** a pass mark: over a multi-step horizon
> values above 1 are normal. To judge whether a model earns its keep, train
> `ts.naive` on the same split and compare. `report.txt` deliberately prints no
> verdict for this reason.
