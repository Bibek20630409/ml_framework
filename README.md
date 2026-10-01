# ML Framework

Production-grade training + serving framework for **tabular**, **image**, **text**,
**time-series**, **audio** and **video** data. Supports binary/multi-class classification, regression,
forecasting, token classification and seq2seq across neural networks (PyTorch
Lightning), transformers (HuggingFace), gradient-boosted trees (XGBoost, LightGBM,
CatBoost) and statistical forecasters (Prophet, ARIMA, seasonal-naive) — driven
end-to-end by a single validated YAML config and a `mlf` CLI.

```
lr finder → select (compare families) → tune (Optuna) → train → serve (FastAPI)
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

All 19 extras:

| Extra | Brings | For |
|---|---|---|
| `lightning` | torch, pytorch-lightning, torchmetrics | `mlp`, `cnn`, `ts.lstm`, every `nlp.*` |
| `image` | torchvision, Pillow | `cnn` and the image source — install *with* `lightning` |
| `audio` | soundfile, torchaudio | the `audio.flac` decoder and `audio.cnn` — install *with* `lightning` |
| `video` | av (PyAV) | the `audio.mp3`, `audio.opus` and `video.h264` decoders |
| `gbdt` | xgboost, lightgbm, catboost | the three tree families |
| `timeseries` | statsmodels, prophet | `ts.arima`, `ts.prophet` |
| `nlp` | transformers, tokenizers, datasets | `nlp.hf_text`, `nlp.hf_token`, `nlp.hf_seq2seq` |
| `serve` | fastapi, uvicorn, prometheus instrumentator | `mlf serve` |
| `security` | slowapi, gunicorn | rate limiting, `--workers N` |
| `monitoring` | scipy, pandera | `/drift`, data contracts |
| `hpo` | optuna, optuna-integration | `mlf tune`, and `mlf train`'s default search |
| `explain` | shap | the middle attribution tier only — native and permutation need no extra |
| `export` | onnx, onnxruntime, onnxscript, skl2onnx | `mlf export --format onnx` |
| `parquet` | pyarrow | Parquet ingestion |
| `fast` | polars, pyarrow | the Polars data backend |
| `mlops` | mlflow, dvc[s3], pyspark | tracking, versioning, the Spark engine |
| `diagnostics` | torch-lr-finder, matplotlib | `mlf lr` |
| `logging` | wandb | `logging.backend: wandb` |
| `dev` | every runtime + the toolchain | running the suite |

> **torch + torchvision are a pair.** Every torchvision release pins one exact
> torch patch, so install `[lightning,image]` together and from one index. The same
> holds for torchaudio and `[lightning,audio]`.

## Quickstart

The short version — no config file at all:

```bash
mlf train --data data/raw/sample.csv
```

That infers the data kind, the target column, the task and the model, logs the
rule behind each, trains, and reports the score **beside a trivial baseline**. See
[Zero-config](#zero-config) for what it will and will not guess.

The long version, when you want the config under version control:

0. **See what this install can train.** These take no config:

   ```bash
   mlf models              # every registered model, its backend, and whether it is ready
   mlf models --show       # plus tasks, data kinds and each model's own search space
   mlf models --all        # plus third-party plugins that failed to *import*
   mlf backends            # the three fit-loop shapes and what each consumes
   mlf data-backends       # the three processing engines: local, polars, spark
   ```

   They deliberately list models whose optional extra is **missing**, each with the
   `pip install` line that fixes it. Listing only what happens to be installed
   would describe the machine rather than the framework, and would make an
   uninstalled extra indistinguishable from a model that does not exist.

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

5. **Compare model families** — score, latency, size and explainability, not
   score alone:
   ```bash
   mlf select -c configs/example_selection.yaml --max-latency-ms 20   # compare only
   mlf train  -c configs/example_selection.yaml --select              # compare, then train the winner
   ```

6. **Serve** — `sample.csv` has six features, so an instance is six numbers:
   ```bash
   mlf serve --artifacts outputs --port 8000
   curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
        -d '{"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}'
   ```

### Config schema v2

The config is organized by *ownership*: fixed blocks for what the framework owns
(`task`, `runtime`, `data`, `fit`, `tune`, `select`, `logging`) and free-form
`params` sub-dicts for what a plugin owns.

| block | holds | validated by |
|---|---|---|
| `runtime` | seed, output_dir, workers, accelerator, precision | the schema |
| `data` | kind, path, target, `split.*` | the schema |
| `data.params` | source-specific knobs (imbalance, img_size…) | the data source |
| `model.params` | architecture knobs (hidden_dims, backbone…) | the model plugin |
| `fit` | budget, patience, batch_size | the schema |
| `fit.params` | loop knobs (lr, weight_decay, LR schedule, clipping) | the backend |
| `tune` | search budget, objective, overrides | the schema |
| `select` | candidate pool, constraints, decision rule | the schema |

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
│   ├── metrics.py       array-based per-task metrics (numpy/sklearn only)
│   ├── lit_model.py     BaseModel: steps, metrics, loss, optimizer
│   ├── evaluate.py      report.txt · predictions.csv · confusion_matrix.txt
│   ├── baseline.py      the trivial predictor every chosen-model run is scored against
│   ├── profile.py       latency (warmed p50/p95/p99) · artifact bytes · fit seconds
│   ├── stall.py         data-wait / GPU-stall profile · stall.json · faults.json
│   ├── explain.py       attribution tiers: native 1.0 → shap 0.8 → permutation 0.5
│   ├── export.py        ONNX · TorchScript · native · pickle
│   ├── inference.py     Inferencer.from_artifacts(dir) — imports no torch
│   └── registry.py      MODELS / BACKENDS / SOURCES / DATA_BACKENDS
├── backends/            one per fit-loop shape — lightning.py owns pl.Trainer · staged.py (epoch/stall/fault callback)
├── plugins/             mlp · cnn · audio · video · gbdt/ (xgboost…) · ts/ (naive…) · nlp/ (hf_text, hf_token, hf_seq2seq)
├── data/
│   ├── backends/        the processing engine: local (pandas) · polars · spark
│   ├── sources/         tabular · image · text · timeseries · audio · video (staged_folder)
│   ├── preprocess/      scaling, imbalance, tokenizers, windows, mel front-end, clip layout
│   ├── streaming/       staged decode: decoders/ · shard index · sampler · materialize · tail probe
│   ├── splitters.py     random · temporal · group · rolling-origin · purged · CPCV
│   └── sniff.py         data-kind / target / task detection
├── pipeline/            train · select · tune · lr_finder · spark_preprocess · contracts · stage_config
├── monitoring/          drift.py (PSI/KS) · model_quality.py (delayed labels)
├── serving/api.py       FastAPI: /health /predict /predict_proba
│                        /predict_with_confidence /drift /metrics
├── utils/               logging, seed, platform-aware workers
└── cli.py               `mlf` entry point
configs/                 example_tabular · example_gbdt · example_selection · example_image · example_timeseries · example_text · example_ner · example_seq2seq · dvc_tabular
benchmarks/              data_backends.py — pandas vs polars, measured not asserted
tests/                   unit · integration · serving · backends · data · pipeline · load
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
  `report.txt` · `predictions.csv` · `confusion_matrix.txt` · `reference_stats.json`
  (the drift baseline) · `hpo.json` · `stall.json` · `faults.json` — plus `cv.json`
  when cross-validating and `selection.json` after a bake-off

## Task reference

| task | loss | metrics |
|---|---|---|
| `binary` | `BCEWithLogitsLoss` (scalar `pos_weight`) | acc, F1, ROC-AUC |
| `multiclass` | `CrossEntropyLoss` (class weights) | acc, macro-F1, ROC-AUC |
| `regression` | `MSELoss` | MAE, RMSE, R² |
| `forecasting` | `MSELoss` (windowed) / per-model | **MASE**, sMAPE, MAE, RMSE |
| `token_classification` | `CrossEntropyLoss` (per position, padding ignored) | **macro-F1**, accuracy — token-level |
| `seq2seq` | `CrossEntropyLoss` (teacher-forced) | **ROUGE-L**, token-F1, exact match |

A task has a row in this table only when the framework can actually run it — a
row that merely lets a config validate would trade an honest refusal at load time
for a confusing failure inside the fit loop. `multilabel` is in the `Task`
vocabulary and deliberately has no row.

## Backends

A **backend** owns the fit loop; a **model** knows only its architecture. Roughly
15 models map onto 3 loop shapes, so there are 3 backends rather than 15 `fit()`
implementations — and adding CatBoost was ~40 lines, not a new backend.

| backend | shape | models |
|---|---|---|
| `lightning` | epoch loop + validation callbacks | `mlp`, `cnn`, `ts.lstm`, `nlp.hf_text`, `nlp.hf_token`, `nlp.hf_seq2seq` |
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

## Data backends

A **training backend** owns the fit loop; a **data backend** owns how the table is read
and reduced. They are different axes, and the engine is a per-run choice:

```bash
mlf data-backends              # every engine, its extra, and whether it is ready
mlf data-backends --show       # plus what each one can and cannot do
mlf train -c configs/example_gbdt.yaml --data-backend polars
```

```yaml
data:
  backend: polars    # local (default) | polars | spark
```

| engine | extra | reach for it when |
|---|---|---|
| `local` | **none** | The default. pandas — `read_table -> pd.DataFrame` is public API |
| `polars` | `[fast]` | The run is parse-bound: a wide or long CSV read repeatedly |
| `spark` | `[mlops]` + a JVM | The table does not fit on one machine |

`--data-backend` applies to `train`, `lr`, `tune` and `select`.

`local` is the only engine with no `requires`, which is what makes "a bare install
trains" unconditional rather than dependent on an extra. Registering the other two costs
a bare install nothing — neither imports its runtime until selected, and CI asserts
exactly that: after listing all three, `pyspark` and `polars` are absent from
`sys.modules`.

**A data backend changes how the table is read, not what the model learns.** The same
config on `local` and `polars` produces the same splits, the same scaler and the same
score — pinned by a cross-engine equality test rather than asserted here.

Two limits worth knowing before you reach for one:

- **Only `tabular` is backend-aware.** Image, text and timeseries read through pandas,
  and a non-local engine is refused **by name** rather than silently ignored.
- **This is not distributed training.** The Spark engine prepares the data; the fit loop
  still runs in one process.

See **[choose.md](choose.md)** for why Polars was gated on a measurement rather than
adopted on reputation, and `benchmarks/data_backends.py` for the measurement:

```bash
make benchmark        # parse CSV, parse Parquet, and full build_bundle, pandas vs polars
```

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
  objective: holdout      # holdout | cv — score a trial across inner folds
  cv_folds: 3             # inner folds for objective: cv
  n_jobs: 1               # concurrent trials (threads)
  overrides:
    model.params.max_depth: {type: int, low: 3, high: 8}
    fit.params.learning_rate: {type: float, low: 0.05, high: 0.2, log: true}
    fit.params.subsample: 0.9        # a bare value pins it
```

## Model selection

`mlf train` tunes **one** model. `mlf select` compares model *families* — tuning
each on its own space, cross-validating it, and measuring the four things a score
does not tell you.

```bash
mlf select --config configs/mine.yaml --max-latency-ms 20 --max-workers 4
```

```
model                  score     +/-   p95 ms      MB  expl  status
-------------------------------------------------------------------
catboost              0.9025  0.0326   1.2235  0.1019  1.00  WINNER
lightgbm              0.8750  0.0179   2.5612  0.1980  1.00  ok
xgboost               0.8724  0.0187   1.3515  0.1747  1.00  ok
mlp                   0.8700  0.0144  24.0520  1.5621  0.50  rejected: p95 latency 24.05 ms exceeds the 20 ms budget
```

Five criteria, all measured in the same run under identical conditions:
**predictive performance** (CV mean ± spread), **inference latency** (warmed
p50/p95/p99 at batch size 1), **memory & compute cost** (serialized artifact
bytes), **explainability** (native importances 1.0 → SHAP 0.8 → permutation 0.5
→ none 0.0), and **maintainability** (fit wall-clock, fold stability, fold
failures).

The default rule is *take the simplest model that is not measurably worse than
the best one*: hard constraints disqualify, then the top score wins unless
something within one standard error of it is cheaper — latency first, then size,
then explainability, then stability. It never trades a real accuracy difference
for speed, and it names the axis that decided:

```
winner: xgboost — xgboost scores 0.8724 against lightgbm's 0.8750 — within the
0.0103 tolerance (std error of the CV mean) — and wins the tie-break on
p95 latency (1.59 ms vs 5.78 ms)
```

`mlf train --select` does the comparison and then trains the winner; the bundle's
`manifest.selection` carries the whole table, so a served model can answer "why
this family?" without the training directory.

Off by default — a bake-off costs one tuning budget per candidate. See
**[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md)** for the configuration
reference, the CV strategy table (stratified · k-fold · walk-forward · purged ·
CPCV), the parallelism options, and the worked examples.

## Cross-validation

`data.split.folds >= 2` cross-validates; `data.split.cv_strategy` decides how the
folds are cut. Using the wrong one does not raise — it reports a *better* score,
which is why the choice has its own config key rather than being inferred from
the model.

| `cv_strategy` | For | Guards against |
|---|---|---|
| `auto` *(default)* | Anything | Resolves from data kind, task and purge settings |
| `stratified` | Tabular classification | Imbalanced folds, high evaluation variance |
| `kfold` | Regression | Stratifying a continuous target |
| `rolling_origin` | Time series | Training on the future |
| `purged` | Overlapping labels | Training on the test set's own observations |
| `cpcv` | Backtests needing a distribution | Judging on one arrangement |

```yaml
data:
  split:
    folds: 5
    cv_strategy: purged
    label_horizon: 10     # a row's label is computed from the next 10 rows
    embargo: 0.01         # drop a further 1% of rows after each test block
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
make install       # editable install with every extra the suite needs
make install-mlops # + mlflow, dvc, pyspark (heavy; Spark tests also need a JVM)
make format        # black + isort
make lint          # ruff + black --check
make type          # mypy
make test          # pytest
make cov           # pytest with coverage (fails under 80%)
make clean         # drop outputs/, caches, build artifacts
```

Demo and ops targets: `make train`, `select`, `serve`, `benchmark`, `docker-serve`,
`stack-up` / `stack-down`, `dvc-repro`, `k8s-validate`, `k8s-deploy`, `airflow-up`.

CI runs lint + type-check + tests on Python **3.10–3.14** at a coverage floor of 80%,
plus three jobs the matrix cannot cover: `gbdt-no-torch` (a tree bundle trains and
serves with no deep-learning stack present), `spark-contract` (the Spark engine against
a real JVM, refusing to pass by skipping) and `mlops-validate` (manifests render,
every YAML parses). Licensed MIT.

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


## Text

Text classification is `binary`/`multiclass` with `data.kind: text` — there is no
`text_classification` task. Task decides loss, metrics and head; kind decides
ingestion. Keeping them orthogonal is what lets the task list grow additively
instead of as a product of the two.

```yaml
task: binary
data:
  kind: text
  path: data/reviews.csv     # CSV, Parquet or JSONL
  target: label              # string labels are encoded; the names come back out
model:
  name: nlp.hf_text
  params: {model_name: distilbert-base-uncased, max_length: 128}
```

```bash
pip install -e '.[nlp,lightning]'
mlf train --config configs/example_text.yaml
curl -X POST localhost:8000/predict -d '{"inputs": ["great movie"]}'
# {"predictions": [1.0], "labels": ["pos"]}
```

`nlp.hf_text` rides the **lightning** backend rather than a fourth one: fine-tuning
a transformer is an epoch loop with validation callbacks, exactly like training a
CNN. What differs is the batch shape and the file format, and both are the model's
business.

**The bundle is self-contained.** The fine-tuned weights are written with
`save_pretrained` into `model/hf_model/` and the tokenizer into
`preprocessor/tokenizer/`. A Lightning checkpoint would round-trip the weights,
but rebuilding the architecture to put them in calls `from_pretrained(model_name)`
— which needs the HuggingFace hub, or a warm cache, *at load time*. That failure
shows up in a serving container rather than in CI. Serving a text bundle needs no
network.

> **The tokenizer is fitted state, not configuration.** It maps strings to ids
> through a vocabulary, and a model fed ids from a *different* vocabulary produces
> confident nonsense: no shape error, no exception, just a model that looks like it
> trained badly. That is why it ships in the bundle, why `model.params.model_name`
> is the only place a checkpoint is named, and why a bundle with no tokenizer
> directory refuses to load rather than quietly re-downloading one.

**Fine-tuning defaults are applied, not documented.** The framework default is
`lr: 1e-3` with Adam — right for a network trained from scratch, and destructive to
a pretrained encoder within the first few steps, while the run looks entirely
healthy and scores near chance. `nlp.hf_text` declares `lr: 2e-5`, AdamW,
`weight_decay: 0.01` and a cosine schedule; the config validator fills in the keys
you did not write, and `config.json` records what actually ran. Tuning narrows the
learning rate to 1e-5..5e-5 for the same reason, instead of inheriting the
backend's from-scratch range.

Requests carry raw strings. Sending token ids would make every client responsible
for using the right vocabulary — the skew above — and would make the endpoint
unusable from curl.

`/drift` answers **501** for text: PSI over token ids is a number without a
meaning, and inventing one would be worse than reporting that there is none.


## Beyond text classification

`data.kind: text` covers three tasks, and the **task** is what decides what a
label is:

| task | model | one input row produces | config |
|---|---|---|---|
| `binary` / `multiclass` | `nlp.hf_text` | one class | `example_text.yaml` |
| `token_classification` | `nlp.hf_token` | one class **per token** | `example_ner.yaml` |
| `seq2seq` | `nlp.hf_seq2seq` | a **string** | `example_seq2seq.yaml` |

All three ride the `lightning` backend. Still three backends.

```bash
curl -X POST localhost:8000/predict -d '{"inputs": ["Ada works at Acme"]}'
# {"tokens": [["Ada","works","at","Acme"]], "labels": [["B-PER","O","O","B-ORG"]]}

curl -X POST localhost:8000/predict -d '{"inputs": ["summarize: ..."]}'
# {"generated": ["..."]}
```

Tags come back per **word**, not per sub-word. The model predicts at sub-word
positions, but you sent words — re-deriving the alignment client-side would mean
owning a copy of the tokenizer, which is the coupling shipping it in the bundle
removed.

> **Token tagging is scored per token, not per entity.** The NER convention
> (seqeval) requires a predicted entity to match the reference in both span *and*
> type. That is strictly harder; token-level figures run several points above it
> and are not comparable with published results. `report.txt` says this in the
> file rather than leaving it to be assumed. Macro-F1 leads rather than accuracy,
> because `O` dominates a tagging corpus — a model answering "not an entity"
> everywhere scores ~90% accuracy and is worth nothing.

> **A seq2seq model is trained one way and evaluated another.** Training is
> teacher-forced: the decoder is fed the reference prefix at every step, so it
> never has to survive its own mistakes. That is what `val/loss` measures, and it
> stays the early-stopping monitor because generating every validation epoch would
> multiply epoch time by the decode length. The reported metrics come from real
> generation, and the two can move in opposite directions. ROUGE-L, token-F1 and
> exact-match are all n-gram overlap against one reference, so a correct
> paraphrase scores near zero — read them as a floor and read the samples in
> `report.txt` as the evidence.

The subtle part of token tagging is the **word-to-sub-word alignment**. A corpus is
tagged per word; the model consumes sub-words, and "Acme" may arrive as
`["ac", "##me"]`. Only the first piece carries the word's tag; continuations are
excluded from both loss and metrics. Repeating the tag across every piece — the
obvious alternative — does not raise. It makes one word's single decision count
once per piece, re-weighting the corpus toward whichever words the tokenizer
fragments most, which is exactly the rare proper nouns NER is about.

## Audio and video

`data.kind: audio` and `data.kind: video` read a folder of clips, one
sub-directory per class. Anything that has to be decoded goes through one offline
pass first — `mlf train` refuses a corpus that has not, and names the command:

```bash
pip install -e '.[lightning,audio]'          # FLAC/WAV; add `video` for MP3, Opus and MP4/H.264
mlf decoders                                  # which formats this install can read, and how each fails
mlf materialize --data ./clips --max-fault-rate 0.02   # decode-probe every sample, write the shard index
mlf train --data ./clips                      # sniffs the kind, picks audio.cnn / video.r3d, trains
```

`materialize` writes `_mlf_shards/` beside the data — `shards.json`,
`entries.jsonl` and `faults.jsonl` — and is the **only** place the fault ceiling
aborts. At training time a corrupt sample is *substituted*, never skipped: under DDP
every rank must produce the same number of batches, and a content-dependent skip is
a collective hang with no error message. What was substituted lands in the bundle's
`faults.json`, and how long the run waited on data in `stall.json`.

Storage → tensor is seven named stages — `read · demux · decode` belong to the
decoder, `construct · transform · h2d · gpu_transform` to the preprocessor. With a
CUDA device the transform (the mel front-end, the clip permute) runs **after** the
host→device copy, so workers stay on IO and the bus carries the smaller tensor;
`--no-device-transform` turns that off to compare. `mlf materialize -c cfg.yaml
--probe-full` pushes one batch through all seven before a run commits to them.

Serving does not accept audio or video yet — `/predict` returns `501` for these
bundles.

## Zero-config

```bash
mlf train --data data/raw/sample.csv          # infer everything, train, report
mlf init  --data data/raw/sample.csv -o configs/mine.yaml   # infer, write, stop
```

`mlf init` writes the config it would have used, with the reasoning attached:

```yaml
# Generated by `mlf init` from sample.csv
#
# Every inferred field carries the rule that produced it. Check them —
# especially any marked GUESS, which are fallbacks rather than evidence.

task: multiclass  # inferred: the target is non-float with 3 distinct values
data:
  kind: tabular  # inferred: columns are numeric or short strings
  target: label  # inferred: column is named 'label'
model:
  name: xgboost  # inferred: gradient boosting is the tabular default
```

**`--data` composes with `--config`.** Synthesis is one layer of an ordinary
precedence chain:

```
plugin defaults < synthesis < YAML file < --set < explicit CLI flags
```

so a YAML that sets two fields overrides exactly those two and inherits the rest.
Nothing downstream knows a value was inferred rather than typed — that is what
makes the two compose instead of being alternatives.

### What it refuses to guess

Two columns named `label` and `target` is not a tie to be broken by column order;
it raises and names them. Two columns of prose, likewise. The one genuinely weak
rule — "no conventional name, so use the last column" — is marked `GUESS` in the
generated file and logged at WARNING.

Evidence is required in both directions. A datetime column must also be
**monotonic** before your data is a time series, because parsing alone would make
a table of birthdays one. A string column must average ≥ 4 words before it is
prose, because a column of colour names is a *feature* — treating it as text would
fine-tune a 66M-parameter encoder on the word "red".

> **An uninstalled family is a refusal, not a substitution.** If `[gbdt]` is
> missing, a tabular run stops with the `pip install` line rather than quietly
> training an MLP. This matters more here than anywhere else: you did not choose
> the model, so a score from the wrong one looks exactly like the score you asked
> for. The one sanctioned exception is seasonal-naive as the forecasting fallback,
> and it is logged at WARNING as the downgrade it is.

### Always a baseline

A run whose model the framework chose also scores the trivial predictor — majority
class, training mean, or repeat-last-season — and records it as `baseline_*`:

```
INFO  acc 0.8750 beats the trivial baseline (0.4000)
```

`test_acc: 0.91` on a dataset that is 91% one class is the most common way a
pipeline looks successful while having learned nothing, and a user who did not
pick the model has nothing else to judge it against. Failing to beat the baseline
is a **warning, not an error** — tying it on a genuinely unpredictable target is
an honest result, and failing the run would be pretending otherwise.

The statistic comes from the *training* split (taking the majority class from test
would make the baseline stronger than anything achievable honestly), and the
comparison follows the task's declared direction (MAE, RMSE and MASE improve by
getting smaller).


## Export and deploy

```bash
mlf export     --artifacts outputs --format onnx -o model.onnx
mlf dockerfile --artifacts outputs -o Dockerfile.serve
```

Export is a **backend method**, so what is possible depends on what the model
physically is:

| backend | formats |
|---|---|
| `lightning` | `onnx`, `torchscript` |
| `gbdt` | `native` — the library's own `.json`/`.cbm`/`.txt` |
| `forecast` | `pickle` |

**Asking for a format a backend cannot produce raises**, and the error names what
it can. A booster has no traced graph, and its own format is what every serving
runtime for that library already reads — converting it would trade exact behaviour
and load speed for portability nobody asked for. Writing *some* file instead would
be discovered at deployment time by a runtime that cannot load it.

> **A traced graph does not include the preprocessing.** The bundle's scaler and
> image transforms stay behind, so whatever loads the artifact must apply them.
> An ONNX file fed raw unscaled features scores confident nonsense with no error —
> which is why `mlf export` prints the caveat rather than burying it in docs.

Exported ONNX is checked against the bundle's own predictions in CI: max absolute
difference ~6e-08, identical argmax, with a dynamic batch axis so the artifact is
not frozen at the shape it was traced with.

### A Dockerfile per bundle

`mlf dockerfile` reads `manifest.requires` — what the model declared at training
time — and installs those extras and no others:

```dockerfile
# Generated by `mlf dockerfile` for the bundle at outputs
#   model    xgboost (gbdt backend)
#   task     multiclass on tabular data
#   extras   gbdt, serve
#
# No torch in this image: this model does not need it. That is roughly
# 500 MB and a cold start the deep-learning image pays and this one does not.
```

The repository's fixed `serve` / `serve-gbdt` targets still exist and still work;
the difference is that this one follows from the bundle rather than from picking
the right target. A third-party plugin that declares its own extra gets a correct
image without the generator knowing the plugin exists.

### Monitoring

Prometheus series carry `backend` and `model` labels. Without them, two containers
scraped into one Prometheus produce timeseries that are indistinguishable and
silently **add together** — a booster's predictions and a transformer's arriving as
one counter.
