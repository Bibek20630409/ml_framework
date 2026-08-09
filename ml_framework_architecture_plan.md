# Unified Production ML Training Framework — Architecture Plan

## Context

`c:\Users\bibek\Desktop\ml_framework` is a ~3,400-line PyTorch Lightning training framework with genuinely strong bones: a decorator registry, a frozen Pydantic config, a self-describing artifact bundle, FastAPI serving, MLflow tracking, drift monitoring, and a full CI/CD + K8s + DVC + Airflow surround. It trains exactly two things: an MLP and a torchvision CNN.

The goal is to extend it into a framework where the user supplies a dataset and a model type and gets a tuned, tracked, deployable model — across MLP, CNN, Transformers/NLP, gradient-boosted trees, and time-series forecasting.

**The single blocking fact:** `BaseModel` *is* a `pl.LightningModule`, and `pipeline/train.py` constructs `pl.Trainer` directly. Every downstream component — `evaluate()`, `Inferencer`, `serving/api.py`, `hpo.py` — assumes torch. XGBoost and Prophet cannot enter through the current registry at any price. The entire plan turns on extracting the fit loop into a swappable object so that everything else consumes **arrays and a manifest** instead of torch objects.

**Decisions locked with the user:**
1. Clean-break v2 config schema; migrating existing configs and tests is acceptable.
2. Build order: **GBDT → time-series → NLP**.
3. HPO runs by default under a budget cap; `--no-tune` disables it.
4. Single `ml_framework` package with optional pip extras (not separate distributions).

---

## 1. Gap Analysis

### 1.1 What exists and is worth preserving

| Asset | Location | Why it survives |
|---|---|---|
| Decorator registry | `core/registry.py` | Right idea; needs to carry metadata, not just classes |
| Frozen config + dotted overrides | `config/schema.py` | `with_overrides()` becomes the HPO trial-application mechanism verbatim |
| Self-describing artifact bundle | `pipeline/train.py:57`, `core/inference.py:58` | Strongest concept in the repo — file-loaded and registry-loaded models take an identical code path |
| Lazy optional imports | throughout | The base install genuinely works without fastapi/mlflow/optuna/torchvision; extend this pattern, don't replace it |
| Data helpers | `core/lit_data.py:44-136` | `read_table`, `detect_imbalance`, `compute_class_weights`, `split_dataset` are framework-agnostic already |
| Production surround | `Dockerfile`, `deploy/`, `.github/workflows/`, `dvc.yaml`, `orchestration/` | Trivy/Syft/cosign, KServe, HPA, NetworkPolicy, Prometheus alerts all real and working |
| Bug-documenting docstrings | throughout | Each module records *why* — preserve this discipline in new modules |

### 1.2 What is missing entirely

- **Any non-Lightning estimator path.** No `fit`/`predict` abstraction. xgboost, lightgbm, catboost, transformers, statsmodels, prophet are absent from the dependency tree.
- **Backend-neutral tracking.** `pipeline/train.py:36` `_build_logger()` returns a *Lightning logger object*; `tracking/mlflow_utils.py:54` reads `logger.run_id` off it. GBDT cannot consume this.
- **Task/metric table.** `task == "binary"` branching is duplicated in `lit_model._shared_step`, `evaluate.py:41-52`, and `inference.py:133-149`. Adding `forecasting` means grepping six files.
- **Temporal splitting.** `split_dataset` (`lit_data.py:90`) is shuffled/stratified. Using it on time-series silently leaks the future into training.
- **Text/timeseries data kinds.** `DataKind = Literal["tabular","image"]`. No tokenizer, vocab, lag-feature, or windowing machinery.
- **Auto-detection / zero-config.** `--config` is *required* on every CLI stage. No data sniffing, no config synthesis.
- **HPO write-back.** `pipeline/hpo.py:73-79` `print()`s the best params for manual copy-paste.
- **Mixed precision, multi-GPU strategy, resume-from-checkpoint, cross-validation as a first-class mode.** `pl.Trainer` is constructed with `accelerator="auto", devices="auto"` and nothing else.
- **Coverage threshold.** `pytest-cov` runs in CI but no minimum is enforced.

### 1.3 What must be refactored (not merely extended)

| Component | Problem | Verdict |
|---|---|---|
| `core/lit_model.py::BaseModel` | Takes whole `ExperimentConfig`; reads `config.model.hidden_dims`, `config.optim.lr` | Change `__init__` to `(task, params, optim, class_weights)`. ~10-line diff; `build_network()` bodies in `mlp.py`/`cnn.py` unchanged |
| `pipeline/train.py` | `pl.Trainer` + callbacks + checkpoint recovery inline | Move verbatim into `backends/lightning.py`. **Acceptance test: `grep pytorch_lightning pipeline/train.py` → 0 hits** |
| `core/lit_data.py` | Lightning-only; `train/val/test_dataloader` + `_workers` duplicated verbatim in both subclasses (`:241-262` and `:321-341`) | Split into `data/sources/*` + `data/preprocess/*` + one `BundleDataModule` adapter |
| `core/registry.py` | Maps name→class only; duplicate registration hard-fails | Spec-carrying registry with capabilities, extras, search spaces |
| `models/__init__.py:16` | Bare `except Exception: pass` hides real bugs behind "torchvision missing" | Replace with `importlib.util.find_spec` availability checks; builtins re-raise |
| `core/inference.py` | Imports torch at module scope; hardcodes `scaler.pkl`; `task == "regression"` gates proba | Manifest-driven dispatch, **no torch import** — a GBDT serving image drops ~2 GB |
| `core/evaluate.py` | Hand-rolled torch predict loop with task branching | Split: backend produces `Predictions`, `core/metrics.py` computes from arrays |
| `pipeline/hpo.py` | Search space hardcoded to MLP shape; silently tunes irrelevant params for a `cnn` | Delete; replace with plugin-declared spaces |
| `serving/api.py:40-49` | Prometheus collectors at module scope → duplicate-timeseries on second `create_app` | Move to a registry-safe `get_collectors()`. Becomes load-bearing once tuning creates apps repeatedly |
| `utils/logging.py:13-14` | `_STREAM_READY`/`_FILE_READY` process globals — second `train()` logs into the *first* run's file | Per-output-dir handler tracking. **Moves onto the critical path** the moment `tune()` calls `train()` N times in one process |
| `cli.py:89-106` | Mutates `os.environ` as an implicit side channel to the ASGI factory | Collapse to `MLF_BUNDLE` + `MLF_API_KEY`; the rest lives in the manifest |

### 1.4 Repo hygiene (outside the package)

- `ml_framework_complete/` is a stale near-duplicate snapshot whose only unique content is a deleted AutoGluon baseline stage — fold the idea into the always-on baseline (§3.8) and delete the directory.
- Two near-identical 900-line HTML docs at repo root, unreferenced by the README.
- The project is **not its own git repo** — `git rev-parse --show-toplevel` returns `C:/Users/bibek`, with one commit. `outputs/`, `.mypy_cache/`, `.ruff_cache/`, `.gui_runs/` are physically in the tree. **Initialize a repo at the project root before Phase 0.**

---

## 2. Proposed Directory & Module Structure

```
ml_framework/
├── src/ml_framework/
│   ├── core/
│   │   ├── types.py            NEW  Task, DataKind, Capabilities, Requirement — dependency-free
│   │   ├── protocols.py        NEW  Estimator, TrainingBackend, Preprocessor, Splitter,
│   │   │                            FitResult, Predictions, RunContext          ← THE CRUX
│   │   ├── plugins.py          NEW  generic PluginRegistry, discovery, MissingExtraError
│   │   ├── task.py             NEW  TaskSpec table: metric, direction, monitor, postprocess
│   │   ├── bundle.py           NEW  Manifest model, write_bundle, read_manifest, check_requirements
│   │   ├── metrics.py          NEW  array-based per-task metrics (numpy/sklearn only)
│   │   ├── registry.py         REWRITE  thin: MODELS/BACKENDS/SOURCES + validate_combination
│   │   ├── inference.py        REWRITE  manifest dispatch; NO torch import
│   │   ├── evaluate.py         REWRITE  consumes Predictions; writes report.txt/predictions.csv
│   │   ├── lit_model.py        MODIFY   __init__(task, params, optim, class_weights)
│   │   └── lit_data.py         DELETE   (helpers re-exported from core/__init__ for compat)
│   ├── backends/               NEW PACKAGE — one per FIT-LOOP SHAPE, not per library
│   │   ├── base.py                  shared budget / early-stop / metric-name helpers
│   │   ├── lightning.py             pl.Trainer lives HERE; LightningEstimator
│   │   ├── gbdt.py                  xgboost / lightgbm / catboost / sklearn
│   │   └── forecast.py              prophet / statsmodels / naive
│   ├── data/
│   │   ├── types.py            NEW  DataBundle, Split, FeatureSchema, Signature
│   │   ├── builders.py         REWRITE  build_bundle(config) -> DataBundle
│   │   ├── splitters.py        NEW  Random / Stratified / Temporal / Group / RollingOrigin
│   │   ├── sniff.py            NEW  data-kind / target / task detection
│   │   ├── lightning_adapter.py NEW BundleDataModule (kills the duplicated dataloader code)
│   │   ├── sources/            NEW  tabular.py image.py text.py timeseries.py
│   │   └── preprocess/         NEW  base.py tabular.py image.py text.py timeseries.py
│   ├── plugins/                NEW PACKAGE (replaces models/)
│   │   ├── __init__.py              explicit builtin list + entry-point discovery
│   │   ├── mlp.py cnn.py            moved; build_network() bodies unchanged
│   │   ├── gbdt/                    xgboost.py lightgbm.py catboost.py
│   │   ├── ts/                      naive.py lstm.py tft.py prophet.py
│   │   └── nlp/                     hf_text.py
│   ├── config/
│   │   ├── schema.py           REWRITE  v2 blocks; plugin-validated params
│   │   ├── defaults.py         NEW  (kind, task, size) → model; per-backend tune budgets
│   │   ├── autoconfig.py       NEW  synthesis from CLI flags + sniff
│   │   └── migrate.py          NEW  v1 → v2 YAML remapper
│   ├── pipeline/
│   │   ├── train.py            REWRITE  orchestration only; zero torch imports
│   │   ├── tune.py             NEW      (hpo.py DELETED)
│   │   ├── lr_finder.py        MODIFY   capability gate
│   │   ├── contracts.py        keep     Pandera validation
│   │   └── spark_preprocess.py keep
│   ├── serving/
│   │   ├── api.py              MODIFY   signature-driven schemas
│   │   ├── schemas.py          NEW      per-data-kind request/response models
│   │   ├── metrics.py          NEW      registry-safe Prometheus collectors
│   │   └── asgi.py             MODIFY   MLF_BUNDLE
│   ├── tracking/
│   │   ├── run_logger.py       NEW      RunLogger protocol + mlflow/wandb/csv/null impls
│   │   └── mlflow_utils.py     MODIFY   log_artifacts(bundle_dir); accepts RunLogger
│   ├── monitoring/             keep     drift.py, model_quality.py (read signature.feature_names)
│   ├── utils/logging.py        MODIFY   per-output-dir handler tracking
│   └── cli.py                  MODIFY   zero-config flags + models/backends/init/migrate
├── configs/                    migrate to v2 + add per-family examples
├── tests/                      + tests/backends/, tests/plugins/, tests/data/
└── (Dockerfile, deploy/, dvc.yaml, orchestration/ — updated for bundle v2)
```

---

## 3. Component-by-Component Architecture

### 3.1 The estimator/backend split — `core/protocols.py`

**Decision: the `TrainingBackend` owns the fit loop. `fit()` does NOT go on the model.**

The cardinality argument decides it: ~15 models map onto **3 fit-loop shapes**. The Lightning loop (~50 lines of Trainer + callbacks + checkpoint recovery) is byte-identical for MLP, CNN, LSTM, TFT and an HF transformer. XGBoost's is ~8 lines and identical across XGB/LGBM/CatBoost. Prophet's is fit-per-series. Putting `fit()` on the model means either 15 duplicates or a base class that owns the loop — and *that is a backend expressed through inheritance*, where you cannot swap it, cannot test it in isolation, and an HF transformer cannot reuse the Lightning loop without subclassing `BaseModel`.

The second argument is this repo's own history: `config/schema.py:1-16` documents that the v1 sin was training code writing derived values onto a shared global. Giving the model a `fit()` re-creates that coupling — the model would need the output dir, checkpoint policy, tracking logger, and early-stopping config. **Keep the model dumb: it knows its architecture and its forward pass.** That is precisely why `mlp.py`/`cnn.py` survive nearly untouched.

| Backend | Shape | Members |
|---|---|---|
| `lightning` | iterative, mini-batch, epoch loop + validation callbacks | MLP, CNN, LSTM, TFT, HF transformer |
| `gbdt` | one-shot `fit(X, y, eval_set=)` + builtin early-stopping callback | XGBoost, LightGBM, CatBoost, sklearn |
| `forecast` | fit-per-series, no X/y, predict-by-horizon | Prophet, statsmodels, seasonal-naive |

**`Estimator` is predict-only** — minimal by design, because it is the only thing that crosses into the serving process:

```
Estimator (Protocol)
    predict(inputs) -> np.ndarray
    predict_proba(inputs) -> np.ndarray      # raises UnsupportedCapability if absent
```

`save`/`load` are deliberately **not** on the estimator — `load` needs registry access to rebuild an architecture before loading weights, which would make every estimator a registry client. They live on the backend:

```
TrainingBackend (Protocol)
    name: ClassVar[str]; capabilities: ClassVar[Capabilities]
    fit(spec, bundle, cfg, *, run: RunContext) -> FitResult
    save(est, dest) -> ArtifactRef                    # {"path": "model/model.json", "format": "xgboost-json"}
    load(bundle_dir, manifest) -> Estimator
    predict_split(est, bundle, split) -> Predictions
    search_space() -> Mapping[str, ParamSpec]         # backend-level knobs (lr, batch_size)
    trial_hooks(trial) -> TrialHooks                  # Optuna pruning, per-backend
    params_model() -> type[BaseModel]                 # pydantic schema for fit.params

FitResult:   estimator, val_metrics: dict[str,float], history, extra_files
Predictions: y_true, y_pred, y_prob, index           # index needed for forecasting reports
RunContext:  output_dir, seed, run_logger, resolved budget, device/precision
```

**`BaseModel` stays a `LightningModule`** — it never becomes an `Estimator`. `LightningBackend.fit()` builds it, wraps the bundle in `BundleDataModule`, constructs `pl.Trainer` (code moved verbatim from `pipeline/train.py:89-124`), fits, reloads the best checkpoint, and returns `FitResult(estimator=LightningEstimator(module, task))`. `LightningEstimator` is ~25 lines holding the module + task and applying canonical head postprocessing — **which collapses the sigmoid/softmax/identity branching currently duplicated across `lit_model._shared_step`, `evaluate.py:41-52`, and `inference.py:133-149` into one place.**

**Payload types are kind-parameterized, not universal.** `predict(X)` is a lying signature for forecasting — Prophet takes a horizon, not a feature matrix. One protocol, payload determined by `data.kind`:

| kind | inputs |
|---|---|
| tabular | `np.ndarray (n,d)` or `pd.DataFrame` |
| image | tensor/PIL batch |
| text | `list[str]` |
| timeseries | `ForecastRequest(horizon, history, exog)` |

### 3.2 Registry evolution — `core/plugins.py` + `core/registry.py`

One generic `PluginRegistry[SpecT]`, three instances: `MODELS`, `BACKENDS`, `SOURCES`. `_DATAMODULE_REGISTRY` dies — datamodules are now a Lightning adapter, not an extension point; **data sources** are.

```
ModelSpec (frozen dataclass)
    name, backend, build: Callable[[BuildContext], Any]
    params_model: type[BaseModel]            # pydantic, frozen, extra="forbid"
    tasks: frozenset[Task]; data_kinds: frozenset[DataKind]
    requires: tuple[Requirement, ...]        # (module, extra, min_version)
    search_space: Mapping[str, ParamSpec]    # keys are DOTTED CONFIG PATHS
    suggest: Callable | None                 # escape hatch for conditional spaces
    capabilities: Capabilities
    auto_priority: int                       # tie-break for zero-config selection
```

`BuildContext` is one frozen dataclass (`task, input_dim, output_dim, n_classes, feature_schema, class_weights, params, optim, seed, device`) so adding a field later doesn't touch 15 plugins.

**Every capability flag has exactly one named consumer** — a flag nothing reads is decoration:

| Flag | Consumer |
|---|---|
| `needs_scaling` | preprocessor skips `StandardScaler` for GBDT (pointless; destroys interpretability) |
| `native_categorical` | passes pandas `category` dtype instead of one-hot |
| `native_missing` | skips imputation — XGB/LGBM handle NaN natively and imputing *hurts* them |
| `supports_sample_weight` | imbalance resolver prefers weights over SMOTE (SMOTE is nonsense for GBDT) |
| `produces_proba` | `/predict_proba` returns 400 from the manifest, replacing `serving/api.py:185`'s hardcoded task check |
| `supports_pruning` | HPO installs a pruning hook or picks a non-pruning `Pruner` |
| `supports_gpu` / `supports_mixed_precision` | device/precision resolution; **warns** instead of silently ignoring `precision: 16` |
| `supports_lr_range_test` | `mlf lr` refuses politely on GBDT instead of crashing inside `torch_lr_finder` |
| `accepts: frozenset[Payload]` | build-time check → *"xgboost cannot consume an image folder"* instead of a shape error 200 lines deep |

**Discovery replaces `try/except Exception: pass` structurally, not with better except clauses.** Rule: *a plugin module must be importable with zero optional deps installed* — heavy imports go inside `build()`/`fit()`, never at module scope. Availability is `importlib.util.find_spec` (no import, no exception), so `mlf models` lists every plugin with an availability column on a bare install.

- Builtins come from an explicit list; a failure there is **our bug** → re-raise with traceback.
- Third-party plugins via `entry_points(group="ml_framework.plugins")`; failure is recorded as `PluginLoadError`, warned once, shown in `mlf models --all`, and **re-raised chained if the user actually selects it**. Never silent.
- Selecting an unavailable plugin raises `MissingExtraError("model 'xgboost' requires xgboost>=2.0 — pip install 'ml-framework[gbdt]'")`. This exact message is the difference between a framework that feels finished and one that doesn't.

`registry.validate_combination(task, data_kind, model_name, payload)` is called from the config validator so incompatible combos fail at load time, not 40 seconds into data loading.

### 3.3 v2 config schema — `config/schema.py`

**Decision: `model.name` + free-form `model.params` validated by the plugin's own Pydantic model. NOT a discriminated union.**

Rejecting the discriminated union on three grounds: (a) a `Literal[...]` discriminator is a **closed set living in core**, so every third-party plugin would require editing `config/schema.py` — destroying the plugin story; (b) Pydantic must import every union member at `ExperimentConfig` import time, pulling xgboost/transformers/prophet eagerly and defeating lazy extras; (c) the discriminator is on the wrong axis — `name` determines params, not `family` (xgboost and lightgbm are one family, different params).

The accepted cost (no IDE completion; `--set model.params.max_depth=8` bypasses the "key must exist" rule) is mitigated:
- An `ExperimentConfig` `model_validator(mode="after")` lazily imports the registry, runs `spec.params_model.model_validate(self.model.params)` (frozen + `extra="forbid"`, so **typos still error at load**), and writes the defaulted dict back via `model_copy(update=...)` — which does not re-run validation, so no recursion. Bonus: `config.json` now records fully-materialized effective params, a reproducibility win over today.
- `with_overrides` gets one narrow change: dotted paths under `model.params.*` / `fit.params.*` / `data.params.*` may create keys; everywhere else today's `KeyError` behavior is preserved, so `test_config.py::test_with_overrides_unknown_key_raises` survives.
- `mlf models --show xgboost` prints `params_model.model_json_schema()` + the search space — better discoverability than IDE completion.

**Circular-import guard:** move `Task`/`DataKind`/`Capabilities` into dependency-free `core/types.py`; both `config` and `plugins` import from there.

**Keep `task` and `DataKind` orthogonal.** Text classification *is* `multiclass` with `data.kind: text` — do not create a `text_classification` task, or the `Literal` becomes combinatorial. Task determines loss/metrics/head; kind determines ingestion.

```
Task     = binary | multiclass | multilabel | regression | forecasting
           | token_classification | seq2seq
DataKind = tabular | image | text | timeseries
```

The `Literal` is only a key. The real growth point is **`core/task.py`**:

```
TaskSpec: name, primary_metric, direction (min|max), monitor,
          output_kind (labels|probabilities|values|series),
          metric_fns, postprocess (sigmoid|softmax|identity)
```

Consumed by early stopping, checkpointing, HPO direction, `evaluate()`, serving response models, and `LightningEstimator.predict_proba`. Adding `forecasting` becomes "add a row + MASE/sMAPE", not a six-file grep.

**v2 shape:**

```yaml
task: multiclass
runtime: {seed: 42, output_dir: outputs, num_workers: -1, accelerator: auto, precision: 32}
data:
  kind: tabular                 # sniffed if omitted
  path: data/train.csv
  target: label
  split: {strategy: auto, val_size: 0.15, test_size: 0.15, time_col: null, gap: 0}
  params: {imbalance_strategy: auto, holdout_threshold: 5000}
model:
  name: xgboost
  params: {max_depth: 6, n_estimators: 500}      # validated by the plugin
fit:
  budget: {max_epochs: 200, max_seconds: null}
  patience: 20
  batch_size: 32
  params: {lr: 1.0e-3, weight_decay: 1.0e-4}     # validated by the backend
tune:
  enabled: true
  max_trials: 20
  max_seconds: 900
  metric: null                  # null → TaskSpec.primary_metric
  refit: best
  overrides: {}                 # narrow a plugin's declared search space
logging: {backend: csv, ...}    # essentially unchanged
```

v1→v2 map: `optim.lr`→`fit.params.lr`; `train.epochs`→`fit.budget.max_epochs`; `train.gradient_clip_val`→`fit.params.gradient_clip_val` (correctly Lightning-only); `data.csv_path`→`data.path`; `data.target_col`→`data.target`; `data.img_size`→`data.params.img_size`; `model.hidden_dims`→`model.params.hidden_dims`; `hpo_n_trials`→`tune.max_trials`; `seed`/`output_dir`→`runtime.*`. Ship `mlf migrate-config` (~60 lines) — it pays for itself immediately.

### 3.4 Data layer — `data/types.py`, `data/sources/`, `data/preprocess/`, `data/splitters.py`

`DataBundle` is the framework-agnostic handoff. **`class_weights` becomes numpy, not torch** — the agnostic layer must not import torch; the Lightning backend converts.

```
Split:         payload (arrays|frame|dataset|series), x, y, index, n
FeatureSchema: feature_names, dtypes, categorical_idx, target_name,
               class_names, time_col, freq
DataBundle:    train/val/test: Split, schema, task, data_kind,
               input_dim, output_dim, class_weights: np.ndarray|None,
               preprocessor (fitted on train ONLY), reference_stats, meta
```

The `payload` tag keeps it honest: image/text splits hold a lazy `Dataset`, not arrays. `Capabilities.accepts` declares what each backend consumes (`gbdt` accepts `{arrays, frame}`), enabling the build-time compatibility error.

`BundleDataModule` (`data/lightning_adapter.py`) is the **only** Lightning datamodule — one implementation of `train/val/test_dataloader`, parameterized by an optional sampler and an optional `collate_fn` from the preprocessor (needed for text padding). This deletes the verbatim duplication at `lit_data.py:241-262` and `:321-341`.

**Preprocessor owns all fitted transform state:**

```
Preprocessor (Protocol): fit(split, schema), transform(x),
                         save(dest) -> manifest fragment,
                         load(src, spec), collate_fn
```

Everything lands in `bundle/preprocessor/` with a `preprocessor.json` naming its dotted class path and files (`scaler.pkl`, `encoder.json`, `tokenizer/`, `series_scalers.pkl`). **Nothing outside the Preprocessor reads that directory** — replacing the hardcoded `scaler.pkl` special-cases at `inference.py:82-86` and `mlflow_utils.py:74`. Keep a ~10-line v1 compat branch (no `preprocessor/` but `scaler.pkl` present → wrap in `TabularPreprocessor`); the clean break was authorized for configs and tests, **not for already-deployed bundles**.

**Temporal splitting is enforced, not conventional:**
- `RandomSplitter` — today's `split_dataset` logic verbatim, so existing split tests survive.
- `TemporalSplitter(time_col, val_size, test_size, gap)` — sort by time, cut contiguously `[train | gap | val | gap | test]`. Never shuffled, never stratified. `gap` prevents leakage through lag features.
- `RollingOriginSplitter(n_folds, horizon, gap, expanding)` for time-series CV.
- `GroupSplitter(group_col)` — repeated-entity leakage; not present today and a common silent defect.

`split.strategy: auto` resolves to temporal whenever `kind == timeseries` **or** `time_col` is set. Explicitly writing `strategy: random` with `kind: timeseries` raises unless `split.allow_temporal_leakage: true` is also set. **Make the leaky path require typing the word "leakage."**

### 3.5 Artifact bundle v2 — `core/bundle.py`

```
<output_dir>/
  manifest.json        ← the ONLY file a loader must understand
  config.json          ← effective config post-defaults/post-HPO. Audit record.
  model/               ← opaque to everything but the backend
      model.ckpt | model.json | model.cbm | hf_model/ | prophet.pkl
  preprocessor/        ← opaque to everything but the Preprocessor
  metrics.json · reference_stats.json · hpo.json
  report.txt · predictions.csv · training.log
```

`manifest.json` carries: `bundle_version`, `framework_version`, `task`, `data_kind`, `model{name, backend, artifact, format, params}`, `preprocessor{class, dir}`, `signature{input{payload, features[], n_features}, output{kind, n_classes, class_names}}`, `requires[]`, `metrics`, `hpo`.

**The invariant that makes heterogeneous model files a non-problem:** the manifest is the only uniform thing, and it *names* the non-uniform things. `model/` may be a file or a directory — the loader never looks; it hands `bundle_dir` + `manifest` to `backend.load()`. `format` is tracked separately from the file extension so serialization can migrate (xgboost json → ubj) without breaking readers.

The `signature` block replaces `serving/api.py:153`'s `getattr(inf.model, "input_dim", None)` — code that reaches into a torch module to learn its own API contract and returns `None` for any non-torch estimator.

**Do not embed the full config in the manifest** (today `metadata.json` does and `inference.py:72` reconstructs `ExperimentConfig` from it). That requires a populated plugin registry *and* every training extra at serving time. `config.json` sits beside it as the audit record; the manifest is the serving contract.

`mlflow_utils.log_and_register` changes from a hardcoded 4-filename loop (`:74`) to `client.log_artifacts(run_id, bundle_dir, "bundle")`.

### 3.6 Inference & serving

`core/inference.py` rewritten to import **neither torch nor pytorch_lightning**:
1. `read_manifest` → reject `bundle_version > 2`
2. `check_requirements` → `MissingExtraError` with the pip command **before** any import attempt
3. `BACKENDS.get(manifest.model.backend)` → `backend.load(dir, manifest)`
4. `load_preprocessor(manifest)` via dotted path
5. `predict = estimator.predict(pre.transform(x))`; `predict_proba` gates on `signature.output.kind`

**A GBDT serving container no longer installs torch** — ~2 GB of image and a real cold-start difference, possible only because `inference.py` stops importing torch at module scope. `from_registry` keeps its shape (`download_bundle` → `from_artifacts`).

`serving/api.py`: request/response models become **manifest-derived**, selected from a table in `serving/schemas.py` keyed by `data_kind`, so FastAPI's OpenAPI docs describe *this specific model* — a genuine feature.
- tabular: accept both `{"instances": [[…]]}` (back-compat with `tests/serving/test_api.py`) and `{"inputs": [{"f0": 1.0}]}`. **Recommend named as the documented production contract** — silent column reordering is the most common serving defect, and feature names are now in the signature to prevent it.
- text `{"inputs": ["…"]}`; image `{"inputs": [{"b64"|"url"}]}`; forecasting `{"horizon", "history", "exog"}` → `{"forecast", "index", "lower", "upper"}`.
- `/drift` is tabular-only → 501 with an explanation for other kinds rather than computing PSI over token ids.
- `/health` keeps `status` and `task` keys (so `test_health` survives) and adds model/backend/versions.
- Fix the module-scope Prometheus collectors; this is now load-bearing because tuning and multi-model serving create apps repeatedly.

### 3.7 HPO driver — `pipeline/tune.py` (deletes `pipeline/hpo.py`)

**Declarative search spaces keyed by dotted config paths**, so applying a trial is exactly `config.with_overrides(trial_values)` — a mechanism that already exists and is already tested.

```
ParamSpec = Float(low, high, log, step) | Int(...) | Categorical(choices) | Const(v)
```

Declarative because it is serializable (into `hpo.json`), narrowable from YAML (`tune.overrides`), and inspectable (`mlf models --show`). **Honest limitation:** conditional spaces (today's `n_layers` → `n_units_l{i}`, `hpo.py:35-38`) don't express well declaratively — plugins get an optional `suggest(trial, cfg)` escape hatch that overrides the declarative space. MLP uses it; XGBoost doesn't need it.

Effective space = `merge(model_spec.search_space, backend.search_space(), cfg.tune.overrides)`, so `lr`/`batch_size` are declared once on the Lightning backend rather than repeated in every neural plugin.

**Pruning is per-backend; the driver never imports `optuna_integration`.** `backend.trial_hooks(trial)` returns `PyTorchLightningPruningCallback` (the dual-import fallback at `hpo.py:29-32` moves into `backends/lightning.py`), `XGBoostPruningCallback`/`LightGBMPruningCallback`, or `TrialHooks.empty()` for forecasting.

Metric selection reads `cfg.tune.metric or TaskSpec.primary_metric` from `FitResult.val_metrics` (a plain dict) with direction from `TaskSpec`. **This fixes a latent bug**: today's objective hardcodes `val/loss` from `trainer.callback_metrics` (`hpo.py:64`), which no non-Lightning backend produces.

**Pushing back on "always on" as literally stated:** 20 trials × 200 epochs on the Lightning path is hours and will make `mlf train` feel broken. Default-on with **backend-aware budgets** in `config/defaults.py`:

| backend | default trials | wall budget |
|---|---|---|
| gbdt | 30 | 300 s |
| lightning | 10 | 900 s, per-trial epochs capped at 25 |
| forecast | 8 | 180 s |

Auto-disable with an INFO log when the merged space is empty. `--no-tune` disables; `--tune-budget 10m` / `--tune-trials N` turn it up as easily as off.

**Write-back closes today's real gap** (`hpo.py:73-79` prints for copy-paste): `train()` calls `tune()`, applies `with_overrides(best_params)`, runs the final fit, and emits `hpo.json` + `manifest.hpo` + tuned `bundle/config.json`, with optional `--emit-config configs/tuned.yaml`. `tune.refit: best|reuse` — `best` (default) retrains at full budget; `reuse` keeps the trial model (cheap, but trained under a reduced budget). Explicit, because both are defensible.

### 3.8 Zero-config layer — `config/autoconfig.py`, `data/sniff.py`, `config/defaults.py`

Synthesis produces a plain **dict**, never a validated config, so it slots into an ordinary precedence chain:

```
plugin/backend defaults < autoconfig synthesis < YAML file < --set overrides < explicit CLI flags
                                      → ExperimentConfig.model_validate(merged)
```

Nothing special-cases synthesized values — that is the entire mechanism, and it is why `--config` and `--data` compose freely.

**`mlf init --data x.csv --target label -o configs/mine.yaml`** writes the synthesized YAML with a comment per inferred field. This is what stops zero-config from being a black box. Every inference also logs the rule that fired: `data.kind=timeseries (column 'date' parses as datetime and is monotonic)`.

Sniffing rules: dir-of-image-dirs → `image`; `.csv`/`.parquet` → `tabular`, upgraded to `timeseries` on a monotonic datetime column (or `--time-col`), or `text` if a string column's mean token count exceeds a threshold; `.jsonl`/`.txt` → `text`. Target: `--target` > a column named `label|target|y` (error if several match — don't guess) > last column with a warning. Task: timeseries → `forecasting`; else 2 unique → `binary`, non-float ≤~20 unique → `multiclass`, else `regression`. Model: a rules table keyed by `(kind, task)` with row/feature thresholds, filtered by availability, tie-broken by `auto_priority` — tabular→`xgboost`, image→`cnn`, text→`hf_text`, timeseries→`lstm` (fallback `seasonal_naive`). An uninstalled extra raises `MissingExtraError`, **never a silent downgrade to MLP**.

**Always fit a baseline.** Whenever the framework picks the model itself, also fit the trivial baseline (majority class / mean / seasonal-naive — the `ts.naive` plugin is needed anyway) and record `baseline_*` in `metrics.json` with a WARNING if the model doesn't beat it. Costs milliseconds; it is the single best guard against zero-config ML quietly shipping a useless model. *(This also subsumes the AutoGluon baseline stage that exists only in the stale `ml_framework_complete/` snapshot.)*

### 3.9 Deep-learning capabilities (folded into `backends/lightning.py`)

Mixed precision (`runtime.precision: 16-mixed|bf16-mixed`, gated on `supports_mixed_precision`), multi-GPU strategy (`ddp`/`ddp_spawn`/`auto`), resume-from-checkpoint (`--resume` reading `bundle/model/last.ckpt` — add `save_last=True`), and gradient accumulation all become `pl.Trainer` kwargs resolved in one place. Cross-validation is a `train()` orchestration mode driven by the `Splitter`, so it works for GBDT and forecasting too — not a Lightning feature.

### 3.10 Deployment & monitoring extensions

Per-extra Docker targets (`serve-gbdt` without torch is the payoff from §3.6); export adapters (`ONNX`/`TorchScript` for Lightning, native `.json`/`.cbm`/`booster` for GBDT, `pickle` for Prophet) declared as a backend method so `mlf export --format onnx` fails loudly where unsupported; existing Prometheus alerts + Grafana dashboard extended with per-backend labels; `monitoring/drift.py` reads `signature.feature_names` instead of guessing.

---

## 4. Technology Stack

| Concern | Choice | Justification |
|---|---|---|
| Config | **Pydantic v2** (keep) | Already frozen + `extra="forbid"`; per-plugin `params_model` reuses the same machinery. **Explicitly not Hydra/OmegaConf** — the precedence chain in §3.8 is ~40 lines and Hydra would fight the plugin registry for ownership of composition |
| DL training | **PyTorch Lightning** (keep) | Already the fit loop; gives AMP/DDP/callbacks for free. Isolated to `backends/lightning.py` |
| GBDT | **XGBoost ≥2.0, LightGBM ≥4.0, CatBoost ≥1.2** | All three: XGB is the default, LGBM wins on wide/large tabular, CatBoost on high-cardinality categoricals. One backend covers all three — marginal cost per library is ~40 lines |
| NLP | **HuggingFace transformers + tokenizers + datasets** | Only credible option. Models wrap into a `LightningModule`, so they reuse the Lightning backend rather than adding a fourth |
| Time-series | **statsmodels + Prophet**, LSTM/TFT native in Lightning | Prophet for interpretable seasonality; statsmodels for ARIMA/ETS + stationarity tests (ADF/KPSS). **Deliberately not `pytorch-forecasting`** — it brings its own datamodule/trainer abstractions that duplicate and conflict with `DataBundle` |
| HPO | **Optuna** (keep) + `optuna-integration` | Already a dependency; TPE Bayesian + median pruning + grid/random samplers all satisfy the search-strategy requirement. **Not Ray Tune** — Ray's value is multi-node scheduling, which this single-package/single-node design doesn't need, and it would own the process model |
| Tracking | **MLflow** primary, W&B optional (keep) | MLflow already wired incl. Model Registry with the low-level `create_model_version` path that keeps the bundle portable. W&B stays a `RunLogger` impl |
| Serving | **FastAPI + uvicorn/gunicorn** (keep) | Already hardened: API key, slowapi rate limiting, size caps, Prometheus, KServe manifests |
| Data validation | **Pandera** (keep) | `pipeline/contracts.py` works; extend to emit `FeatureSchema` |
| Drift | **scipy** PSI/KS (keep) | Sufficient. **Not Evidently/NannyML** — heavy deps for capability already present; revisit only if drift reporting becomes a product surface |
| Preprocessing | **scikit-learn + pandas** (keep) | `StandardScaler`/imputers/encoders; sklearn estimators also drop into the `gbdt` backend free |
| Imbalance | **imbalanced-learn** (keep) | Capability-gated: SMOTE only where `supports_sample_weight` is False |
| Distributed prep | **PySpark** (keep, optional) | Already used for large-parquet preprocessing |
| Pipeline/versioning | **DVC + Airflow** (keep) | Both wired; `dvc.yaml` outs point at the bundle dir |
| Testing | **pytest + pytest-cov** (keep) | Add `importorskip` guards per extra; enforce a coverage floor |

**New pyproject extras** — the `extra` field in each `Requirement` must match these names verbatim, since that is what makes the error messages actionable:
```
gbdt       = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2"]
timeseries = ["statsmodels>=0.14", "prophet>=1.1"]
nlp        = ["transformers>=4.40", "tokenizers", "datasets"]
export     = ["onnx", "onnxruntime", "skl2onnx"]
```

---

## 5. Implementation Roadmap

> **This is the original ten-phase roadmap, kept as written.** P0–P9 below were scoped
> before any of them were built. Three further phases — P10 (exporter migration), P11
> (model selection) and P12 (pluggable data backends) — were scoped afterwards, in
> response to what the earlier ones surfaced, and are deliberately **not** backfilled
> here: a plan edited to predict what actually happened is no longer evidence of what was
> intended. For the live state of all twelve, see
> [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md).

Strictly dependency-ordered. **Phases 0–3 are the load-bearing work**; everything after is additive.

| Phase | Content | Exit gate |
|---|---|---|
| **P0 — Foundations** | Init a git repo at the project root; delete `ml_framework_complete/` + duplicate HTML. Add `core/types.py`, `protocols.py`, `plugins.py`, `task.py`, `bundle.py`, `metrics.py`, `tracking/run_logger.py`. New registries populated *alongside* the old ones. **Zero behavior change.** | Full existing suite green, untouched |
| **P1 — Data + backend extraction** ⚠️ | `DataBundle`, sources, preprocessors, splitters, `BundleDataModule`; `backends/lightning.py`; `pipeline/train.py` → pure orchestration; `evaluate` consumes `Predictions`; bundle v2 written; fix `utils/logging.py` globals | Integration tests pass with only artifact-path edits. **`grep pytorch_lightning src/ml_framework/pipeline/train.py` → 0** |
| **P2 — v2 config** | `schema.py` rewrite, `migrate.py`, migrate 3 configs + `conftest.make_config`, `lit_model` param injection, `models/` → `plugins/`, non-swallowing discovery | All configs + fixtures migrated; `test_config.py` rewritten; `mlf migrate-config` round-trips |
| **P3 — GBDT** | `backends/gbdt.py` + xgboost/lightgbm/catboost plugins; torch-free `inference.py`; manifest-driven serving + `serving/schemas.py` + Prometheus fix; `gbdt` extra; `serve-gbdt` Docker target | **`mlf train --data x.csv --model xgboost` produces a bundle that serves correctly in an environment where torch is not installed.** That single test proves the whole abstraction |
| **P4 — AutoML** | `pipeline/tune.py`, declarative spaces, per-backend pruning, budget defaults, write-back, `hpo.json`; delete `hpo.py` | `mlf train` on a small CSV tunes and finishes within the default budget; best params land in `bundle/config.json` |
| **P5 — DL hardening** | Mixed precision, DDP strategy, resume-from-checkpoint, gradient accumulation, cross-validation as an orchestration mode, configurable optimizer/scheduler | AMP run matches FP32 metrics within tolerance; `--resume` continues from `last.ckpt`; CV works for GBDT too |
| **P6 — Time-series** | `forecasting` TaskSpec row, `TemporalSplitter`/`RollingOrigin`, timeseries source + preprocessor (lags, windows, ADF/KPSS), `ts.naive`/`ts.lstm`/`ts.prophet`, `backends/forecast.py`, MASE/sMAPE, forecast serving schema | **Leakage test: `strategy: random` on `kind: timeseries` raises.** Temporal split beats shuffled-CV on a sanity check |
| **P7 — NLP** | Text source + tokenizer preprocessor, `nlp.hf_text` on the Lightning backend, HF-directory artifact format, fine-tuning defaults | Tokenizer round-trips through the bundle; text `/predict` accepts raw strings |
| **P8 — Zero-config** | `sniff.py`, `autoconfig.py`, `defaults.py`, `mlf init`, always-on baseline, `mlf models/backends` | **`mlf train --data x.csv` with no YAML and no flags produces a bundle and reports vs. baseline** |
| **P9 — Deployment polish** | ONNX/TorchScript export adapters, per-model Dockerfile generation, monitoring labels, docs rewrite, coverage floor in CI | Exported ONNX matches native predictions; CI enforces ≥80% |

**P1 is the highest-risk phase** — it changes the data layer and the fit loop simultaneously. If de-risking is wanted, split it: **P1a** introduces `DataBundle` + `BundleDataModule` feeding the *existing* `train()`; **P1b** extracts `LightningBackend`. Each half is independently revertible.

**CI guardrail from P0 onward:** the suite must pass on an install *without* the new extras. Add one explicit test asserting the registry lists an unavailable plugin without importing it and raises `MissingExtraError` on selection. That test is the guardrail for the entire plugin design.

---

## 6. Key Design Decisions & Trade-offs

| # | Decision | Alternative rejected | Trade-off accepted |
|---|---|---|---|
| 1 | **Backend owns the fit loop; model stays dumb** | `fit()` on the model / a base class per family | One more indirection layer to learn. Bought: 15 models → 3 loops, loops testable in isolation, HF transformers reuse the Lightning loop without subclassing `BaseModel` |
| 2 | **One backend per fit-loop *shape*, not per library** | One backend per library (xgboost/lightgbm/catboost separately) | A backend must handle small per-library differences internally. Bought: adding CatBoost is ~40 lines, not a new backend |
| 3 | **Free-form `model.params` + plugin Pydantic validation** | Discriminated union on `model.family` | No IDE completion; `--set` typos caught at validation rather than parse. Bought: third-party plugins never edit core, and lazy extras survive |
| 4 | **`Estimator` is predict-only; save/load on the backend** | Full `fit/predict/save/load` on the estimator | Two protocols instead of one. Bought: the serving process needs neither the registry nor training deps |
| 5 | **Manifest names non-uniform artifacts rather than forcing a uniform format** | Convert everything to ONNX/pickle at save time | Loading requires the originating library. Bought: no lossy conversion, native early-stopping/feature-importance survive, format migrations don't break readers |
| 6 | **Manifest does NOT embed the full training config** | Keep today's `metadata.json` behavior | Serving can't reconstruct `ExperimentConfig` (kept alongside as `config.json` for audit) | Bought: serving needs neither a populated registry nor training extras |
| 7 | **Task and DataKind stay orthogonal** | `text_classification`, `image_classification` as tasks | Users must set two fields (sniffing sets both). Bought: the `Literal` grows additively instead of combinatorially |
| 8 | **Declarative search spaces + a `suggest()` escape hatch** | Pure `Callable[[Trial], dict]` | Two ways to declare a space. Bought: spaces are serializable, narrowable from YAML, and printable — with conditional spaces still expressible |
| 9 | **Tune-on-by-default with backend-aware budgets** | Literal always-on with uniform trial counts | Users may still be surprised by runtime. Bought: `mlf train` finishes in minutes instead of hours; `--no-tune` and `--tune-budget` are both one flag |
| 10 | **Temporal leakage is blocked, overridable only via `allow_temporal_leakage: true`** | Warn and proceed | A deliberate friction point. Bought: the most damaging silent failure in time-series ML requires typing the word "leakage" |
| 11 | **Always fit a trivial baseline in zero-config mode** | Trust the chosen model | Milliseconds of compute + one more metrics row. Bought: the main failure mode of AutoML — silently shipping a model worse than the mean — becomes a WARNING |
| 12 | **Single package + extras** *(user decision)* | Core + entry-point plugin distributions | One release cycle can't ship a plugin fix independently. Bought: one repo, one CI, one version — the entry-point hook is still built, so splitting later stays possible |
| 13 | **Keep Optuna; don't adopt Ray Tune** | Ray Tune | No multi-node HPO. Bought: no competing process/scheduler model; Optuna is already a working dependency |
| 14 | **Keep Pydantic; don't adopt Hydra** | Hydra/OmegaConf | Hand-rolled precedence chain (~40 lines). Bought: no fight with Hydra over composition ownership; frozen validation preserved |
| 15 | **v1 bundle compat retained even though configs break** | Clean break everywhere | ~10 lines of legacy branch in the preprocessor loader. Bought: already-deployed bundles keep serving |

---

## 7. Verification

Each phase has a mechanical gate; these are the end-to-end checks that prove the architecture rather than the code.

1. **Abstraction proof (after P3)** — in a fresh venv with `pip install -e '.[gbdt,serve]'` and **no torch**: `mlf train --data data/raw/sample.csv --model xgboost --task multiclass`, then `mlf serve --artifacts outputs` and POST `/predict`. If this passes, the backend split is real.
2. **Lightning regression (after P1/P2)** — `mlf train -c configs/example_tabular.yaml` reproduces the pre-refactor `outputs/smoke/metrics.json` value (`test_acc: 0.75`) at the same seed.
3. **No-torch-in-orchestration** — `grep -r "pytorch_lightning\|import torch" src/ml_framework/pipeline/train.py src/ml_framework/core/inference.py` returns nothing.
4. **Plugin isolation** — on a bare install, `mlf models --all` lists every plugin with availability; selecting an uninstalled one raises `MissingExtraError` naming the pip extra; a deliberately broken builtin raises with its real traceback, not a swallowed pass.
5. **Leakage guard (after P6)** — `strategy: random` + `kind: timeseries` raises; with `allow_temporal_leakage: true` it proceeds.
6. **Tuning budget (after P4)** — `mlf train --data sample.csv --model xgboost` completes inside the 300 s default and `bundle/config.json` contains tuned params differing from defaults.
7. **Zero-config (after P8)** — `mlf train --data data/raw/sample.csv` with no other arguments produces a bundle, logs each inference rule, and reports `baseline_*` alongside real metrics.
8. **Bundle back-compat** — `Inferencer.from_artifacts("outputs/smoke")` (a v1 bundle already on disk) still loads and predicts.
9. **Suite + coverage** — `pytest` green with and without extras installed; `make cov` ≥ 80% enforced in `ci.yml`.

---

## 8. Feature Preservation Audit

Every capability in the current framework, mapped to its destination. **Nothing is dropped.** Items marked ⚠️ were *not* explicitly named in §§1–7 and are now specified here — they are binding requirements, not optional.

### 8.1 Config layer

| Current | Destination |
|---|---|
| Pydantic v2 `frozen=True, extra="forbid"` | Preserved at every level, incl. per-plugin `params_model` |
| `from_yaml`, `with_overrides` (dotted) | Preserved verbatim; `with_overrides` becomes the HPO trial mechanism |
| `DataConfig._check_required_by_kind`, `val_size + test_size < 1.0` | Moves to the v2 `data` block validator; same errors |
| `ModelConfig._check_dims` (positive `hidden_dims`) | Moves into the MLP plugin's `params_model` |
| ⚠️ `OptimConfig.lr_patience`, `lr_factor` ([schema.py:89-90](src/ml_framework/config/schema.py#L89)) | **Must land in `fit.params` as `lr_patience`/`lr_factor`.** The §3.3 YAML example showed only `lr`/`weight_decay` — an illustrative excerpt, not the full set. `ReduceLROnPlateau` config survives; P5 makes the scheduler *choice* configurable on top of it |
| `TrainConfig.*`, `LoggingConfig.*` (incl. all `mlflow_*`, `registered_model_name`, `wandb_*`, `log_model`) | Mapped per the §3.3 v1→v2 table; `logging` block essentially unchanged |
| `hpo_n_trials`, `hpo_timeout` | → `tune.max_trials`, `tune.max_seconds` |

### 8.2 Core

| Current | Destination |
|---|---|
| `BaseModel.build_network()` hook | Unchanged — the whole point of §3.1 |
| `_build_criterion`: BCEWithLogits **scalar `pos_weight`**, CE `weight`, MSE | Preserved. The scalar-`pos_weight` fix is guarded by [test_model_and_registry.py:38](tests/unit/test_model_and_registry.py#L38) (`# the core bug fix`) — that test must stay green |
| `_build_metrics` (torchmetrics Accuracy/F1/MAE/MSE) | Moves into `core/task.py` `TaskSpec.metric_fns`; torchmetrics still used inside the Lightning backend |
| Adam + `ReduceLROnPlateau` on `val/loss` | Preserved as the default; P5 adds configurability |
| ⚠️ `BaseModel.count_parameters()` ([lit_model.py:160](src/ml_framework/core/lit_model.py#L160), logged at [train.py:87](src/ml_framework/pipeline/train.py#L87)) | **Generalize, don't drop.** Becomes optional `TrainingBackend.model_size(est) -> dict` (params for Lightning, tree/leaf counts for GBDT) logged by the orchestrator and written into `manifest.model` |
| `read_table`, `detect_imbalance`, `compute_class_weights`, `split_dataset` | Re-exported from `core/__init__` with unchanged signatures |
| ⚠️ `apply_smote` ([lit_data.py:66](src/ml_framework/core/lit_data.py#L66)) | **Add to that re-export list** — omitted from §3.4. Logic moves to `data/preprocess/tabular.py`, capability-gated by `supports_sample_weight` |
| StandardScaler, class weights, KFold-vs-holdout at `holdout_threshold` | `RandomSplitter` + `TabularPreprocessor`, logic verbatim |
| ImageFolder, ImageNet norm constants, augmentation stack, `WeightedRandomSampler`, pretrained backbone head-resize (`fc`/`classifier`), Kaiming init | All preserved — moved to `data/sources/image.py`, `data/preprocess/image.py`, and the unchanged `plugins/cnn.py` / `plugins/mlp.py` bodies |
| ⚠️ `instantiate_model`, `instantiate_datamodule`, `available_models`, `available_datamodules` (public in [core/__init__.py:14-19](src/ml_framework/core/__init__.py#L14)) | **Keep all four as thin shims** over the new registries. `available_models()` is directly asserted by [test_model_and_registry.py:24](tests/unit/test_model_and_registry.py#L24). `available_datamodules()` maps onto `SOURCES` |
| ⚠️ `Inferencer.predict_with_confidence` ([inference.py:152](src/ml_framework/core/inference.py#L152)) | **The one genuine near-miss — not mentioned anywhere in §§1–7.** Preserve on `Inferencer`, implemented via `predict_proba().max(axis=1)`, gated on `signature.output.kind == "probabilities"`. Also worth exposing as an optional `/predict_with_confidence` route |
| ⚠️ `confusion_matrix.txt` ([evaluate.py:66](src/ml_framework/core/evaluate.py#L66)) | **Missing from the §3.5 bundle listing.** Still written for classification tasks |
| ⚠️ `predictions.csv` column schema: `label`, `prediction`, + `prob_class_{i}` (multiclass) / `probability` (binary) ([evaluate.py:75-81](src/ml_framework/core/evaluate.py#L75)) | **Exact schema preserved**, extended with an `index` column for forecasting |
| `report.txt` (`classification_report` digits=4, zero_division=0, `target_names`; MAE/RMSE for regression) | Preserved; sklearn call unchanged |
| `Inferencer.from_artifacts` / `from_registry` symmetry | Preserved — the framework's best idea |

### 8.3 Pipeline, serving, tracking, monitoring, utils

| Current | Destination |
|---|---|
| `find_lr` (torch_lr_finder, `lr_finder_plot.png`, suggestion = argmin-loss lr / 10) | Preserved, capability-gated on `supports_lr_range_test` |
| `contracts.py` (Pandera `build_schema`/`validate_dataframe`/`validate_file` + `python -m` entry) | Kept as-is; additionally emits `FeatureSchema` |
| `spark_preprocess.py` (+ `python -m` entry) | Kept unchanged |
| Optuna MedianPruner + Lightning pruning callback + `optuna_integration` fallback import | Preserved, relocated into `backends/lightning.py` |
| All 5 serving routes, API-key `hmac.compare_digest`, slowapi rate limit, `max_instances` cap, `lifespan`, `/metrics` | All preserved; only the *schemas* become manifest-derived |
| `asgi.py` env-var factory | Preserved, simplified to `MLF_BUNDLE` + `MLF_API_KEY` |
| `resolve_tracking_uri`, `get_mlflow_logger`, `log_and_register` (low-level `create_model_version` on `runs:/…/bundle`), `download_bundle` | All preserved; `log_and_register` accepts a `RunLogger` and logs the whole bundle dir |
| `drift.py` (`build_reference`, `psi_from_reference`, `ks_statistic`, `compute_drift`, `DriftTracker` w/ window & min_samples) | Unchanged; reads `signature.feature_names` instead of guessing |
| `model_quality.py` (`evaluate_against_labels` + its own CLI) | Unchanged |
| `setup_logging` UTF-8 Windows stdout fix | Preserved (only the module-global handler tracking changes) |
| `resolve_num_workers` (Windows→0), `seed_everything` | Unchanged |
| Documented bug fixes: scalar `pos_weight`, KFold-for-regression, `Subset` label recovery, MLflow file-store deprecation, `optuna_integration` move | All preserved **and still covered by their existing regression tests**, which the plan treats as non-negotiable |

### 8.4 Infrastructure (unchanged unless noted)

Dockerfile multi-stage (gains per-extra serve targets) · docker-compose (api/train/mlflow/minio/prometheus/grafana) · 10 K8s manifests incl. KServe, NetworkPolicy default-deny, PDB, HPA 2–10 @70%, ResourceQuota, cert-manager TLS · `ci.yml` (ruff/black/mypy/pytest 3.10–3.12 + `mlops-validate`) [†] · `cd.yml` (buildx → Trivy → Syft SBOM → GHCR → cosign → gated deploy) · DVC 2 stages + `params.yaml` + metrics (outs repointed at the bundle dir) · Prometheus 5 alert rules + Grafana dashboard · locust + k6 load tests · pre-commit · Makefile · `.env.example`.

[†] **As of this plan.** CI has since grown to a **3.10–3.14** matrix and four jobs:
`test`, `gbdt-no-torch` (a tree bundle trains and serves with no deep-learning stack
present), `spark-contract` (the Spark engine against a real JVM, refusing to pass by
skipping) and `mlops-validate`. The line above is left as written for the same reason as
§5 — it records the surface being preserved at the time, not the surface today. Current
state: [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md) and [README.md](README.md).

⚠️ **Airflow DAG** ([orchestration/airflow/dags/ml_pipeline.py](orchestration/airflow/dags/ml_pipeline.py)): the 7-task chain, `_evaluate_gate` (`ACCURACY_GATE` against `metrics.json`) and `_promote_model` (MLflow 3 `set_registered_model_alias`) are **preserved**. Only two edits: paths point at the bundle dir, and the gate reads `TaskSpec.primary_metric` instead of hardcoding accuracy — so it works for regression and forecasting too. Replace its `print()` calls with the logger while there.

### 8.5 Deliberately removed (with justification)

| Removed | Why it is not a loss |
|---|---|
| `pipeline/hpo.py` as a standalone module | Every capability moves into `pipeline/tune.py` and gains: per-plugin spaces, non-MLP models actually tunable, and automatic write-back replacing the copy-paste `print()` |
| `core/lit_data.py` as a file | Split, not deleted; all public helpers re-exported at their original import paths |
| `models/` package name | Becomes `plugins/`; `mlp.py`/`cnn.py` bodies unchanged |
| `_DATAMODULE_REGISTRY` | Superseded by `SOURCES`; `register_datamodule`/`available_datamodules` kept as shims |
| `ml_framework_complete/` snapshot | Stale duplicate. Its only unique content — the AutoGluon baseline stage — is superseded by the always-on baseline (§3.8), which is strictly better: it runs every time, costs milliseconds, and needs no extra dependency |
| Duplicate root HTML docs | Unreferenced by the README; supersede with one regenerated doc in P9 |

### 8.6 Guardrail

**The existing test suite is the contract.** Tests may be *edited* only for (a) v2 config field names and (b) bundle artifact paths. Any test whose *assertion semantics* must change is treated as a design defect in the plan, not a test to update — with one deliberate exception: `tests/unit/test_config.py` is a full rewrite, since the schema break is the explicitly approved decision.
