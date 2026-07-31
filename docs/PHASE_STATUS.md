# Phase status

Tracks progress against [ml_framework_architecture_plan.md](../ml_framework_architecture_plan.md)
§5 (Implementation Roadmap). Update this when a phase lands.

| Phase | State | Commit |
|---|---|---|
| P0 — Foundations | **done** | `a8692a1` |
| P1 — Data + backend extraction | **done** | `95c4359` (P1a) · `2ac9d45` (P1b) |
| P2 — v2 config | **done** | `5bddf2a` |
| P3 — GBDT | **done** | `3b46f4a` |
| P4 — AutoML | **done** | `a300c38` |
| P5 — DL hardening | **done** | `3296920` |
| P6 — Time-series | **done** | this branch |
| P7 — NLP | not started | — |
| P8 — Zero-config | not started | — |
| P9 — Deployment polish | not started | — |

## Version

**2.0.0 as of P3.** 1.0.0 was claiming a compatibility the package no longer has:
P2 broke the config schema, P3 broke the install contract (torch is an extra) and
the bundle layout (no root `model.ckpt`/`scaler.pkl`/`metadata.json`). The string
is not cosmetic — it lands in every bundle's `manifest.json` as
`framework_version`, which is the field someone reads to explain why an old bundle
behaves differently, and `serving/api.py` reports it as the OpenAPI version.

v1 *bundles* still load (`test_v1_bundle_compat.py`); v1 *configs* do not, and
`mlf migrate-config` converts them.

## Test baseline

**458 passed, 1 skipped** with every declared extra installed except DVC
(46/4 before P0 → 130/4 after P0 → 249/4 after P1 → 258/1 → 284/1 after P2 → 355/1 after P3 → 395/1 after P4 → 429/1 after P5 →
**458/1**). Every phase gate is measured against this number — a phase that ends
with fewer passing tests than it started with has regressed something, regardless
of what its own new tests say.

**Read the 257 → 256 step carefully: it was not a regression** (2 parquet-guard tests then took it to 258). Installing
`torchvision` makes `MODELS.is_available("cnn")` true, so
`test_validate_combination_raises_missing_extra_for_a_sound_but_uninstalled_model`
self-skips ("torchvision installed — nothing to refuse") because the condition it
exists to test no longer holds. The companion
`test_models_for_lists_only_installed_compatible_models` simply took its `["cnn"]`
branch instead of `[]`. Both tests are availability-aware by design; do **not**
pin them to either world.

**That test is the only expected skip.** Any *other* skip means a package went
missing.

P2's own version of the same check — `test_config.py::
test_an_uninstalled_model_reports_the_pip_extra_at_load`, which exercises the
refusal one layer up now that the config validator resolves plugins — is
deliberately **not** availability-gated. Its `unavailable_model` fixture registers
a spec whose requirement can never be satisfied and restores the original
afterwards, so the assertion runs everywhere rather than only on a bare install.
Gating it would have left the P2 addition untested in exactly the environment most
people develop in. The pre-existing `test_plugins.py` check keeps its
availability-aware form; do **not** pin that one to either world.

### The bare-install guardrail still holds

The plan's §5 guardrail — the suite must pass on an install *without* the extras —
can no longer be exercised locally now that the extras are present. It is still
enforced in CI, and the design guardrail itself does not depend on torchvision
being absent: `test_plugins.py::test_get_refuses_a_plugin_whose_extra_is_missing`
and `test_bundle.py::test_bundle_requirements_are_checked_before_any_import_is_attempted`
both use synthetic package names that are never installed. Do not convert an
`importorskip` into a hard import.

**Test edits are confined to what §8.6 permits.** P0 and P1 edited nothing. P2
edited existing tests in exactly three mechanical ways — v2 config field names
(`cfg.output_dir` → `cfg.runtime.output_dir`, `data.target_col` → `data.target`,
…), the `ml_framework.models` → `ml_framework.plugins` import, and the authorized
full rewrite of `tests/unit/test_config.py`. **No assertion semantics changed**
except one test that existed to pin a transitional behaviour P2 was scheduled to
close — see "What P2 landed" below.

P3 edited existing tests only for bundle **artifact paths** (`model.ckpt` →
`model/model.ckpt`, `scaler.pkl` → `preprocessor/scaler.pkl`) and the
`inf.model.input_dim` → `inf.n_features` accessor rename — both §8.6 category (b).
Two assertions inverted, and both existed to pin transitional behaviour with P3
named as the owner: `test_v1_artifacts_remain_at_the_bundle_root` (now asserts
they are gone) and `test_builtin_model_specs_are_registered_alongside_the_v1_registry`
(the v1 class registry is Lightning-only by construction once a plugin registers a
build *function* rather than a class — see `available_models` below).

Verification commands (all clean):

```
pytest                        # 458 passed, 1 skipped
ruff check src tests
black --check src tests
isort --check-only src tests
mypy src
```

### Environment notes

- **torch is an extra now, not a base dependency** (P3). `pip install -e .` gets
  you the data layer, the config, the bundle loader and the GBDT path;
  `[lightning]` adds the deep-learning runtime. `[dev]` installs everything, so
  the local suite is unaffected. The one thing to remember: a lean checkout runs
  `pytest` with the neural tests failing at import unless `[lightning]` is present
  — which is why `dev` lists it explicitly rather than relying on the base.
- **torch and torchvision must be installed together, from one index.** Every
  torchvision release pins one torch patch *exactly* (0.25.0 requires
  torch==2.10.0). The extras therefore keep an unbounded floor — an upper bound
  there would pin the user's torch and contradict the `lightning` extra. The constraint
  lives in CI instead, which installs both from the CPU index in a single
  resolution. Locally, install the pair: a bare `pip install -e '.[image]'`
  resolves the newest torchvision and replaces torch to match it, silently and
  expensively. Here that is torch 2.10.0+cpu / torchvision 0.25.0+cpu.
- **DVC is deliberately not installed.** It is the only declared dependency that
  forces major upgrades of shared libraries (`urllib3` 1.26→2.7,
  `cryptography` 46→49) in a user site-packages shared with unrelated projects. No
  Python code imports it — only `Makefile`, `ci.yml`, `dvc.yaml` and `docs/MLOPS.md`
  — so `dvc repro` does not run locally. CI installs its own.
- **pyspark is installed but cannot run**: no JVM on PATH. `spark_preprocess.py`
  imports and type-checks; a real `SparkSession` needs Java.
- **Parquet now declares its own engine.** It used to arrive only because mlflow
  requires pyarrow unconditionally — a coupling that breaks precisely where it
  matters, in a **P3** serving image built without the mlops extra. There is now a
  `parquet` extra, and `read_table` gates that branch with the framework's own
  `check_requirements`, so a lean install gets
  `pip install 'ml-framework[parquet]'` instead of a pandas "unable to find a
  usable engine" ImportError. CSV reading never consults it.

## What P0 landed

Framework-agnostic contracts added *alongside* the v1 Lightning implementation,
with zero behavior change. Rationale for each decision is in the module
docstrings rather than here — read the source, not this file.

- `core/types.py` — Task/DataKind/Payload vocabulary, `Requirement`, `Capabilities`.
  Dependency-free, and there is a test that keeps it that way.
- `core/protocols.py` — `Estimator`, `TrainingBackend`, `Preprocessor`, `Splitter`,
  `FitResult`, `Predictions`, `RunContext`, `BuildContext`, `ParamSpec`, `TrialHooks`.
- `core/plugins.py` — `PluginRegistry`, `ModelSpec`/`BackendSpec`/`SourceSpec`,
  `MissingExtraError`, entry-point discovery.
- `core/task.py` — `TaskSpec` table. Rows for binary/multiclass/regression only.
- `core/bundle.py` — bundle v2 `Manifest`, read/write, v1 bundle detection.
- `core/metrics.py` — array-based per-task metrics (numpy/sklearn only).
- `tracking/run_logger.py` — `RunLogger` protocol + null/csv/mlflow/wandb impls.

None of the seven imports torch or Lightning.

## Deliberate loose ends P1/P2 must close — **all closed**

These looked like omissions and were not. Kept here as the record of who closed what.

1. ~~**`BACKENDS` is registered but empty.**~~ **Closed by P1**, which registered the
   `lightning` backend once that module landed.
2. ~~**`ModelSpec.build` and `SourceSpec.build` are typed `Callable[..., Any]`.**~~
   **Closed by P1** (`SourceSpec.build` returns a `DataBundle`) and **P2**
   (`ModelSpec.build` is `Callable[[BuildContext], Any]`).
3. ~~**`ModelSpec.params_model` is `None` on both builtin specs.**~~ **Closed by P2**:
   both carry one, `ModelConfig._check_dims` moved into `MLPParams`, and the config
   validator runs them.
4. **Search-space keys are dotted paths against the v2 schema**
   (`model.params.dropout`). The schema now exists and
   `test_search_space_paths_are_applicable_as_overrides` proves a trial applies
   cleanly, but nothing *drives* them until **P4**.
5. ~~**`models/__init__.py` still swallows a failed `cnn` import.**~~ **Closed by P2**:
   the package is `plugins/`, both builtins import unconditionally, and
   `test_no_builtin_plugin_imports_an_optional_dependency_at_module_scope` holds the
   rule that makes that safe.
6. ~~**`config/schema.py` keeps its own narrow `Task`/`DataKind` literals.**~~
   **Closed by P2.** The schema imports the wide `core/types.py` vocabulary and
   refuses a task with no `TaskSpec` row, so what validates is unchanged.

## What P1 landed

Split as the plan suggests, each half independently revertible.

**P1a — the data layer** (`95c4359`). `DataBundle` (arrays + schema, no torch) with
`data/sources/`, `data/preprocess/`, `data/splitters.py` and one `BundleDataModule`
replacing the two v1 datamodules and their verbatim-duplicated dataloader methods.
The existing `pl.Trainer` path kept running unchanged.

**P1b — the fit loop.** `backends/lightning.py` owns `pl.Trainer`, its callbacks,
checkpoint recovery and the Optuna pruning import. `pipeline/train.py` is
orchestration through protocols. `core/evaluate.py` consumes `Predictions` arrays
and imports no torch. Bundle v2 is written. `utils/logging.py` tracks handlers per
output directory.

### Gates met

- `grep pytorch_lightning src/ml_framework/pipeline/train.py` → **0 hits**
  (`import torch` → 0 as well). The file must not spell the module name even in
  prose, which its docstring notes.
- Integration tests pass **unedited** — the plan allowed artifact-path edits and
  none were needed.
- End-to-end run at seed 42 against `b195f77` produces byte-identical
  `predictions.csv`, `report.txt`, `confusion_matrix.txt`, `metrics.json` and
  fitted scaler. `RandomSplitter` additionally has a test running the inlined v1
  split body as an oracle across both branches, three tasks and two seeds.

### Deliberate loose ends P2/P3 must close

1. **`_write_v1_artifacts` in `pipeline/train.py`** writes `model.ckpt`,
   `scaler.pkl` and `metadata.json` at the bundle root beside the v2 layout.
   `Inferencer`, `serving/api.py` and `mlflow_utils.log_and_register` still read
   them. **Removal owner: P3**, which rewrites the loader to be manifest-driven and
   torch-free — the same edit. Doing half of it here would be churn P3 undoes.
2. ~~**`LightningBackend.load` reads `config.json`.**~~ **Closed by P2.** The
   manifest carries `model.params` and the signature, which is everything the
   architecture needs, so `config.json` is purely the audit record.
3. ~~**`ModelSpec.build` is still called with the legacy kwargs.**~~ **Closed by
   P2**: one `BuildContext`.
4. **`search_space()` keys are dotted paths against the v2 schema**
   (`fit.params.lr`). The schema exists now; the driver that consumes them is
   **P4**.
5. **`pipeline/hpo.py` and `lr_finder.py` still build their own `pl.Trainer`.**
   `hpo.py` is deleted in **P4**; `lr_finder.py` gains a capability gate in P3/P5.
   P2 moved both onto v2 config paths and no further.
6. **`mlflow_utils.log_and_register` still logs a hardcoded 4-filename list.**
   `train()` now also logs the whole bundle dir through the `RunLogger`, so the
   registered bundle is complete; collapsing the two belongs with **P3**'s serving
   work.
7. **`TemporalSplitter`/`GroupSplitter` had no config path to reach them.** **P2
   built the path** (`data.split.strategy`, resolved in the tabular source, with
   `time_col`/`group_col` excluded from the feature matrix). The *guard* — refusing
   an explicitly shuffled split on time-series data — is **P6**, which is what gives
   it something to guard. `RollingOriginSplitter` waits for the time-series CV that
   consumes it.

### MLflow run ownership, verified end to end

P1b moved run ownership: the orchestrator creates the MLflow run and the backend
attaches to it by `run_id`. Getting that backwards produces two runs per training,
which no assertion in the suite would have caught.

Verified against real MLflow 3.14 on a SQLite store: **one** run per training,
holding both the Lightning per-epoch series (`val/loss`, `val/acc`, `test/acc`) and
the orchestrator's final `test_acc`, 43 flattened config params, and the whole v2
bundle under `bundle/` — manifest.json, model/, preprocessor/ and the transitional
root files. v1 logged 4 files and no params, so this is strictly more than before.

`test_mlflow_logger_attaches_to_the_orchestrators_run` keeps the wiring covered
with a stubbed logger for installs without the `[mlops]` extra; the end-to-end
proof is `tests/integration/test_mlflow.py`, which now runs here.

### Note on tracking

`train()` now holds a backend-neutral `RunLogger` and the Lightning logger lives
inside the backend. For MLflow the two are bound to the **same run** via `run_id`:
the orchestrator creates the run, the backend attaches to it. v1 had the Lightning
logger own the run, which is why `log_and_register` reached into it for a `run_id`
that a GBDT run would never have.

## What P2 landed

The clean-break v2 config schema, and the plugin surface it exists to serve.
Rationale for each decision is in the module docstrings rather than here.

- **`config/schema.py` rewritten.** Blocks organized by *ownership*: fixed
  (`task`, `runtime`, `data`, `data.split`, `fit`, `fit.budget`, `tune`,
  `logging`) versus plugin-owned free-form `params` dicts. Task/DataKind come from
  `core/types.py` now, with a `TaskSpec` existence check so what validates is
  unchanged.
- **`model.params` is validated by the plugin's own frozen `extra="forbid"`
  schema** at config-load time, and the defaulted values are written back with
  `model_copy(update=...)` (no re-validation, so no recursion). `config.json` and
  `manifest.model.params` therefore record the *effective* params.
- **`config/migrate.py` + `mlf migrate-config`.** Every v1 key has an explicit
  destination; an unmapped key is an error naming it, never a silent drop.
- **`core/lit_model.py` takes `(input_dim, output_dim, task, params, optim,
  class_weights)`** and no longer imports `config`. `OptimSettings` holds the
  optimizer defaults once; `LightningFitParams` builds its schema from them.
- **`models/` → `plugins/`** with `MLPParams`/`CNNParams`, module-level
  `build(ctx)` functions, and unconditional builtin imports.
- **`data.split.strategy`** resolves `auto` → random/temporal/group and is wired
  through the tabular source.

### Gates met

- `mlf migrate-config` round-trips all three v1 configs from `fdd33f4`; the two
  tabular ones validate and match the hand-written v2 files. The image one is
  reported as invalid — correctly: v1 carried `model.dropout` (the CNN never read
  it) and `data.imbalance_strategy` (the image path always uses a
  `WeightedRandomSampler`). Both are dead settings the per-plugin/per-source
  schemas now name. That is the schema working, not the migrator failing.
- **End-to-end run at seed 42 against `fdd33f4` produces byte-identical
  `predictions.csv`, `report.txt`, `confusion_matrix.txt`, `metrics.json`,
  `reference_stats.json` and fitted scaler.** The config surface changed; no
  number did.
- `grep pytorch_lightning src/ml_framework/pipeline/train.py` → still **0**.
- `grep -r "ml_framework.models"` across src, tests, configs and docs → **0**.

### The one assertion that changed, and why

`test_backend_load_refuses_a_bundle_without_its_config` asserted that
`LightningBackend.load` raises when `config.json` is missing. That test existed to
pin **P1 loose end 2** — a behaviour P1 documented as transitional and named P2 as
the owner of. It is replaced by
`test_backend_load_needs_only_the_manifest_not_the_training_config`, which deletes
`config.json` and asserts the load still reproduces the same predictions. Under
§8.6 this is the intended kind of change: the assertion tracked a loose end, and
the loose end is closed.

### Where each `params` block is validated

Deliberately asymmetric, and the reason is import cost:

| block | validated | by |
|---|---|---|
| `model.params` | config-load time | `ModelSpec.params_model`, from the registry |
| `fit.params` | `LightningBackend.fit` | `LightningFitParams` |
| `data.params` | `build_*_bundle` | `TabularSourceParams` / `ImageSourceParams` |

Resolving a model spec costs one import of `ml_framework.plugins`, whose modules
are required to be importable with zero optional dependencies. Resolving a backend
and a source would mean importing the whole data layer and every backend to
validate a YAML file. The guarantee is the same either way — frozen,
`extra="forbid"` — one step later for two of the three.

## What P3 landed

The GBDT family, and the torch-free serving path that was the point of building
the backend split in the first place.

- **`backends/gbdt.py`** — one-shot `fit(X, y, eval_set=…)` with library-native
  early stopping. Per-library differences (serialization format, the spelling of
  early stopping, the pruning callback) live in one adapter table, which is why
  CatBoost was ~40 lines rather than a fourth backend.
- **`plugins/gbdt/`** — xgboost, lightgbm, catboost. Each declares *tree shape* in
  its own params schema; the boosting loop's knobs (`learning_rate`,
  `n_estimators`, `subsample`, `colsample_bytree`) are declared once on the
  backend, exactly as `lr`/`batch_size` are on the Lightning backend.
- **`core/inference.py` rewritten** — manifest-driven, no torch. Five steps, all
  from `manifest.json`: version check, requirement check, `backend.load`,
  `load_preprocessor`, predict.
- **`serving/schemas.py` + `serving/metrics.py`** — request/response models keyed
  by `data.kind`, and Prometheus collectors that survive a second `create_app`.
- **`core/__init__` and `data/__init__` are lazy** (PEP 562). Without this the
  package `__init__` would import torch for every serving process no matter how
  clean `inference.py` was.
- **torch left the base dependencies** for a `lightning` extra — see below.

### Capability flags that now do real work

Each of these had exactly one named consumer in the plan; P3 is where three of
them acquired one.

| flag | consumer | observable effect |
|---|---|---|
| `needs_scaling=False` | `TabularPreprocessor` | a tree bundle has no fitted `StandardScaler` |
| `supports_sample_weight` | the imbalance resolver | `imbalance_strategy: auto` yields per-row weights, not SMOTE |
| `produces_proba` / `output.kind` | `/predict_proba` | the 400 comes from the manifest, not from `task == "regression"` |

### Gates met

- **The phase gate, twice.** `tests/integration/test_torch_free_serving.py` trains
  an XGBoost bundle and serves it in a subprocess where `import torch` *raises* —
  a stricter condition than absence, because it also catches an import that would
  have succeeded by accident. And it was then run for real:

  ```
  python -m venv .venv && .venv/bin/pip install -e '.[gbdt,serve]'
  mlf train --config configs/example_gbdt.yaml     # test_acc 0.90
  # POST /predict -> {"predictions": [2.0, 2.0]}
  # torch imported: False    pytorch_lightning imported: False
  ```

  **Measured saving: 498 MB** (torch 471 + torchvision 15 + torchmetrics 8 +
  pytorch-lightning 4, CPU wheels). The plan estimated "~2 GB"; that figure comes
  from CUDA builds. 498 MB is what a CPU serving image actually drops, and the
  honest number is the one worth recording. The `gbdt-no-torch` CI job keeps it
  enforced.
- `grep pytorch_lightning src/ml_framework/pipeline/train.py` → still **0**.
- **v1 bundles still load** (`test_v1_bundle_compat.py`), including the case where
  v1's flat `ModelConfig` recorded `backbone`/`pretrained` on an MLP.
- All three libraries train → save → reload → predict identically
  (`test_each_library_trains_saves_and_reloads_identically`).

### The dependency finding, and what it forced

The plan's gate reads "a fresh venv with `pip install -e '.[gbdt,serve]'` and **no
torch**". That could not pass as written: `torch`, `pytorch-lightning` and
`torchmetrics` were **base** dependencies, so the lean install pulled 533 MB of
torch and the `serve-gbdt` image would have shipped it too — leaving the concrete
payoff the plan names unrealized.

They moved to a `lightning` extra. The consequences were followed through rather
than papered over:

- `mlp`/`cnn` declare `Requirement("torch", extra="lightning")`, so on a
  GBDT-only install they are **listed** by the registry and refuse selection with
  `pip install 'ml-framework[lightning]'`. This is the plugin design doing exactly
  what it was built for, and the `gbdt-no-torch` CI job asserts it.
- `pip install ml-framework` alone no longer trains an MLP. The README leads with
  the extras rather than burying them.
- The `serve` and `train` Docker targets gained `lightning`; `serve-gbdt` builds
  from a torch-free `src` stage.

### Deliberate loose ends P4+ must close

1. **`pipeline/hpo.py` still builds its own `pl.Trainer`** and tunes an MLP shape.
   **P4** deletes it for `pipeline/tune.py`, which consumes the declarative spaces
   both backends now expose.
2. **`GbdtBackend.trial_hooks` reports rather than prunes.** The per-library
   Optuna callbacks are written (`_xgboost_adapter`/`_lightgbm_adapter`) but the
   driver that would install them is **P4**; CatBoost has no such callback at all,
   and returning an empty hook set is the honest answer.
3. **`fit.budget.max_seconds` is carried but not enforced.** The per-backend
   wall-clock caps that make tuning-on-by-default finish in minutes belong to
   **P4**.
4. **`native_categorical` has no consumer yet.** The three libraries declare it,
   but the tabular source still builds a float matrix; passing a pandas
   `category` dtype through needs the source to stop calling `.values.astype`,
   which is a data-layer change with its own tests.
5. **`lr_finder.py` has no capability gate.** `mlf lr` on a GBDT config will still
   crash inside `torch_lr_finder` rather than refusing politely.
   `supports_lr_range_test` is declared `False` and unread — **P5**.

## What P4 landed

Hyperparameter search that is declared by plugins, driven by the task table, and
**applied** rather than printed.

- **`pipeline/tune.py`** replaces `pipeline/hpo.py`, which is deleted. The old
  module had three defects and each was silent: its space was hardcoded to the
  MLP's shape (running it on a CNN tuned `hidden_dims`, a parameter that model
  does not have, and reported a meaningless "best"); its objective read
  `val/loss` off `trainer.callback_metrics`, an object no non-Lightning backend
  produces; and it `print()`ed the winner for copy-paste.
- **`config/defaults.py`** — per-backend budgets. Trees get 30 trials / 300 s;
  neural nets 10 / 900 s with a 25-epoch per-trial cap. Without that cap one slow
  trial consumes the whole wall budget and the search degenerates to one sample.
- **Write-back** — the winner lands in `bundle/config.json`, `bundle/hpo.json`
  and `manifest.hpo`; `--emit-config` writes a committable YAML.
- **`FitResult.metric()`** absorbs a real wart: Lightning reports `val/acc` (its
  logger convention groups with a slash) and array-computed metrics come back as
  `val_acc`. Normalizing at the source would rename keys the trackers already
  publish, so the *lookup* handles it — in one place.
- **`fit.budget.max_seconds` is enforced**, not merely carried (P3 loose end 3),
  via Lightning's `Timer`. It stops at an epoch boundary: a half-finished epoch
  produces no usable checkpoint, so a hard kill would be worse.

### Gates met

```
mlf train --config configs/example_gbdt.yaml
  tuning xgboost/gbdt on 'acc' (max): 30 trials, 300s budget, 7 parameters
  best acc=0.9000 after 30 trials (0 pruned) in 22s
  Test accuracy: 0.9250            # untuned was 0.9000
  total elapsed 41s                # default budget 300s
```

`bundle/config.json` holds `max_depth: 11` against the YAML's `6`, and
`hpo.json` records the range it came from (`Int(low=3, high=12)`) alongside all
30 trials — "max_depth=11" is uninterpretable without knowing what was searched,
and a winner sitting at a boundary is telling you the range was wrong.

### Decisions worth knowing

**Tuning is on for users and off for tests.** `conftest.make_config` sets
`tune.enabled: false`: a test of the bundle layout should not spend 300 s
searching. The tuning path has its own tests, which opt back in explicitly.

**A skipped search still writes `hpo.json`.** Tuning off, an empty space, optuna
absent — each is recorded with its reason. An absent file would be
indistinguishable from a bundle written before P4.

**Optuna missing degrades to a plain fit with a warning**, rather than failing.
Tuning is on by default, so a hard failure would break every install without the
`[hpo]` extra.

**Pruning attachment is per-library, not just the callback.** xgboost ≥ 2.0 takes
`callbacks` on the *constructor*; lightgbm and catboost take it on `fit`. Passing
it to `XGBClassifier.fit()` is a `TypeError` — found by the torch-free training
test, which exercises the default (tuning) path. It now lives in the adapter
table alongside the other per-library differences.

**Correction to a P3 note:** that entry claimed CatBoost has no Optuna pruning
callback. `optuna_integration.catboost.CatBoostPruningCallback` exists and is now
wired; all three tree libraries prune.

### Deliberate loose ends P5+ must close

1. **`native_categorical` still has no consumer.** All three tree plugins declare
   it, but the tabular source builds a float matrix via `.values.astype`. Passing
   a pandas `category` dtype through is a data-layer change with its own tests.
2. **`lr_finder.py` has no capability gate.** `mlf lr` on a GBDT config still
   crashes inside `torch_lr_finder`; `supports_lr_range_test` is declared `False`
   and unread — **P5**.
3. **`max_seconds` is unenforced on the GBDT path.** A one-shot `fit` cannot be
   interrupted at a round boundary without a callback per library. The *search*
   budget is enforced (Optuna's `timeout`); a single overrunning fit is not.
4. **Cross-validation is not a `train()` mode yet** — **P5** owns it, and the
   plan is explicit that it belongs to the orchestrator so GBDT and forecasting
   get it too rather than it being a Lightning feature.

## What P5 landed

The deep-learning capabilities the Lightning path was missing, plus the one
"DL feature" that deliberately is not one.

- **Mixed precision** — `runtime.precision`, resolved once in
  `backends/base.resolve_precision`. `16`/`bf16` are normalized to Lightning's
  `-mixed` spellings rather than rejected.
- **Multi-device strategy** — `runtime.strategy` (`auto`/`ddp`/`ddp_spawn`).
- **Resume** — `ModelCheckpoint(save_last=True)`, `last.ckpt` copied into the
  bundle, `train(resume=...)` / `mlf train --resume`.
- **Gradient accumulation** — `fit.params.accumulate_grad_batches`.
- **Configurable optimizer/scheduler** — `adam|adamw|sgd` ×
  `plateau|cosine|step|none`. v1's Adam + ReduceLROnPlateau are the defaults, so
  an existing config trains exactly as it did.
- **Cross-validation** — `data.split.folds`, run by `train()`.
- **`mlf lr` is capability-gated** (P3/P4 loose end), so it refuses a GBDT config
  instead of crashing inside `torch_lr_finder`.

### Gates met

**AMP matches FP32 within tolerance.** `test_amp_matches_fp32_within_tolerance`
trains the same config twice, once at `bf16-mixed`. bf16 rather than fp16 because
this machine is CPU-only and fp16 has no gradient scaler there — which the
resolution logic detects and downgrades, with its own test.

**`--resume` continues rather than restarting.** The obvious assertion does not
work: a resumed run and a fresh one *end* at the same epoch, so the final number
proves nothing. The test resumes into an **already-exhausted** budget and asserts
zero further steps ran — true only if the epoch counter, optimizer and scheduler
all came back. A restart would have trained the full 2 epochs again.

**Cross-validation works for GBDT.** `--folds 3` on an xgboost config produces
`cv_acc_mean`/`cv_acc_std` and a normal GBDT bundle, because CV drives the
splitter and the same `fit`/`predict_split` calls every backend implements.

### Decisions worth knowing

**`data.split.folds`, not `fit.cv`.** k-fold is a way of *cutting the data*, and
putting it in the split block avoids a second `strategy` field that would have to
be kept in step with the first.

**CV estimates; it does not produce the model.** k folds run first, then the usual
single fit writes the bundle. So there is one bundle-writing path regardless of
how the score was estimated, and `cv_*` metrics sit *beside* `test_acc` rather
than replacing it — they answer different questions.

**Each fold re-fits its own preprocessor.** `build_cv_bundles` re-runs the whole
source pipeline per fold. Fitting a scaler once and sharing it would leak every
fold's test set into every other fold's preprocessing, producing a CV estimate
that looks better than the model is.

**`cv.json` records per-fold scores, not just the mean.** A mean of 0.85 across
0.84/0.86 and across 0.70/1.00 are the same number and completely different
results.

**`last.ckpt` is not the manifest's artifact.** The manifest still points at the
*best* checkpoint: a loader wants the best weights, and only a resuming trainer
wants the last optimizer state. Both live in `model/`.

### Deliberate loose ends P6+ must close

1. **Cross-validation is tabular-only.** `build_cv_bundles` raises
   `NotImplementedError` for other kinds rather than silently cross-validating
   something else; threading fold indices through `ImageFolder` is its own change.
2. **`native_categorical` still has no consumer.** All three tree plugins declare
   it, but the tabular source builds a float matrix via `.values.astype`.
3. **`max_seconds` is unenforced on the GBDT path.** A one-shot `fit` cannot be
   interrupted at a round boundary without a per-library callback. The *search*
   budget is enforced; a single overrunning fit is not.
4. **`ddp` is wired but untested here** — this machine has one CPU device, so the
   strategy field is passed through and never exercised against real multi-GPU.

## What P6 landed

Forecasting: the task row, the temporal splitters, four models across two
backends, and the guard that makes the whole thing trustworthy.

- **`forecasting` TaskSpec** with MASE as its objective, plus `mase`/`smape` in
  `core/metrics.py`.
- **`RollingOriginSplitter`** — k origins, each training only on its past.
  Expanding (a production retrain) or sliding (old data is misleading).
- **The leakage guard** — `strategy: random` on `kind: timeseries` raises.
- **`data/sources/timeseries.py`** — one source, two payloads.
- **`backends/forecast.py`** — the third fit-loop shape.
- **`plugins/ts/`** — `ts.naive`, `ts.arima`, `ts.prophet`, `ts.lstm`.
- **Forecast serving** — `POST /predict {"horizon": 7}` through the schema that
  had been declared since P3.

### Gates met

**`strategy: random` on `kind: timeseries` raises.** Refusing rather than warning
is the decision: nothing crashes on a shuffled series, the score simply comes back
*better*. The escape hatch is `allow_temporal_leakage: true`, which costs typing
the word — roughly the deliberation the decision deserves.

**A temporal split scores honestly where a shuffled one flatters.** Demonstrated
rather than asserted: a one-nearest-neighbour forecast scores *better* under a
shuffled split, because training points sit interleaved among the test points and
the "nearest neighbour" is often the adjacent timestamp. That is the leak, and it
is why the guard exists.

### Decisions worth knowing

**One source, two payloads.** `ts.prophet` declares `accepts={"series"}` and gets
the ordered values; `ts.lstm` declares `accepts={"arrays"}` and gets sliding
windows. The source reads the declaration — no config flag. This required
generalizing `DEFAULT_PAYLOAD` (kind → one payload) into `KIND_PAYLOADS` (kind →
the set it can be materialized as), because a data *kind* is a statement about the
data, not about the shape a model wants it in.

**`ts.lstm` rides the `lightning` backend, not `forecast`.** An LSTM forecaster
trains in mini-batches over epochs exactly as an MLP does. Putting it on the
forecast backend would have meant reimplementing the Lightning loop for one model
— the outcome the per-shape split exists to avoid.

**`ts.naive` declares no requirements.** MASE is defined against it, so it has to
run wherever forecasting does, including an install with neither prophet nor
statsmodels. It is also the baseline the zero-config work will use.

**Time-series CV is rolling-origin, never k-fold.** Shuffled folds here would be
the same leakage the validator refuses, wearing a different hat.

**Fit-per-series with one series.** `ForecastEstimator` holds `{series_id: model}`
and the backend iterates. With one series the mapping has one entry — but
multi-series then becomes a *source* change rather than a backend rewrite.

### A correction I made to my own work

The first draft of `mase()` documented "1.0 is the line: below it the model beats
doing nothing". The smoke test then scored **every** model above 1, including the
seasonal-naive baseline itself — which that reading says is impossible.

The reading is wrong for multi-step horizons. MASE scales by the average *one-step*
change; a model forecasting 45 steps ahead is being asked a harder question than
the denominator measures, so values above 1 are normal. The docstring now says
what the function computes, and `report.txt` deliberately prints **no verdict** —
it points at the honest comparison instead (train `ts.naive` on the same split).
Shipping the confident-sounding version would have been worse than shipping
nothing.

### Deliberate loose ends P7+ must close

1. **Multi-series is not wired.** The backend iterates, but the source emits one
   series; a `series_col` is the missing piece.
2. **`exog` is carried, not consumed.** The source validates and passes exogenous
   columns; none of the four models currently uses them.
3. **Prophet synthesizes a daily date range** when the index is positional. It
   needs *a* time axis; that affects the labels of the seasonality it finds, not
   whether it finds one.
4. **`native_categorical` still has no consumer** (from P4).
5. **CV remains unimplemented for image data** (from P5).
