# Phase status

Tracks progress against [ml_framework_architecture_plan.md](../ml_framework_architecture_plan.md)
§5 (Implementation Roadmap). Update this when a phase lands.

**P0–P9 come from that roadmap; P10, P11 and P12 do not** — they were scoped after it,
once the earlier phases had landed and shown what was missing. The plan is therefore a
record of what was *intended* at the outset, not a live index of the work. This file is
the live index. Neither was backfilled to match the other, because a plan edited to
predict what actually happened stops being evidence of anything.

| Phase | State | Commit |
|---|---|---|
| P0 — Foundations | **done** | `a8692a1` |
| P1 — Data + backend extraction | **done** | `95c4359` (P1a) · `2ac9d45` (P1b) |
| P2 — v2 config | **done** | `5bddf2a` |
| P3 — GBDT | **done** | `3b46f4a` |
| P4 — AutoML | **done** | `a300c38` |
| P5 — DL hardening | **done** | `3296920` |
| P6 — Time-series | **done** | `01bd2aa` |
| P7 — NLP | **done** | `e06c299` |
| P8 — Zero-config | **done** | `62672eb` |
| P9 — Deployment polish | **done** | `1dc3a0c` |
| P10 — Exporter migration | **partial** — ONNX done, TorchScript open | see below |
| P11 — Model selection | **done** | `c42dc04` |
| P12 — Pluggable data backends | **done** — all five phases | `c42dc04` |
| P13 — Staged decode pipeline | **in progress** — P13a done; b–e open | see below |

P11 and P12 share a commit. They were developed in sequence but could not be split
into two: `cli.py`, `config/schema.py` and `core/registry.py` each carry changes for
both, so a P11-only commit would not have been independently green.

**The P11 row previously read `08529e8`, which was wrong.** That commit touched only
`ci.yml` and this file — the row was written when the phase was finished rather than
when it was committed, and the code sat uncommitted for two further commits. Worth
recording because the phase table is the thing someone reads to find out when a
behaviour changed, and a row pointing at a commit that does not contain the feature
is worse than no row at all.

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

**922 passed, 18 skipped** with every declared extra installed except DVC
(46/4 before P0 → 130/4 after P0 → 249/4 after P1 → 258/1 → 284/1 after P2 → 355/1 after P3 → 395/1 after P4 → 429/1 after P5 → 458/1 after P6 → 471/1 after the image-augmentation fix → 511/1 after P7 → 549/1 after the two NLP tasks → 555/1 after `mlf models` → 563/1 after the HF cache pin → 621/1 after P8 →
641/1 after P9 → 643/1 after the P10 ONNX half → 811/1 after P11 → **922/18** after P12). Every phase gate is measured against this number — a phase that ends
with fewer passing tests than it started with has regressed something, regardless
of what its own new tests say.

**The skip count jumped 1 → 18 at P12, and that is expected rather than a
regression.** Every one of the new skips is a Spark test gated on a JVM this
machine does not have (`shutil.which("java")`), not on an uninstalled package —
pyspark itself imports fine here, which is exactly why `importorskip` is not the
gate. They run in the `spark-contract` CI job, which refuses to pass by skipping.
The Polars backend added in P12's last phase needs no JVM, so its share of the
same suites runs here rather than skipping.

**Read the 257 → 256 step carefully: it was not a regression** (2 parquet-guard tests then took it to 258). Installing
`torchvision` makes `MODELS.is_available("cnn")` true, so
`test_validate_combination_raises_missing_extra_for_a_sound_but_uninstalled_model`
self-skips ("torchvision installed — nothing to refuse") because the condition it
exists to test no longer holds. The companion
`test_models_for_lists_only_installed_compatible_models` simply took its `["cnn"]`
branch instead of `[]`. Both tests are availability-aware by design; do **not**
pin them to either world.

**Outside the Spark suites, that test is the only expected skip.** The 17 others
are the JVM-gated Spark tests described above. Any skip that is neither of those
means a package went missing.

### The suite does not need the network (after the first run)

The NLP tests fine-tune ~90 KB random checkpoints pulled from the HuggingFace hub.
`from_pretrained` revalidates against the hub on **every** call, even for a fully
cached model — a HEAD request per file — and when DNS fails rather than answering
cleanly, `huggingface_hub` retries five times with exponential backoff and then
raises. Measured against an unreachable endpoint: **~37 s of backoff and a
`ConnectionError` for one cached tokenizer**.

That is not hypothetical. A blip mid-run once turned six passing NLP tests into
errors and stretched the suite from 5m37s to 11m42s.

`tests/conftest.py` now sets `HF_HUB_OFFLINE=1` at **import** time, once it has
confirmed on disk that every test checkpoint is cached. Import time because the
variable is read into a module constant when `huggingface_hub` is imported —
setting it one line later does nothing, silently. A cold or partial cache is left
online so the first run can download.

Verified by pointing `HF_ENDPOINT` at an unreachable address: 49 NLP tests pass
without touching it.

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
pytest                        # 641 passed, 1 skipped, 91% coverage
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
5. ~~CV remains unimplemented for image data~~ — closed below, and text followed
   in P7, so cross-validation now covers all four kinds.

## Follow-up: cross-validation for image data

Not a phase — a loose end from P5, closed on request. Cross-validation now covers
**tabular, timeseries and image**; `text` followed in P7, once there was a text
source to partition.

The blocker was never mechanical. The image source reads two separate directories
— `ImageFolder(data.path)` for train and `ImageFolder(params.test_dir)` for test —
and k-fold needs one pool to repartition. Two readings were available and neither
is obviously right:

* fold over the train directory only, leaving `test_dir` untouched, or
* pool train + test and repartition.

**Chosen: fold over the train directory.** `test_dir` is an explicit statement
about which images are held back, and silently folding it into the pool would
override a decision made on disk. The cost is that "test" means different things
in the two places, so the bundle records `meta["cv_test_source"] = "train_dir"`
and the README says it outright: the CV estimate comes from folds of `data.path`,
`test_acc` comes from `test_dir`.

Two per-fold details that would have been invisible if wrong:

* **Val and test images get the *eval* transforms.** Augmentation exists to make
  training harder; measuring on augmented images measures the augmentation. That
  needs a second `ImageFolder` over the same directory, because a transform
  belongs to the dataset rather than to the `Subset`.
* **Sample weights are recomputed per fold.** Reusing one vector would weight each
  fold by another fold's class balance — the same category of mistake as sharing a
  fitted scaler.

`train_labels()` reads labels from the directory tree rather than decoding pixels,
since stratification needs them up front.

**One test inverted**, and it existed to pin exactly this limitation:
`test_cross_validation_on_images_says_it_is_not_implemented` became
`test_cross_validation_refuses_a_kind_it_cannot_partition`, now using `text` —
which is genuinely unpartitionable rather than merely unimplemented.

467 passed, 1 skipped (was 458/1); 471/1 after the augmentation fix below.

### The related wart, now fixed

The **non-CV** image path carved its validation set with `random_split` over the
*augmented* dataset, so every holdout validation image arrived randomly cropped
and flipped. It was recorded here as deliberately unfixed; on request it is fixed.

A transform belongs to the dataset, not to a `Subset` of it, so the fix is an
un-augmented view of the same corpus — `eval_view()`, a shallow copy that rebinds
one attribute rather than walking the directory tree a second time. The CV path
now uses the same helper instead of building a second `ImageFolder`.

**The partition is unchanged.** The same generator and the same `random_split`
call decide which images land in validation at a given seed; only the pipeline
each side goes through differs. There is a test that reproduces v1's split and
compares indices.

**Expect validation metrics on image runs to change, and to improve slightly.**
They were previously measured on deliberately degraded inputs. This is not
cosmetic: early stopping and `ModelCheckpoint` both read `val/loss`, so the fix
changes which epoch gets selected as well as what the number says. Test metrics
are unaffected — `test_dir` always used the eval transforms.

471 passed, 1 skipped.

## P7 — NLP — **done**

Text classification as `multiclass`/`binary` with `data.kind: text`. No
`text_classification` task: task decides loss, metrics and head; kind decides
ingestion, and keeping them orthogonal is what stops the task `Literal` from
growing as a product.

- **`data/preprocess/text.py`** — the tokenizer, as a preprocessor.
- **`data/sources/text.py`** — CSV/Parquet/JSONL → lazy `TextDataset` of strings.
- **`plugins/nlp/hf_text.py`** — fine-tunes a HuggingFace encoder, on the
  **Lightning backend**. No fourth backend: a fine-tune is an epoch loop with
  validation callbacks, which is what `lightning` already is.
- **HF-directory artifact format** — `model/hf_model/` via `save_pretrained`.
- **`ModelSpec.fit_defaults`** — fine-tuning hyperparameters, applied not printed.
- **Text CV** — came free; the source accepts injected indices like the other three.

### Gates met

**The tokenizer round-trips through the bundle.** Proved the only way that
distinguishes it from a silent re-download: the test *adds a token* to the
tokenizer before saving and asserts the loaded one still knows it, and that a
fresh `from_pretrained(model_name)` does not. Asserting merely that a tokenizer
loads would pass just as happily if the loader went back to the hub.

**Text `/predict` accepts raw strings.** `{"inputs": ["great movie"]}` → labels.
Asking clients to send token ids would make each of them responsible for using the
right vocabulary — the exact skew the bundled tokenizer prevents — and would make
the endpoint unusable from curl.

### Decisions worth knowing

**The tokenizer is fitted state, not configuration.** It maps strings to ids
through a vocabulary, and a model fed ids from a different vocabulary produces
confident nonsense — no shape error, no exception, just a model that looks like it
trained badly. So it lives in `preprocessor/tokenizer/` and is loaded from there.
A missing directory **raises** rather than falling back to the hub: failing loudly
at load time beats scoring wrong at request time.

**A checkpoint directory, not a Lightning checkpoint.** A `.ckpt` round-trips the
weights, but rebuilding the architecture to put them in calls
`from_pretrained(model_name)` — which needs the hub, or a warm cache, *at load
time*. That failure appears in a serving container, not in CI. The backend
discovers `save_artifact`/`load_artifact` by name and uses the checkpoint path for
every model that does not define them, so this is a model capability rather than a
branch in the backend. `last.ckpt` still rides along: resuming is a property of
the loop, not of the file the loop produced.

**Splits hold strings; tokenization happens per batch.** Padding then follows the
batch rather than the corpus, and attention cost grows with the padded length — so
on text with a long tail, which is all text, up-front tokenization is most of the
compute. It also keeps a fold cheap: k folds would otherwise re-tokenize the whole
corpus k times to produce k partitions of it.

**`ModelSpec.fit_defaults`, and why a model gets to have an opinion about the
loop.** The framework default is `lr: 1e-3` with Adam. That is correct for a
network trained from scratch and destroys a pretrained encoder in the first few
steps — while the run looks entirely healthy and scores near chance. `nlp.hf_text`
declares `lr: 2e-5`, AdamW, `weight_decay: 0.01`, cosine; the config validator
applies them to keys the user did not write, so `config.json` records what actually
ran. Scoped to `fit.params` deliberately: `batch_size` and `budget` are typed
fields whose defaults are indistinguishable from an explicit value, so "the user
did not set it" is not answerable for them.

**The search-space merge order flipped, to backend-then-model.** Previously the
backend won, which made a model unable to narrow `fit.params.lr`. The Lightning
backend proposes 1e-4..1e-2 — a range in which most trials would wreck a
pretrained encoder, so a search would spend its budget confirming that. Now the
more specific declaration wins. No existing plugin declared a `fit.params.*` key,
so the change is behaviour-preserving for everything shipped before P7, and a test
pins it.

**Two guesses the source refuses to make.** Which column holds the text (with
several candidates and no conventional name it raises and lists them) and which
tokenizer (always the model's checkpoint, never a separate `data.params` knob).
Both wrong guesses produce a model that scores badly rather than one that fails.

**String labels are encoded in sorted order**, not order of appearance — otherwise
shuffling the input file renumbers the classes and two runs over the same data
produce models whose class 0 means different things.

### Deliberate loose ends P8+ must close

1. ~~`token_classification` and `seq2seq` have no `TaskSpec` row~~ — closed below.
2. **No multi-label text.** `multilabel` has no `TaskSpec`, and by the rule stated
   below that is correct until it has a source, a model and a loss.
3. **Drift for text is a 501.** PSI over token ids is a number without a meaning;
   embedding-distance drift is the real answer and is not built.
4. **`native_categorical` still has no consumer** (from P4).
5. **Multi-series forecasting and `exog` consumption** (from P6).

## Follow-up: token classification and seq2seq

Not a phase — the first loose end from P7, closed on request. The NLP surface is
no longer classification-only.

### The rule this settled

**A `TaskSpec` row means the framework can run the task.** Adding two rows is a
ten-line change and would have made both configs validate — and then failed
somewhere inside the fit loop, which is strictly worse than the refusal it
replaced. So the rows arrived with what makes them true:

| piece | `token_classification` | `seq2seq` |
|---|---|---|
| source | `read_token_corpus` — words + tags | `read_seq2seq_corpus` — source + target |
| preprocessor | `TokenTextPreprocessor` (alignment) | `Seq2SeqPreprocessor` (two lengths) |
| model | `nlp.hf_token` | `nlp.hf_seq2seq` |
| loss | CE over positions, `ignore_index` | CE over positions, teacher-forced |
| metrics | macro-F1, token-level | ROUGE-L / token-F1 / exact-match |
| serving | one tag per **word** | generated strings |

`multilabel` is still unregistered, for exactly this reason, and a test now pins
that so the rule is not quietly abandoned the next time somebody wants a config to
validate.

### The two things that are actually hard

**Word-to-sub-word alignment.** A corpus is tagged per *word*; the model consumes
*sub-words*. Only the first piece of each word carries its tag — every
continuation gets `IGNORE_INDEX`. The tempting alternative, repeating the tag
across all pieces, does not raise: it makes one word's single decision count once
per piece, silently re-weighting the corpus toward whichever words the tokenizer
fragments most, which is exactly the rare proper nouns NER is about. It also makes
the number incomparable with anything published. A test asserts the scored count
equals the *word* count while the piece count is strictly larger.

**Teacher forcing is not generation.** A seq2seq model trains with the reference
prefix fed to the decoder at every step and is evaluated by generating without
one. `val/loss` stays the early-stopping monitor because generating each
validation epoch would multiply epoch time by the decode length — but it is *not*
what the reported metrics measure, and the two can move in opposite directions.
`report.txt` says so rather than leaving it to be discovered.

### Decisions worth knowing

**`OutputKind` grew two values and `Postprocess` grew one.** `token_labels` and
`text` are not decoration on `labels`: a tagger emits one label per position in a
variable-length sequence and a generator emits a string, so `evaluate()`,
`predictions.csv` and the serving response each have to know which they hold.
`Postprocess.generate` is honest about being the odd one — the other three are
functions of the logits, while generation is an autoregressive loop that needs the
*inputs*.

**Ragged batches flatten rather than pad together.** Batches are padded to their
own longest sequence, so `(B, T)` arrays from different batches cannot be
concatenated. Dropping the ignored positions flattens both sides to 1-D over real
tokens — which is also the only granularity at which the metrics mean anything.
`predictions.csv` then has one row per token, and the sentence index is dropped
rather than mislabelling every row.

**One tokenizer in the bundle, not two.** Generation needs to turn ids back into
text, which needs a tokenizer the model does not own. Rather than have the model
carry a second copy — which could disagree with the preprocessor's about the
vocabulary — `Inferencer` binds the loaded preprocessor onto the estimator, and
the estimator borrows it. A test asserts `estimator.preprocessor is
inf.preprocessor` and that no tokenizer exists under `model/`.

**The tagging report states that it is token-level.** The NER convention (seqeval)
is *entity*-level F1: a predicted entity must match the reference in both span and
type. That is strictly harder, token-level figures run several points above it,
and printing this under the bare name "F1" would invite a comparison that flatters
it. Entity-level scoring is not implemented.

**Macro-F1 leads for tagging, not accuracy.** `O` dominates a tagging corpus; a
model that answers "not an entity" everywhere scores ~90% accuracy and is worth
nothing.

**Three generated-text metrics, all shallow, and the report says so.** ROUGE-L,
token-F1 and exact-match are n-gram overlap against one reference, so a correct
paraphrase scores near zero. Reported anyway because a number honest about being
shallow beats no number, and because the alternatives (BERTScore, an LLM judge)
are a model dependency this framework should not acquire by default. ROUGE-L and
token-F1 are both reported because they disagree in a specific way — token-F1
ignores word order — and a test pins that disagreement.

**Stratification is now a property of the task, not "is it regression".** Four
call sites in `splitters.py` spelled "don't stratify" as `task == "regression"`,
which quietly asserted that every other task has one class label per row. True
until token tagging, then wrong. Now a `STRATIFIED_TASKS` table.

### Newly deliberate loose ends

1. **Entity-level (seqeval) scoring for NER.** Token-level is what is computed and
   the report says so.
2. **`num_beams > 1` is honoured but untuned**, and beam search is not in the
   search space — it costs linearly on every evaluation pass.
3. **A seq2seq model cannot resize its vocabulary.** Adding tokens to the
   preprocessor's tokenizer would leave the embedding table behind.
4. **Drift for text remains a 501** (from P7).

## P8 — Zero-config — **done**

`mlf train --data x.csv` with no YAML and no flags produces a bundle and reports
against a baseline. That is the stated exit gate, and it is the first test in
`tests/integration/test_zero_config.py`.

- **`data/sniff.py`** — kind, target, task, each with the rule that produced it.
- **`config/defaults.py`** — the model rules table, keyed by `(kind, task)`.
- **`config/autoconfig.py`** — synthesis to a plain dict, and the merge.
- **`core/baseline.py`** — the trivial model, scored beside the real one.
- **`mlf init`** — the synthesized YAML, with a comment per inferred field.
- **`mlf backends`** — the other half of the plugin surface (`mlf models` landed
  just before this phase).

### The three decisions that make it trustworthy rather than merely convenient

**Synthesis produces a plain dict, never a validated config.** That single choice
is the whole mechanism: it lets synthesized values sit in an ordinary precedence
chain instead of being special-cased anywhere downstream.

    plugin defaults < synthesis < YAML file < --set < explicit CLI flags

Nothing below that line can tell which layer a value came from, so `--data` and
`--config` *compose* rather than being alternatives — point at a CSV, keep a YAML
that overrides two fields, and exactly those two are overridden. The merge is
recursive for the same reason: a shallow one would make a YAML that sets only
`model.name` discard the synthesized `data.kind` beside it.

**An uninstalled family is a refusal, not a substitution.** If `[gbdt]` is missing,
a tabular run raises `MissingExtraError` with the pip command rather than quietly
training an MLP. This matters more here than anywhere else in the framework: the
user did not choose the model, so a score from the wrong one looks exactly like
the score they asked for. The single sanctioned exception — seasonal-naive as the
forecasting fallback — is marked `downgrade=True` in the table and logged at
WARNING.

**Every inference carries its rule.** The CLI logs each one (weak ones at WARNING)
and `mlf init` writes them into the generated file:

    task: multiclass  # inferred: the target is non-float with 3 distinct values
    data:
      kind: tabular  # inferred: columns are numeric or short strings
      target: label  # inferred: column is named 'label'

Fallbacks are marked `GUESS` rather than `inferred`, in the file and in the header
block. A framework that decides things for you and will not say why is worse than
one that makes you type.

### Refusals, and the one guess it does make

Two columns named `label` and `target` is not a tie to be broken by column order —
it raises and names them. Two columns of prose, likewise. The one genuinely weak
rule is "no conventional name, so use the last column", which is right often
enough to be worth doing and wrong often enough to say out loud.

`data.kind` is decided on positive evidence in both directions: a datetime column
must also be **monotonic** before the data is a time series (parsing alone would
make a table of birthdays one), and a string column must average ≥ 4 whitespace
tokens before it is prose (a column of colour names is a *feature*, and treating
it as text would fine-tune a 66M-parameter encoder on the word "red").

### The baseline, and why it is on by default here only

`test_acc: 0.91` on a dataset that is 91% one class is the most common way a
pipeline looks successful while having learned nothing. A user who named the model
has their own frame of reference; a zero-config user has none. So whenever the
*framework* chose the model, the trivial predictor is scored too — majority class,
training mean, or repeat-last-season — and recorded under `baseline_*`.

Failing to beat it is a WARNING, not an error. A model that ties the baseline on a
genuinely unpredictable target is an honest result, and failing the run would be
pretending otherwise.

Two details that are easy to get backwards and produce a number that looks fine:
the statistic comes from the **training** split (taking the majority class from
test would make the baseline stronger than anything achievable honestly), and the
comparison uses the task's **declared direction** (MAE, RMSE and MASE improve by
getting smaller).

Whether the framework chose is decided against the *surviving* value, not by
whether synthesis ran: a YAML, a `--set` or an explicit `--model` all mean the user
chose, and then the baseline is not the point.

### A pre-existing bug this phase had to fix

`--set model.name=catboost` failed on **any** config, including a plain YAML with
no zero-config involved. `_resolve_plugin_params` writes every default back into
`model.params` at load time, so a config validated once carries xgboost's
`tree_method` — and re-validating it as catboost failed with a wall of "extra
inputs are not permitted".

Fixed by clearing `model.params` when `model.name` is overridden, *before* the
overrides are applied so that `--set model.name=catboost --set model.params.depth=5`
lands `depth` on an empty dict. Clearing afterwards would either keep the stale
keys or discard the value just set. Fixed here rather than deferred because
`--set` is a documented layer of the chain this phase ships, and a layer that
cannot change the model is a broken layer.

### Also fixed while wiring it up

A synthesized tabular config sets `data.params.imbalance_strategy: auto`. The
schema default is `smote`, kept so an existing v1 config trains exactly as it did —
but zero-config always picks a tree, which consumes sample weights natively and is
measurably hurt by synthetic neighbours. Left at the default, *every* zero-config
tabular run emitted a warning telling the user to set the value synthesis now sets.

Pointing `--data` at an arbitrary directory used to surface a raw pyarrow schema
error, because a directory of parquet part-files is a legitimate table. The sniffer
now owns that message.

### Deliberate loose ends P9+ must close

1. **`--kind` cannot be forced.** `--text-col` and `--time-col` force text and
   timeseries respectively, but there is no direct override for `data.kind`.
2. **Image zero-config reuses the training folder as `test_dir`**, recorded as a
   weak inference. There is no way to infer a held-out image directory.
3. **No baseline for `seq2seq`** — "always emit the most common string" is not a
   comparison anybody would make.
4. **Drift for text remains a 501** (from P7).
5. **Entity-level NER scoring, seq2seq vocabulary resizing** (from P7).
6. **Multi-series forecasting and `exog` consumption** (from P6).
7. **`native_categorical` still has no consumer** (from P4).

## P9 — Deployment polish — **done**

The exit gate is two claims and both are now checked by tests: **exported ONNX
matches native predictions**, and **CI enforces ≥80% coverage**.

- **`core/export.py`** — the format vocabulary and the refusal.
- **`backend.export()`** — on all three backends, plus a refusing default.
- **`mlf export --format onnx|torchscript|native|pickle`**
- **`deploy.py` + `mlf dockerfile`** — an image built from what the bundle declares.
- **Per-backend Prometheus labels** on both collectors.
- **Coverage floor** — measured 91%, enforced at 80%.

### The exit gate, measured

Seven rows through a graph traced at batch size one:

    max abs diff   5.96e-08
    argmax agrees  yes
    tolerance      1e-5 (stated, not tuned until green)

Seven rows rather than one is deliberate — it proves the dynamic batch axis.
Without `dynamic_axes` the artifact is frozen at whatever was traced and fails on
the second row, which a one-row test would never catch.

`opset_version` is pinned at 17 rather than left to torch's default, which moves
between releases and would silently change what a deployment target must support.

### Export is a backend method, and refusing is a normal outcome

Only the backend knows what its estimator physically is — a checkpoint, a booster,
a pickled statsmodels object. A central exporter with `if backend == …` in it would
need editing for every new backend, which is the coupling the plugin design exists
to remove.

| backend | formats | why not more |
|---|---|---|
| `lightning` | onnx, torchscript | — |
| `gbdt` | native | a booster has no traced graph, and its own `.json`/`.cbm` is what every runtime for that library already reads |
| `forecast` | pickle | Prophet and statsmodels expose no portable form of a *fitted* model |

**An unsupported combination raises.** Writing *some* file when the user asked for
ONNX would be discovered at deployment time by a runtime that cannot load it — or
worse, by one that loads it and scores differently. The error names what the
backend *can* produce, turning a dead end into a next step.

### Two things the export path does not silently assume

**Tracing exports the graph, not the preprocessing.** The bundle's scaler and image
transforms stay behind. An ONNX file fed raw unscaled features produces confident
nonsense with no error, so the note travels back through `ExportResult.notes` and
the CLI prints it.

**The traced object is verified before anything is written.** `jit.trace` cannot
walk a `LightningModule` — its `trainer` property raises when detached, which a
loaded bundle always is — so the *inner network* is traced instead. That
substitution assumes `forward` is exactly `self.network(x)`. Rather than assume it,
the module's own output and the network's are compared first, and export refuses if
they differ. One extra forward pass buys the difference between an assumption and a
check.

Text bundles are refused earlier still: tracing a tokenized batch would bake that
batch's sequence length into the graph, so every future request would be silently
truncated or padded to it. The HuggingFace directory is already portable.

### Per-bundle Dockerfiles

The repository Dockerfile's fixed targets (`serve`, `serve-gbdt`) are guesses about
which case you are in. A bundle already knows: `manifest.requires` lists exactly
what its backend needs, recorded at training time by the plugin that needed it.
`mlf dockerfile` reads that and installs those extras and no others — so a booster
image has no torch in it because the *bundle* says so, not because someone picked
the right target. A third-party plugin declaring its own extra gets a correct image
without `deploy.py` knowing the plugin exists.

Generated rather than committed: a Dockerfile checked in next to the code goes
stale the moment a plugin's requirements change, and the staleness is invisible
until an image fails to serve.

### Monitoring

Both collectors gained `backend` and `model` labels. Without them, two containers
scraped into one Prometheus produce indistinguishable timeseries that **silently
add together** — a booster's predictions and a transformer's arriving as one
counter. Cardinality is bounded: one value pair per served bundle, fixed for the
life of the process.

`monitoring/drift.py` reading `signature.feature_names` was already true — it has
read `manifest.signature.input.features` since the signature landed, so that plan
item needed verifying rather than implementing.

### Coverage

**91% measured, 80% enforced.** Before this, `ci.yml` ran `--cov` and enforced
nothing, so the number could drift down indefinitely without failing a build. The
floor is now in both `ci.yml` and `pyproject.toml`, so a local `pytest --cov` fails
for the same reason CI does rather than passing quietly and surprising someone on
push.

The plan's "duplicate root HTML docs" item was already moot — no such files exist
in the repository.

### Deliberate loose ends

1. **GBDT has no ONNX path.** Deliberate, per the table above, and it means
   `mlf export --format onnx` works for exactly one of three backends.
2. **`skl2onnx` is declared in the `export` extra and unused**, like `datasets` in
   `nlp`. Left alone.
3. **The generated Dockerfile is not built in CI**, so it is checked for content
   rather than for actually producing a working image.
4. **Text and seq2seq cannot be traced** — refused with a reason.
5. **Both export paths run on machinery PyTorch has deprecated** — measured below,
   and P10's to close.
6. Everything still open from P4–P8: `native_categorical` has no consumer,
   multi-series forecasting, `exog`, entity-level NER scoring, seq2seq vocabulary
   resizing, text drift.

### The deprecation P10 inherits, measured rather than guessed

Both formats sit on TorchScript, which PyTorch is retiring. On torch 2.10 /
Python 3.14 the suite is green but noisy:

    torch/jit/_trace.py:994          torch.jit.trace is not supported in Python 3.14+
    torch/onnx/.../torchscript_exporter/utils.py:218   the feature will be removed

The second one is the surprise: `dynamo=False` at `lightning.py` routes **ONNX**
through the TorchScript exporter too, so this is not a TorchScript-format-only
problem. That pin was correct when written — the dynamo exporter needs
`onnxscript`, which the `export` extra does not declare, so the torch 2.9+ default
would fail on a correctly-installed machine.

**The blocker is one missing package, and the opset pin survives.** Measured with
`onnxscript` 0.7.1 present, exporting and verifying in one process, through
onnxruntime at 1/7/64 rows:

| requested | artifact `opset_import` | batch axis | max abs diff |
|---|---|---|---|
| 17 | 17 | `'batch'` | **5.96e-08** |
| 18 | 18 | `'batch'` | 1.19e-07 |

5.96e-08 is the *same* figure the legacy exporter produces in the gate above, so
the migration does not move the number, and `ONNX_OPSET = 17` needs no
renegotiation with deployment targets.

**One caveat that belongs in the test, not in a comment.** Torch builds at opset 18
and down-converts, warning that conversion "may not be successful". It succeeded
here for `Gemm`/`Relu`; the CNN adds `Conv`/`MaxPool`, all long-settled ops, so the
risk is low — but it is **op-dependent, not a blanket guarantee**. The export test
should assert `opset_import` on the produced file, so an op that fails to
down-convert fails the build instead of silently shipping an opset-18 artifact to a
runtime that was promised 17.

**The trap in the migration.** `export()` traces at batch size **1**
(`torch.zeros(shape)`). Harmless for TorchScript; fatal for `torch.export`, which
treats 0 and 1 as special values and specializes the dimension to a constant —
`ConstraintViolationError: You marked batch as dynamic but your code specialized it
to be a constant (1)`. The example tensor must be batch ≥ 2. This is exactly the
frozen-batch failure the seven-row gate exists to catch, so the test would catch
it; the point is to not spend the debugging.

**TorchScript has no fix.** `torch.jit` is end-of-life and there is no shim that
keeps `trace` working on 3.14+. The replacement is `torch.export` + `.pt2`, which
round-trips exactly (verified, dynamic batch intact). That makes it a
user-visible format decision on `mlf export --format torchscript` rather than a
code edit, which is why it is not being done as a P9 amendment: the warnings are
noise today, not breakage, and P9's gates pass as committed.

## P10 — the ONNX half — **done**

Split deliberately. The ONNX migration is an internal swap with no user-visible
surface; the TorchScript one changes what `mlf export --format torchscript`
writes. Doing them together would have hidden a format decision inside a cleanup.

- **`export` declares `onnxscript>=0.7`**, and `ONNX_REQUIREMENTS` checks it — so
  a missing copy names the pip extra instead of failing from inside torch.
- **`dynamo=True`** in `backends/lightning.py`. The pin it replaces was correct
  when written and had no future.
- **`ONNX_OPSET = 17` is unchanged and now asserted on the produced file.**

### Gates met

`test_the_exported_file_declares_the_opset_we_promised` reads `opset_import` off
the artifact. The exporter builds at 18 and down-converts, warning that it "may
not be successful" — for an MLP's `Gemm`/`Relu` it succeeds, but that is
op-dependent, so the assertion is what stops a future op silently shipping an
opset-18 file to a runtime promised 17. It also asserts the batch axis is a
symbol rather than the width it was traced at.

The P9 parity gate is untouched and still passes: 7 rows, `argmax` agreeing,
inside the stated 1e-5.

### Two corrections to the scoping commit above

**The batch-size trap did not materialize, and no code changed for it.**
`ccfa12b` predicted `ConstraintViolationError` from tracing at batch 1, because
`torch.export` specializes 0 and 1 to constants. Measured: exporting at batch
**1** and at batch **2** both produce `dim_param='batch'` and both score
correctly at 1, 7 and 64 rows. The specialization applies to raw `torch.export`
with `dynamic_shapes`; `torch.onnx.export(dynamo=True, dynamic_axes=…)` goes
through a compat shim that handles it. So `example_input_shape` keeps returning
a leading `1` — changing it would have been churn against a prediction that did
not hold. **If the TorchScript half later moves to raw `torch.export`, the trap
returns and the prediction becomes correct again.**

**The real blocker was one nobody predicted, and it is platform-specific.** The
dynamo exporter writes U+2705 to stdout on success. On Windows, where the console
is cp1252, encoding it raises `UnicodeEncodeError` *from inside an otherwise
successful export* — a correct artifact reported as a failure. `verbose=False`
suppresses it. `test_export_survives_a_console_that_cannot_encode_a_check_mark`
pins this by redirecting stdout through a strict cp1252 stream, so it reproduces
on every platform rather than only where it bites; verified to fail without the
flag. This is the same class of bug as the UTF-8 stdout fix in `utils/logging.py`,
arriving through a dependency instead of our own logging.

### What P10 still owes

1. **`--format torchscript` still calls `torch.jit.trace`**, which is deprecated
   and unsupported on Python 3.14+. Still a warning, not breakage. The decision it
   needs is a product one — rename the format, keep the flag and write `.pt2`, or
   drop it — not a code edit.
2. ~~**CI does not test the Python versions where this bites.**~~ **Closed here.**
   `requires-python` claims `>=3.10` with no ceiling while `ci.yml` stopped at
   3.12, so the `torch.jit` deprecation that motivated this migration fired on a
   version no job ran. The matrix is now 3.10–3.14. **3.14 is the load-bearing
   entry** — 3.13 would not have caught it, since the deprecation is 3.14+.

   Evidence is asymmetric and worth stating: 3.14 is backed by the full suite
   passing locally on 3.14.2 / torch 2.10.0+cpu, which is where every number in
   this document was measured. **3.13 has not been run anywhere** — no
   interpreter for it here. If it fails on the first CI run, that is the matrix
   doing its job and the fix is a real one, not a revert.
3. `skl2onnx` remains declared and unused (from P9).

---

## P11 — Model selection

Full documentation: [MODEL_SELECTION.md](MODEL_SELECTION.md).

### What P11 landed

Cross-family model selection: tune *every* eligible candidate, then choose on
five measured criteria rather than on the score alone.

1. **`pipeline/select.py`** — the driver. Gates the candidate pool, tunes each
   survivor on its own space, cross-validates and profiles it, disqualifies on
   measured constraints, and applies a decision rule. **Returns a config**, like
   `tune` does, so `train()` keeps one bundle-writing path whether a bake-off
   ran, only tuning ran, or neither did.
2. **`core/profile.py`** — the four measurements that were not being taken:
   p50/p95/p99 single-row latency and batched throughput, serialized artifact
   bytes, the explainability tier, and fold stability / fit wall-clock / fold
   failures.
3. **`core/explain.py`** — tiered attribution: native `feature_importances_`
   (1.0) → SHAP for trees (0.8) → `permutation_importance` (0.5) → none (0.0).
   The score feeds `min_explainability`; the values go to
   `feature_importance.json`.
4. **Purged and combinatorial-purged CV** in `data/splitters.py`, joining the
   list-returning family in the new `CV_SPLITTERS` registry. `label_spans`,
   `purge_and_embargo` and `embargo_rows` are the shared primitives.
5. **`data.split.cv_strategy`** — `auto | stratified | kfold | rolling_origin |
   purged | cpcv`. `auto` reproduces the pre-P11 rule exactly, so an existing
   config keeps the folds it already had.
6. **`tune.objective: cv`** — a trial scored as the mean across `tune.cv_folds`
   inner folds, which is what makes "model and hyperparameters selected jointly,
   on cross-validation" literally true. `tune.n_jobs` for concurrent trials.
7. **Parallel candidates**, two ways: `select.max_workers` (a
   `ProcessPoolExecutor` on one machine) and Airflow dynamic task mapping (one
   mapped task per family, then a reduce task).
8. **`mlf select`**, plus `mlf train --select`. `--candidate`/`--collect` split
   the fan-out from the decision so the same rule runs in both topologies.
9. **`manifest.selection`** — the whole bake-off inside the served bundle, so a
   deployed model can answer "why this family?" without the training directory.

### Gates met

- **811 passed, 1 skipped** (from 643/1). Coverage **89.7%**, floor 80.
- `ruff`, `black --check`, `isort`, `mypy src` all clean.
- New modules: `select.py` 87%, `profile.py` 93%, `explain.py` 90%,
  `splitters.py` 95%.
- Verified end to end on a real dataset: three tree families plus a torch MLP
  compared on one table, cross-backend, with constraint disqualification, the
  weighted objective, purged CV, CPCV (C(5,2)=10 folds), a 3-worker process pool,
  and the Airflow fan-out/collect round trip through JSON.

### Decisions worth knowing

**`select` returns a config, exactly like `tune`.** It does not return a fitted
model and does not write a bundle. The alternative — having `train()` branch on
whether a bake-off happened — would have duplicated the bundle-writing path,
which is the thing P1 existed to collapse. The winner is refit at full budget by
the code that was already there.

**Gating is two-phase because it has to be.** Compatibility, availability and
row-count rules are answerable before training and *skip* work. Latency and size
cannot be known until a model exists, so they disqualify after the fact — with
the number missed by, because "too slow" is not actionable and "p95 24.10 ms
exceeds the 20.00 ms budget" is.

**The default decision rule never trades accuracy for speed silently.** Hard
constraints disqualify; among survivors, the best score wins unless something
*statistically tied* with it (within one standard error of the CV mean, by
default) is cheaper. The tie-break order is fixed — latency, size,
explainability, stability — and the reason string names the axis that actually
decided, because listing the whole vector invites the reading that the winner is
better on all of it, which it usually is not.

**An unmeasured latency sorts last, never first.** A model whose speed could not
be measured must not win a tie *because* it is unknown; that would make a
measurement failure look like a measurement of zero. Same rule for size.

**The weighted objective exists but is not the default.** It makes accuracy and
milliseconds commensurable, which they are not, and its composite has no meaning
outside the run that produced it (normalization is within the bake-off). It is
there for regulated contexts that need an auditable weight table.

**Purging needs the label span, and there are two honest ways to say it.**
`label_horizon` states it in rows and needs nothing from the source.
`label_end_col` is exact and requires `time_col` alongside it — mapping a label
*end time* onto a row position is impossible without the rows' own observation
times, and the first implementation of this got it wrong by searching the sorted
label-end array against itself. The test that caught it is
`test_label_spans_maps_an_end_time_column_to_positions`.

**Processes, not threads, for candidate parallelism.** A fit is CPU-bound and
holds the GIL, and two Lightning trainers in one interpreter share global state
(seed, logger, accelerator registry) in ways that make results depend on
interleaving. The pool falls back to sequential — loudly — where one cannot be
created, because failing a run over a scheduling detail is worse than being slow.

**Airflow gets N mapped tasks rather than one task with a loop.** Independent
retries and independent failures: a family whose extra is missing on one worker
should be one red square, not a dead pipeline. Reports cross as JSON, not
pickles, because writer and reader are different processes on different machines.

### A pre-existing bug this surfaced

**`mlf train --model lightgbm` with tuning on a classification task failed
outright, on `main`, before any P11 code.** Verified by stashing the P11 changes
and reproducing it.

`_PRUNING_METRICS` in `backends/gbdt.py` maps every task to the library's own
training loss (`binary_logloss`, `validation_0-mlogloss`, `RMSE`) — all
*minimized*. The Optuna study direction comes from the task's primary metric,
which for every classification task is accuracy — *maximized*. LightGBM's
`LightGBMPruningCallback` detects the mismatch and raises:

```
ValueError: The intermediate values are inconsistent with the objective values
in terms of study directions.
```

xgboost's and catboost's callbacks do **not** detect it, so they pruned in the
wrong direction in silence — concluding that a *rising* loss was progress and
abandoning exactly the trials that were working. That is the worse of the two
failures, and nothing would have reported it.

Fixed by having `pruning_callback` read `trial.study.direction` and attach the
callback only when the directions agree, logging why when they do not. Pruning is
worth less on this backend anyway — the module's own docstring already said a
boosting trial costs seconds, so there is little to abandon — and a search that
runs every trial honestly beats one that discards the good ones quickly.

The bake-off is what exposed it: nothing before P11 ran lightgbm and xgboost
through the same tuning path in one command.

## P12 — Pluggable data backends

Design and full execution flow in [choose.md](../choose.md).

`data.backend` (or `--data-backend`) selects the engine that reads and reduces the
table, per run: `local` (pandas, the default), `polars`, or `spark`. A fourth
registry, `DATA_BACKENDS`, mirrors `BACKENDS` exactly — spec plus lazy factory, so
`mlf data-backends` lists every engine on an install without them and selecting one
yields `pip install 'ml-framework[mlops]'` (or `[fast]`) rather than an ImportError.

`local` and `polars` are **peers**: same contract, different parser. pandas stays
the base dependency and the default because `read_table -> pd.DataFrame` is public
API consumed by pandera in `pipeline/contracts.py`.

**The three pipeline call sites did not change.** `train.py`, `tune.py` and
`select.py` all call `build_bundle(config)`; the choice rides inside the config.
That was the point.

### The boundary, stated plainly

A data backend chooses **how the table is read and reduced, not how the model is
trained**. `DataBundle` holds numpy arrays and the splitters index into them, so
the framework collects a materialized bundle and fits on one node under either
engine. Spark's job ends at the feature matrix. What it buys:

* **Fold planning without the feature matrix.** `_cv_population` uses exactly
  `read_table`, `n_rows`, `column` — verified by tracing, not asserted.
* **Failing on the schema.** A mistyped `data.target` now costs a metadata lookup
  instead of a full materialization that then throws.
* **One Spark codebase.** `pipeline/spark_preprocess.py` no longer imports
  pyspark; it runs on the protocol, so the DVC stage and the training read share
  one session configuration — and the same cleaning runs under `local` with no
  JVM, which was impossible before.

### The correctness rule the protocol enforces

Only two methods collect: `column` (standalone arrays) and `to_pandas` (the atomic
one). *Arrays that must line up row-for-row come out of a single collect.* Each
collect re-executes the query plan, and pyspark documents
`monotonically_increasing_id` — which the sort tie-break relies on — as
non-deterministic; separate collects could therefore pair feature rows with the
wrong labels, silently. An earlier draft had a `matrix()` method that invited
exactly that, and it was removed rather than patched.

### The measurement that settled the Polars question

`benchmarks/data_backends.py`, median of 5 runs on this machine: CSV parse
**19–21×**, Parquet parse ~1.3×, and the whole `build_bundle` **2.8–3.7×**. The
design doc had argued Polars was not worth adding because the win was "confined to
parse time"; the parse turns out to dominate `build_bundle`, so the argument was
right about the location and wrong about the size. The benchmark is committed so
the ratios can be re-measured rather than trusted.

### What P12 still owes

1. **Nothing Spark-specific has been proven on a machine with a JVM.** The sort
   tie-break and the multi-collect alignment rule skip without one. The
   `spark-contract` CI job exists to run them and refuses to pass by skipping, but
   it has not yet run green. (The cross-engine bundle-equality tests *do* now run,
   for Polars — that engine needs no JVM.)
2. **Only `tabular` is backend-aware.** `image`, `text` and `timeseries` read
   through pandas; a non-local backend is refused by name for them.
3. **No distributed training.** A `spark` *training* backend consuming a `frame`
   payload is separate work on a different registry: `GBDTBackend.fit` routes
   through `_as_matrix`, which sends anything non-pandas to `np.asarray`.

### Deliberate loose ends P12+ must close

1. **Forecasting candidates are not latency-profiled.** A forecaster takes a
   horizon, not rows, so "ms per row" is not a quantity it has; the profile says
   "not measured" and the tie-break sorts it last. A latency constraint therefore
   disqualifies *every* forecaster, which is correct-but-blunt. A horizon-based
   latency measure is the obvious fix.
2. **Latency and size are measured on the training machine.** The ranking
   transfers under identical conditions; the absolute number does not. A
   production-representative benchmark host would make `max_latency_p95_ms`
   portable rather than relative.
3. **No warm-starting between candidates.** Each family's study starts cold.
   Sharing information across families (a meta-learner over past runs) is real
   AutoML and deliberately out of scope.
4. **`select.max_workers` is CPU-oriented.** On one GPU, concurrent candidates
   contend and wall-clock gets worse. There is no device-aware default; the
   documentation says to leave it at 1 and fan out with Airflow instead.
5. **CPCV validation folds come off the end of the surviving training block** —
   simple and slightly conservative rather than optimal.
6. **`label_end_col` is tabular/timeseries only.** Image and text folds use
   `label_horizon` or nothing.
7. **SHAP is tree-only.** `shap.Explainer`'s sampling fallback takes minutes on a
   non-tree model, inside a routine that is also timing inference. Those models
   get permutation importance.

---

## P13 — Staged decode pipeline

**P13a landed. P13b–P13e are open.** Test baseline after P13a: **984 passed, 18 skipped**
(up from 922; the 18 remain the JVM-gated Spark tests — P13a added 62 and skipped none,
because its two zero-dependency decoders are exactly the ones a bare install can run).

Storage → tensor is eleven stages, and this framework previously expressed about
three of them. The two claims driving the phase:

1. **Read, demux and decode are three distinct stages, and decode is independent of
   tensor construction.** A pre-tokenized `.bin` shard has neither demux nor decode
   but does have tensor construction; a GPU decoder has decode and *no* tensor
   construction at all. A design with one "load" stage cannot say either thing.
2. **A corrupt sample must be substituted, never skipped.** Under DDP every rank
   must produce an identical number of batches or the next collective hangs with no
   error message, so `continue` is the one response that cannot be allowed. Worse
   are the formats that fail *silently* — MP3 resync, mid-stream H.264 artifacts,
   NVDEC garbage frames — which yield valid-shaped tensors that degrade the model
   without tripping any handler.

### P13a — decoder registry and stage vocabulary (done)

Zero behaviour change: nothing outside `data/streaming/` imports it yet.

* `core/types.py` gains `Stage`, `Layout`, `LandsIn`, `Integrity` beside `DataKind`
  and `Payload` — vocabulary, in the dependency-free module, because
  `core/plugins.DecoderSpec` is typed against them and core may not import the data
  layer. `DataKind` gains `audio` and `video`; `DEFAULT_PAYLOAD`/`KIND_PAYLOADS` map
  both to `dataset`. Their sources and models arrive in P13d.
* `core/plugins.DecoderSpec` + `core/registry.DECODERS` / `register_decoder` /
  `get_decoder`, modelled on `DataBackendSpec` exactly: a spec plus a lazy factory,
  never an import of the codec.
* `data/streaming/` — `stages.py` (`SampleRef`/`Blob`/`Packet`/`Decoded`/
  `DecodeContext`/`BlobSource`), `integrity.py` (the four classifications,
  `SampleFault`, `FaultLog`), `decoders/` (nine specs, seven modules).
* `mlf decoders [--all] [--show]`, off the existing generic `_print_plugins`.
* Extras `audio = [soundfile, torchaudio]` and `video = [av]`.

**The decoder table, as `mlf decoders --show` renders it:**

| decoder | stages | output | lands | integrity |
|---|---|---|---|---|
| `audio.pcm` | read, decode | int16 pcm | host | loud — **the oracle**, no requirements |
| `audio.flac` | read, decode | int16 pcm | host | **checked** — per-frame CRC-16 |
| `audio.mp3` | read, demux, decode | float32 pcm | host | **silent** — resyncs past damage |
| `audio.opus` | read, demux, decode | float32 pcm | host | checked — Ogg page CRC |
| `image.jpeg` | read, decode | uint8 hwc | host | loud *because we make it so* |
| `image.png` | read, decode | uint8 hwc | host | checked — Adler-32 + CRC-32 |
| `video.h264` | read, demux, decode | uint8 thwc | host | loud — but artifacts are silent |
| `text.tokens` | **read** | **uint16** tokens | host | **none** — a bit flip is a valid id |
| `fake.device` | read, decode | uint8 hwc | **device** | silent — the seam under test |

Four decisions in that table worth keeping:

* **`read` is on every row.** Bytes always come off a device. What varies is the
  other two, which is why `text.tokens` declaring `{"read"}` *alone* is the
  informative case. A test pins that only that row claims decode costs nothing.
* **`audio.pcm` has no requirements on purpose.** It is the oracle the
  cross-decoder equivalence tests compare against — the role `local` plays for data
  backends — and that claim is only worth making if the reference path is present on
  a bare install.
* **JPEG's `loud` is a deviation, not a default.** libjpeg treats a truncated file
  as a *warning* and returns a partial image with the missing scanlines filled grey;
  a test pins that Pillow really does this (31/64 flat rows on a half-truncated
  64×64), that the decoder raises instead, and that it still raises when something
  else has set `LOAD_TRUNCATED_IMAGES = True` globally.
* **`fake.device` ships with no CUDA.** The `lands_in` seam changes loader
  construction in two places and is invisible in a third; a decoder that allocates
  no device memory is what lets CI execute all three. A real DALI/NVDEC decoder
  registers identically.

### P13b–P13e — open

* **P13b — shards, sampler, dataset, state, materialize.** `ShardIndex`
  (`shards.json` + streamed `entries.jsonl`), `BlobSource` implementations,
  `ShardShuffleSampler`, `StagedDataset`, `LoaderState`, `mlf materialize`.
  Testable with `audio.pcm` + `text.tokens` alone, so it needs no optional
  dependency. The load-bearing invariant: `len(dataset)` is read from the index and
  `__getitem__` is *total*, so batch count is invariant to how many samples are
  corrupt — which is how DDP parity holds with no rank-aware code anywhere in `src/`.
* **P13c — the transport tail.** `pin_memory` / `persistent_workers` /
  `prefetch_factor` (none of which exist in the repo today), zero-copy `from_numpy`
  when the buffer is already float32/C-contiguous/writeable, and the
  `lands_in == "device"` branch. Ships alone, and its gate is byte-identical
  artifacts against the previous commit at seed 42.
* **P13d — `audio` and `video` sources, preprocessors and models.** The vocabulary
  landed in P13a; this adds `SourceSpec`s, `AudioSourceParams`/`VideoSourceParams`,
  the mel/clip preprocessors, `audio.cnn` and `video.r3d`, `serving/schemas.py`, and
  `sniff.py` detection.
* **P13e — stall profiler and the fault→tracker path.** `core/stall.py`,
  `StagedDataCallback`, `bundle/faults.json` and `bundle/stall.json` always written.

### Known gaps P13a leaves open, deliberately

1. **Nothing consumes the registry yet.** `decoder_for` resolves and instantiates,
   but no source calls it — that is P13b/P13d. The phase is a foundation, and its
   gate was "`mlf decoders` prints the table truthfully on a bare install".
2. **`audio`/`video` are in `DataKind` with no source behind them.** Setting
   `data.kind: audio` today reaches `SOURCES` and fails with "unknown source", which
   is a clear error but not the eventual one. `_check_required_by_kind` grows its
   cases in P13d.
3. **The compressed-audio and video decoders are untested against real files.** They
   are gated on `av`, which is not installed here. `audio.mp3`'s silent-resync
   detection needs the duration cross-check in `mlf materialize` (P13b) plus a
   committed ~2 KB truncated MP3 fixture to be proven at all.
4. **`mlf materialize` does not exist**, so the refusal to train on a `silent`/`none`
   corpus without an offline pass is designed but not enforced. That enforcement is
   P13b's, and until it lands nothing checks `REQUIRES_MATERIALIZATION`.
5. **Parquet is deliberately not a decoder row.** Its read/demux/decode is already
   owned by `DataBackend` (P12), and the reference table itself notes the
   Thrift-footer→chunk-offset step is "same API, no seam". A second owner for one
   read path is what `DataBackendSpec`'s docstring forbids.
