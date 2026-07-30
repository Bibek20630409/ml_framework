# Phase status

Tracks progress against [ml_framework_architecture_plan.md](../ml_framework_architecture_plan.md)
§5 (Implementation Roadmap). Update this when a phase lands.

| Phase | State | Commit |
|---|---|---|
| P0 — Foundations | **done** | `a8692a1` |
| P1 — Data + backend extraction | **done** | `95c4359` (P1a) · P1b |
| P2 — v2 config | not started | — |
| P3 — GBDT | not started | — |
| P4 — AutoML | not started | — |
| P5 — DL hardening | not started | — |
| P6 — Time-series | not started | — |
| P7 — NLP | not started | — |
| P8 — Zero-config | not started | — |
| P9 — Deployment polish | not started | — |

## Test baseline

**256 passed, 1 skipped** with every declared extra installed except DVC
(46/4 before P0 → 130/4 after P0 → 249/4 after P1 → 257/0 → **256/1**). Every
phase gate is measured against this number — a phase that ends with fewer passing
tests than it started with has regressed something, regardless of what its own new
tests say.

**Read the 257 → 256 step carefully: it is not a regression.** Installing
`torchvision` makes `MODELS.is_available("cnn")` true, so
`test_validate_combination_raises_missing_extra_for_a_sound_but_uninstalled_model`
self-skips ("torchvision installed — nothing to refuse") because the condition it
exists to test no longer holds. The companion
`test_models_for_lists_only_installed_compatible_models` simply took its `["cnn"]`
branch instead of `[]`. Both tests are availability-aware by design; do **not**
pin them to either world.

The one expected skip is therefore `tests/unit/test_plugins.py:253`. Any *other*
skip means a package went missing.

### The bare-install guardrail still holds

The plan's §5 guardrail — the suite must pass on an install *without* the extras —
can no longer be exercised locally now that the extras are present. It is still
enforced in CI, and the design guardrail itself does not depend on torchvision
being absent: `test_plugins.py::test_get_refuses_a_plugin_whose_extra_is_missing`
and `test_bundle.py::test_bundle_requirements_are_checked_before_any_import_is_attempted`
both use synthetic package names that are never installed. Do not convert an
`importorskip` into a hard import.

**No existing test has been edited in any phase so far.** The plan permits edits
for v2 config field names and bundle artifact paths (§8.6); P1 needed neither,
because the v1 artifacts stay at the bundle root until P3 rewrites the loader.

Verification commands (all clean):

```
pytest                        # 256 passed, 1 skipped
ruff check src tests
black --check src tests
isort --check-only src tests
mypy src
```

### Environment notes

- **torch and torchvision must be installed together, from one index.** Every
  torchvision release pins one torch patch *exactly* (0.25.0 requires
  torch==2.10.0). The extras therefore keep an unbounded floor — an upper bound
  there would pin the user's torch and contradict `torch>=2.0.0`. The constraint
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
- **`pyarrow` is undeclared but not actually missing**: `mlflow` requires it
  unconditionally (`pyarrow<25,>=4.0.0`), and every environment that runs the suite
  installs `.[dev]`, which includes mlflow. So parquet support and
  `test_read_table.py` work here and in CI today. The hidden coupling is worth
  fixing before **P3**, whose whole point is a serving image without the heavy
  deps: drop mlflow from such an image and `read_table`'s parquet branch loses its
  engine with no declaration to explain why.

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

## Deliberate loose ends P1/P2 must close

These look like omissions and are not. Do not "fix" them out of order.

1. **`BACKENDS` is registered but empty.** P0 would have had to point a
   `BackendSpec.factory` at `backends/lightning.py`, which did not exist yet. **P1**
   registers the `lightning` backend once that module lands.
2. **`ModelSpec.build` and `SourceSpec.build` are typed `Callable[..., Any]`** and
   currently hold the legacy callables (the model class; `build_datamodule`). Their
   docstrings name the real target contracts: **P1** tightens `SourceSpec.build` to
   return a `DataBundle`; **P2** switches `ModelSpec.build` to `build(BuildContext)`.
3. **`ModelSpec.params_model` is `None` on both builtin specs.** Per-plugin Pydantic
   validation of `model.params` belongs to the v2 config schema, so **P2** wires it —
   at which point `ModelConfig._check_dims` moves into the MLP plugin's params model.
4. **Search-space keys are dotted paths against the *v2* schema**
   (`model.params.dropout`), which does not exist until P2. Nothing consumes them
   until **P4**, so they are declared, unused and correct.
5. **`models/__init__.py` still swallows a failed `cnn` import** (`except Exception:
   pass`). The v2 registry already reports `cnn` honestly via `find_spec`, but the v1
   path keeps the old behavior until **P2** replaces it with non-swallowing discovery.
6. **`config/schema.py` keeps its own narrow `Task`/`DataKind` literals** rather than
   importing the wider ones from `core/types.py`. Switching in P0 would have widened
   what validates — a behavior change. **P2** does it as part of the schema rewrite.

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
2. **`LightningBackend.load` reads `config.json`** to rebuild an architecture,
   because v1's `BaseModel.__init__` takes a whole `ExperimentConfig`. **P2** makes
   `ModelSpec.build` take a `BuildContext`, after which the manifest signature is
   sufficient alone.
3. **`ModelSpec.build` is still called with the legacy kwargs**
   (`input_dim`, `output_dim`, `config`, `class_weights`) from
   `LightningBackend.fit`. Same owner: **P2**.
4. **`search_space()` keys are dotted paths against the v2 schema**
   (`fit.params.lr`), which does not exist yet. Declared, unused, correct —
   consumed by **P4**.
5. **`pipeline/hpo.py` and `lr_finder.py` still build their own `pl.Trainer`.**
   `hpo.py` is deleted in **P4**; `lr_finder.py` gains a capability gate in P3/P5.
   Neither is on the P1 gate.
6. **`mlflow_utils.log_and_register` still logs a hardcoded 4-filename list.**
   `train()` now also logs the whole bundle dir through the `RunLogger`, so the
   registered bundle is complete; collapsing the two belongs with **P3**'s serving
   work.
7. **`TemporalSplitter`/`GroupSplitter` have no config path to reach them.** That
   needs `split.strategy` in the v2 schema (**P2**) and, for the leakage guard,
   **P6**. `RollingOriginSplitter` waits for the time-series CV that consumes it.

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
