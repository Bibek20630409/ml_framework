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

**249 passed, 4 skipped** after P1 (46/4 before P0 → 130/4 after P0 → 249/4 after P1).
Every phase gate is measured against this number — a phase that ends with fewer
passing tests than it started with has regressed something, regardless of what
its own new tests say.

**No existing test has been edited in either phase.** The plan permits edits for
v2 config field names and bundle artifact paths (§8.6); P1 needed neither, because
the v1 artifacts stay at the bundle root until P3 rewrites the loader.

Verification commands (all clean):

```
pytest                        # 249 passed, 4 skipped
ruff check src tests
black --check src tests
isort --check-only src tests
mypy src
```

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

### Coverage caveat

The 4 skips are pre-existing and unchanged: `pandera`, `mlflow`, `slowapi` and
`prometheus_fastapi_instrumentator` are not installed here, so
`tests/integration/test_mlflow.py` does **not** run in this environment. The
MLflow run-ownership change is therefore covered by a stubbed unit test of the
wiring (`test_mlflow_logger_attaches_to_the_orchestrators_run`) rather than by an
end-to-end MLflow run. Install the `[mlops]` extra to exercise the real path.

### Note on tracking

`train()` now holds a backend-neutral `RunLogger` and the Lightning logger lives
inside the backend. For MLflow the two are bound to the **same run** via `run_id`:
the orchestrator creates the run, the backend attaches to it. v1 had the Lightning
logger own the run, which is why `log_and_register` reached into it for a `run_id`
that a GBDT run would never have.
