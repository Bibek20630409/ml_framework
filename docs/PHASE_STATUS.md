# Phase status

Tracks progress against [ml_framework_architecture_plan.md](../ml_framework_architecture_plan.md)
§5 (Implementation Roadmap). Update this when a phase lands.

| Phase | State | Commit |
|---|---|---|
| P0 — Foundations | **done** | `a8692a1` |
| P1 — Data + backend extraction | not started | — |
| P2 — v2 config | not started | — |
| P3 — GBDT | not started | — |
| P4 — AutoML | not started | — |
| P5 — DL hardening | not started | — |
| P6 — Time-series | not started | — |
| P7 — NLP | not started | — |
| P8 — Zero-config | not started | — |
| P9 — Deployment polish | not started | — |

## Test baseline

**130 passed, 4 skipped** as of `a8692a1` (46 passed / 4 skipped before P0; P0 added
84 tests and edited none). Every phase gate is measured against this number — a
phase that ends with fewer passing tests than it started with has regressed
something, regardless of what its own new tests say.

Verification commands (all clean at `a8692a1`):

```
pytest                        # 130 passed, 4 skipped
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

## P1 notes

The plan calls P1 the highest-risk phase: it changes the data layer and the fit loop
simultaneously. §5 offers a split, and it is worth taking:

- **P1a** — introduce `DataBundle` + `BundleDataModule`, feeding the *existing*
  `train()`.
- **P1b** — extract `LightningBackend`; `pipeline/train.py` becomes pure
  orchestration.

Each half is independently revertible against `a8692a1`.

Mechanical gate for P1: `grep pytorch_lightning src/ml_framework/pipeline/train.py`
returns **0 hits**.
