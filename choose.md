# Choosing a Data Backend

**Status: all five phases (0–4) are implemented and green.**
**Goal:** let a single run say *"use the Spark implementation"* or *"use the local implementation"* without changing the rest of the framework.

> **What shipped.** `mlf data-backends` lists three engines; `--data-backend` selects one per run; the engine reaches inside `build_bundle`; `pipeline/spark_preprocess.py` no longer imports pyspark at all; and Polars ships as a peer of pandas after the benchmark §7 gated it on came back **2.8–3.7× on `build_bundle`**. The three pipeline call sites were never touched. Gate at time of writing: ruff / black / mypy clean, **922 passed, 18 skipped, 90.3% coverage** (up from an 811/1 baseline; the extra skips are all JVM-gated Spark tests). Where the build diverged from this document, the document has been corrected and the change is called out in a **Built:** note. See §13 for the full list.
>
> **Phase 2 changed the protocol.** Implementing it surfaced a correctness bug in this plan: the design had several methods collecting independently, and on a distributed engine separate collects can disagree on row order — which would pair feature rows with the wrong labels, silently. `matrix()` was replaced by a lazy `select()`, and the rule *"arrays that must line up come out of one collect"* is now part of the protocol. See §3 and §13 row 8.

```
                          Framework
                              |
                         Data Backend
                              |
        +---------------------+---------------------+
        |                     |                     |
   LocalBackend        PolarsBackend          SparkBackend
        |                     |                     |
     pandas                polars                 Spark
    (default)              (peer)             (distributed)
```

```bash
mlf train --config configs/example_tabular.yaml                        # local (default)
mlf train --config configs/example_tabular.yaml --data-backend polars  # Polars
mlf train --config configs/example_tabular.yaml --data-backend spark   # Spark
```

The original sketch drew `LocalBackend → Polars`. What shipped is `local` **and** `polars` as peers, which is strictly better: pandas stays the default and `read_table`'s public contract never moves, while Polars is a choice a run makes rather than a decree — or worse, a silent behaviour change keyed on what happens to be installed. See §7.

---

## 1. Verdict

**Yes — and it is a small change, because the machinery already exists.**

The framework already has a generic plugin registry that does exactly this for three other things. It has simply never been pointed at the data-processing engine.

| Piece we need | Already in the repo |
|---|---|
| Name → implementation registry | `PluginRegistry(Generic[SpecT])`, `core/plugins.py:224` |
| Three live instances of it | `MODELS` / `BACKENDS` / `SOURCES`, `core/registry.py:131-133` |
| Registration without importing the heavy dep | `_lazy_factory`, `backends/__init__.py:25-32` |
| "Missing extra" as a pip line, not an ImportError | `Requirement` + `check_requirements`, `core/types.py:134`, `core/plugins.py:95` |
| Per-run selection from config/CLI | `--set`, `with_overrides`, `_load_config`, `cli.py:98` |
| Spark that already works | `pipeline/spark_preprocess.py` |

And there is exactly **one** hard pandas boundary in the whole data layer:

```python
# src/ml_framework/data/sources/tabular.py:73-84
def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir() or p.suffix.lower() in (".parquet", ".pq"):
        check_requirements((PARQUET_REQUIREMENT,), what=f"reading parquet from '{path}'")
        return pd.read_parquet(path)
    return pd.read_csv(path)
```

Threading a backend through that one function delivers the entire user-facing promise. The three pipeline call sites — `train.py:126`, `tune.py:568`, `select.py:500` — never change, because they all call `build_bundle(config)` and the choice rides inside the config.

---

## 2. The honest boundary

Before the design, the limit — stated plainly, because the alternative is a feature that oversells itself.

**A data backend chooses how the table is read and reduced, not how the model is trained.**

Three facts make this non-negotiable:

1. `DataBundle` holds **numpy arrays** (`data/types.py` — `Split.x`, `Split.y`, `class_weights`). It is a frozen, materialized snapshot.
2. `splitters.py` (789 lines) is `split(n: int, *, y: np.ndarray) -> SplitIndices` — index arithmetic over in-memory arrays. Then `build_tabular_bundle` does `x[parts.train]` (`tabular.py:250`), which requires `x` in RAM.
3. All three training backends — `lightning`, `gbdt`, `forecast` — fit on a single node. So do `TabularPreprocessor`, SMOTE (`resolve_imbalance`), and the drift reference builder.

So the line falls at exactly one point, the feature matrix:

```
Spark:   read ──► filter ──► sort ──► project              [distributed, cluster-scale]
                                        │
                                        ▼
                              collect / .toPandas()         ◄── THE COLLECT POINT
                                        │                       tabular.py:237
                                        ▼
         DataBundle (numpy) ──► backend.fit(...)            [single node, unchanged]
```

### What this genuinely buys

Three wins survive the framing, and they are the justification for the feature:

1. **CV fold planning without collecting features.** `_cv_population` needs only a row count and one label column. Under Spark that is a `count()` plus a one-column collect — the feature matrix is never touched. A 100 GB table can have its folds planned. **Verified after Phase 2**: tracing the engine through `_cv_population` shows exactly `read_table`, `n_rows`, `column` — `to_pandas` is never called.
2. **Fail fast on the schema, before collecting anything.** `build_tabular_bundle` validates `data.target` and the split columns against `engine.columns(table)`. A mistyped target now costs a metadata lookup instead of a full materialization that then throws.
3. **Column projection pushdown** — but only where the caller genuinely needs a subset. `_label_columns` reads 2 of N columns, so `select` pushes down into Parquet and the driver receives two columns.

> **Built: win 2 was originally stated as pushdown in `build_tabular_bundle`, and that was wrong.** The claim was that the builder "drops `target`/`time_col`/`group_col` and *then* materializes, so the driver receives only the columns it trains on". But `feature_cols = all − reserved`, and the bundle needs the features **and** the target **and** the time column **and** the group column — `feature_cols ∪ reserved` is *every* column. There is nothing to project away there, so its collect is a plain `to_pandas`. The pushdown win is real for `_label_columns` and not for the builder; fail-fast validation is what the builder actually gained.

### What it does not buy

You still cannot train on data larger than the driver. Say so in the docs; do not let a user discover it via an OOM kill.

### The door left open

A genuinely distributed path exists, but it is a **different registry**: a `TrainingBackend` named `spark` (SynapseML, `xgboost4j-spark`, `spark.ml`) consuming a bundle whose `payload == "frame"`. `Payload` already includes `"frame"` (`core/types.py:49`) and `gbdt`'s `Capabilities.accepts` already includes it (`backends/__init__.py:74`). That is separate work. Do not conflate the two.

---

## 3. The `DataBackend` protocol

Lives in `core/protocols.py`, beside `TrainingBackend` (L276) and `Splitter` (L355). Same shape as its neighbours: `@runtime_checkable` `Protocol` with a `ClassVar[str] name`. The repo uses no ABCs anywhere in `src/`, so this does not introduce one.

The design mirrors `TrainingBackend` deliberately: a **flat, stateless backend that passes an opaque handle back into every method**, exactly as `TrainingBackend` passes an `Estimator`. The alternative — a `Table` protocol that third parties must also implement — doubles the conformance surface for no gain.

```python
# An opaque handle to a table owned by a DataBackend: a `pd.DataFrame` under
# `local`, a `pyspark.sql.DataFrame` under `spark`. `Any` rather than a union,
# for the same reason `Split.x` is `Any` — naming the alternatives would make
# this module import pandas and pyspark.
Table = Any


@runtime_checkable
class DataBackend(Protocol):
    """The engine that reads and reduces a table, chosen per run by `data.backend`.

    A data backend decides *how the bytes become a matrix*, not how the model is
    trained. Every method below replaces one pandas idiom that exists in the data
    layer today; there are no methods here without a call site.

    **Exactly two methods collect: `column` and `to_pandas`.** The split between
    them is a correctness rule, not a convenience:

        Arrays that must line up row-for-row have to come out of a single collect.

    Each collect re-executes the plan on a distributed engine, and two executions
    need not agree on row order. So a caller needing features *and* labels calls
    `to_pandas` once and slices it; `column` is for the standalone array with no
    partner to fall out of step with. `select` narrows the table *without*
    collecting, so that one materialization stays cheap.
    """

    name: ClassVar[str]
    engine: ClassVar[str]          # "pandas" | "pyspark" — what `mlf data-backends` prints

    # ── read ──
    def read_table(self, path: str) -> Table: ...

    # ── inspect (nothing leaves the cluster) ──
    def columns(self, table: Table) -> tuple[str, ...]: ...
    def n_rows(self, table: Table) -> int: ...
    def dtypes(self, table: Table, columns: Sequence[str]) -> Mapping[str, str]:
        """Normalized **numpy-style** dtype names, identically across engines."""

    # ── reduce (still lazy) ──
    def sort_by(self, table: Table, column: str) -> Table:
        """**Stable** ascending sort. Ties broken deterministically — see Risk 1."""

    def select(self, table: Table, columns: Sequence[str]) -> Table:
        """Narrow the table. **Lazy** — pushes down into Parquet, does not collect."""

    # ── collect: the only two methods that move data to the driver ──
    def column(self, table: Table, name: str, *, dtype: str | None = None) -> np.ndarray:
        """One *standalone* column. Two calls are two collects — see the rule above."""

    def to_pandas(self, table: Table) -> Any:
        """The atomic collect. Everything sliced from the result is row-aligned."""
```

> **Built: `matrix()` was replaced by `select()` in Phase 2, and this is the most substantive correction in the document.** The original design had `build_tabular_bundle` call `matrix()` for features and `column()` three more times for the target, time and group columns — four independent collects whose results were then indexed together. On a distributed engine that is unsound: each collect re-executes the query plan, and pyspark explicitly documents `monotonically_increasing_id` — which `sort_by` uses for its tie-break — as *"non-deterministic because its result depends on partition IDs"*. Two executions of a sorted plan may therefore return different row orders, and the builder would pair feature rows with the wrong labels **silently**, which is the worst failure this layer can have.
>
> The fix makes the collect atomic: one `to_pandas`, sliced locally. That left `matrix()` with no consumer, so by the house rule it goes; `select()` takes its place as the *lazy* projection that keeps the single collect affordable. `column()` survives because `_cv_population`'s label vector genuinely stands alone.

### Every method earns its place

`core/types.py:211-213` states the house rule: *a capability without exactly one named consumer is a review failure.* Applied here — each method replaces a real line that exists today.

Consumers as **built** at the end of Phase 2:

| Method | Replaces | Consumers |
|---|---|---|
| `read_table` | `pd.read_parquet` / `pd.read_csv` | `read_table` shim; `build_tabular_bundle`; `_label_columns`; `_cv_population` |
| `columns` | `df.columns` membership / list-comp | `build_tabular_bundle` (target + split-column validation, **before the collect**); `_label_columns` |
| `n_rows` | `len(frame)` | `_cv_population` (feeds `Splitter.split(n)`); `_label_columns` (row-count check) |
| `dtypes` | `str(df[c].dtype)` | `build_tabular_bundle` → `FeatureSchema.dtypes` → bundle manifest → serving signature |
| `sort_by` | `frame.sort_values(col, kind="stable")` | `_label_columns` |
| `select` | `df[cols]` | `_label_columns` (2 of N columns, pushed down) |
| `column` | `frame[c].to_numpy()` | `_cv_population` — the one array with no alignment partner |
| `to_pandas` | identity under `local` | `read_table` shim; `contracts.py` (pandera); `build_tabular_bundle`; `_label_columns` |

`column` is deliberately not named `labels` — but note it ended Phase 2 with a *single* consumer, because every other array it might have served turned out to need a partner array in the same row order and therefore had to go through `to_pandas`. If a future change removes `_cv_population`'s use, `column` should go with it rather than linger.

### Deliberately excluded

- **`join` / `groupby` / `agg`.** The sketch says "load/filter/join/aggregate", but grep the data layer: it never joins and never aggregates. Adding them would be decoration.
- **`split()`.** `splitters.py` is already engine-agnostic (`n: int` in, numpy indices out), and `splitters.py:713-715` explicitly rejects registry ceremony there: *"splitters take no optional dependencies and need no capability metadata."* Splitting does not move.
- **`write_parquet` / `filter_notnull` / `drop_duplicates`.** Real methods with a real consumer — but that consumer (`spark_preprocess.py`) is not inside `build_bundle`. **Added in Phase 3**, together with it, never before: five methods (`filter_notnull`, `drop_all_null_rows`, `drop_duplicates`, `cast`, `write_parquet`), each with exactly one call site in `preprocess()`.

---

## 4. Registry wiring

### 4.1 `DataBackendSpec` — `core/plugins.py`

Add after `SourceSpec` (L197-217) and extend the `SpecT` bound at L220.

```python
@dataclass(frozen=True, slots=True)
class DataBackendSpec:
    """A registered data-processing engine, selected by ``data.backend``.

    ``factory`` is a zero-arg callable returning the
    :class:`~ml_framework.core.protocols.DataBackend`; it imports pyspark inside
    itself, so registering the spark backend costs a bare install nothing.

    **No ``capabilities`` field.** Every flag on :class:`Capabilities` is about a
    model or a fit loop (``needs_scaling``, ``produces_proba``); a
    default-constructed one here would be exactly the decoration that class's
    docstring forbids. ``engine`` is the one honest column, and
    ``mlf data-backends`` is its single named consumer.
    """

    name: str
    factory: Callable[[], Any]
    engine: str = ""
    requires: tuple[Requirement, ...] = ()
    description: str = ""


SpecT = TypeVar("SpecT", bound="ModelSpec | BackendSpec | SourceSpec | DataBackendSpec")
```

### 4.2 Registry + accessor — `core/registry.py`

Mirrors `get_backend` (L153-167) exactly.

```python
DATA_BACKENDS: PluginRegistry[DataBackendSpec] = PluginRegistry("data backend")


def register_data_backend(spec: DataBackendSpec, *, override: bool = False) -> DataBackendSpec:
    return DATA_BACKENDS.register(spec, override=override)


def get_data_backend(name: str) -> Any:
    """The instantiated :class:`DataBackend` for ``name``.

    Populates :data:`DATA_BACKENDS` first, for the same reason
    :func:`get_backend` does: registration is an import side effect, and callers
    here (``read_table``, reached from ``contracts.py`` and ``core.lit_data``) do
    not otherwise import the data package.

    ``params`` is ``data.backend_params`` verbatim, passed through unvalidated on
    purpose: validating it here would mean this function knowing every engine's
    knobs, which is the coupling the registry exists to remove. The backend
    rejects what it does not recognize.
    """
    import ml_framework.data.backends  # noqa: F401  (registration side effect)

    return DATA_BACKENDS.get(name).factory(**params)
```

> **Built:** `get_data_backend` takes `**params` and `DataBackendSpec.factory` is `Callable[..., Any]`, not the zero-arg `Callable[[], Any]` this document first specified. `BackendSpec`'s factory can be zero-arg because a training backend's params are validated in `fit()`; a data backend has no equivalent later hook, so `backend_params` has to arrive at construction.

The registry `kind` is `"data backend"` — with a space — because `PluginRegistry` interpolates it into prose. The payoff of the whole design is this sentence, produced for free by `check_requirements`:

```
data backend 'spark' requires pyspark>=3.5. Install it with: pip install 'ml-framework[mlops]'
```

### 4.3 The package — `src/ml_framework/data/backends/__init__.py`

A line-for-line analogue of `backends/__init__.py`.

```python
"""Data backends: one per data-processing **engine**, not per file format.

Registration is a spec plus a lazy factory, never an import of the engine. That
is what lets ``mlf data-backends`` list ``spark`` alongside ``local`` on a bare
install, and what keeps ``read_table`` — which every ingestion path calls — free
of pyspark.

    local   pandas, in-process       (read, sort, project, collect)
    spark   pyspark, distributed     (same, at cluster scale, then collects)

A data backend chooses how the table is read and reduced, not how the model is
trained: the framework collects a materialized bundle and trains on one node.
"""

from __future__ import annotations

import importlib
from typing import Any

# Three dots, not two: this module is `ml_framework.data.backends`, so `..` is
# `ml_framework.data` and `...` is the package root.
from ...core.plugins import DataBackendSpec
from ...core.registry import register_data_backend
from ...core.types import Requirement


def _lazy_factory(module: str, attr: str = "build_data_backend"):
    """Defer the import to call time, so registering never imports pyspark."""

    def _build(**params: Any) -> Any:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)(**params)

    return _build


# No `requires`: pandas is a BASE dependency and `local` is the default. Giving
# the default backend an optional requirement would make the framework's central
# guarantee — "a bare install trains" — conditional on an extra.
register_data_backend(
    DataBackendSpec(
        name="local",
        factory=_lazy_factory(".local"),
        engine="pandas",
        requires=(),
        description="In-process pandas: CSV/Parquet read, stable sort, column projection.",
    )
)

register_data_backend(
    DataBackendSpec(
        name="spark",
        factory=_lazy_factory(".spark"),
        engine="pyspark",
        requires=(Requirement("pyspark", extra="mlops", min_version="3.5"),),
        description="Distributed read/filter/sort; collects a materialized bundle to the driver.",
    )
)

__all__ = ["DataBackendSpec"]
```

### 4.4 The import-cycle hazard — read this before writing `local.py`

`core/__init__.py:91` lazily re-exports `read_table` from `.lit_data`, which imports `data.sources.tabular`. `builders.py:22-29` documents this exact trap.

Therefore:

- `data/backends/*` must import `...core.plugins` and `...core.registry` **as submodules** — never `from ...core import X`.
- `local.py` must **not** import `data.sources.tabular`, because tabular will call back into it.
- `PARQUET_REQUIREMENT`, which lived in `tabular.py`, **moves to `local.py`** with the pandas read it guards.

> **Built: the re-export was dropped, not kept.** This document originally said `tabular.py` re-imports `PARQUET_REQUIREMENT` "for compatibility". It does not. Once the constant moved, nothing in the repo imported it from the old path — `tests/unit/test_read_table.py` was re-pointed at `local` (the module that now *reads* it, which is the correct patch target anyway), and the `docs/PHASE_STATUS.md` reference is prose, not code. Ruff flagged the alias as an unused import and was right: a compatibility shim for a name with zero importers only ever costs. A comment at `tabular.py` records where it went.

---

## 5. Config surface

`DataConfig` (`schema.py:251`) gains two fields:

```python
    # Which engine reads and reduces the table. A plain `str`, not a `Literal`:
    # closing the set would put every third-party engine in this file, the same
    # reason `model.params` is not a discriminated union. An unknown name
    # surfaces as UnknownPluginError from `get_data_backend`, listing what is
    # registered.
    backend: str = "local"
    backend_params: dict[str, Any] = Field(default_factory=dict)
```

### Why `backend_params` is a new field, not `data.params`

Three reasons, decisive first:

1. **`data.params` is validated by the source with `extra="forbid"`.** `tabular.py:196` runs `TabularSourceParams.model_validate(dict(config.data.params))`, and that model is `{"frozen": True, "extra": "forbid"}` (`tabular.py:57`). Putting `shuffle_partitions` in `data.params` makes the source **reject the config**. The only fix would be punching a hole in the strictness that currently catches `imbalance_strategy: smoate` — trading a real guarantee for a naming convenience.
2. **One owner per params dict.** The three-tier rule (`schema.py:26-37`) *is* "each free-form dict has exactly one validator". The source owns `data.params`; the backend would be a second owner of the same dict.
3. **Orthogonality becomes checkable.** The same `data.params` must produce the same bundle under either engine — that is the conformance test's central assertion. Mixing engine knobs in makes the property unstatable.

### Keep the three-tier rule intact

Do **not** resolve the backend in `_resolve_plugin_params`. Add one row to the table at `schema.py:26-31`:

```
    model.params         here, at load time, via ``ModelSpec.params_model``
    fit.params           by the backend, in ``fit()``
    data.params          by the source, in its ``build_*_bundle``
    data.backend_params  by the data backend, at first use
```

and one clause to the rationale at L32-37: resolving a data backend would mean importing pandas or pyspark to validate a YAML file — the same argument, one more instance.

### `_CREATABLE_PREFIXES`

Add `"data.backend_params."` to the tuple at `schema.py:90-100`. `data.backend` itself is a typed field, so `--set data.backend=spark` already satisfies the "key must already exist" rule and needs nothing.

### Two free wins from existing machinery

- `_BundleCache._key` (`tune.py:549-560`) serializes `config.data.model_dump()`, so `backend` and `backend_params` land in the cache key automatically. Flipping the backend mid-search correctly invalidates. **No change needed.**
- `tests/conftest.py`'s `make_config(csv, task, **{"data.backend": "spark"})` works with no fixture change, because it routes through `with_overrides`.

Every existing `configs/*.yaml` keeps validating — the field is defaulted.

---

## 6. CLI

### `--data-backend` goes on `_add_config_args`, not `_add_data_args`

`_add_data_args` (`cli.py:83`) is the *synthesis* surface, also used by `mlf init`. The engine is an execution choice about **this run**, not a fact inferred from the file, so it belongs beside `--set` in `_add_config_args` (`cli.py:154`):

```python
    sub.add_argument(
        "--data-backend",
        default=None,
        metavar="NAME",
        help="Data-processing engine for this run: local (default) or spark",
    )
```

`default=None` preserves the "nobody said" vs "the user asked for local" distinction that `_apply_tune_args` documents at `cli.py:196-199`.

Applied in `_load_config` (`cli.py:98`) **after** the `args.set` block, honouring the documented chain at `cli.py:40` — *plugin defaults < synthesis < YAML < `--set` < explicit CLI flags*:

```python
    if getattr(args, "data_backend", None):
        cfg = cfg.with_overrides({"data.backend": args.data_backend})
```

### `mlf data-backends` as a new command, not a flag on `mlf backends`

1. `mlf backends` answers *"what fit-loop shapes exist"*. A flag that silently makes it answer a different question is worse than a second verb.
2. It appears in `mlf --help`; a flag on another command does not.
3. The dispatch at `cli.py:744-753` gets **smaller**, not larger — the ternary becomes a lookup:

```python
    if args.command in ("models", "backends", "data-backends"):
        import ml_framework.plugins  # noqa: F401

        from . import backends as _backends  # noqa: F401
        from .core.registry import BACKENDS, DATA_BACKENDS, MODELS
        from .data import backends as _data_backends  # noqa: F401

        registry = {
            "models": MODELS,
            "backends": BACKENDS,
            "data-backends": DATA_BACKENDS,
        }[args.command]
        return _print_plugins(registry, include_failed=args.all, show_detail=args.show)
```

That new `from .data import backends` import must not pull pandas or pyspark. It does not — `_lazy_factory` defers both — and §9 makes CI assert it.

### Four edits inside `_print_plugins`, three of which fix latent bugs

Adding a fourth registry is what makes these reachable; all three were already wrong for `mlf backends`.

| Was | Fix | Now at |
|---|---|---|
| `_second_column`: `return ",".join(sorted(spec.capabilities.accepts))` | add `if kind == "data backend": return spec.engine` — otherwise **`AttributeError`**, since `DataBackendSpec` has no `capabilities` | `cli.py:397` |
| `print("no models registered")` | `print(f"no {registry.kind}s registered")` — `mlf backends` on an empty registry said *"no models registered"* | `cli.py:357` |
| `verb = "models" if registry.kind == "model" else "backends"` | derive from `registry.kind` (`"data backend"` → `mlf data-backends --all`) | `cli.py:388` |
| `width = max(len(row["name"]) for row in rows)` | include the header: `max(*(len(r["name"]) for r in rows), len(label))` | `cli.py:363` |

> **Built: the fourth edit was not in the original plan.** `width` sized the name column on the names alone, which happened to be wide enough for `MODEL` and `BACKEND`. `DATA BACKEND` is 12 characters and longer than `local`/`spark`, so the header ran into the `READY` column and ragged-edged every row under it. Caught by running the command, not by a test.

Plus a `data backend` branch in `_format_detail` (`cli.py:411`) printing `engine: {spec.engine}` — there are no capability flags to print.

Expected output on a dev install with no pyspark:

```
$ mlf data-backends
DATA BACKEND  READY  ENGINE          DESCRIPTION
local         yes    pandas          In-process pandas: CSV/Parquet read, stable sort, ...
polars        NO     polars          In-process Polars: multithreaded CSV/Parquet parse ...
                     polars is not installed
                     fix: pip install 'ml-framework[fast]'
spark         NO     pyspark         Distributed read/filter/sort; collects a materialized ...
                     pyspark is not installed
                     fix: pip install 'ml-framework[mlops]'
```

Every engine but `local` is listed-but-unavailable on a bare install, each naming its own extra. That one screen is the whole design working: three engines registered, none imported.

---

## 7. Why the local backend stays on pandas

The sketch says `LocalBackend → Polars`. **Recommendation: keep pandas. Register Polars later as a third peer, not as a migration.**

Four arguments, strongest first:

1. **`read_table -> pd.DataFrame` is a load-bearing public contract, and one consumer is pandas-only.** It is re-exported from `core/__init__.py` (L79, L91, L123), `data/sources/__init__.py` and `core/lit_data.py:42`; it has its own test file `tests/unit/test_read_table.py`; `ml_framework_architecture_plan.md:537` records it as *"re-exported from `core/__init__` with unchanged signatures"*. And `pipeline/contracts.py:57` hands the result straight to **pandera** `schema.validate(df)` — a Polars frame breaks it outright. Changing the return type is a breaking change to a documented promise, for a benefit nobody has measured.

2. **pandas is a BASE dependency; Polars would not be.** Either Polars becomes a second base dep — contradicting the extras discipline the entire plugin system exists to enforce — or it goes in an extra, in which case the **default** data backend acquires a `requires`, becomes conditionally unavailable, and the `gbdt-no-torch` CI job (the standing bare-install guardrail) has to install it. That is a real regression in the framework's central guarantee, traded for a faster CSV parse.

3. ~~**The profile does not support the claim.**~~ **Refuted by measurement — see below.** The argument was that a Polars read hands the identical numpy matrix to identical downstream code, so "the win is confined to parse time, and it is unmeasured". The first half is true and the conclusion drawn from it was wrong.

4. **Reject "Polars quietly inside LocalBackend" specifically.** Making `read_table`'s return type depend on whether Polars happens to be installed is worse than either a migration or the status quo: it is a silent behaviour change keyed on the environment — precisely the failure mode `Requirement.is_installed()` and the whole availability design were built to eliminate.

**The diagram is satisfied structurally anyway.** `Framework → DataBackend → {Local, Spark}` is exactly what this delivers. Polars later becomes:

```python
DataBackendSpec(
    name="polars",
    factory=_lazy_factory(".polars"),
    engine="polars",
    requires=(Requirement("polars", extra="fast"),),
    description="In-process Polars: multithreaded read and projection.",
)
```

— a *peer*, selectable per run by the same `--data-backend` flag, with zero changes anywhere else. That is a strictly better outcome than a migration: it makes Polars a **choice** rather than a decree, and `read_table` never moves.

### The benchmark this was gated on — run, in Phase 4

`benchmarks/data_backends.py`. Median of 5 runs after a discarded warm-up; this machine (Windows, Python 3.14). Absolute numbers are machine-specific — only the ratios travel.

| shape | operation | pandas | polars | speedup |
|---|---|---|---|---|
| 50 000 × 20 | parse CSV | 381 ms | 20 ms | **19.5×** |
| 50 000 × 20 | parse Parquet | 23 ms | 16 ms | 1.5× |
| 50 000 × 20 | **`build_bundle`** | 537 ms | 173 ms | **3.1×** |
| 500 000 × 20 | parse CSV | 3 481 ms | 185 ms | **18.8×** |
| 500 000 × 20 | parse Parquet | 131 ms | 104 ms | 1.3× |
| 500 000 × 20 | **`build_bundle`** | 4 957 ms | 1 764 ms | **2.8×** |
| 200 000 × 50 | parse CSV | 3 765 ms | 177 ms | **21.2×** |
| 200 000 × 50 | parse Parquet | 145 ms | 116 ms | 1.3× |
| 200 000 × 50 | **`build_bundle`** | 4 948 ms | 1 353 ms | **3.7×** |

**Argument 3 was wrong, and the gate was worth having precisely because it caught that.** The win *is* confined to the parse, exactly as argued — but the CSV parse **dominates** `build_bundle`, so confining it there does not make it small. End to end the whole bundle build is **2.8–3.7× faster**, which is not the "faster CSV parse" the section dismissed.

Two honest qualifications:

- **Parquet barely moves** (1.3–1.5×). Anything already reading Parquet — which includes everything downstream of the `preprocess` stage — gains almost nothing. The 19× is a *CSV* number.
- **This is one machine.** pandas' CSV reader here is slower than published figures, so the ratio may be flattering. That is what the committed benchmark is for.

**Arguments 1, 2 and 4 stand, and they are the ones that shaped the design.** They never said "don't build it" — they said *don't make it the default, don't change `read_table`'s return type, and don't do it implicitly*. Phase 4 honours all three: pandas remains the base dependency and the default, `read_table -> pd.DataFrame` is unchanged, and Polars is reached only by an explicit `--data-backend polars`.

---

## 8. Execution flow

### Phase 0 — plumbing, zero behaviour change

**Files**

| Action | Path |
|---|---|
| edit | `core/plugins.py` — `DataBackendSpec`, extend `SpecT` (L220) |
| edit | `core/protocols.py` — `Table` alias, `DataBackend` protocol |
| edit | `core/registry.py` — `DATA_BACKENDS`, `register_data_backend`, `get_data_backend` |
| new | `data/backends/__init__.py`, `data/backends/local.py` |
| edit | `data/sources/tabular.py` — `read_table` body moves to `local.py`; `PARQUET_REQUIREMENT` moves with it |

**The keystone edit.** `read_table` keeps its signature *and* its return type:

```python
def read_table(path: str, *, backend: str = "local") -> pd.DataFrame:
    """Read a tabular dataset as CSV or Parquet.

    Always returns a **pandas** DataFrame, under every backend: this is public
    API (re-exported from `core`, consumed by `pipeline/contracts.py` and
    pandera). `backend` chooses who does the reading — under `spark` the read is
    distributed and this function is the collect point.
    """
    from ...core.registry import get_data_backend

    engine = get_data_backend(backend)
    return engine.to_pandas(engine.read_table(path))
```

Under `local`, `to_pandas` is the identity — so this is byte-for-byte the old behaviour plus one dict lookup. All six call sites, the three public re-exports and `contracts.py:57` are untouched and pass unchanged.

**Exit criterion:** full suite green with **at most** incidental test edits; `mlf data-backends` lists `local` (ready) and `spark` (NO + pip hint).

> **Built: one test edit, against the original "no test edits" criterion.** Two tests in `tests/unit/test_read_table.py` monkeypatched `tabular.PARQUET_REQUIREMENT`, and the constant moved to the module that reads it. The patch target is re-pointed at `local`; the assertions are unchanged, and the new target is the more correct one — patching where a constant is *read* rather than where it happens to be re-exported. The criterion was too strong: a test that reaches in via a module attribute is coupled to layout, and this refactor moves layout by design.

---

### Phase 1 — the MVP that delivers the promise

| Action | Path | Change |
|---|---|---|
| edit | `config/schema.py` | `DataConfig.backend`, `.backend_params`; `_CREATABLE_PREFIXES`; docstring row |
| edit | `cli.py` | `--data-backend`; `_load_config` application; `data-backends` command; 4 `_print_plugins` edits |
| new | `data/backends/spark.py` | implements the protocol |
| edit | 3 call sites | thread `backend=config.data.backend`: `tabular.py:196`, `builders.py:240`, `builders.py:277` |
| edit | `pipeline/contracts.py:53` | `validate_file(..., backend: str = "local")` — standalone CLI, no config |

> **Built: three call sites, not five.** This document originally listed `timeseries.py:156` and `text.py:154` as well. Both are unreachable with a non-local backend — the guard below refuses those kinds by name — so threading the parameter there would have been plumbing that can only ever carry the value `"local"`. They read through the default and the guard keeps them honest. They get threaded in Phase 2, when they become backend-aware and it means something.

Plus a guard, mirroring the `_CV_BUILDERS` refusal idiom at `builders.py:110-114`:

```python
_BACKEND_AWARE_KINDS = frozenset({"tabular"})


def _check_backend_supported(config) -> None:
    backend = config.data.backend
    if backend != "local" and config.data.kind not in _BACKEND_AWARE_KINDS:
        raise FrameworkError(
            f"data.backend '{backend}' is implemented for "
            f"{sorted(_BACKEND_AWARE_KINDS)} data; got kind '{config.data.kind}'. "
            f"Use data.backend: local."
        )
```

Refusing by name beats silently collecting the whole table and pretending it was distributed.

> **Built: the guard is called from `build_cv_bundles` too, not only `build_bundle`.** The original plan put it in `build_bundle` alone. That leaks: `build_cv_bundles` calls the source builders **directly** (`_CV_BUILDERS[kind](config, indices=...)`) and never goes through `build_bundle`, so every fold of a cross-validated run would have slipped past the refusal the single-holdout path enforces. Extracted to `_check_backend_supported` and called from both. `test_the_cross_validation_path_refuses_it_too` covers it.

**The three pipeline call sites are not touched.** `train.py:126`, `tune.py:568` and `select.py:500` all call `build_bundle(config)`; the backend rides inside the config. `_BundleCache`'s DI seam is untouched. `gbdt-no-torch` is untouched — `local` has no `requires`, and registration imports nothing.

**End state:** `mlf train -c cfg.yaml --data-backend spark` reads through Spark and collects. Roughly **400 new lines and ~30 changed**. This is the whole user-facing promise.

---

### Phase 2 — push real work into the engine

**Built.** Ordered by (distributed win ÷ blast radius):

1. **`builders._cv_population`** — biggest win, smallest function. Becomes `t = eng.read_table(p); return eng.n_rows(t), eng.column(t, target, dtype=...)`. The feature matrix is never collected, and that is now *verified* rather than asserted: tracing every attribute the engine receives during a fold-planning run yields exactly `['read_table', 'n_rows', 'column']`.
2. **`builders._label_columns`** — `eng.columns` for the existence check, `eng.n_rows` for the count assertion, then `eng.select(eng.sort_by(table, time_col), [col, time_col])` and **one** `eng.to_pandas`. The pair is sliced from that single frame.
3. **`tabular.build_tabular_bundle`** — `eng.columns` for target/split validation *before* any collect, `eng.dtypes` for the schema, then one `eng.to_pandas`. Everything after slices that frame exactly as before.

A shared helper resolves the engine so no source has to remember to forward `backend_params`:

```python
def engine_for(config: Any) -> Any:
    """The DataBackend this run selected, with its params applied."""
    from ...core.registry import get_data_backend

    return get_data_backend(config.data.backend, **dict(config.data.backend_params))
```

> **Built: steps 2 and 3 do *not* use `eng.column` per array, as this document originally specified.** That was the misalignment bug — see §3. `_label_columns` needs its label-end and observation-time arrays paired row-for-row (`label_spans` reads them together), and `build_tabular_bundle` needs `x`/`y`/`time`/`groups` paired. Each therefore takes exactly one collect.

> **Built: `build_tabular_bundle` gained less than expected, and gained something else instead.** Because it needs every column (§2), there is no projection to push down and its collect is a plain `to_pandas` — Phase 2 does not make its data movement smaller. What it does buy is **failing on the schema**: `data.target 'labl' not in the table's columns` now arrives from a metadata lookup rather than after materializing a table that was never going to work.

`timeseries`, `text` and `image` stay on `to_pandas()` / numpy. `timeseries._series_bundle` uses `exog.iloc[rows]` and `pd.to_datetime`, and forecasting is inherently per-series — porting it buys nothing. The Phase 1 guard already refuses non-local for them by name, so the "they get threaded in Phase 2" note under Phase 1 did **not** come due: they remain unthreaded, deliberately.

---

### Phase 3 — fold the Spark fork

**Built.** `preprocess()` is now engine-agnostic: five methods joined the protocol *with* their one consumer — `filter_notnull`, `drop_all_null_rows`, `drop_duplicates`, `cast`, `write_parquet`.

```python
engine = get_data_backend(backend)
table = engine.read_table(input_path)
table = engine.filter_notnull(table, target_col)
if dropna:
    table = engine.drop_all_null_rows(table, [c for c in engine.columns(table) if c != target_col])
if deduplicate:
    table = engine.drop_duplicates(table)
table = engine.cast(table, target_col, "float64")
engine.write_parquet(table, output_path)
```

**Payoff, realized:** the repo carries one Spark codebase instead of two, with one session configuration. And the same cleaning runs under `local` with no JVM — which turned the stage from `py_compile`-only into **100% covered**, because its logic is now testable everywhere and only the engine underneath varies.

**Constraint honoured.** `--data-backend` is additive and defaults to `spark`, so `python -m ml_framework.pipeline.spark_preprocess --input X --output Y --target-col Z` — the exact string in both `dvc.yaml` and `ml_pipeline.py:159` — is unchanged. A test asserts that command still works and that the default is still `spark`; making `local` the default would have quietly changed what the DVC/Airflow pipeline *does*.

> **Built: `write_parquet` must produce a directory on every backend, including `local`.** Not in the plan, and it would have been a silent breakage. `read_table` distinguishes Parquet from CSV by inspecting the path, and the DVC stage declares `outs: data/processed` — a *directory*. Had `local` written a bare suffix-less file there, the next read would have fallen through to `pd.read_csv` on Parquet bytes. `local` therefore mkdirs and writes `part-0.parquet`, mirroring Spark's `coalesce(1).write.parquet(dir)` shape exactly.

> **Built: `local` has to emulate `mode("overwrite")`.** Spark clears the output directory; pandas does not. Without deleting stale `*.parquet` first, a rerun over fewer rows would leave the previous run's part-file in place and the next `read_table` would silently union two runs. Covered by `test_rerunning_over_fewer_rows_does_not_leave_the_previous_output_behind`.

> **Built: `drop_all_null_rows`, not `drop_nulls`.** The plan listed "`filter_notnull`". The stage actually needs two different null policies — *any* null in the target drops the row, but only a row null across *every* feature does — and one method named for the general idea would have hidden that. Two methods, two names, two consumers.

---

### Phase 4 — the Polars peer

**Built — the first item. The second is refused, on evidence.**

**Polars as a third peer backend.** `data/backends/polars_backend.py` implements all fifteen protocol members; a `fast` extra carries `polars>=1.0` and `pyarrow` (the latter because `to_pandas` — the handoff every backend goes through — routes via Arrow). Registration is the same spec-plus-lazy-factory as the others, so a bare install still lists it and still refuses it with a pip line.

The gate §7 put on it is now **resolved by measurement**, not waived: `build_bundle` is **2.8–3.7× faster** end to end. §7 records the table and marks its own third argument refuted.

Two properties made it cheap, and both are consequences of earlier phases:

- **The protocol was already stable.** Phase 2 settled the collect rule and Phase 3 added the clean/write half, so Polars implemented a fixed contract rather than negotiating one. It passed the entire conformance suite on the first run — including the sort-stability and dtype-normalization tests, the two most likely to catch an engine out.
- **`local` is the oracle.** Every Polars assertion is *"equals what pandas produced"*, so the suite needed a fixture parameter, not new tests.

**The one genuinely new thing it bought:** Polars needs no JVM, so §9.3's cross-engine bundle-equality tests **now execute on an ordinary dev machine** instead of skipping. That converts the feature's central claim — same bundle, different engine — from believed to tested.

> **Built: two real bugs, found by running the stage rather than reading it.** Both were pre-existing and neither was caught by the suite, because no test had used a table whose only column is the target.
>
> 1. **`local` silently produced an empty output.** `pandas.dropna(how="all", subset=[])` drops *every* row — vacuously, all zero of the named columns are null in each. `preprocess` hits this whenever `feature_cols` comes out empty. This was a **Phase 3 data-loss bug** that shipped, and Phase 4 only found it because a second engine raised where pandas silently succeeded.
> 2. **The engines disagreed on a quoted empty CSV field.** `pandas.to_csv` writes a lone missing value in a single-column frame as `""`; pandas reads that back as NaN, polars read it as the *string* `""` and inferred the column as String, so the target cast failed. Fixed by `null_values=[""]` on the polars reader — `local` is the oracle, so the reader is aligned to it rather than the divergence being documented and left in.
>
> Both now have regression tests that run under **every** in-process engine.

**The `frame` payload straight to `gbdt` — refused, with the evidence.** `gbdt`'s `Capabilities.accepts` includes `"frame"`, which made this look like a small win. It is not available: `GBDTBackend.fit` calls `model.fit(_as_matrix(x_train), ...)`, and `_as_matrix` passes a pandas frame through but sends everything else to `np.asarray(inputs, dtype="float32")`. A Spark DataFrame dies there. Skipping the collect for tree models therefore needs a Spark-native trainer (SynapseML, `xgboost4j-spark`) — which is **distributed training**, a stated non-goal (§11) on a different registry (§2, "the door left open"). `accepts` including `"frame"` describes what the *payload vocabulary* permits, not what this backend implements.

---

## 9. Testing

Constraints, all real: 80% coverage floor (`[tool.coverage.report] fail_under = 80` and `--cov-fail-under=80` in CI); markers `unit` / `integration` / `serving` only; `pyspark` lives in the `mlops` extra but **not** `dev`; and **no JVM anywhere in the default matrix, or on this machine** — `docs/PHASE_STATUS.md` records *"pyspark is installed but cannot run: no JVM on PATH."*

That last constraint is what §9.6's `spark-contract` job exists to lift, and it lifts it *only there*: everything below is about staying honest in the jobs that still have no JVM.

### 9.1 The right skip gate

`importorskip("pyspark")` is **not enough** — pyspark imports fine and then dies at `getOrCreate()`. Encode the distinction in `tests/unit/test_data_backends.py`:

```python
def _spark_runnable() -> bool:
    """pyspark importable AND a JVM on PATH. The second half is the one that
    matters: pyspark imports fine without Java and fails at getOrCreate()."""
    import importlib.util
    import os
    import shutil

    if importlib.util.find_spec("pyspark") is None:
        return False
    return bool(shutil.which("java") or os.environ.get("JAVA_HOME"))


requires_spark = pytest.mark.skipif(not _spark_runnable(), reason="pyspark needs a JVM on PATH")
```

> **Built: it lives in the test module, not `tests/conftest.py`.** The plan put it in the root conftest for sharing, but the test tree has no `__init__.py`, so `from ..conftest import requires_spark` fails with *"attempted relative import with no known parent package"* — conftest supplies **fixtures** by injection, not importable module-level constants, and a `skipif` needs a value at decoration time. It sits in the one module that uses it; promoting it is the right move at the moment a second module needs it, not before.

### 9.2 Conformance suite — `tests/unit/test_data_backends.py`

Parameterized over whichever backends can actually run. **`local` is the oracle**: every Spark assertion is *"equals what local produced."*

```python
@pytest.fixture(params=["local", pytest.param("spark", marks=requires_spark)])
def engine(request):
    return get_data_backend(request.param)
```

One test per protocol method against the existing `tabular_csv` / `binary_csv` fixtures, plus:

- a directory of Parquet part-files (mirrors `tests/unit/test_read_table.py:34`);
- **a table with duplicate values in `time_col`** — the sort-stability test, and the most important one in the file (Risk 1);
- `dtypes()` returning identical normalized strings on both engines — this protects `FeatureSchema.dtypes`, which reaches the serving signature.

Names follow the repo convention:
`test_sorting_by_a_column_with_duplicate_keys_keeps_the_original_relative_order`.

### 9.3 The one test that proves the feature

`tests/integration/`, marked `integration` + `requires_spark`: build a `DataBundle` from one CSV under `local` and under `spark` with the same seed, then assert `np.allclose(train.x)`, `array_equal(train.y)`, and equality of `schema`, `input_dim`, `output_dim`.

Without this, "choose an implementation per run" is two products, not one choice.

> **Built in Phase 2** — `tests/integration/test_data_backend_equivalence.py`, three tests: identical `x`/`y` across all three splits plus `input_dim`/`output_dim`; identical `FeatureSchema` (so no artifact records which engine built it); and identical `_cv_population` output, since fold planning is the one place Spark works without collecting features.
>
> It skips without a JVM, so its **assertions** were verified separately by running the same bodies local-vs-local — a test whose only evidence is a skip proves nothing about whether it would pass. `spark-contract` runs it for real.

### 9.4 Meaningful Spark coverage with no JVM

Three mechanisms:

1. **Registry-level tests that need no pyspark** — these cover the half users actually hit, and run in the default job:
   - `DATA_BACKENDS.names() == ["local", "spark"]`
   - `DATA_BACKENDS.get_spec("spark")` succeeds — listing works on a bare install
   - `DATA_BACKENDS.is_available("spark") is False`
   - `get_data_backend("spark")` raises `MissingExtraError` containing `ml-framework[mlops]`
   - `"pandas" not in sys.modules` and `"pyspark" not in sys.modules` immediately after `import ml_framework.data.backends`

   Same assertion shape as the `mlp` / `MissingExtraError` block already in `gbdt-no-torch`.

2. **A stub-session translation test** — `tests/unit/test_spark_backend_translation.py`, **32 cases from 16 functions** after Phase 3. `spark.py` is written so the session comes from exactly one seam (`def _session(self)`, the only pyspark import in the class), so a `SimpleNamespace` stub reaches everything else. This is honestly a **translation** test, not a correctness test — the module docstring says so — and it pins the parquet-vs-csv branch, the dtype normalization in *both* directions, the collect guard, `select`'s laziness, the `coalesce(1)`/overwrite write, and the fact that the tie-break bookkeeping column never surfaces.

   The sharpest ones are structural rather than incidental. `test_a_collect_over_the_ceiling_is_refused_before_it_is_attempted`: the stub table has **no `toPandas` attribute at all**, so a guard that let the call through fails with `AttributeError` instead of passing. `test_select_projects_without_collecting`: same trick, so a `select` that materialized could not pass.

3. **`mlops-validate` grows two lines**: `py_compile` for `data/backends/spark.py` and `data/backends/local.py`, beside the existing `spark_preprocess.py` compile.

4. **Phase 3 made the preprocessing stage testable outright** — `tests/unit/test_preprocess_stage.py`, 8 tests. Because `preprocess()` no longer speaks pyspark, its *logic* (which rows survive, the target cast, the on-disk shape, overwrite semantics, the DVC/Airflow command contract) runs under `local` everywhere. This is the strongest of the four mechanisms: it is not a stub or a proxy, it is the real code path with a different engine underneath.

**Coverage arithmetic — measured, not estimated.** `spark.py` came in at 64 statements after Phase 1 and 78 after Phase 3. With only the JVM-gated tests it sat at **0%**; the stub tests took it to **81%**, and the 15 remaining lines are exactly the ones needing a live session. `pipeline/spark_preprocess.py` went the other way entirely — from `py_compile`-only to **100%**. Full-suite coverage is **90.2%** against the 80% floor, so the drag never came close to biting. `# pragma: no cover` is on the `_session()` body **only**, never on the translation logic — that is exactly the code the stub tests exist to reach.

### 9.5 Extend `gbdt-no-torch` (~8 lines)

Add the four registry assertions from 9.4(1) to the *"Unavailable plugins are listed, not hidden"* step, plus `assert "pyspark" not in sys.modules` after the train step, directly beside the existing `assert "torch" not in sys.modules`.

That is what proves registering a Spark backend costs a bare install nothing — the same guardrail, one axis wider.

### 9.6 The 4th CI job — the only way to close the gap for real

**Built.** This is the difference between *"we believe the Spark path works"* and *"CI proved it"*: everywhere else in `ci.yml`, pyspark is either absent (`gbdt-no-torch`) or present without a JVM (the test matrix), so `sort_by`'s tie-break and the two collects never execute.

```yaml
  spark-contract:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install with the mlops extra
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,mlops]"
      - name: Refuse to pass by skipping
        run: |
          python - <<'PY'
          import importlib.util, shutil, sys
          if importlib.util.find_spec("pyspark") is None:
              sys.exit("pyspark did not install; the conformance suite would skip")
          java = shutil.which("java")
          if not java:
              sys.exit("no JVM on PATH; the conformance suite would skip")
          print("java on PATH:", java)
          from pyspark.sql import SparkSession
          spark = SparkSession.builder.appName("ci-probe").getOrCreate()
          print("spark session:", spark.version)
          spark.stop()
          PY
      - name: Conformance suite against a real Spark session
        run: |
          pytest -p no:cacheprovider -v -rs \
            tests/unit/test_data_backends.py \
            tests/unit/test_spark_backend_translation.py \
            tests/unit/test_data_backend_selection.py \
            tests/unit/test_preprocess_stage.py \
            tests/integration/test_data_backend_equivalence.py
      - name: The preprocessing stage runs on a real Spark session
        # Its unit tests pin the cleaning logic under `local`; this is the only
        # place the same code path executes against pyspark, which is the engine
        # the DVC stage and the Airflow DAG actually use.
        run: |
          python - <<'PY'
          ... build a messy CSV, run `main([...])` with the default engine,
          ... assert 6 rows in -> 3 rows out and label dtype float64
          PY
```

> **Built: the "Refuse to pass by skipping" step was not in the original sketch, and it is the step that makes the job worth having.** Every Spark test is `skipif`-gated on *(pyspark importable AND java on PATH)*. If `setup-java` silently failed, all of them would skip and the job would report **green having proven nothing** — reproducing, inside the job added to close the gap, the exact gap it exists to close. The probe asserts the gate is open first, and actually builds a `SparkSession` so a JVM that installs but cannot start is caught here rather than as a confusing skip downstream.
>
> The draft also selected tests with `-k data_backend`, which silently **misses** `test_spark_backend_translation.py`. Naming the files is less clever and cannot drift — and the list grew to five as later phases added suites, which a `-k` pattern would have silently failed to pick up.
>
> **Built in Phase 3: the separate preprocess step.** `test_preprocess_stage.py` runs the stage under `local`, so adding the file to the pytest list proves the *logic* but never touches pyspark. The extra step invokes `main()` with the default engine, which is the only place in CI the DVC/Airflow code path executes against a real Spark session.

---

## 10. Risks

Worst first.

### 1. Sort stability — the single most likely correctness bug

`_label_columns` (`builders.py:250`) relies on `sort_values(kind="stable")` so that the positions the splitter computes line up with the rows it splits; its own docstring says so at L219-220. `timeseries.py:167` does the same.

Spark's `orderBy` gives a total order but does **not** resolve ties deterministically across partitions or runs. With duplicate timestamps, local and Spark yield different row orders → different folds → different scores, **silently**.

**Mitigation:** `sort_by` must break ties deterministically (`zipWithIndex` before sorting, or a secondary key), and the conformance suite must include a duplicate-key table. Treat this as a **blocking** acceptance criterion for `spark.py`.

### 2. `to_pandas()` / `column()` is a loaded gun

The one call that can turn a distributed run into an OOM-killed driver with no traceback.

**Mitigation:** `max_collect_rows` in `data.backend_params` (default ~5e6), checked **before** `.toPandas()`, raising a `FrameworkError` that names the actual row count and the knob that raises it. Failing loudly beats a SIGKILL.

### 3. dtype string drift

`FeatureSchema.dtypes` was `{c: str(df[c].dtype)}` and is now `dict(engine.dtypes(...))` (`tabular.py:280`) — `"int64"` / `"float64"` under pandas, `"bigint"` / `"double"` natively under Spark. It reaches the bundle manifest and the serving signature, so flipping the backend would silently change the artifact and could make a serving-side column check reject valid input.

**Mitigation:** `dtypes()` returns normalized numpy-style names on every backend, pinned by a cross-engine equality assertion.

### 3b. Separate collects can disagree on row order — **found while building Phase 2**

Not in the original risk list, and it belonged near the top. Every method that collects re-executes the query plan; two executions are not contractually order-equivalent, and `sort_by`'s tie-break uses `monotonically_increasing_id`, which pyspark documents as *"non-deterministic because its result depends on partition IDs"*. The design's `matrix()` + three `column()` calls would therefore have indexed four independently-ordered results together — pairing feature rows with the wrong labels, with no error and no symptom except a worse model.

**Mitigation:** structural, not defensive. The protocol now permits exactly two collecting methods, and the rule *"arrays that must line up come out of one collect"* is stated in the `DataBackend` docstring. `matrix()` was removed rather than fixed, because its very existence invited a second collect. Covered by `test_arrays_that_must_line_up_come_from_a_single_collect`.

### 4. Row order after a Spark read is not guaranteed

Parquet part-file ordering is arbitrary. `RandomSplitter` is seeded, but it seeds a permutation of *positions* — a different underlying row order means a different split for the same seed.

**Mitigation:** document that reproducibility is guaranteed *within* a backend, not across; record `data.backend` in the run manifest so a rerun can be pinned. Where an explicit `time_col` sort exists, order is defined.

### 5. SparkSession lifetime

`getOrCreate()` returns a process-global singleton. `mlf tune` (via `_BundleCache`) and `mlf select` build many bundles in one process, so the session must be created once and reused, and `spark.stop()` must **not** be called per bundle — which is exactly what `spark_preprocess.py:76` does, correctly, for a one-shot job.

**Mitigation:** per-instance lazy session in `data/backends/spark.py`, no `stop()`, with a comment naming the deliberate divergence. (Per-instance rather than module-global: `getOrCreate()` is already a process-wide singleton underneath, so caching on the instance reuses the same JVM without adding a second layer of global state that tests would have to reset.)

### 6. CI can never prove the Spark path in the default matrix

**Closed** by the `spark-contract` job (§9.6) — with the conformance suite, the stub translation tests and `py_compile` as defence in depth. The residual risk is now that the job itself degrades into skipping, which is what its "Refuse to pass by skipping" step exists to prevent.

### 7. Two Spark codebases until Phase 3

**Closed.** `spark_preprocess.py` no longer imports pyspark; it runs on the protocol, so there is one session configuration. The temporary state lasted two phases, as scheduled.

### 8. Forward-incompatible configs

`ExperimentConfig` is `extra="forbid"`, so a YAML containing `data.backend` is *rejected* by an older install. Additive, so it is a one-way concern — but worth a line in the release notes.

---

## 11. Non-goals

State these in the module docstring so nobody infers otherwise.

- **No distributed training.** `lightning` / `gbdt` / `forecast` all fit on one node. Spark's job ends at the collect.
- **No out-of-core splitting.** `splitters.py` takes `n: int` and returns numpy index arrays; `x[parts.train]` needs `x` in RAM. The label vector, the index vector and the feature matrix must fit on the driver — always.
- **No streaming or incremental training.** `DataBundle` is a frozen snapshot.
- **No lazy `DataBundle`.** `Split.x` *could* hold a Spark DataFrame under `payload="frame"`, and `gbdt.accepts` even includes `"frame"` — but `gbdt.fit` expects pandas/numpy. Do not ship a payload nothing consumes.
- **Not every data kind.** Only `tabular` is backend-aware. `image` is torchvision; `text` and `timeseries` stay on `to_pandas()`. The Phase 1 guard refuses the rest by name.
- ~~**No Polars.**~~ **Built in Phase 4** as a peer, after the benchmark §7 gated it on came back 2.8–3.7× — see §7. pandas remains the base dependency and the default; Polars is opt-in per run.
- **Polars is not the default, and `read_table` still returns pandas.** The non-goal that survives is the *migration*, not the engine: `read_table -> pd.DataFrame` is public API and `pipeline/contracts.py` hands it to pandera.

---

## 12. Files at a glance

**New**

```
src/ml_framework/data/backends/__init__.py       registration, lazy factories, engine_for (18 stmts, 100%)
src/ml_framework/data/backends/local.py          pandas; read_table's body lives here     (56 stmts, 100%)
src/ml_framework/data/backends/polars_backend.py polars; peer of local                    (76 stmts, 100%)
src/ml_framework/data/backends/spark.py          pyspark; lazy session, collect guard     (78 stmts,  81%)
benchmarks/data_backends.py                      resolves §7's gate; not a pytest module
tests/unit/test_data_backends.py                 conformance suite, local as oracle
tests/unit/test_data_backend_selection.py        config + CLI + the refusal paths
tests/unit/test_spark_backend_translation.py     what spark.py says to Spark, no JVM
tests/unit/test_polars_backend.py                dtype tables + refusals with no oracle
tests/unit/test_preprocess_stage.py              the stage, over every in-process engine
tests/integration/test_data_backend_equivalence.py  one CSV, every engine, one bundle (§9.3)
```

Named `polars_backend.py`, not `polars.py`: a module shadowing the library it wraps is legal under Python 3's absolute imports but misleads every reader and some tooling. The registered backend is still `polars`.

`pipeline/spark_preprocess.py` went from `py_compile`-only to **100% covered** — the clearest single measure of what Phase 3 bought.

> **Built: three test files more than planned.** §12 originally listed only the conformance suite. The selection path (config field, `--set`, CLI precedence, both guards, the cache key) is a different concern from protocol conformance and belongs in its own file; the translation tests are what took `spark.py` from 0% to 81% without a JVM; and Phase 3's stage tests exist because `preprocess()` became testable at all.

**Modified**

| File | Change |
|---|---|
| `core/plugins.py` | P0: `DataBackendSpec`; extend `SpecT` (L220) |
| `core/protocols.py` | P0: `Table` alias, `DataBackend` protocol · P2: `matrix()` → lazy `select()`, the atomic-collect rule in the docstring · P3: the five clean/write methods |
| `core/registry.py` | P0: `DATA_BACKENDS`; `register_data_backend`; `get_data_backend(name, **params)` |
| `config/schema.py` | P1: `DataConfig.backend` / `.backend_params` (L251); `_CREATABLE_PREFIXES` (L90); docstring (L26-37) |
| `cli.py` | P1: `--data-backend` (L177, in `_add_config_args` L160); `_load_config` (L98); dispatch (L781); `_print_plugins` (L357/363/388/397/411) |
| `data/sources/tabular.py` | P0: `read_table` becomes the shim · P2: `build_tabular_bundle` validates on the schema, then one collect |
| `data/builders.py` | P1: `_check_backend_supported` from `build_bundle` **and** `build_cv_bundles` · P2: `_label_columns` and `_cv_population` on the engine |
| `data/backends/__init__.py` | P2: `engine_for(config)` — one place resolves backend + params |
| `pipeline/contracts.py` | P1: `validate_file(..., backend="local")` + a `--data-backend` flag on its standalone CLI |
| `pipeline/spark_preprocess.py` | P3: no longer imports pyspark; runs on the protocol, `--data-backend` defaults to `spark` |
| `tests/unit/test_read_table.py` | P0: monkeypatch target re-pointed from `tabular` to `local` |
| `.github/workflows/ci.yml` | P1: `gbdt-no-torch` assertions, `mlops-validate` py_compile, **`spark-contract` job** · P3: the real-session preprocess step |
| `docs/PHASE_STATUS.md` | **P12 row + section written**, test baseline moved to 880/17 |

**Untouched — this is the point**

```
pipeline/train.py:126      bundle = build_bundle(config)
pipeline/tune.py:568       _BundleCache -> build_bundle(config)
pipeline/select.py:500     build_cv_bundles(config)
data/splitters.py          engine-agnostic already
data/types.py              the handoff contract does not move
```

---

## 13. What the build changed about this plan

All five phases are implemented. Sixteen places where contact with the code corrected the design — each is marked with a **Built:** note in context above. The most consequential are #8 (a silent label-misalignment bug the protocol would have shipped) and #14–15 (a benchmark that refuted this document's own argument, and a data-loss bug it exposed).

| # | Plan said | Built | Why |
|---|---|---|---|
| 1 | `tabular.py` re-imports `PARQUET_REQUIREMENT` for compatibility | Re-export dropped | Zero importers of the old path once the tests were re-pointed. Ruff flagged it as unused and was right — a shim nobody imports only ever costs. |
| 2 | Phase 0 exits with **no test edits** | One edit, 2 lines | `test_read_table.py` patched `tabular.PARQUET_REQUIREMENT`; the constant moved to the module that reads it. New target is the more correct one. The criterion was too strong for a refactor that moves layout by design. |
| 3 | Thread the backend through **5** call sites | **3** | `timeseries` and `text` are refused by the guard, so the parameter could only ever carry `"local"` there. Dead plumbing. |
| 4 | Guard in `build_bundle` | Guard in `build_bundle` **and** `build_cv_bundles` | The CV path calls source builders directly and never goes through `build_bundle` — every fold would have slipped past the refusal. |
| 5 | `DataBackendSpec.factory` is `Callable[[], Any]` | `Callable[..., Any]`, `get_data_backend(name, **params)` | A training backend validates its params in `fit()`; a data backend has no later hook, so `backend_params` must arrive at construction. |
| 6 | **3** `_print_plugins` edits | **4** | `width` sized on names alone, so the 12-character `DATA BACKEND` header collided with the `READY` column. Found by running the command. |
| 7 | `requires_spark` in `tests/conftest.py`; `-k data_backend` in CI | In the test module; three files named explicitly | No `__init__.py` in the test tree, so `from ..conftest import ...` fails — conftest injects fixtures, it is not importable for a decoration-time constant. And `-k data_backend` silently misses `test_spark_backend_translation.py`. |

Phase 2 added three more:

| # | Plan said | Built | Why |
|---|---|---|---|
| 8 | `matrix()` collects features; `column()` ×3 collects target/time/group | **`matrix()` removed**, replaced by a lazy `select()`; one atomic `to_pandas` | Separate collects re-execute the plan and need not agree on row order (`monotonically_increasing_id` is documented non-deterministic), so the builder would have paired feature rows with the wrong labels, silently. The single worst bug this design could have shipped. See §3, Risk 3b. |
| 9 | Win #2 is "column projection pushdown" in `build_tabular_bundle` | Pushdown applies to `_label_columns`; the builder gained **fail-fast schema validation** instead | `feature_cols ∪ reserved` is every column, so the builder has nothing to project away. The claim was arithmetically wrong. |
| 10 | `timeseries`/`text` "get threaded in Phase 2" (Phase 1 note) | Still unthreaded | They remain non-backend-aware and the guard still refuses them by name, so threading would still be plumbing that can only carry `"local"`. The note anticipated a change Phase 2 gave no reason to make. |

Phase 3 added three more:

| # | Plan said | Built | Why |
|---|---|---|---|
| 11 | `write_parquet` writes Parquet | Writes a **directory** of part-files on every backend, `local` included | `read_table` tells Parquet from CSV by inspecting the path, and DVC declares the output as a directory. A bare suffix-less file from `local` would have been read back as a CSV. |
| 12 | (not mentioned) | `local` deletes stale `*.parquet` before writing | Spark's `mode("overwrite")` clears the directory; pandas does not. A rerun over fewer rows would otherwise union with its own previous output. |
| 13 | One `filter_notnull` | `filter_notnull` **and** `drop_all_null_rows` | The stage needs two different null policies — any null target drops a row, but only an all-null *feature* row does. One method named for the general idea would have hidden the distinction. |

Phase 4 added three more:

| # | Plan said | Built | Why |
|---|---|---|---|
| 14 | Polars is not worth building — "the win is confined to parse time, and it is unmeasured" (§7 arg 3) | **Built**; the benchmark refutes the argument | The win *is* confined to the parse, but the CSV parse dominates `build_bundle`, so end to end the whole build is **2.8–3.7×** faster. Arguments 1/2/4 — don't change the default, the return type, or do it implicitly — all stand and shaped the design. |
| 15 | (not anticipated) | Two real bugs found, both pre-existing | `pandas.dropna(how="all", subset=[])` drops every row, so the Phase 3 stage silently emitted an empty output whenever the target was the only column — **shipped data loss**. And the engines disagreed on a quoted-empty CSV field. A second in-process engine is what surfaced both: polars raised where pandas silently succeeded. |
| 16 | §9.3's equivalence tests skip without a JVM | They **run** now | Polars needs no JVM, so parameterizing the equivalence suite over every non-default engine made the feature's central claim — same bundle, different engine — actually execute on a dev machine rather than skip. |

### Still outstanding

- **Spark specifically remains unproven on this machine.** `sort_by`'s tie-break (Risk 1) and the multi-collect alignment rule (Risk 3b) rest on tests that **skip without a JVM**. Phase 4 narrowed this materially — §9.3's equivalence suite now *executes* for Polars, so the cross-engine claim is tested rather than believed, and the shared conformance suite has a second real engine holding it honest. But the Spark path itself is still asserted by construction. `spark-contract` converts that into evidence and has not yet run.
- **The `frame` payload straight to `gbdt` stays refused**, on evidence rather than reluctance: `GBDTBackend.fit` calls `_as_matrix`, which sends anything that is not pandas to `np.asarray(..., dtype="float32")`. Skipping the collect needs a Spark-native trainer, i.e. distributed training — a non-goal (§11) on a different registry (§2).
- **Only `tabular` is backend-aware.** `image`, `text` and `timeseries` read through pandas and the guard refuses a non-local engine for them by name. Widening that is a phase of its own, not a loose end.
- **The benchmark is one machine's numbers.** `benchmarks/data_backends.py` is committed so the ratios can be re-measured rather than trusted; pandas' CSV reader is slower here than published figures, which may flatter the comparison.
