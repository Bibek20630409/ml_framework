"""
data/backends/
──────────────
Data backends: one per data-processing **engine**, not per file format.

    local   pandas, in-process       (read, sort, project, collect) — the default
    spark   pyspark, distributed     (the same, at cluster scale, then collects)
    polars  polars, in-process       (the same as local, multithreaded parse)

``local`` and ``polars`` are **peers**: same contract, different parser. pandas is
the default and the base dependency because ``read_table -> pd.DataFrame`` is
public API, so Polars is a choice a run makes rather than a migration or a silent
behaviour change keyed on what happens to be installed.

Registration is a spec plus a lazy factory, never an import of the engine. That
is what lets ``mlf data-backends`` list ``spark`` and ``polars`` alongside
``local`` on a bare install, and what keeps ``read_table`` — which every ingestion
path calls — free of both.

**A data backend chooses how the table is read and reduced, not how the model is
trained.** :class:`~ml_framework.data.types.DataBundle` holds numpy arrays and
the splitters index into them, so the framework collects a materialized bundle
and fits on one node under every backend. Spark's job ends at the feature matrix;
what it buys is planning folds and projecting columns over a table too large to
land on the driver whole.

Selected per run by ``data.backend`` (or ``--data-backend``), which is why the
engine can change without anything upstream of ``build_bundle`` changing.
"""

from __future__ import annotations

import importlib
from typing import Any

# Imported as submodules, never `from ..core import X`: `core/__init__` lazily
# re-exports `read_table`, which imports `data.sources.tabular`, which reaches
# back here. See the same warning in `data/builders.py`.
from ...core.plugins import DataBackendSpec
from ...core.registry import register_data_backend
from ...core.types import Requirement


def engine_for(config: Any) -> Any:
    """The :class:`DataBackend` this run selected, with its params applied.

    One place resolves ``data.backend`` + ``data.backend_params`` into an engine,
    so a source never has to remember to pass the second one — forgetting it would
    silently drop ``max_collect_rows`` and turn a guarded collect into an OOM.
    """
    from ...core.registry import get_data_backend

    return get_data_backend(config.data.backend, **dict(config.data.backend_params))


def _lazy_factory(module: str, attr: str = "build_data_backend"):
    """Defer the import to call time, so registering never imports pyspark."""

    def _build(**params: Any) -> Any:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)(**params)

    return _build


# No `requires`: pandas is a *base* dependency and `local` is the default. Giving
# the default backend an optional requirement would make the framework's central
# guarantee — a bare install trains — conditional on an extra.
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

# A *peer* of `local`, not a replacement: pandas stays the base dependency and the
# default engine because `read_table -> pd.DataFrame` is public API. Registering
# Polars here rather than hiding it inside `local` is what makes it a choice a run
# can make, instead of a silent behaviour change keyed on what happens to be
# installed. `pyarrow` is in `requires` because `to_pandas` -- the handoff every
# backend goes through -- routes via Arrow.
register_data_backend(
    DataBackendSpec(
        name="polars",
        factory=_lazy_factory(".polars_backend"),
        engine="polars",
        requires=(
            Requirement("polars", extra="fast", min_version="1.0"),
            Requirement("pyarrow", extra="fast", min_version="10.0.1"),
        ),
        description="In-process Polars: multithreaded CSV/Parquet parse and projection.",
    )
)

__all__ = ["DataBackendSpec", "engine_for"]
