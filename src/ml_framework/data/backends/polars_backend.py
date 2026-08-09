"""
data/backends/polars_backend.py
───────────────────────────────
The Polars data backend: in-process like ``local``, multithreaded underneath.

**A peer of ``local``, not a replacement for it.** pandas remains the base
dependency and the default engine, because ``read_table -> pd.DataFrame`` is
public API consumed by ``pipeline/contracts.py`` (pandera is pandas-only). This
backend hands off through :meth:`to_pandas` like every other, so nothing
downstream can tell which engine read the bytes — that is precisely what makes it
selectable per run rather than a migration.

Named ``polars_backend`` rather than ``polars``: a module called ``polars.py``
inside this package is legal (Python 3 has no implicit relative imports) but it
shadows the real library for any reader, and for tooling that resolves by name.
The registered backend is still ``polars``; only the filename differs.

Where it is faster: the CSV/Parquet parse, and column projection. Where it is
not: everything after the handoff, which is the same numpy the other backends
produce. See choose.md §7 for the measured numbers rather than the assumption.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ...core.plugins import check_requirements
from ...core.types import FrameworkError, Requirement

log = logging.getLogger(__name__)

# Polars itself, plus pyarrow: `to_pandas()` goes through Arrow, and that call is
# how this backend hands off to the rest of the framework. Declaring only polars
# would let the backend register as available and then fail at the handoff.
POLARS_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("polars", extra="fast", min_version="1.0"),
    Requirement("pyarrow", extra="fast", min_version="10.0.1"),
)

# Polars' type names → the numpy spellings pandas would have produced. The same
# normalization `spark.py` does, for the same reason: `FeatureSchema.dtypes`
# reaches the bundle manifest and the serving signature, so an artifact must not
# record which engine happened to build it.
_POLARS_TO_NUMPY: Mapping[str, str] = {
    "boolean": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "string": "object",
    "utf8": "object",
    "categorical": "object",
    "date": "datetime64[ns]",
    "datetime": "datetime64[ns]",
}

_NUMPY_TO_POLARS: Mapping[str, str] = {
    "bool": "Boolean",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "object": "String",
    "datetime64[ns]": "Datetime",
}


def _normalize_dtype(polars_type: str) -> str:
    """``Int64`` → ``int64``, ``String`` → ``object``, and so on."""
    base = polars_type.lower().split("(")[0].strip()
    if base.startswith("decimal"):
        return "float64"
    return _POLARS_TO_NUMPY.get(base, base)


def _to_polars_dtype(numpy_dtype: str) -> Any:
    """The numpy spelling a caller uses → the Polars dtype object for ``cast``."""
    import polars as pl

    name = _NUMPY_TO_POLARS.get(numpy_dtype.lower())
    if name is None:
        raise FrameworkError(
            f"the polars backend cannot cast to '{numpy_dtype}'. "
            f"Known: {sorted(_NUMPY_TO_POLARS)}"
        )
    return getattr(pl, name)


class PolarsBackend:
    """Polars. Same contract as ``local``, different parser underneath."""

    name: ClassVar[str] = "polars"
    engine: ClassVar[str] = "polars"

    def __init__(self, **params: Any) -> None:
        # Availability is checked here rather than at import: the module has to be
        # importable for `mlf data-backends` to describe it on an install without
        # polars, and this is the first point at which it would actually be used.
        check_requirements(POLARS_REQUIREMENTS, what="data backend 'polars'")
        if params:
            raise FrameworkError(
                f"the polars data backend takes no data.backend_params; got {sorted(params)}"
            )

    # ── read ──
    def read_table(self, path: str) -> Any:
        import polars as pl

        p = Path(path)
        if p.is_dir():
            # Same "directory of part-files" shape the other backends read and
            # write. Polars takes a glob rather than a directory.
            return pl.read_parquet(str(p / "*.parquet"))
        if p.suffix.lower() in (".parquet", ".pq"):
            return pl.read_parquet(path)
        # `null_values=[""]` matches what pandas does by default. Without it a
        # *quoted* empty field — which is exactly what `pandas.to_csv` writes for a
        # missing value in a single-column frame — reads as the string `""` rather
        # than null, and polars then infers the whole column as String where pandas
        # infers a float. `local` is the oracle, so the reader is aligned to it
        # rather than the divergence being documented and left in.
        return pl.read_csv(path, null_values=[""])

    # ── inspect ──
    def columns(self, table: Any) -> tuple[str, ...]:
        return tuple(table.columns)

    def n_rows(self, table: Any) -> int:
        return int(table.height)

    def dtypes(self, table: Any, columns: Sequence[str]) -> Mapping[str, str]:
        native = dict(zip(table.columns, (str(t) for t in table.dtypes), strict=True))
        return {c: _normalize_dtype(native[c]) for c in columns}

    # ── reduce ──
    def sort_by(self, table: Any, column: str) -> Any:
        # `maintain_order=True` is the contract, not a preference: equal keys must
        # keep their original relative order, or the positions a splitter computes
        # stop lining up with the rows it splits. Polars sorts multithreaded and
        # is *not* stable without this.
        return table.sort(column, maintain_order=True)

    def select(self, table: Any, columns: Sequence[str]) -> Any:
        return table.select(list(columns))

    # ── collect ──
    def column(self, table: Any, name: str, *, dtype: str | None = None) -> np.ndarray:
        values = table[name].to_numpy()
        return values if dtype is None else values.astype(dtype)

    def to_pandas(self, table: Any) -> Any:
        # `use_pyarrow_extension_array` stays at its default of False, which yields
        # numpy-backed columns whose dtypes match a pandas read exactly. Flipping
        # it would hand the framework Arrow extension dtypes and make
        # `FeatureSchema.dtypes` depend on the engine after all.
        return table.to_pandas()

    # ── clean + write (for `pipeline.spark_preprocess`) ──
    def filter_notnull(self, table: Any, column: str) -> Any:
        import polars as pl

        return table.filter(pl.col(column).is_not_null())

    def drop_all_null_rows(self, table: Any, columns: Sequence[str]) -> Any:
        import polars as pl

        cols = list(columns)
        if not cols:
            return table
        return table.filter(~pl.all_horizontal(pl.col(c).is_null() for c in cols))

    def drop_duplicates(self, table: Any) -> Any:
        # `maintain_order=True` for the same reason as `sort_by`: polars' default
        # `keep="any"` with an unordered result would make the surviving row order
        # vary between runs on identical input.
        return table.unique(maintain_order=True)

    def cast(self, table: Any, column: str, dtype: str) -> Any:
        import polars as pl

        return table.with_columns(pl.col(column).cast(_to_polars_dtype(dtype)))

    def write_parquet(self, table: Any, path: str) -> None:
        dest = Path(path)
        # A directory holding one part-file, identical to `local` and to Spark's
        # `coalesce(1)` output — `read_table` decides Parquet-vs-CSV by inspecting
        # the path, so a bare suffix-less file would be read back as a CSV.
        dest.mkdir(parents=True, exist_ok=True)
        # Overwrite semantics: without this a rerun over fewer rows would leave the
        # previous run's part-file behind and the next read would union the two.
        for stale in dest.glob("*.parquet"):
            stale.unlink()
        table.write_parquet(dest / "part-0.parquet")


def build_data_backend(**params: Any) -> Any:
    return PolarsBackend(**params)
