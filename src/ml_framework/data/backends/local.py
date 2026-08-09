"""
data/backends/local.py
──────────────────────
The default data backend: pandas, in-process.

This is where ``read_table``'s body lives now. The function itself stays in
``data/sources/tabular.py`` as a thin shim, because it is public API — re-exported
from ``core``, consumed by ``pipeline/contracts.py`` — and its pandas return type
is part of that contract.

Deliberately does **not** import ``data.sources.tabular``: tabular calls back into
this module, and `core/__init__` lazily re-exports ``read_table`` from
``core.lit_data``, which imports tabular. ``PARQUET_REQUIREMENT`` therefore lives
here and tabular re-imports it, rather than the other way round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...core.plugins import check_requirements
from ...core.types import FrameworkError, Requirement

# pandas needs an engine to read parquet, and it is not a pandas dependency.
# Declaring it here rather than leaning on `mlflow` (which happens to require
# pyarrow) keeps the coupling visible: the parquet path must fail with a pip
# command, not with a pandas ImportError, in exactly the environment where the
# dependency is most likely absent — a serving image built without the mlops extra.
PARQUET_REQUIREMENT = Requirement("pyarrow", extra="parquet", min_version="10.0.1")


class LocalBackend:
    """pandas. The reference implementation, and the oracle the others are tested against."""

    name: ClassVar[str] = "local"
    engine: ClassVar[str] = "pandas"

    def __init__(self, **params: Any) -> None:
        # Takes no knobs, but refuses unknown ones rather than ignoring them: a
        # `backend_params` block that silently does nothing after the engine was
        # switched back to `local` is exactly the kind of dead config that later
        # gets copied into a run where it *would* have mattered.
        if params:
            raise FrameworkError(
                f"the local data backend takes no data.backend_params; got {sorted(params)}"
            )

    # ── read ──
    def read_table(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.is_dir() or p.suffix.lower() in (".parquet", ".pq"):
            check_requirements((PARQUET_REQUIREMENT,), what=f"reading parquet from '{path}'")
            return pd.read_parquet(path)
        return pd.read_csv(path)

    # ── inspect ──
    def columns(self, table: pd.DataFrame) -> tuple[str, ...]:
        return tuple(str(c) for c in table.columns)

    def n_rows(self, table: pd.DataFrame) -> int:
        return int(len(table))

    def dtypes(self, table: pd.DataFrame, columns: Sequence[str]) -> Mapping[str, str]:
        return {c: str(table[c].dtype) for c in columns}

    # ── reduce ──
    def sort_by(self, table: pd.DataFrame, column: str) -> pd.DataFrame:
        # `kind="stable"` is the contract, not a preference: the positions a
        # splitter computes must line up with the rows it splits, so equal keys
        # may not reorder between one read and the next.
        return table.sort_values(column, kind="stable")

    def select(self, table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        return table[list(columns)]

    # ── collect (free here; the whole table is already in memory) ──
    def column(self, table: pd.DataFrame, name: str, *, dtype: str | None = None) -> np.ndarray:
        values = table[name].to_numpy()
        return values if dtype is None else values.astype(dtype)

    def to_pandas(self, table: pd.DataFrame) -> pd.DataFrame:
        return table

    # ── clean + write (for `pipeline.spark_preprocess`) ──
    def filter_notnull(self, table: pd.DataFrame, column: str) -> pd.DataFrame:
        return table[table[column].notna()]

    def drop_all_null_rows(self, table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        cols = list(columns)
        if not cols:
            # `dropna(how="all", subset=[])` drops **every** row — vacuously, all
            # zero of the named columns are null in each. Reachable from
            # `preprocess` whenever the target is the table's only column, where it
            # would have silently produced an empty output.
            return table
        return table.dropna(how="all", subset=cols)

    def drop_duplicates(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.drop_duplicates()

    def cast(self, table: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
        # `assign` rather than in-place: the caller's table is not ours to mutate.
        return table.assign(**{column: table[column].astype(dtype)})

    def write_parquet(self, table: pd.DataFrame, path: str) -> None:
        check_requirements((PARQUET_REQUIREMENT,), what=f"writing parquet to '{path}'")
        dest = Path(path)
        # A *directory* holding one part-file, mirroring what Spark's
        # `coalesce(1).write.parquet(dir)` produces. Writing a bare file here
        # would be read back as a CSV, because `read_table` decides by inspecting
        # the path and a suffix-less file is not recognizable as Parquet.
        dest.mkdir(parents=True, exist_ok=True)
        # Overwrite semantics, matching Spark's `mode("overwrite")`. Without this
        # a rerun over fewer rows would leave the previous run's part-files in
        # place and the next read would silently union the two.
        for stale in dest.glob("*.parquet"):
            stale.unlink()
        table.to_parquet(dest / "part-0.parquet", index=False)


def build_data_backend(**params: Any) -> Any:
    return LocalBackend(**params)
