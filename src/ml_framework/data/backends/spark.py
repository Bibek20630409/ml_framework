"""
data/backends/spark.py
──────────────────────
The distributed data backend: pyspark reads, filters, sorts and projects; the
driver collects what training needs.

**Where the distribution stops.** :meth:`SparkBackend.matrix` and
:meth:`SparkBackend.to_pandas` are the collect points. Everything downstream —
splitters, preprocessors, SMOTE, every training backend — is single-node numpy.
What Spark buys is the work *before* the collect: planning cross-validation folds
from a row count and one label column without ever touching the feature matrix,
and pushing a column projection into Parquet so the driver receives only the
columns being trained on.

**Two divergences from ``pipeline/spark_preprocess.py``, both deliberate:**

* The session is created once per process and **never stopped**. That module is a
  one-shot job and stopping in a ``finally`` is right for it; here ``mlf tune``
  and ``mlf select`` build many bundles in one process, and a stopped session
  would take the next bundle down with it.
* Sorting breaks ties explicitly. See :meth:`sort_by`.

Requires a JVM (Java 11/17) and ``pip install -e ".[mlops]"``. Note that pyspark
imports fine *without* a JVM and only fails at ``getOrCreate()``, which is why
the tests gate on ``shutil.which("java")`` rather than on importability.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ...core.types import FrameworkError

log = logging.getLogger(__name__)

# Collecting is the one operation here that can kill the driver with no
# traceback, so it is capped rather than attempted and hoped for. Override with
# `data.backend_params.max_collect_rows`.
DEFAULT_MAX_COLLECT_ROWS = 5_000_000

# Spark's own dtype names would reach `FeatureSchema.dtypes`, and from there the
# bundle manifest and the serving signature — so an artifact would record
# `bigint` or `double` purely because of which engine happened to build it.
# Normalized to the numpy spellings pandas produces.
_SPARK_TO_NUMPY: Mapping[str, str] = {
    "boolean": "bool",
    "tinyint": "int8",
    "smallint": "int16",
    "int": "int32",
    "integer": "int32",
    "bigint": "int64",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "string": "object",
    "date": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
}

# The column `sort_by` adds to make ties deterministic. Prefixed so it cannot
# collide with a real column, and dropped before anything is collected.
_ROW_ORDER = "__mlf_row_order__"


def _normalize_dtype(spark_type: str) -> str:
    """Spark's type name → the numpy spelling pandas would have produced."""
    base = spark_type.lower().split("(")[0].strip()
    if base.startswith("decimal"):
        return "float64"
    return _SPARK_TO_NUMPY.get(base, base)


# The inverse, for `cast`. Built from the table above rather than written out a
# second time, so the two can never drift; the aliases that collide on the same
# numpy name (`int`/`integer`, `long`/`bigint`) resolve to whichever Spark spells
# canonically, which is why those are pinned explicitly afterwards.
_NUMPY_TO_SPARK: dict[str, str] = {v: k for k, v in _SPARK_TO_NUMPY.items()}
_NUMPY_TO_SPARK.update({"int32": "int", "int64": "bigint", "datetime64[ns]": "timestamp"})


def _to_spark_type(numpy_dtype: str) -> str:
    """The numpy spelling a caller uses → the type name Spark's ``cast`` wants."""
    return _NUMPY_TO_SPARK.get(numpy_dtype.lower(), numpy_dtype)


class SparkBackend:
    """pyspark. Reads and reduces at cluster scale, then collects."""

    name: ClassVar[str] = "spark"
    engine: ClassVar[str] = "pyspark"

    def __init__(self, **params: Any) -> None:
        self.max_collect_rows = int(params.pop("max_collect_rows", DEFAULT_MAX_COLLECT_ROWS))
        self.shuffle_partitions = int(params.pop("shuffle_partitions", 8))
        self.app_name = str(params.pop("app_name", "ml_framework"))
        if params:
            raise FrameworkError(
                f"unknown data.backend_params for the spark backend: {sorted(params)}. "
                f"Known: max_collect_rows, shuffle_partitions, app_name"
            )
        self._spark: Any = None

    # ── the single pyspark seam ──
    def _session(self) -> Any:  # pragma: no cover - needs a JVM
        """The process-wide SparkSession, created on first use.

        The only place this class imports pyspark, which is what makes the rest
        of the module testable against a stub.
        """
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder.appName(self.app_name)
                .config("spark.sql.shuffle.partitions", str(self.shuffle_partitions))
                .getOrCreate()
            )
        return self._spark

    # ── read ──
    def read_table(self, path: str) -> Any:
        # Same parquet-vs-csv rule as the local backend, including the directory
        # case: a bundle must not change shape because the engine changed.
        p = Path(path)
        reader = self._session().read.option("header", True).option("inferSchema", True)
        if p.is_dir() or p.suffix.lower() in (".parquet", ".pq"):
            return reader.parquet(path)
        return reader.csv(path)

    # ── inspect: nothing leaves the cluster ──
    def columns(self, table: Any) -> tuple[str, ...]:
        return tuple(c for c in table.columns if c != _ROW_ORDER)

    def n_rows(self, table: Any) -> int:
        return int(table.count())

    def dtypes(self, table: Any, columns: Sequence[str]) -> Mapping[str, str]:
        native = dict(table.dtypes)
        return {c: _normalize_dtype(native[c]) for c in columns}

    # ── reduce ──
    def sort_by(self, table: Any, column: str) -> Any:
        """Stable ascending sort with a deterministic tie-break.

        Spark's ``orderBy`` gives a total order but does not fix the order of
        *equal* keys, and it can differ between runs on the same data. That is not
        cosmetic here: ``builders._label_columns`` sorts so the positions a
        splitter computes line up with the rows it splits, so a table with
        duplicate timestamps would otherwise produce different folds — and
        different scores — from one run to the next, silently.

        A monotonic row id captured *before* sorting supplies the tie-break, which
        reproduces pandas' ``kind="stable"``: equal keys keep their original
        relative order.
        """
        from pyspark.sql import functions as F

        ordered = table
        if _ROW_ORDER not in table.columns:
            ordered = table.withColumn(_ROW_ORDER, F.monotonically_increasing_id())
        return ordered.orderBy(F.col(column).asc(), F.col(_ROW_ORDER).asc())

    def select(self, table: Any, columns: Sequence[str]) -> Any:
        """Lazy projection. Pushes down into Parquet; nothing is collected here."""
        return table.select(*list(columns))

    # ── collect: the only two methods that move data to the driver ──
    def column(self, table: Any, name: str, *, dtype: str | None = None) -> np.ndarray:
        """One standalone column.

        Safe precisely because it stands alone: nothing else has to line up with
        it. Callers needing two arrays in the same row order use
        :meth:`to_pandas`, because two calls here are two collects — and a
        `sort_by`-ordered plan re-executed twice need not agree on row order.
        """
        self._guard_collect(table, what=f"column '{name}'")
        rows = table.select(name).collect()
        values = np.array([r[0] for r in rows])
        return values if dtype is None else values.astype(dtype)

    def to_pandas(self, table: Any) -> Any:
        """The atomic collect. One `toPandas`, so every slice of it is aligned."""
        self._guard_collect(table, what="the whole table")
        frame = table.toPandas()
        return frame.drop(columns=[_ROW_ORDER]) if _ROW_ORDER in frame.columns else frame

    # ── clean + write (for `pipeline.spark_preprocess`) ──
    def filter_notnull(self, table: Any, column: str) -> Any:
        from pyspark.sql import functions as F

        return table.where(F.col(column).isNotNull())

    def drop_all_null_rows(self, table: Any, columns: Sequence[str]) -> Any:
        return table.dropna(how="all", subset=list(columns))

    def drop_duplicates(self, table: Any) -> Any:
        return table.dropDuplicates()

    def cast(self, table: Any, column: str, dtype: str) -> Any:
        from pyspark.sql import functions as F

        return table.withColumn(column, F.col(column).cast(_to_spark_type(dtype)))

    def write_parquet(self, table: Any, path: str) -> None:
        # `coalesce(1)` gives a single deterministic output partition, which is
        # what the DVC stage's checksum depends on: N part-files whose row split
        # varies between runs would make an unchanged input look changed. Drop it
        # for genuinely large outputs and read the directory directly.
        table.coalesce(1).write.mode("overwrite").parquet(path)

    # ── the guard that makes a collect fail loudly ──
    def _guard_collect(self, table: Any, *, what: str) -> None:
        """Refuse a collect that would not fit, *before* attempting it.

        An OOM-killed driver produces no traceback and no useful message; a row
        count and the name of the knob that raises the ceiling does.
        """
        n = self.n_rows(table)
        if n > self.max_collect_rows:
            raise FrameworkError(
                f"refusing to collect {what}: {n:,} rows exceeds max_collect_rows="
                f"{self.max_collect_rows:,}. Collecting happens on the driver, so this "
                f"would likely exhaust its memory. Either filter upstream, or raise the "
                f"ceiling with data.backend_params.max_collect_rows."
            )


def build_data_backend(**params: Any) -> Any:
    return SparkBackend(**params)
