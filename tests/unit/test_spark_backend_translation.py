"""What the Spark backend *says to Spark*, checked without a JVM.

These are **translation** tests, not correctness tests, and the distinction is the
point. CI has no JVM (``docs/PHASE_STATUS.md`` records pyspark as installed but
unrunnable), so the conformance suite in ``test_data_backends.py`` skips there and
``spark.py`` would otherwise ship entirely unexercised.

What is reachable without a session is the translation layer: the branch that
picks parquet over csv, the dtype normalization that keeps a bundle's schema
independent of the engine, and the guard that refuses a collect *before*
attempting it. Those are also the three things most likely to be wrong. Anything
needing a real ``Column`` — the ``sort_by`` tie-break, the collects themselves —
is left to the conformance suite, which activates the moment a JVM appears.

The backend is written so a stub can reach all of this: ``_session()`` is the only
place it imports pyspark.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml_framework.core.types import FrameworkError
from ml_framework.data.backends.spark import (
    DEFAULT_MAX_COLLECT_ROWS,
    SparkBackend,
    _normalize_dtype,
    _to_spark_type,
)

pytestmark = pytest.mark.unit


class _StubReader:
    """Records which reader method the backend chose, and with what path."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def option(self, *_args, **_kwargs) -> _StubReader:
        return self

    def parquet(self, path: str) -> str:
        self.calls.append(("parquet", path))
        return "TABLE"

    def csv(self, path: str) -> str:
        self.calls.append(("csv", path))
        return "TABLE"


@pytest.fixture
def stub_backend(monkeypatch):
    """A SparkBackend whose only pyspark seam is replaced by a recorder."""
    backend = SparkBackend()
    reader = _StubReader()
    monkeypatch.setattr(backend, "_session", lambda: SimpleNamespace(read=reader))
    return backend, reader


# ── the parquet-vs-csv rule must match the local backend exactly ──────
@pytest.mark.parametrize(
    "filename,expected",
    [("d.csv", "csv"), ("d.parquet", "parquet"), ("d.pq", "parquet"), ("d.PARQUET", "parquet")],
)
def test_the_reader_is_chosen_by_the_same_rule_as_the_local_backend(
    stub_backend, tmp_path, filename, expected
):
    """A bundle must not change shape because the engine changed."""
    backend, reader = stub_backend
    backend.read_table(str(tmp_path / filename))
    assert reader.calls[0][0] == expected


def test_a_directory_reads_as_parquet_part_files(stub_backend, tmp_path):
    """What the Spark preprocessing stage writes."""
    backend, reader = stub_backend
    d = tmp_path / "processed"
    d.mkdir()
    backend.read_table(str(d))
    assert reader.calls[0][0] == "parquet"


# ── dtype normalization: the artifact must not name the engine ────────
@pytest.mark.parametrize(
    "spark_type,expected",
    [
        ("bigint", "int64"),
        ("double", "float64"),
        ("int", "int32"),
        ("float", "float32"),
        ("boolean", "bool"),
        ("string", "object"),
        ("timestamp", "datetime64[ns]"),
        ("decimal(10,2)", "float64"),
    ],
)
def test_spark_types_are_reported_in_numpy_spelling(spark_type, expected):
    """`FeatureSchema.dtypes` reaches the bundle manifest and the serving
    signature, so `bigint` leaking through would make the artifact depend on which
    engine happened to build it."""
    assert _normalize_dtype(spark_type) == expected


def test_an_unmapped_spark_type_passes_through_rather_than_raising():
    """An unknown type is a gap in the table, not a reason to fail a run that was
    otherwise going to succeed."""
    assert _normalize_dtype("interval") == "interval"


def test_dtypes_reads_the_tables_own_schema(stub_backend):
    backend, _ = stub_backend
    table = SimpleNamespace(dtypes=[("f0", "double"), ("label", "bigint")])
    assert backend.dtypes(table, ["f0", "label"]) == {"f0": "float64", "label": "int64"}


# ── the collect guard: refuse before attempting ───────────────────────
def test_a_collect_over_the_ceiling_is_refused_before_it_is_attempted(stub_backend):
    """An OOM-killed driver leaves no traceback. `toPandas` must never be reached,
    which is why the stub table has no `toPandas` at all: if the guard let the call
    through, this would fail with AttributeError instead of passing."""
    backend, _ = stub_backend
    backend.max_collect_rows = 10
    table = SimpleNamespace(count=lambda: 11, columns=["f0"])
    with pytest.raises(FrameworkError) as excinfo:
        backend.to_pandas(table)
    message = str(excinfo.value)
    assert "11" in message
    assert "max_collect_rows" in message


def test_a_collect_at_the_ceiling_is_allowed(stub_backend):
    """The bound is inclusive: a table of exactly `max_collect_rows` still fits."""
    backend, _ = stub_backend
    backend.max_collect_rows = 10
    frame = SimpleNamespace(columns=[])
    table = SimpleNamespace(count=lambda: 10, toPandas=lambda: frame)
    assert backend.to_pandas(table) is frame


def test_the_ceiling_is_configurable_through_backend_params():
    assert SparkBackend().max_collect_rows == DEFAULT_MAX_COLLECT_ROWS
    assert SparkBackend(max_collect_rows=7).max_collect_rows == 7


def test_an_unknown_backend_param_names_the_ones_that_exist():
    with pytest.raises(FrameworkError) as excinfo:
        SparkBackend(shuffel_partitions=8)
    message = str(excinfo.value)
    assert "shuffel_partitions" in message
    assert "shuffle_partitions" in message


# ── select must stay lazy ─────────────────────────────────────────────
def test_select_projects_without_collecting(stub_backend):
    """`select` earns its place by *not* collecting: it narrows the table so the
    single `to_pandas` that follows carries less.

    The stub records the projection and exposes no `toPandas`/`collect`, so a
    `select` that materialized would fail with AttributeError rather than pass.
    """
    backend, _ = stub_backend
    projected: list[tuple[str, ...]] = []
    table = SimpleNamespace(
        columns=["f0", "f1", "label"],
        select=lambda *cols: projected.append(cols) or SimpleNamespace(columns=list(cols)),
    )
    narrowed = backend.select(table, ["f0", "label"])
    assert projected == [("f0", "label")]
    assert backend.columns(narrowed) == ("f0", "label")


# ── the clean/write half (Phase 3) ────────────────────────────────────
@pytest.mark.parametrize(
    "numpy_dtype,expected",
    [
        ("float64", "double"),
        ("float32", "float"),
        ("int64", "bigint"),
        ("int32", "int"),
        ("bool", "boolean"),
        ("object", "string"),
        ("datetime64[ns]", "timestamp"),
    ],
)
def test_a_numpy_dtype_name_casts_to_the_right_spark_type(numpy_dtype, expected):
    """`cast` takes numpy spellings so callers need not know Spark's. Getting the
    inverse wrong would silently write the target column as the wrong type."""
    assert _to_spark_type(numpy_dtype) == expected


def test_the_dtype_mapping_round_trips():
    """The forward and inverse tables are derived from one another; this is what
    catches them drifting apart if either is edited by hand."""
    for spark_name in ("double", "bigint", "boolean", "string"):
        assert _to_spark_type(_normalize_dtype(spark_name)) == spark_name


def test_dropping_all_null_rows_is_subset_scoped_not_global(stub_backend):
    """`how="all"` over the *feature* columns only. Passing no subset would let a
    row survive purely because its target was populated."""
    backend, _ = stub_backend
    seen: dict = {}
    table = SimpleNamespace(dropna=lambda **kw: seen.update(kw) or "CLEANED")
    assert backend.drop_all_null_rows(table, ["f0", "f1"]) == "CLEANED"
    assert seen == {"how": "all", "subset": ["f0", "f1"]}


def test_writing_coalesces_to_one_partition_and_overwrites(stub_backend):
    """`coalesce(1)` keeps the DVC output checksum stable — N part-files whose row
    split varies between runs would make an unchanged input look changed."""
    backend, _ = stub_backend
    calls: list = []

    class _Writer:
        def mode(self, m):
            calls.append(("mode", m))
            return self

        def parquet(self, p):
            calls.append(("parquet", p))

    table = SimpleNamespace(
        coalesce=lambda n: calls.append(("coalesce", n)) or SimpleNamespace(write=_Writer())
    )
    backend.write_parquet(table, "out/dir")
    assert calls == [("coalesce", 1), ("mode", "overwrite"), ("parquet", "out/dir")]


# ── the tie-break bookkeeping column must never surface ───────────────
def test_the_tie_break_column_is_hidden_from_the_column_list(stub_backend):
    """`sort_by` adds a row-order column to make ties deterministic. It is
    bookkeeping, and a caller listing feature columns must not see it."""
    backend, _ = stub_backend
    table = SimpleNamespace(columns=["f0", "label", "__mlf_row_order__"])
    assert backend.columns(table) == ("f0", "label")


def test_the_tie_break_column_is_dropped_before_the_frame_is_returned(stub_backend):
    import pandas as pd

    backend, _ = stub_backend
    frame = pd.DataFrame({"f0": [1.0], "__mlf_row_order__": [0]})
    table = SimpleNamespace(count=lambda: 1, toPandas=lambda: frame)
    assert list(backend.to_pandas(table).columns) == ["f0"]
