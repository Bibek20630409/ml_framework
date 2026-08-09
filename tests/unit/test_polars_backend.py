"""Polars-specific edges the shared conformance suite cannot reach.

Everything about *behaviour* belongs in `test_data_backends.py`, where `local` is
the oracle and Polars is just another fixture parameter — that is the whole point
of a peer backend. What is left here are the paths with no pandas counterpart to
compare against: the two dtype-translation tables and the refusals.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from ml_framework.core.registry import get_data_backend  # noqa: E402
from ml_framework.core.types import FrameworkError  # noqa: E402
from ml_framework.data.backends.polars_backend import (  # noqa: E402
    PolarsBackend,
    _normalize_dtype,
    _to_polars_dtype,
)

pytestmark = pytest.mark.unit


# ── dtype translation: the artifact must not name the engine ───────────
@pytest.mark.parametrize(
    "polars_type,expected",
    [
        ("Int64", "int64"),
        ("Int32", "int32"),
        ("Float64", "float64"),
        ("Float32", "float32"),
        ("Boolean", "bool"),
        ("String", "object"),
        ("Utf8", "object"),
        ("Categorical", "object"),
        ("Datetime(time_unit='us', time_zone=None)", "datetime64[ns]"),
        ("Decimal(precision=10, scale=2)", "float64"),
    ],
)
def test_polars_types_are_reported_in_numpy_spelling(polars_type, expected):
    """`FeatureSchema.dtypes` reaches the bundle manifest and the serving
    signature, so `Int64` leaking through would make the artifact depend on which
    engine happened to build it."""
    assert _normalize_dtype(polars_type) == expected


def test_an_unmapped_polars_type_passes_through_rather_than_raising():
    """An unknown type is a gap in the table, not a reason to fail a run that was
    otherwise going to succeed."""
    assert _normalize_dtype("Struct") == "struct"


def test_casting_to_an_unknown_dtype_names_what_is_supported():
    """The inverse table is smaller than the forward one, so a caller can ask for
    something unmappable. Refusing loudly beats casting to the wrong type."""
    with pytest.raises(FrameworkError) as excinfo:
        _to_polars_dtype("complex128")
    message = str(excinfo.value)
    assert "complex128" in message
    assert "float64" in message  # lists what it does know


def test_the_forward_and_inverse_dtype_tables_agree():
    """Both are hand-written here (unlike spark.py, where one is derived), so this
    is what catches them drifting apart."""
    import polars as pl

    for numpy_name in ("int64", "float64", "bool", "object"):
        polars_type = _to_polars_dtype(numpy_name)
        frame = pl.DataFrame({"c": []}, schema={"c": polars_type})
        assert _normalize_dtype(str(frame.dtypes[0])) == numpy_name


# ── refusals ──────────────────────────────────────────────────────────
def test_the_polars_backend_takes_no_backend_params():
    """Same rule as `local`: a knob that silently does nothing is how dead config
    survives into a run where it would have mattered."""
    with pytest.raises(FrameworkError) as excinfo:
        get_data_backend("polars", max_collect_rows=10)
    assert "max_collect_rows" in str(excinfo.value)


# ── the read paths, which differ from pandas in shape ──────────────────
def test_a_parquet_file_and_a_parquet_directory_both_read(tmp_path):
    """Polars takes a glob rather than a directory, so the two branches are
    genuinely different code — unlike pandas, where `read_parquet` handles both."""
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"f0": [1.0, 2.0], "label": [0, 1]})

    single = tmp_path / "one.parquet"
    frame.to_parquet(single, index=False)

    directory = tmp_path / "parts"
    directory.mkdir()
    frame.to_parquet(directory / "part-0.parquet", index=False)

    engine = PolarsBackend()
    assert engine.n_rows(engine.read_table(str(single))) == 2
    assert engine.n_rows(engine.read_table(str(directory))) == 2


def test_dropping_all_null_rows_over_no_columns_is_a_no_op(tmp_path):
    """Reachable from `preprocess` on a single-column table, where the target is
    the only column and the feature list comes out empty. `all_horizontal` over an
    empty expression list would otherwise drop every row."""
    src = tmp_path / "only_target.csv"
    pd.DataFrame({"label": [0, 1]}).to_csv(src, index=False)

    engine = PolarsBackend()
    table = engine.read_table(str(src))
    assert engine.n_rows(engine.drop_all_null_rows(table, [])) == 2
