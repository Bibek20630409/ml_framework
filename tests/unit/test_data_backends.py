"""The data-backend registry, and the conformance suite every engine must pass.

Two halves, deliberately separated:

* **Registry tests** need no engine at all. They cover the half a user on a bare
  install actually hits — that ``spark`` is *listed* rather than hidden, that
  selecting it without the extra names the pip command, and that importing the
  package costs nothing.
* **Conformance tests** run against every backend that can actually run here, with
  ``local`` as the **oracle**: a Spark assertion is "equals what pandas produced".
  Without that framing, "choose an implementation per run" would be two products
  rather than one choice.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import ml_framework.data.backends  # noqa: F401  (populates DATA_BACKENDS)
from ml_framework.core.plugins import MissingExtraError, UnknownPluginError
from ml_framework.core.protocols import DataBackend
from ml_framework.core.registry import DATA_BACKENDS, get_data_backend
from ml_framework.core.types import FrameworkError

pytestmark = pytest.mark.unit


# ── Spark: importable is not the same as runnable ─────────────────────
def _spark_runnable() -> bool:
    """pyspark importable **and** a JVM on PATH.

    The second half is the one that matters, and it is why this is not a plain
    ``importorskip``: pyspark imports perfectly well without Java and only fails
    at ``getOrCreate()``, which would turn a clean skip into a suite-wide error.
    ``docs/PHASE_STATUS.md`` records this environment as exactly that case.
    """
    import importlib.util
    import os
    import shutil

    if importlib.util.find_spec("pyspark") is None:
        return False
    return bool(shutil.which("java") or os.environ.get("JAVA_HOME"))


requires_spark = pytest.mark.skipif(not _spark_runnable(), reason="pyspark needs a JVM on PATH")


# ── The registry: no engine required ──────────────────────────────────
def test_every_engine_is_registered():
    assert DATA_BACKENDS.names() == ["local", "polars", "spark"]


def test_local_and_polars_are_peers_not_a_default_and_a_fallback():
    """The point of registering Polars rather than hiding it inside `local`: both
    are ordinary entries, and `local` is the default only because
    `DataConfig.backend` says so — not because the other is second-class."""
    local, polars = DATA_BACKENDS.get_spec("local"), DATA_BACKENDS.get_spec("polars")
    assert (local.engine, polars.engine) == ("pandas", "polars")
    # pandas is a base dependency, so `local` alone may carry no requirement —
    # that is what keeps "a bare install trains" unconditional.
    assert local.requires == ()
    assert polars.requires != ()


def test_an_uninstalled_engine_is_listed_rather_than_hidden():
    """`mlf data-backends` must show spark on a bare install, not omit it.

    Listing is `get_spec`, which deliberately does not check availability — the
    same contract the model registry has.
    """

    assert DATA_BACKENDS.get_spec("spark").engine == "pyspark"


def test_selecting_an_unregistered_engine_names_the_ones_that_exist():
    with pytest.raises(UnknownPluginError) as excinfo:
        get_data_backend("dask")
    assert "dask" in str(excinfo.value)


def test_importing_the_backends_package_imports_no_engine():
    """Registration is a spec plus a lazy factory, never an import of the engine.

    This is what lets `mlf data-backends` list spark on a bare install, and it is
    the property the `gbdt-no-torch` CI job asserts one axis wider.

    Run in a **fresh interpreter** rather than by evicting `sys.modules`: this
    module has already imported the package, and re-importing it would re-run
    registration against the still-populated global registry. A subprocess is also
    the honest question — "does importing this cost pyspark", from cold.
    """
    code = (
        "import sys; import ml_framework.data.backends; "
        "assert 'pyspark' not in sys.modules, 'importing the package imported pyspark'"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_local_engine_never_requires_an_extra():
    """The default backend must not be conditionally available.

    `local` carrying a `requires` would make "a bare install trains" depend on an
    extra, which is the guarantee the whole plugin system exists to protect.
    """

    assert DATA_BACKENDS.get_spec("local").requires == ()
    assert DATA_BACKENDS.is_available("local")


def test_an_unknown_backend_param_is_refused_rather_than_ignored():
    """A knob that silently does nothing is how dead config survives into a run
    where it would have mattered."""
    with pytest.raises(FrameworkError) as excinfo:
        get_data_backend("local", shuffle_partitions=8)
    assert "shuffle_partitions" in str(excinfo.value)


@pytest.mark.skipif(
    DATA_BACKENDS.is_available("spark"),
    reason="pyspark is installed here, so the missing-extra path cannot be exercised",
)
def test_selecting_spark_without_the_extra_names_the_pip_command():
    """The payoff of routing engine selection through the registry: a bare install
    gets a pip command, not an ImportError from inside a factory."""
    with pytest.raises(MissingExtraError) as excinfo:
        get_data_backend("spark")
    assert "ml-framework[mlops]" in str(excinfo.value)


def test_the_spark_requirement_produces_the_documented_pip_hint():
    """Asserted on the spec rather than by selecting, so it runs *here*, where
    pyspark happens to be installed and the test above can only skip."""
    from ml_framework.core.plugins import check_requirements
    from ml_framework.core.types import Requirement

    spec = DATA_BACKENDS.get_spec("spark")
    assert spec.requires == (Requirement("pyspark", extra="mlops", min_version="3.5"),)
    with pytest.raises(MissingExtraError) as excinfo:
        check_requirements(
            (Requirement("definitely_not_pyspark_xyz", extra="mlops", min_version="3.5"),),
            what="data backend 'spark'",
        )
    assert "pip install 'ml-framework[mlops]'" in str(excinfo.value)


# ── Conformance: local is the oracle ──────────────────────────────────
requires_polars = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None, reason="polars is not installed"
)


@pytest.fixture(
    params=[
        "local",
        pytest.param("spark", marks=requires_spark),
        pytest.param("polars", marks=requires_polars),
    ]
)
def engine(request):
    """Every engine that can run here. `local` is the oracle for the others."""
    return get_data_backend(request.param)


@pytest.fixture
def table_csv(tmp_path):
    """A table with a duplicate time key — see the sort-stability test."""
    frame = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [10.0, 20.0, 30.0, 40.0],
            "t": [3, 1, 2, 1],
            "label": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_every_backend_satisfies_the_protocol(engine):
    assert isinstance(engine, DataBackend)


def test_columns_and_row_count_agree_with_the_file(engine, table_csv):
    table = engine.read_table(table_csv)
    assert engine.columns(table) == ("f0", "f1", "t", "label")
    assert engine.n_rows(table) == 4


def test_select_narrows_the_table_without_collecting(engine, table_csv):
    """`select` is lazy on every backend: it returns a table, not an array."""
    table = engine.read_table(table_csv)
    narrowed = engine.select(table, ["f0", "f1"])
    assert engine.columns(narrowed) == ("f0", "f1")
    assert engine.n_rows(narrowed) == 4


def test_the_feature_matrix_comes_out_of_one_materialization(engine, table_csv):
    """What `matrix()` used to do, now through the atomic collect.

    Slicing one frame is the *only* safe way to get several row-aligned arrays —
    see the alignment test below for why.
    """
    table = engine.read_table(table_csv)
    frame = engine.to_pandas(engine.select(table, ["f0", "f1"]))
    matrix = frame[["f0", "f1"]].values.astype("float32")
    assert matrix.shape == (4, 2)
    assert matrix.dtype == np.dtype("float32")
    assert np.allclose(matrix[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_arrays_that_must_line_up_come_from_a_single_collect(engine, table_csv):
    """The rule that shaped this protocol.

    Each collect re-executes the plan on a distributed engine, and two executions
    need not agree on row order — pyspark documents
    `monotonically_increasing_id`, which `sort_by` relies on, as
    non-deterministic. So features must not be collected separately from labels:
    the pairing would break silently, which is the worst failure this layer could
    have. One `to_pandas`, then slice.
    """
    table = engine.read_table(table_csv)
    frame = engine.to_pandas(engine.sort_by(table, "t"))
    # f1 identifies the row; label is what must still be attached to it.
    pairs = list(zip(frame["f1"].tolist(), frame["label"].tolist(), strict=True))
    assert pairs == [(20.0, 1.0), (40.0, 1.0), (30.0, 0.0), (10.0, 0.0)]


def test_a_column_collects_as_numpy_with_the_requested_dtype(engine, table_csv):
    table = engine.read_table(table_csv)
    labels = engine.column(table, "label", dtype="int64")
    assert labels.dtype == np.dtype("int64")
    assert labels.tolist() == [0, 1, 0, 1]


def test_dtypes_are_reported_in_numpy_spelling_on_every_backend(engine, table_csv):
    """These reach `FeatureSchema.dtypes`, the bundle manifest and the serving
    signature — so Spark's native `bigint`/`double` would make the artifact depend
    on which engine happened to build it."""
    table = engine.read_table(table_csv)
    dtypes = engine.dtypes(table, ["f0", "label"])
    assert dtypes["f0"] == "float64"
    assert dtypes["label"] == "int64"


def test_sorting_by_a_column_with_duplicate_keys_keeps_the_original_relative_order(
    engine, table_csv
):
    """The most important test in this file.

    `builders._label_columns` sorts so the positions a splitter computes line up
    with the rows it splits. Spark's `orderBy` fixes the order of *keys* but not of
    equal keys, so without a deterministic tie-break a table with duplicate
    timestamps would produce different folds — and different scores — from one run
    to the next, silently. Rows 1 and 3 both have t=1; row 1 must stay first.
    """
    table = engine.read_table(table_csv)
    ordered = engine.sort_by(table, "t")
    assert engine.column(ordered, "t", dtype="int64").tolist() == [1, 1, 2, 3]
    # f1 is the row identity: 20.0 came before 40.0 in the file and must stay there.
    assert engine.column(ordered, "f1").tolist() == [20.0, 40.0, 30.0, 10.0]


def test_to_pandas_returns_the_whole_table_as_pandas(engine, table_csv):
    frame = engine.to_pandas(engine.read_table(table_csv))
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 4
    assert list(frame.columns) == ["f0", "f1", "t", "label"]


def test_a_directory_of_parquet_part_files_reads_as_one_table(engine, tmp_path):
    """What Spark writes. Both engines must read it identically."""
    pytest.importorskip("pyarrow")
    d = tmp_path / "processed"
    d.mkdir()
    pd.DataFrame({"f0": [1.0, 2.0], "label": [0, 1]}).to_parquet(d / "part-0.parquet")
    pd.DataFrame({"f0": [3.0, 4.0], "label": [1, 0]}).to_parquet(d / "part-1.parquet")
    assert engine.n_rows(engine.read_table(str(d))) == 4


# ── Spark-specific: the collect guard ─────────────────────────────────
@requires_spark
def test_spark_refuses_a_collect_that_would_not_fit(table_csv):
    """An OOM-killed driver leaves no traceback; a row count and the name of the
    knob that raises the ceiling does."""
    engine = get_data_backend("spark", max_collect_rows=2)
    table = engine.read_table(table_csv)
    with pytest.raises(FrameworkError) as excinfo:
        engine.to_pandas(engine.select(table, ["f0", "f1"]))
    assert "max_collect_rows" in str(excinfo.value)


@requires_spark
def test_the_guard_also_covers_the_single_column_collect(table_csv):
    """Both collecting methods are guarded, not just the big one."""
    engine = get_data_backend("spark", max_collect_rows=2)
    with pytest.raises(FrameworkError):
        engine.column(engine.read_table(table_csv), "label")
