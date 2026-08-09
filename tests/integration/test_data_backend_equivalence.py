"""One CSV, several engines, one bundle — the test that makes this a choice.

Everything in ``tests/unit/test_data_backends.py`` is per-*method*: it asserts
that ``sort_by`` sorts and ``dtypes`` normalizes. This file asserts the thing a
user actually cares about, which no per-method test implies: that selecting a
different engine changes **how** the data is read and **nothing about what comes
out**.

Without it, "for this run, use the Spark implementation" describes several
products that happen to share a CLI flag rather than one product with a switch.

Deferred until Phase 2 on purpose. Before it, ``read_table`` collected to pandas
immediately and every engine ran byte-identical code after the read, so this would
have asserted a tautology. Now the engine reaches into ``build_bundle`` — schema
validation, dtype normalization and the collect all go through it — and the
assertion has something to catch.

**Phase 4 is what made it run.** Parameterized over every non-default engine, and
Polars needs no JVM, so on an ordinary dev machine these assertions now *execute*
rather than skip. That is the difference between a property believed and a property
tested; the Spark parameter still skips without a JVM and is covered by
`spark-contract`.
"""

from __future__ import annotations

import importlib.util
import os
import shutil

import numpy as np
import pytest

from ml_framework.data.builders import _cv_population, build_bundle

pytestmark = pytest.mark.integration


def _spark_runnable() -> bool:
    """pyspark importable **and** a JVM on PATH. See test_data_backends.py."""
    if importlib.util.find_spec("pyspark") is None:
        return False
    return bool(shutil.which("java") or os.environ.get("JAVA_HOME"))


requires_spark = pytest.mark.skipif(not _spark_runnable(), reason="pyspark needs a JVM on PATH")
requires_polars = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None, reason="polars is not installed"
)


@pytest.fixture(
    params=[
        pytest.param("spark", marks=requires_spark),
        pytest.param("polars", marks=requires_polars),
    ]
)
def other(request):
    """Each non-default engine in turn. `local` is always the oracle."""
    return request.param


def _bundles(make_config, csv, other):
    """The same config under `local` and under `other`."""
    base = {"task": "multiclass", "model": "xgboost"}
    local = build_bundle(make_config(csv, **base))
    alt = build_bundle(make_config(csv, **base, **{"data.backend": other}))
    return local, alt


def test_every_engine_builds_the_same_bundle_from_the_same_csv(make_config, tabular_csv, other):
    """Same file, same seed, different engine → indistinguishable bundles."""
    local, alt = _bundles(make_config, tabular_csv, other)

    for name in ("train", "val", "test"):
        got, want = getattr(alt, name), getattr(local, name)
        assert np.allclose(got.x, want.x), f"{other}: {name}.x differs from local"
        assert np.array_equal(got.y, want.y), f"{other}: {name}.y differs from local"

    assert alt.input_dim == local.input_dim
    assert alt.output_dim == local.output_dim


def test_the_schema_does_not_record_which_engine_built_it(make_config, tabular_csv, other):
    """`FeatureSchema.dtypes` reaches the bundle manifest and the serving
    signature. If an engine's native type names leaked through — Spark's
    `bigint`/`double`, Polars' `Int64`/`String` — an artifact would differ by
    engine and a serving-side column check could reject valid input."""
    local, alt = _bundles(make_config, tabular_csv, other)

    assert alt.schema.feature_names == local.schema.feature_names
    assert alt.schema.dtypes == local.schema.dtypes
    assert alt.schema.target_name == local.schema.target_name


def test_cross_validation_folds_agree_between_engines(make_config, tabular_csv, other):
    """Fold planning is the one place an engine works without collecting the
    feature matrix, so it is the one place its row count and label vector have to
    match pandas exactly — a disagreement here silently changes every fold."""
    base = {"task": "multiclass", "model": "xgboost"}
    n_local, y_local = _cv_population(make_config(tabular_csv, **base))
    n_alt, y_alt = _cv_population(make_config(tabular_csv, **base, **{"data.backend": other}))

    assert n_alt == n_local
    assert np.array_equal(np.sort(y_alt), np.sort(y_local))


def test_a_regression_target_survives_the_round_trip_identically(
    make_config, regression_csv, other
):
    """Float targets take a different `dtype=None` path through `column()` than the
    int cast a classification label gets, so they are worth asserting separately."""
    base = {"task": "regression", "model": "xgboost"}
    local = build_bundle(make_config(regression_csv, **base))
    alt = build_bundle(make_config(regression_csv, **base, **{"data.backend": other}))

    assert np.allclose(alt.train.x, local.train.x)
    assert np.allclose(alt.train.y, local.train.y)
    assert alt.schema.dtypes == local.schema.dtypes
