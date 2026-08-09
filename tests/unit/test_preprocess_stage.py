"""The preprocessing stage, now expressed on the DataBackend protocol.

Before Phase 3 this module imported pyspark directly, so none of it could be
tested without a JVM and the DVC/Airflow stage was covered by `py_compile` alone.
Running it through the protocol is what makes the cleaning itself testable: these
tests exercise the real `preprocess`, so the *logic* — which rows survive, what the
target is cast to, what the output looks like on disk — is verified, and only the
engine underneath varies.

Parameterized over every **in-process** engine, so Phase 4's Polars backend has to
clean identically to pandas rather than merely register. `spark` is excluded here
on purpose: it needs a JVM, and `spark-contract` runs the same code path against a
real session.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from ml_framework.core import read_table
from ml_framework.pipeline.spark_preprocess import main, preprocess

pytestmark = pytest.mark.unit

requires_polars = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None, reason="polars is not installed"
)


@pytest.fixture(params=["local", pytest.param("polars", marks=requires_polars)])
def engine_name(request):
    """The in-process engines. Every assertion below must hold for each."""
    return request.param


@pytest.fixture
def messy_csv(tmp_path):
    """One row of each thing the stage is supposed to remove, plus survivors."""
    frame = pd.DataFrame(
        {
            "f0": [1.0, 2.0, np.nan, 4.0, 4.0, 6.0],
            "f1": [10.0, 20.0, np.nan, 40.0, 40.0, 60.0],
            "label": [0, 1, 0, 1, 1, None],
        }
    )
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_the_stage_drops_null_targets_all_null_rows_and_duplicates(
    messy_csv, tmp_path, engine_name
):
    out = str(tmp_path / "processed")
    preprocess(messy_csv, out, "label", backend=engine_name)

    frame = read_table(out)
    # Of six rows: one has a null target, one is null across every feature, one
    # duplicates its predecessor. Three survive.
    assert len(frame) == 3
    assert frame["f0"].tolist() == [1.0, 2.0, 4.0]
    assert not frame["label"].isna().any()


def test_a_single_missing_feature_is_not_grounds_for_dropping_a_row(tmp_path, engine_name):
    """`drop_all_null_rows` is "all", not "any" — a partially observed row still
    carries signal, and dropping it would quietly shrink the dataset."""
    src = tmp_path / "partial.csv"
    pd.DataFrame({"f0": [1.0, np.nan], "f1": [np.nan, 20.0], "label": [0, 1]}).to_csv(
        src, index=False
    )
    out = str(tmp_path / "processed")
    preprocess(str(src), out, "label", backend=engine_name)
    assert len(read_table(out)) == 2


def test_the_target_is_cast_to_a_stable_type(messy_csv, tmp_path, engine_name):
    """Downstream reads should not have to care whether the raw file typed the
    label as int or float."""
    out = str(tmp_path / "processed")
    preprocess(messy_csv, out, "label", backend=engine_name)
    assert str(read_table(out)["label"].dtype) == "float64"


def test_the_output_is_a_directory_of_part_files_on_every_backend(messy_csv, tmp_path, engine_name):
    """The shape matters, not just the bytes: `read_table` tells Parquet from CSV
    by inspecting the path, so a bare suffix-less file would be read as a CSV."""
    out = tmp_path / "processed"
    preprocess(messy_csv, str(out), "label", backend=engine_name)
    assert out.is_dir()
    assert list(out.glob("*.parquet"))


def test_rerunning_over_fewer_rows_does_not_leave_the_previous_output_behind(
    messy_csv, tmp_path, engine_name
):
    """Spark's `mode("overwrite")` clears the directory. `local` has to as well —
    otherwise a rerun unions with its own previous output, silently."""
    out = str(tmp_path / "processed")
    preprocess(messy_csv, out, "label", backend=engine_name)

    smaller = tmp_path / "one.csv"
    pd.DataFrame({"f0": [1.0], "f1": [10.0], "label": [0]}).to_csv(smaller, index=False)
    preprocess(str(smaller), out, "label", backend=engine_name)

    assert len(read_table(out)) == 1


def test_the_cleaning_steps_can_be_turned_off(messy_csv, tmp_path, engine_name):
    """`--no-dropna` / `--no-dedup` still reach the engine."""
    out = str(tmp_path / "processed")
    preprocess(messy_csv, out, "label", dropna=False, deduplicate=False, backend=engine_name)
    # Only the null-target row goes; the all-null row and the duplicate stay.
    assert len(read_table(out)) == 5


# ── the CLI contract DVC and Airflow depend on ────────────────────────
def test_the_documented_command_line_still_works(messy_csv, tmp_path, engine_name):
    """`dvc.yaml` and the Airflow BashOperator both invoke exactly this shape.
    The new flag had to be additive and defaulted or both would break."""
    out = str(tmp_path / "processed")
    assert (
        main(
            [
                "--input",
                messy_csv,
                "--output",
                out,
                "--target-col",
                "label",
                "--data-backend",
                engine_name,
            ]
        )
        == 0
    )
    assert len(read_table(out)) == 3


def test_the_engine_defaults_to_spark_so_the_pipeline_keeps_its_meaning(
    messy_csv, tmp_path, monkeypatch
):
    """This is the *distributed* stage. Making `local` the default would quietly
    change what the DVC/Airflow pipeline does, so the default stays `spark` and
    running locally is opt-in.

    `preprocess` is stubbed rather than run: asserting the default means invoking
    `main` without `--data-backend`, and actually honouring that default would
    reach for a JVM this machine does not have.
    """
    import inspect

    import ml_framework.pipeline.spark_preprocess as stage

    assert inspect.signature(preprocess).parameters["backend"].default == "spark"

    seen: dict = {}
    monkeypatch.setattr(stage, "preprocess", lambda *args, **kwargs: seen.update(kwargs))
    stage.main(["--input", messy_csv, "--output", str(tmp_path / "o"), "--target-col", "label"])
    assert seen["backend"] == "spark"


# ── regressions found by running the stage, not by reading it ──────────
def test_a_table_whose_only_column_is_the_target_keeps_its_rows(tmp_path, engine_name):
    """`feature_cols` comes out empty here, and both engines got this wrong.

    `pandas.dropna(how="all", subset=[])` drops **every** row — vacuously, all zero
    of the named columns are null in each — so the stage silently produced an empty
    output. Polars raised instead. Both now short-circuit on an empty subset.
    """
    src = tmp_path / "only_target.csv"
    pd.DataFrame({"label": [0, 1, 1, None]}).to_csv(src, index=False)
    out = str(tmp_path / "processed")

    preprocess(str(src), out, "label", backend=engine_name)

    frame = read_table(out)
    # 4 rows in: one null target dropped, then the duplicate `1` de-duplicated.
    assert frame["label"].tolist() == [0.0, 1.0]


def test_a_quoted_empty_field_is_a_missing_value_on_every_engine(tmp_path, engine_name):
    """`pandas.to_csv` writes a lone missing value in a single-column frame as `""`.

    pandas reads that back as NaN; polars read it as the *string* `""` and inferred
    the column as String, so the target cast then failed. The polars reader is
    aligned to pandas with `null_values=[""]` — `local` is the oracle.
    """
    src = tmp_path / "quoted.csv"
    src.write_text('label\n0.0\n1.0\n""\n', encoding="utf-8")
    out = str(tmp_path / "processed")

    preprocess(str(src), out, "label", backend=engine_name)

    frame = read_table(out)
    assert str(frame["label"].dtype) == "float64"
    assert frame["label"].tolist() == [0.0, 1.0]
