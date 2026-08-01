"""Dataset sniffing: what it infers, and what it refuses to guess.

Most of this file is about the refusals. A sniffer that always answers is worse
than one that sometimes asks, because a wrong answer here is not an error — it is
a model trained on the wrong column, reporting a plausible score. So two columns
named `label` and `target` raise, two prose columns raise, and the one genuinely
weak rule (fall back to the last column) is marked `weak` so the CLI can say it
loudly and `mlf init` can write GUESS beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_framework.data.sniff import (
    MAX_CLASSES,
    TEXT_TOKEN_THRESHOLD,
    SniffError,
    sniff,
)


def write_csv(path: Path, **columns) -> Path:
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


# ── Kind ──────────────────────────────────────────────────
@pytest.mark.unit
def test_a_plain_table_is_tabular(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0, 2.0, 3.0, 4.0], label=[0, 1, 0, 1])

    found = sniff(csv)
    assert found.kind == "tabular"
    assert found.target == "label"


@pytest.mark.unit
def test_a_sorted_datetime_column_makes_it_a_time_series(tmp_path):
    csv = write_csv(
        tmp_path / "d.csv",
        date=pd.date_range("2024-01-01", periods=10).astype(str),
        value=np.arange(10.0),
    )

    found = sniff(csv)
    assert found.kind == "timeseries"
    assert found.time_col == "date"
    assert found.task == "forecasting"


@pytest.mark.unit
def test_an_unsorted_datetime_column_is_not_a_time_series(tmp_path):
    """Parsing alone would make a table of birthdays a series. Monotonicity is
    what says the rows are *ordered by* time, which is the thing that makes
    forecasting meaningful and a shuffled split wrong."""
    dates = list(pd.date_range("2024-01-01", periods=6).astype(str))
    csv = write_csv(tmp_path / "d.csv", date=dates[::-1], label=[0, 1, 0, 1, 0, 1])

    assert sniff(csv).kind == "tabular"


@pytest.mark.unit
def test_a_declared_time_col_that_is_unsorted_is_an_error(tmp_path):
    """Silently treating it as tabular would ignore what the user asked for."""
    dates = list(pd.date_range("2024-01-01", periods=6).astype(str))
    csv = write_csv(tmp_path / "d.csv", when=dates[::-1], value=np.arange(6.0))

    with pytest.raises(SniffError, match="not sorted"):
        sniff(csv, time_col="when")


@pytest.mark.unit
def test_an_integer_id_column_is_not_mistaken_for_a_timestamp(tmp_path):
    """Every integer parses as an epoch. Only a *declared* numeric column is
    treated as time, because otherwise every table with a row id is a series."""
    csv = write_csv(tmp_path / "d.csv", row_id=list(range(8)), label=[0, 1] * 4)

    assert sniff(csv).kind == "tabular"


@pytest.mark.unit
def test_a_prose_column_makes_it_text(tmp_path):
    csv = write_csv(
        tmp_path / "d.csv",
        text=["this is a much longer sentence of real prose"] * 6,
        label=["a", "b"] * 3,
    )

    found = sniff(csv)
    assert found.kind == "text"
    assert found.text_col == "text"


@pytest.mark.unit
def test_a_short_string_column_stays_a_tabular_feature(tmp_path):
    """The threshold is doing real work: a column of category names is a
    *feature*, and treating it as the text a transformer should read would
    fine-tune a 66M-parameter encoder on the word "red"."""
    csv = write_csv(tmp_path / "d.csv", colour=["red", "blue"] * 4, label=[0, 1] * 4)

    assert sniff(csv).kind == "tabular"


@pytest.mark.unit
def test_jsonl_is_text_whatever_is_inside_it(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "a b", "label": 0}\n{"text": "c d", "label": 1}\n', encoding="utf-8")

    assert sniff(path).kind == "text"


@pytest.mark.unit
def test_a_directory_of_class_directories_of_images_is_image(tmp_path):
    pytest.importorskip("PIL", reason="the image extra is not installed")
    from PIL import Image

    root = tmp_path / "corpus"
    for name in ("cat", "dog"):
        (root / name).mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype="uint8")).save(root / name / "0.png")

    found = sniff(root)
    assert found.kind == "image"
    assert found.task == "binary"  # two class directories


@pytest.mark.unit
def test_a_directory_with_no_images_is_not_an_image_folder(tmp_path):
    """Checked positively, because a directory of parquet part-files is also a
    directory and it *is* a table (Spark writes them that way).

    So this folder falls through to the table reader and fails there — and the
    message has to come from the sniffer rather than from pyarrow, which would
    report a schema error that says nothing about what the user should do.
    """
    root = tmp_path / "parts"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "part-0.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SniffError, match="could not read"):
        sniff(root)


# ── Target ────────────────────────────────────────────────
@pytest.mark.unit
def test_a_conventionally_named_column_is_the_target(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 4, y=[0, 1, 0, 1])

    assert sniff(csv).target == "y"


@pytest.mark.unit
def test_two_conventionally_named_columns_is_a_refusal(tmp_path):
    """Not a tie to be broken by column order — a question only the user can
    answer, and answering it wrong trains on the wrong column."""
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 4, label=[0, 1, 0, 1], target=[1, 0, 1, 0])

    with pytest.raises(SniffError, match="several columns could be the target"):
        sniff(csv)


@pytest.mark.unit
def test_an_explicit_target_settles_it(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 4, label=[0, 1, 0, 1], target=[1, 0, 1, 0])

    found = sniff(csv, target="target")
    assert found.target == "target"
    assert any(i.field == "data.target" and i.rule == "given explicitly" for i in found.inferences)


@pytest.mark.unit
def test_the_last_column_fallback_is_marked_weak(tmp_path):
    """Right often enough to be worth doing, wrong often enough to say out loud.
    The `weak` flag is what makes the CLI log it at WARNING and `mlf init` write
    GUESS beside it."""
    csv = write_csv(tmp_path / "d.csv", a=[1.0] * 4, b=[2.0] * 4, outcome=[0, 1, 0, 1])

    found = sniff(csv)
    target = next(i for i in found.inferences if i.field == "data.target")
    assert target.value == "outcome"
    assert target.weak is True


@pytest.mark.unit
def test_an_unknown_target_column_is_an_error(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 4, label=[0, 1, 0, 1])

    with pytest.raises(SniffError, match="not a column"):
        sniff(csv, target="nope")


# ── Task ──────────────────────────────────────────────────
@pytest.mark.unit
def test_two_distinct_values_is_binary(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 6, label=[0, 1] * 3)

    assert sniff(csv).task == "binary"


@pytest.mark.unit
def test_a_few_non_float_values_is_multiclass(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 6, label=[0, 1, 2] * 2)

    assert sniff(csv).task == "multiclass"


@pytest.mark.unit
def test_a_float_target_is_regression(tmp_path):
    """Even with few distinct values. A float column is a quantity; three of them
    is a small sample, not three classes."""
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 6, label=[0.5, 1.5, 2.5] * 2)

    assert sniff(csv).task == "regression"


@pytest.mark.unit
def test_many_integer_values_is_regression_not_a_huge_head(tmp_path):
    """Above the threshold, treating it as classification would build a head with
    hundreds of logits for what is obviously a count."""
    csv = write_csv(
        tmp_path / "d.csv",
        f0=[1.0] * (MAX_CLASSES + 5),
        label=list(range(MAX_CLASSES + 5)),
    )

    assert sniff(csv).task == "regression"


@pytest.mark.unit
def test_a_constant_target_is_refused(tmp_path):
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 4, label=[7, 7, 7, 7])

    with pytest.raises(SniffError, match="nothing to learn"):
        sniff(csv)


# ── The record it leaves ──────────────────────────────────
@pytest.mark.unit
def test_every_inference_carries_the_rule_that_produced_it(tmp_path):
    """The difference between zero-config being a convenience and a black box."""
    csv = write_csv(tmp_path / "d.csv", f0=[1.0] * 6, label=[0, 1] * 3)

    found = sniff(csv)
    fields = {i.field for i in found.inferences}
    assert {"data.kind", "data.target", "task"} <= fields
    assert all(i.rule for i in found.inferences), "an inference with no reason is a black box"


@pytest.mark.unit
def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(SniffError, match="no such dataset"):
        sniff(tmp_path / "absent.csv")


@pytest.mark.unit
def test_the_text_threshold_sits_between_prose_and_categories():
    """Pinned because moving it silently reroutes datasets to a different source.
    A sentiment corpus averages 15-30 tokens; a colour column averages 1."""
    assert 1 < TEXT_TOKEN_THRESHOLD < 10
