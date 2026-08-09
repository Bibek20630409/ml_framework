import numpy as np
import pandas as pd
import pytest

from ml_framework.core import read_table
from ml_framework.core.plugins import MissingExtraError
from ml_framework.core.types import Requirement

# The parquet guard is patched on the module that *reads* it. `read_table` is now
# a shim over the selected data backend, and the requirement moved to the local
# backend with the pandas read it guards; `tabular` only re-exports the name.
from ml_framework.data.backends import local


@pytest.mark.unit
def test_read_table_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out = read_table(str(p))
    assert list(out.columns) == ["a", "label"]
    assert len(out) == 2


@pytest.mark.unit
def test_read_table_parquet_file(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"f0": np.arange(5, dtype="float32"), "label": [0, 1, 2, 0, 1]})
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    out = read_table(str(p))
    assert len(out) == 5
    assert "label" in out.columns


@pytest.mark.unit
def test_read_table_parquet_dir(tmp_path):
    """A directory of parquet part-files (what Spark writes) is read transparently."""
    pytest.importorskip("pyarrow")
    d = tmp_path / "processed"
    d.mkdir()
    pd.DataFrame({"f0": [1.0, 2.0], "label": [0, 1]}).to_parquet(d / "part-0.parquet")
    pd.DataFrame({"f0": [3.0, 4.0], "label": [1, 0]}).to_parquet(d / "part-1.parquet")
    out = read_table(str(d))
    assert len(out) == 4


@pytest.mark.unit
def test_missing_parquet_engine_names_the_extra_not_a_pandas_import_error(monkeypatch, tmp_path):
    """Parquet needs an engine that is not a pandas dependency.

    The failure a user actually hits is on a lean install — a serving image built
    without the mlops extra, where pyarrow no longer arrives via mlflow. They
    should get the pip command, not `ImportError: Unable to find a usable engine`.
    """
    monkeypatch.setattr(
        local,
        "PARQUET_REQUIREMENT",
        Requirement("definitely_not_pyarrow_xyz", extra="parquet", min_version="10.0.1"),
    )
    with pytest.raises(MissingExtraError) as excinfo:
        read_table(str(tmp_path / "d.parquet"))
    assert "pip install 'ml-framework[parquet]'" in str(excinfo.value)


@pytest.mark.unit
def test_csv_reading_never_consults_the_parquet_engine(monkeypatch, tmp_path):
    """The guard must gate only the parquet branch — a CSV-only install stays free
    of pyarrow entirely."""
    monkeypatch.setattr(
        local,
        "PARQUET_REQUIREMENT",
        Requirement("definitely_not_pyarrow_xyz", extra="parquet"),
    )
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1], "label": [0]}).to_csv(p, index=False)
    assert len(read_table(str(p))) == 1
