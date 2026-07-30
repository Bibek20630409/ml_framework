import numpy as np
import pandas as pd
import pytest

from ml_framework.core import read_table


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
