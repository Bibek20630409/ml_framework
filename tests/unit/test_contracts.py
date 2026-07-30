import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")

from ml_framework.pipeline.contracts import validate_dataframe  # noqa: E402


def _good_frame(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "label": rng.integers(0, 3, size=n),
        }
    )


@pytest.mark.unit
def test_valid_frame_passes():
    df = _good_frame()
    out = validate_dataframe(df, "label")
    assert len(out) == len(df)


@pytest.mark.unit
def test_null_target_fails():
    import pandera.errors as pae

    df = _good_frame()
    df.loc[0, "label"] = np.nan
    with pytest.raises(pae.SchemaError):
        validate_dataframe(df, "label")


@pytest.mark.unit
def test_null_feature_fails():
    import pandera.errors as pae

    df = _good_frame()
    df.loc[1, "f0"] = np.nan
    with pytest.raises(pae.SchemaError):
        validate_dataframe(df, "label")


@pytest.mark.unit
def test_duplicate_rows_fail():
    import pandera.errors as pae

    df = _good_frame(10)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # inject a duplicate
    with pytest.raises(pae.SchemaError):
        validate_dataframe(df, "label")


@pytest.mark.unit
def test_validate_file_csv(tmp_path):
    from ml_framework.pipeline.contracts import validate_file

    df = _good_frame(20)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    assert validate_file(str(path), "label") == 20
