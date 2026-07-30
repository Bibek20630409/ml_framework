from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_framework import cli


def _write_dataset_and_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4)).astype("float32")
    w = rng.normal(size=(4, 3))
    y = (x @ w).argmax(axis=1)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    df["label"] = y
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    cfg = {
        "task": "multiclass",
        "output_dir": str(tmp_path / "outputs"),
        "data": {"kind": "tabular", "csv_path": str(csv), "target_col": "label"},
        "model": {"name": "mlp", "hidden_dims": [16, 8]},
        "train": {"epochs": 1, "batch_size": 16, "num_workers": 0},
        "logging": {"backend": "none"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.mark.unit
def test_parse_override_types():
    assert cli._parse_override("optim.lr=0.01") == ("optim.lr", 0.01)
    assert cli._parse_override("train.epochs=5") == ("train.epochs", 5)
    assert cli._parse_override("logging.backend=csv") == ("logging.backend", "csv")


@pytest.mark.unit
def test_parse_override_bad_format_raises():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_override("no_equals_sign")


@pytest.mark.unit
def test_cli_train_end_to_end(tmp_path):
    cfg_path = _write_dataset_and_config(tmp_path)
    rc = cli.main(["train", "--config", str(cfg_path), "--set", "train.epochs=1"])
    assert rc == 0
    out = tmp_path / "outputs"
    assert (out / "model.ckpt").exists()
    assert (out / "metadata.json").exists()


@pytest.mark.unit
def test_cli_serve_invokes_uvicorn(monkeypatch, tmp_path):
    called = {}

    def fake_run(app, host, port):  # noqa: ANN001
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    # create_app is imported lazily inside main; stub it so no artifacts needed.
    monkeypatch.setattr("ml_framework.serving.api.create_app", lambda d, **kw: object())

    rc = cli.main(["serve", "--artifacts", str(tmp_path), "--host", "1.2.3.4", "--port", "9999"])
    assert rc == 0
    assert called == {"host": "1.2.3.4", "port": 9999}
