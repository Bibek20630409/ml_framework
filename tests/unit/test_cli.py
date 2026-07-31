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
        "runtime": {"output_dir": str(tmp_path / "outputs"), "num_workers": 0},
        "data": {"kind": "tabular", "path": str(csv), "target": "label"},
        "model": {"name": "mlp", "params": {"hidden_dims": [16, 8]}},
        "fit": {"budget": {"max_epochs": 1}, "batch_size": 16},
        "logging": {"backend": "none"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _v1_config(tmp_path: Path, csv: Path) -> Path:
    """A v1 config file, for the migration path."""
    cfg = {
        "task": "multiclass",
        "seed": 7,
        "output_dir": str(tmp_path / "outputs"),
        "data": {"kind": "tabular", "csv_path": str(csv), "target_col": "label"},
        "model": {"name": "mlp", "hidden_dims": [16, 8], "dropout": 0.25},
        "optim": {"lr": 0.002, "weight_decay": 0.0003},
        "train": {"epochs": 3, "batch_size": 8, "num_workers": 0, "patience": 4},
        "logging": {"backend": "none"},
        "hpo_n_trials": 11,
    }
    path = tmp_path / "v1.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.unit
def test_parse_override_types():
    assert cli._parse_override("fit.params.lr=0.01") == ("fit.params.lr", 0.01)
    assert cli._parse_override("fit.budget.max_epochs=5") == ("fit.budget.max_epochs", 5)
    assert cli._parse_override("logging.backend=csv") == ("logging.backend", "csv")


@pytest.mark.unit
def test_parse_override_bad_format_raises():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_override("no_equals_sign")


@pytest.mark.unit
def test_cli_train_end_to_end(tmp_path):
    cfg_path = _write_dataset_and_config(tmp_path)
    rc = cli.main(["train", "--config", str(cfg_path), "--set", "fit.budget.max_epochs=1"])
    assert rc == 0
    out = tmp_path / "outputs"
    assert (out / "manifest.json").exists()
    assert (out / "model" / "model.ckpt").exists()


# ── migrate-config ────────────────────────────────────────
@pytest.mark.unit
def test_migrate_config_round_trips_a_v1_file_into_a_trainable_v2_config(tmp_path):
    """The phase gate: a v1 config goes in, a config that trains comes out."""
    from ml_framework.config import ExperimentConfig

    csv = _write_dataset_and_config(tmp_path).parent / "data.csv"
    src = _v1_config(tmp_path, csv)
    dest = tmp_path / "v2.yaml"

    assert cli.main(["migrate-config", "-i", str(src), "-o", str(dest)]) == 0

    cfg = ExperimentConfig.from_yaml(dest)
    assert cfg.runtime.seed == 7
    assert cfg.data.path == str(csv)
    assert cfg.data.target == "label"
    assert cfg.model.params["hidden_dims"] == [16, 8]
    assert cfg.fit.params["lr"] == pytest.approx(0.002)
    assert cfg.fit.budget.max_epochs == 3
    assert cfg.fit.patience == 4
    assert cfg.tune.max_trials == 11
    assert cfg.logging.backend == "none"


@pytest.mark.unit
def test_migrate_config_refuses_to_overwrite_without_force(tmp_path):
    csv = _write_dataset_and_config(tmp_path).parent / "data.csv"
    src = _v1_config(tmp_path, csv)
    dest = tmp_path / "v2.yaml"
    dest.write_text("existing: true", encoding="utf-8")

    assert cli.main(["migrate-config", "-i", str(src), "-o", str(dest)]) == 1
    assert dest.read_text(encoding="utf-8") == "existing: true"
    assert cli.main(["migrate-config", "-i", str(src), "-o", str(dest), "--force"]) == 0


@pytest.mark.unit
def test_migrate_config_reports_an_unmappable_key_instead_of_dropping_it(tmp_path):
    src = tmp_path / "weird.yaml"
    src.write_text(
        yaml.safe_dump({"task": "binary", "train": {"epochs": 1}, "mystery_knob": 3}),
        encoding="utf-8",
    )
    assert cli.main(["migrate-config", "-i", str(src), "-o", str(tmp_path / "out.yaml")]) == 1
    assert not (tmp_path / "out.yaml").exists()


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


# ── Tuning flags ──────────────────────────────────────────
@pytest.mark.unit
def test_no_tune_switches_tuning_off():
    """Turning it off is one flag — and so is turning it up."""
    import argparse

    from ml_framework.cli import _apply_tune_args
    from ml_framework.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(
        {"task": "binary", "data": {"kind": "tabular", "path": "d.csv", "target": "y"}}
    )
    assert cfg.tune.enabled is True

    args = argparse.Namespace(tune=False, tune_trials=None, tune_budget=None)
    assert _apply_tune_args(cfg, args).tune.enabled is False


@pytest.mark.unit
def test_tune_budget_accepts_a_readable_duration():
    """`--tune-budget 10m` rather than 600: a wall-clock budget typed as a bare
    number is easy to misread by a factor of sixty."""
    import argparse

    from ml_framework.cli import _apply_tune_args
    from ml_framework.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(
        {"task": "binary", "data": {"kind": "tabular", "path": "d.csv", "target": "y"}}
    )
    args = argparse.Namespace(tune=None, tune_trials=25, tune_budget="10m")
    tuned = _apply_tune_args(cfg, args)
    assert tuned.tune.max_trials == 25
    assert tuned.tune.max_seconds == 600.0
    assert tuned.tune.enabled is True  # untouched by the other two flags


@pytest.mark.unit
def test_cli_train_defaults_to_tuning_and_no_tune_skips_it(tmp_path):
    """The default is on; the flag is how you say otherwise."""
    import json

    from ml_framework.pipeline.tune import HPO_FILE

    cfg_path = _write_dataset_and_config(tmp_path)
    assert cli.main(["train", "--config", str(cfg_path), "--no-tune"]) == 0
    hpo = json.loads((tmp_path / "outputs" / HPO_FILE).read_text(encoding="utf-8"))
    assert hpo["ran"] is False


@pytest.mark.unit
def test_mlf_hpo_still_works_as_an_alias(tmp_path, caplog):
    """Removing a verb people have in scripts is a gratuitous break; the new
    driver answers the same question."""
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    import yaml

    csv = _write_dataset_and_config(tmp_path).parent / "data.csv"
    cfg = {
        "task": "multiclass",
        "runtime": {"output_dir": str(tmp_path / "out"), "num_workers": 0},
        "data": {"kind": "tabular", "path": str(csv), "target": "label"},
        "model": {"name": "xgboost"},
        "fit": {"params": {"n_estimators": 10, "early_stopping_rounds": 0}},
        "tune": {"enabled": True, "max_trials": 2},
        "logging": {"backend": "none"},
    }
    path = tmp_path / "gbdt.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert cli.main(["hpo", "--config", str(path)]) == 0
    assert any("use `mlf tune`" in r.message for r in caplog.records)


@pytest.mark.unit
def test_tune_emit_config_writes_the_winner(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    import yaml

    from ml_framework.config import ExperimentConfig

    csv = _write_dataset_and_config(tmp_path).parent / "data.csv"
    cfg = {
        "task": "multiclass",
        "runtime": {"output_dir": str(tmp_path / "out"), "num_workers": 0},
        "data": {"kind": "tabular", "path": str(csv), "target": "label"},
        "model": {"name": "xgboost"},
        "fit": {"params": {"n_estimators": 10, "early_stopping_rounds": 0}},
        "tune": {"enabled": True, "max_trials": 2},
        "logging": {"backend": "none"},
    }
    path = tmp_path / "gbdt.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    dest = tmp_path / "tuned.yaml"

    assert cli.main(["tune", "--config", str(path), "--emit-config", str(dest)]) == 0
    assert ExperimentConfig.from_yaml(dest).model.name == "xgboost"
