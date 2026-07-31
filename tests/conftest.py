"""Shared fixtures: synthetic datasets and configs written to tmp dirs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_framework.config import ExperimentConfig


def _make_classification_csv(
    path: Path, n: int, n_features: int, n_classes: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype("float32")
    # separable-ish signal so training actually reduces loss
    w = rng.normal(size=(n_features, n_classes))
    logits = x @ w
    y = logits.argmax(axis=1)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    df.to_csv(path, index=False)


def _make_regression_csv(path: Path, n: int, n_features: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype("float32")
    w = rng.normal(size=n_features)
    y = (x @ w + rng.normal(scale=0.1, size=n)).astype("float32")
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    df.to_csv(path, index=False)


@pytest.fixture
def tabular_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "data.csv"
    _make_classification_csv(csv, n=200, n_features=6, n_classes=3, seed=0)
    return csv


@pytest.fixture
def binary_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "binary.csv"
    _make_classification_csv(csv, n=200, n_features=5, n_classes=2, seed=1)
    return csv


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "reg.csv"
    _make_regression_csv(csv, n=200, n_features=5, seed=2)
    return csv


@pytest.fixture
def make_config(tmp_path: Path):
    """A minimal valid v2 config, overridable with dotted keys.

    Overrides go through ``with_overrides`` rather than a dict merge on purpose:
    every test that tweaks a config exercises the same mechanism HPO uses to
    apply a trial.
    """

    def _factory(csv: Path, task: str, *, model: str = "mlp", **overrides) -> ExperimentConfig:
        # `model` is set in the raw dict rather than passed as an override on
        # purpose: switching model.name afterwards would leave the previous
        # plugin's params in place, and the config validator would then — quite
        # correctly — reject `hidden_dims` as not a knob xgboost has.
        raw = {
            "task": task,
            "runtime": {
                "seed": 42,
                "output_dir": str(tmp_path / "outputs"),
                "num_workers": 0,
            },
            "data": {"kind": "tabular", "path": str(csv), "target": "label"},
            "model": {"name": model},
            "fit": {"budget": {"max_epochs": 2}, "batch_size": 16},
            "logging": {"backend": "none"},
        }
        cfg = ExperimentConfig.model_validate(raw)
        return cfg.with_overrides(overrides) if overrides else cfg

    return _factory
