"""Shared fixtures: synthetic datasets and configs written to tmp dirs.

Also pins HuggingFace to its local cache — see :func:`_pin_hf_to_its_cache`, which
runs at *import* time and therefore before anything else in the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_framework.config import ExperimentConfig

# ── HuggingFace: cache-only once the checkpoints are present ──────────
# The tiny checkpoints the NLP tests fine-tune. Named here rather than imported
# from the test modules because this has to run before those modules are even
# collected.
HF_TEST_CHECKPOINTS: tuple[str, ...] = (
    "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
    "hf-internal-testing/tiny-random-BartForConditionalGeneration",
)


def _hf_cache_dir() -> Path:
    """Where `huggingface_hub` keeps its blobs, resolved **without importing it**.

    Importing the library to ask would defeat the purpose: `HF_HUB_OFFLINE` is
    read into a module constant at import time, so it has to be set before the
    first import or it does nothing at all. That single fact is what shapes this
    whole function — the precedence below mirrors the library's own.
    """
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _pin_hf_to_its_cache() -> bool:
    """Set ``HF_HUB_OFFLINE=1`` when every test checkpoint is already downloaded.

    **Why this exists.** ``from_pretrained`` revalidates against the hub on every
    call, even for a fully cached model. That is a HEAD request per file, and when
    DNS fails rather than returning cleanly, `huggingface_hub` retries five times
    with exponential backoff and then errors. A single network blip during a run
    therefore turns green NLP tests into errors — which is exactly what happened
    once, six of them, and cost eleven minutes of retry backoff on the way.

    Nothing about those tests needs the network: the checkpoints are ~90 KB and
    cached after the first run. So once they are on disk, the suite is told to
    stop asking.

    Deliberately **not** unconditional: on a cold machine the first run has to
    download, and forcing offline there would fail with a cache miss rather than
    fetching. Absence of the cache is the signal to leave the network alone.

    Returns whether the pin was applied, so a test can assert the mechanism.
    """
    # "0" is a *string* and therefore truthy, so a bare truthiness check here
    # would read `HF_HUB_OFFLINE=0` — which means "stay online" — as "already
    # pinned". Matching the library's own reading of the variable.
    if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False"):
        return True  # already pinned by the caller (CI, or a developer)

    cache = _hf_cache_dir()
    for repo in HF_TEST_CHECKPOINTS:
        # `models--org--name` is the on-disk layout, and checking it is pure
        # filesystem — no import, no network, no exception to catch.
        if not (cache / f"models--{repo.replace('/', '--')}").is_dir():
            return False

    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


HF_PINNED = _pin_hf_to_its_cache()


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
            # Tuning is on by default for users and off by default for tests: a
            # test of the bundle layout should not spend 300 s searching. The
            # tuning path has its own tests, which opt back in explicitly.
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
        cfg = ExperimentConfig.model_validate(raw)
        return cfg.with_overrides(overrides) if overrides else cfg

    return _factory
