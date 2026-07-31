"""Bundles written before the manifest existed still load and predict.

Plan §7.8 and decision 15: the clean break was authorized for **configs and
tests**, not for bundles already deployed. A v1 bundle is `metadata.json` +
`model.ckpt` + `scaler.pkl` at the root, with no `manifest.json` — and P3 removed
the code that *wrote* that layout, so the only way to keep the promise honest is
to build one the way v1 did and load it with today's code.

The v1 layout is reconstructed here from a current bundle rather than checked in
as a binary fixture, so it cannot silently rot into something v1 never produced.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.bundle import detect_bundle_version, is_v1_bundle
from ml_framework.core.inference import Inferencer
from ml_framework.pipeline import train

pytestmark = pytest.mark.integration


@pytest.fixture
def v1_bundle(tabular_csv, make_config, tmp_path) -> tuple[Path, np.ndarray]:
    """A v1-layout bundle, plus the predictions the v2 bundle makes from it."""
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    v2 = Path(cfg.runtime.output_dir)

    expected = Inferencer.from_artifacts(v2).predict(np.zeros((4, 6), dtype="float32"))

    # Rebuild exactly what v1's `_write_v1_artifacts` produced.
    legacy = tmp_path / "v1_bundle"
    legacy.mkdir()
    shutil.copyfile(v2 / "model" / "model.ckpt", legacy / "model.ckpt")
    shutil.copyfile(v2 / "preprocessor" / "scaler.pkl", legacy / "scaler.pkl")
    shutil.copyfile(v2 / "reference_stats.json", legacy / "reference_stats.json")

    config = json.loads((v2 / "config.json").read_text(encoding="utf-8"))
    # v1's ModelConfig was flat and carried *every* model's knobs for every model —
    # an MLP bundle recorded `backbone` and `pretrained`. Reproduced faithfully,
    # because the loader has to survive it.
    config["model"] = {
        "name": "mlp",
        **config["model"]["params"],
        "backbone": "resnet18",
        "pretrained": True,
    }
    (legacy / "metadata.json").write_text(
        json.dumps(
            {
                "task": "multiclass",
                "input_dim": 6,
                "output_dim": 3,
                "feature_cols": [f"f{i}" for i in range(6)],
                "class_names": None,
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return legacy, expected


def test_the_fixture_really_is_a_v1_bundle(v1_bundle):
    """Guards the guard: if this stopped being detected as v1, the tests below
    would be exercising the v2 path and proving nothing."""
    legacy, _ = v1_bundle
    assert is_v1_bundle(legacy)
    assert detect_bundle_version(legacy) == 1
    assert not (legacy / "manifest.json").exists()


def test_a_v1_bundle_loads_and_predicts_identically(v1_bundle):
    legacy, expected = v1_bundle
    inf = Inferencer.from_artifacts(legacy)
    assert np.array_equal(inf.predict(np.zeros((4, 6), dtype="float32")), expected)


def test_a_v1_bundle_gets_a_synthesized_signature(v1_bundle):
    """Everything downstream — the serving schemas, the proba gate, /health —
    reads the manifest, so a v1 bundle has to present one."""
    legacy, _ = v1_bundle
    inf = Inferencer.from_artifacts(legacy)
    assert inf.manifest.bundle_version == 1
    assert inf.backend_name == "lightning"
    assert inf.n_features == 6
    assert inf.feature_cols == [f"f{i}" for i in range(6)]
    assert inf.signature.output.n_classes == 3
    assert inf.produces_proba


def test_a_v1_bundle_still_applies_its_scaler(v1_bundle):
    """The `scaler.pkl` special case is gone from the loader, so the legacy branch
    has to route through the preprocessor like everything else."""
    legacy, _ = v1_bundle
    inf = Inferencer.from_artifacts(legacy)
    assert inf.preprocessor is not None
    raw = np.full((2, 6), 5.0, dtype="float32")
    assert not np.allclose(inf.preprocessor.transform(raw), raw)


def test_v1_model_params_are_filtered_to_what_the_plugin_declares(v1_bundle):
    """v1 recorded `backbone`/`pretrained` on an MLP. Passing those to the v2
    params schema would fail `extra="forbid"`, so the loader keeps only the keys
    the plugin actually declares."""
    legacy, _ = v1_bundle
    manifest = Inferencer.from_artifacts(legacy).manifest
    assert set(manifest.model.params) == {"hidden_dims", "dropout"}


def test_a_v1_bundle_serves(v1_bundle):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ml_framework.serving.api import create_app

    legacy, _ = v1_bundle
    with TestClient(create_app(legacy)) as client:
        assert client.get("/health").json()["bundle_version"] == 1
        res = client.post("/predict", json={"instances": [[0.0] * 6]})
        assert res.status_code == 200
