"""**The P3 phase gate**: a GBDT bundle loads and serves without torch.

The plan states it as one sentence — "``mlf train --model xgboost`` produces a
bundle that serves correctly in an environment where torch is not installed" —
and calls it the single test that proves the whole abstraction. If the backend
split is real, ``core/inference.py`` never reaches for a checkpoint loader; if it
is not, this fails.

Torch *is* installed in this environment, so asserting "torch is absent" here
would prove nothing. Instead each check runs in a subprocess with a meta-path
finder that makes ``import torch`` raise ``ModuleNotFoundError`` — a stricter
condition than absence, because it also catches an import that would have
succeeded by accident. The genuinely torch-free run belongs to CI's
``serve-gbdt`` image, and to a fresh venv (see docs/PHASE_STATUS.md).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost", reason="the gbdt extra is not installed")

pytestmark = pytest.mark.integration

# Installed ahead of everything else, so any torch import below is fatal rather
# than merely unnecessary.
_BAN_TORCH = """
import sys

BANNED = ("torch", "pytorch_lightning", "torchvision", "torchmetrics")


class _Ban:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ModuleNotFoundError(f"{name} is banned in this process")
        return None


sys.meta_path.insert(0, _Ban())
for name in list(sys.modules):
    if name.split(".")[0] in BANNED:
        del sys.modules[name]
"""


def _run_without_torch(body: str, cwd: Path) -> subprocess.CompletedProcess:
    script = _BAN_TORCH + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(cwd)
    )


@pytest.fixture(scope="module")
def gbdt_bundle(tmp_path_factory) -> Path:
    """A real XGBoost bundle, trained once for the whole module."""
    import pandas as pd

    from ml_framework.config import ExperimentConfig
    from ml_framework.pipeline import train

    root = tmp_path_factory.mktemp("gbdt_serving")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)).astype("float32")
    y = (x @ rng.normal(size=(6, 3))).argmax(axis=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["label"] = y
    csv = root / "data.csv"
    frame.to_csv(csv, index=False)

    cfg = ExperimentConfig.model_validate(
        {
            "task": "multiclass",
            "runtime": {"output_dir": str(root / "bundle"), "seed": 42, "num_workers": 0},
            "data": {"kind": "tabular", "path": str(csv), "target": "label"},
            "model": {"name": "xgboost"},
            "fit": {"params": {"n_estimators": 20, "early_stopping_rounds": 0}},
            "logging": {"backend": "none"},
        }
    )
    train(cfg)
    return root / "bundle"


# ── The gate ──────────────────────────────────────────────
def test_the_ban_actually_bans_torch(tmp_path):
    """Guards the guard: if the meta-path finder stopped working, every test below
    would pass for the wrong reason."""
    proc = _run_without_torch("import torch", tmp_path)
    assert proc.returncode != 0
    assert "banned" in proc.stderr


def test_a_gbdt_bundle_predicts_with_torch_unimportable(gbdt_bundle):
    proc = _run_without_torch(
        f"""
        import numpy as np
        from ml_framework.core.inference import Inferencer

        inf = Inferencer.from_artifacts(r"{gbdt_bundle}")
        assert inf.backend_name == "gbdt", inf.backend_name
        assert inf.n_features == 6

        x = np.zeros((4, 6), dtype="float32")
        preds = inf.predict(x)
        probs = inf.predict_proba(x)
        assert preds.shape == (4,), preds.shape
        assert probs.shape == (4, 3), probs.shape

        import sys
        assert "torch" not in sys.modules
        print("ok")
        """,
        gbdt_bundle.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_the_serving_app_answers_predict_with_torch_unimportable(gbdt_bundle):
    """The full gate: create_app → POST /predict, no deep-learning stack present."""
    pytest.importorskip("fastapi")
    proc = _run_without_torch(
        f"""
        from fastapi.testclient import TestClient
        from ml_framework.serving.api import create_app

        client = TestClient(create_app(r"{gbdt_bundle}", api_key=None, rate_limit=None))

        health = client.get("/health").json()
        assert health["status"] == "ok", health
        assert health["backend"] == "gbdt", health
        assert health["task"] == "multiclass", health

        body = {{"instances": [[0.0] * 6, [1.0] * 6]}}
        res = client.post("/predict", json=body)
        assert res.status_code == 200, res.text
        assert len(res.json()["predictions"]) == 2

        res = client.post("/predict_proba", json=body)
        assert res.status_code == 200, res.text
        assert len(res.json()["probabilities"][0]) == 3

        import sys
        assert "torch" not in sys.modules
        print("ok")
        """,
        gbdt_bundle.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_named_inputs_serve_without_torch_too(gbdt_bundle):
    """The recommended production contract, on the torch-free path."""
    pytest.importorskip("fastapi")
    proc = _run_without_torch(
        f"""
        from fastapi.testclient import TestClient
        from ml_framework.serving.api import create_app

        client = TestClient(create_app(r"{gbdt_bundle}", api_key=None, rate_limit=None))
        row = {{f"f{{i}}": 0.5 for i in range(6)}}
        res = client.post("/predict", json={{"inputs": [row]}})
        assert res.status_code == 200, res.text
        assert len(res.json()["predictions"]) == 1
        print("ok")
        """,
        gbdt_bundle.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_importing_the_inference_path_does_not_import_torch(tmp_path):
    """The structural half of the gate: the module graph itself is clean.

    A bundle that happens to load is not proof — a package `__init__` that grows an
    eager import later would reintroduce ~500 MB with nothing noticing. Every entry
    point a torch-free process touches is listed here, and this is not theoretical:
    building the first lean venv found three of them (`pipeline/__init__` importing
    `hpo`, `data/builders` importing the Lightning adapter, `core/registry` never
    populating BACKENDS).
    """
    proc = _run_without_torch(
        """
        import ml_framework
        import ml_framework.cli                     # noqa: F401  `mlf` entry point
        import ml_framework.pipeline                # noqa: F401  `mlf train`
        import ml_framework.pipeline.train          # noqa: F401
        import ml_framework.core.inference          # noqa: F401
        import ml_framework.serving.api             # noqa: F401
        import ml_framework.data.preprocess         # noqa: F401
        from ml_framework.config import ExperimentConfig  # noqa: F401
        print("ok")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_the_whole_training_path_runs_without_torch(tmp_path):
    """`mlf train --model xgboost` on an install with no deep-learning stack.

    The serving half of the gate can pass while training still fails at import —
    which is exactly what happened before `pipeline/__init__` and `data/builders`
    were made lazy. Both halves are gated.
    """
    import pandas as pd

    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 4)).astype("float32")
    y = (x @ rng.normal(size=(4, 2))).argmax(axis=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["label"] = y
    csv = tmp_path / "d.csv"
    frame.to_csv(csv, index=False)

    proc = _run_without_torch(
        f"""
        from ml_framework.config import ExperimentConfig
        from ml_framework.pipeline import train

        cfg = ExperimentConfig.model_validate({{
            "task": "binary",
            "runtime": {{"output_dir": r"{tmp_path / 'out'}", "seed": 0, "num_workers": 0}},
            "data": {{"kind": "tabular", "path": r"{csv}", "target": "label"}},
            "model": {{"name": "xgboost"}},
            "fit": {{"params": {{"n_estimators": 10, "early_stopping_rounds": 0}}}},
            "logging": {{"backend": "none"}},
        }})
        metrics = train(cfg)
        assert "test_acc" in metrics, metrics
        print("ok")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_selecting_a_neural_model_without_torch_reports_the_pip_extra(tmp_path):
    """torch is optional from P3, so `mlp` is refused the way any uninstalled
    plugin is — listed by the registry, with the command that fixes it."""
    proc = _run_without_torch(
        """
        from ml_framework.core.plugins import MissingExtraError
        from ml_framework.core.registry import MODELS
        import ml_framework.plugins  # noqa: F401

        assert "mlp" in MODELS.names(), "an unavailable model must still be listed"
        assert not MODELS.is_available("mlp")
        try:
            MODELS.get("mlp")
        except MissingExtraError as exc:
            assert "ml-framework[lightning]" in str(exc), exc
            print("ok")
        else:
            raise AssertionError("selecting mlp without torch must raise")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
