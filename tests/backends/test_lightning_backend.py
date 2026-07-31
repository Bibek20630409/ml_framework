"""The Lightning backend: the fit loop, the estimator, and the bundle round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml_framework.backends.lightning import (
    ARTIFACT_FORMAT,
    LightningBackend,
    LightningEstimator,
)
from ml_framework.core.bundle import MODEL_DIR, read_manifest
from ml_framework.core.protocols import Predictions, RunContext
from ml_framework.core.registry import MODELS, get_backend
from ml_framework.core.types import UnsupportedCapability
from ml_framework.data import build_bundle


@pytest.fixture
def fitted(tabular_csv, make_config):
    """A real 2-epoch fit, reused by the tests that need a trained estimator."""
    cfg = make_config(tabular_csv, "multiclass")
    bundle = build_bundle(cfg)
    backend = LightningBackend()
    run = RunContext(output_dir=Path(cfg.runtime.output_dir), seed=cfg.runtime.seed)
    result = backend.fit(MODELS.get("mlp"), bundle, cfg, run=run)
    return backend, result, bundle, cfg


# ── Registration ──────────────────────────────────────────
@pytest.mark.unit
def test_lightning_backend_is_registered_and_buildable():
    """P0 left BACKENDS deliberately empty because no backend module existed."""
    backend = get_backend("lightning")
    assert isinstance(backend, LightningBackend)
    assert backend.name == "lightning"


@pytest.mark.unit
def test_registering_the_backend_does_not_import_torch():
    """The spec carries a lazy factory so `mlf backends` works on a bare install."""
    import subprocess
    import sys

    code = (
        "import sys; import ml_framework.backends;"
        "assert 'ml_framework.backends.lightning' not in sys.modules, 'backend module imported';"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.unit
def test_spec_capabilities_match_the_backend_class():
    """The spec duplicates them to avoid importing the class; keep them in step."""
    from ml_framework.core.registry import BACKENDS

    assert BACKENDS.get_spec("lightning").capabilities == LightningBackend.capabilities


# ── Estimator ─────────────────────────────────────────────
class _Head(torch.nn.Module):
    """A fixed linear head, so the decoded values are predictable."""

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, out_features)

    def forward(self, x):
        return self.linear(x)


def _estimator(task: str, out_features: int) -> LightningEstimator:
    return LightningEstimator(_Head(out_features), task, device=torch.device("cpu"))


@pytest.mark.unit
def test_binary_head_decodes_to_labels_and_two_column_probabilities():
    est = _estimator("binary", 1)
    x = np.random.default_rng(0).normal(size=(7, 3)).astype("float32")
    preds, probs = est.decode(est.logits(x))
    assert preds.shape == (7,) and set(np.unique(preds)) <= {0, 1}
    assert probs.shape == (7, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.unit
def test_multiclass_head_decodes_to_argmax_and_softmax():
    est = _estimator("multiclass", 4)
    x = np.random.default_rng(0).normal(size=(7, 3)).astype("float32")
    preds, probs = est.decode(est.logits(x))
    assert probs.shape == (7, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(preds, probs.argmax(axis=1))


@pytest.mark.unit
def test_regression_head_is_identity_and_refuses_probabilities():
    """`predict_proba` on a regression model is the capability refusal the
    UnsupportedCapability class exists for — not a ValueError from deep inside."""
    est = _estimator("regression", 1)
    x = np.random.default_rng(0).normal(size=(5, 3)).astype("float32")
    assert est.predict(x).shape == (5,)
    with pytest.raises(UnsupportedCapability, match="not probabilities"):
        est.predict_proba(x)


@pytest.mark.unit
def test_predict_and_predict_proba_agree_with_decode():
    """One implementation of postprocessing; three entry points into it."""
    est = _estimator("multiclass", 3)
    x = np.random.default_rng(1).normal(size=(6, 3)).astype("float32")
    preds, probs = est.decode(est.logits(x))
    assert np.array_equal(est.predict(x), preds)
    assert np.allclose(est.predict_proba(x), probs)


@pytest.mark.unit
def test_estimator_accepts_a_tensor_as_well_as_an_array():
    est = _estimator("multiclass", 3)
    x = np.random.default_rng(2).normal(size=(4, 3)).astype("float32")
    assert np.array_equal(est.predict(x), est.predict(torch.tensor(x)))


# ── fit ───────────────────────────────────────────────────
@pytest.mark.integration
def test_fit_returns_plain_float_val_metrics(fitted):
    """FitResult.val_metrics is a dict, not Lightning's callback_metrics — that is
    what lets the tuning driver read an objective from any backend."""
    _, result, _, _ = fitted
    assert isinstance(result.val_metrics, dict)
    assert result.val_metrics
    assert all(isinstance(v, float) for v in result.val_metrics.values())
    assert "val/loss" in result.val_metrics


@pytest.mark.integration
def test_fit_returns_an_estimator_holding_its_best_checkpoint(fitted):
    _, result, _, cfg = fitted
    est = result.estimator
    assert isinstance(est, LightningEstimator)
    assert est.checkpoint_path is not None and est.checkpoint_path.exists()
    assert est.checkpoint_path.parent == Path(cfg.runtime.output_dir) / "checkpoints"


@pytest.mark.integration
def test_model_size_generalizes_count_parameters(fitted):
    """count_parameters() becomes a backend method so a GBDT can report tree counts
    through the same manifest field."""
    backend, result, _, _ = fitted
    size = backend.model_size(result.estimator)
    assert size["trainable_parameters"] > 0


# ── predict_split ─────────────────────────────────────────
@pytest.mark.integration
def test_predict_split_returns_aligned_arrays(fitted):
    backend, result, bundle, _ = fitted
    preds = backend.predict_split(result.estimator, bundle, "test")
    assert isinstance(preds, Predictions)
    assert preds.n == bundle.test.n
    assert preds.y_true is not None and len(preds.y_true) == preds.n
    assert preds.y_prob is not None and preds.y_prob.shape == (preds.n, bundle.output_dim)


@pytest.mark.integration
def test_predict_split_preserves_the_split_row_order(fitted):
    """predictions.csv rows must line up with their labels, which only holds if the
    evaluation loader does not shuffle."""
    backend, result, bundle, _ = fitted
    preds = backend.predict_split(result.estimator, bundle, "test")
    assert np.array_equal(preds.y_true, bundle.test.y)
    assert preds.index is not None and np.array_equal(preds.index, bundle.test.index)


@pytest.mark.integration
def test_predict_split_works_for_every_split(fitted):
    backend, result, bundle, _ = fitted
    for name in ("train", "val", "test"):
        assert backend.predict_split(result.estimator, bundle, name).n == bundle.split(name).n


# ── save / load ───────────────────────────────────────────
@pytest.mark.integration
def test_save_returns_a_bundle_relative_artifact_ref(fitted, tmp_path):
    backend, result, _, _ = fitted
    ref = backend.save(result.estimator, tmp_path / MODEL_DIR)
    assert ref.path == f"{MODEL_DIR}/model.ckpt"
    assert ref.format == ARTIFACT_FORMAT
    assert (tmp_path / ref.path).exists()


@pytest.mark.integration
def test_save_refuses_an_estimator_that_did_not_come_from_fit(tmp_path):
    backend = LightningBackend()
    with pytest.raises(FileNotFoundError, match="no checkpoint to save"):
        backend.save(_estimator("binary", 1), tmp_path / MODEL_DIR)


@pytest.mark.integration
def test_backend_load_reproduces_the_trained_predictions(tabular_csv, make_config):
    """The bundle v2 round-trip: the manifest names the artifact, the backend loads
    it, and nothing in between needs to know it is a Lightning checkpoint."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)

    out = Path(cfg.runtime.output_dir)
    manifest = read_manifest(out)
    backend = get_backend(manifest.model.backend)
    est = backend.load(out, manifest)

    import pandas as pd

    bundle = build_bundle(cfg)
    expected = pd.read_csv(out / "predictions.csv")["prediction"].to_numpy()
    assert np.array_equal(est.predict(bundle.test.x), expected)


@pytest.mark.integration
def test_backend_load_needs_only_the_manifest_not_the_training_config(tabular_csv, make_config):
    """P1's `load` re-validated config.json to rebuild the architecture, because
    v1's BaseModel took a whole ExperimentConfig. That made loading a bundle depend
    on the training config still parsing under the current schema. The manifest now
    carries `model.params` and the signature, which is everything the network
    needs — and config.json goes back to being purely the audit record."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)

    out = Path(cfg.runtime.output_dir)
    manifest = read_manifest(out)
    expected = get_backend("lightning").load(out, manifest).predict(build_bundle(cfg).test.x)

    (out / "config.json").unlink()
    est = get_backend("lightning").load(out, manifest)
    assert np.array_equal(est.predict(build_bundle(cfg).test.x), expected)


@pytest.mark.integration
def test_manifest_records_the_effective_model_params(tabular_csv, make_config):
    """Which is what makes the load above possible: post-defaults params, not the
    subset the user happened to type."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    params = read_manifest(Path(cfg.runtime.output_dir)).model.params
    assert set(params) == {"hidden_dims", "dropout"}


# ── Logger wiring ─────────────────────────────────────────
@pytest.mark.unit
def test_mlflow_logger_attaches_to_the_orchestrators_run(monkeypatch, tabular_csv, make_config):
    """The orchestrator creates the MLflow run; the backend joins it.

    v1 had the Lightning logger own the run, which is why `log_and_register` read
    a `run_id` off it — an attribute a GBDT run would never have. Getting this
    backwards would silently produce two runs per training.

    Exercised with a stub because the real path needs the [mlops] extra, and
    tests/integration/test_mlflow.py skips without it.
    """
    import pytorch_lightning.loggers as pl_loggers

    captured: dict[str, object] = {}

    class _FakeMLFlowLogger:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pl_loggers, "MLFlowLogger", _FakeMLFlowLogger)

    class _RunLogger:
        run_id = "run-abc123"

    cfg = make_config(tabular_csv, "multiclass", **{"logging.backend": "mlflow"})
    run = RunContext(output_dir=Path(cfg.runtime.output_dir), run_logger=_RunLogger())
    logger = LightningBackend()._build_logger(cfg, run)

    assert isinstance(logger, _FakeMLFlowLogger)
    assert captured["run_id"] == "run-abc123"
    assert captured["log_model"] is False  # the bundle is logged instead
    assert captured["experiment_name"] == cfg.logging.mlflow_experiment


@pytest.mark.unit
def test_no_lightning_logger_when_tracking_is_disabled(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")  # logging.backend == "none"
    run = RunContext(output_dir=Path(cfg.runtime.output_dir))
    assert LightningBackend()._build_logger(cfg, run) is False


# ── HPO surface ───────────────────────────────────────────
@pytest.mark.unit
def test_search_space_declares_loop_knobs_as_dotted_v2_paths():
    """lr and batch_size belong to the loop, so every neural plugin stops repeating
    them. Keys are config paths so applying a trial is with_overrides()."""
    space = LightningBackend().search_space()
    assert set(space) == {"fit.params.lr", "fit.params.weight_decay", "fit.batch_size"}


@pytest.mark.unit
def test_trial_hooks_are_empty_without_a_trial():
    assert not LightningBackend().trial_hooks(None)


@pytest.mark.unit
def test_fit_params_model_keeps_the_scheduler_configuration():
    """lr_patience/lr_factor configure ReduceLROnPlateau; dropping them in the v2
    move would silently change the schedule."""
    params = LightningBackend().params_model()
    assert params is not None
    assert set(params.model_fields) == {
        "lr",
        "weight_decay",
        "lr_patience",
        "lr_factor",
        "gradient_clip_val",
        # P5: v1's hardcoded Adam + ReduceLROnPlateau became defaults rather than
        # the only option, and gradient accumulation joined them.
        "optimizer",
        "scheduler",
        "accumulate_grad_batches",
    }
    # The v1 behaviour is still what you get by default.
    defaults = params()
    assert (defaults.optimizer, defaults.scheduler) == ("adam", "plateau")
    assert defaults.accumulate_grad_batches == 1
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        params(lr=1e-3, typo=1)
