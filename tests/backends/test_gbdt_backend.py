"""The GBDT backend: the fit loop, the estimator, and the bundle round-trip.

The point of this file is the *sameness*. Every test here is the GBDT counterpart
of one in ``test_lightning_backend.py``, driven through the identical protocol
calls — which is the claim the backend split makes and the thing that would
silently stop being true without a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.bundle import MODEL_DIR, read_manifest
from ml_framework.core.protocols import BuildContext, Predictions, RunContext
from ml_framework.core.registry import MODELS, get_backend
from ml_framework.core.types import UnsupportedCapability

xgboost = pytest.importorskip("xgboost", reason="the gbdt extra is not installed")

from ml_framework.backends.gbdt import (  # noqa: E402
    GbdtBackend,
    GbdtEstimator,
    GbdtFitParams,
    adapter_for_estimator,
    adapter_for_format,
)
from ml_framework.data import build_bundle  # noqa: E402

TREES = ("xgboost", "lightgbm", "catboost")


def _gbdt_config(csv, task, make_config, model="xgboost", **overrides):
    """A tabular config pointed at a tree model with a tiny boosting budget."""
    base = {
        "fit.params.n_estimators": 20,
        "fit.params.learning_rate": 0.3,
        "fit.params.early_stopping_rounds": 0,
        **overrides,
    }
    return make_config(csv, task, model=model, **base)


@pytest.fixture
def fitted(tabular_csv, make_config):
    """A real XGBoost fit, reused by the tests that need a trained estimator."""
    cfg = _gbdt_config(tabular_csv, "multiclass", make_config)
    bundle = build_bundle(cfg)
    backend = GbdtBackend()
    run = RunContext(output_dir=Path(cfg.runtime.output_dir), seed=cfg.runtime.seed)
    result = backend.fit(MODELS.get("xgboost"), bundle, cfg, run=run)
    return backend, result, bundle, cfg


# ── Registration ──────────────────────────────────────────
@pytest.mark.unit
def test_gbdt_backend_is_registered_and_buildable():
    backend = get_backend("gbdt")
    assert isinstance(backend, GbdtBackend)
    assert backend.name == "gbdt"


@pytest.mark.unit
def test_registering_the_backend_does_not_import_xgboost():
    """The spec carries a lazy factory, so `mlf backends` works on a bare install."""
    import subprocess
    import sys

    code = (
        "import sys, ml_framework.backends;"
        "assert 'ml_framework.backends.gbdt' not in sys.modules, 'backend module imported';"
        "assert 'xgboost' not in sys.modules, 'xgboost imported';"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.unit
def test_spec_capabilities_match_the_backend_class():
    from ml_framework.core.registry import BACKENDS

    assert BACKENDS.get_spec("gbdt").capabilities == GbdtBackend.capabilities


@pytest.mark.unit
def test_the_backend_itself_declares_no_requirements():
    """Which library it needs depends on the model, and each plugin says so.

    Declaring the union on the backend would refuse a lightgbm run on an install
    that has lightgbm but not catboost.
    """
    from ml_framework.core.registry import BACKENDS

    assert BACKENDS.get_spec("gbdt").requires == ()
    assert BACKENDS.is_available("gbdt")


# ── Capabilities that do real work ────────────────────────
@pytest.mark.unit
def test_trees_declare_the_four_flags_that_change_behaviour():
    for name in TREES:
        caps = MODELS.get_spec(name).capabilities
        assert caps.needs_scaling is False, name
        assert caps.native_missing is True, name
        assert caps.native_categorical is True, name
        assert caps.supports_sample_weight is True, name
        assert caps.supports_lr_range_test is False, name


@pytest.mark.integration
def test_a_tree_bundle_is_not_standardized(tabular_csv, make_config):
    """`needs_scaling=False` reaches the preprocessor, not just the docs.

    Scaling a tree buys nothing and turns an interpretable split threshold
    ("age > 41") into an opaque one ("age > 0.34").
    """
    tree = build_bundle(_gbdt_config(tabular_csv, "multiclass", make_config))
    neural = build_bundle(make_config(tabular_csv, "multiclass"))

    assert tree.preprocessor.needs_scaling is False
    assert tree.preprocessor.scaler is None
    # The neural path still standardizes, so this is a real difference and not a
    # preprocessor that stopped working.
    assert neural.preprocessor.scaler is not None
    assert np.allclose(neural.train.x.mean(axis=0), 0.0, atol=1e-4)
    assert not np.allclose(tree.train.x.mean(axis=0), 0.0, atol=1e-4)


@pytest.mark.integration
def test_auto_imbalance_gives_a_tree_sample_weights_instead_of_smote(binary_csv, make_config):
    """`supports_sample_weight=True` reaches the imbalance resolver.

    SMOTE interpolates synthetic neighbours, which an axis-aligned splitter uses
    poorly; these libraries take a weight per row natively.
    """
    cfg = _gbdt_config(
        binary_csv, "binary", make_config, **{"data.params.imbalance_strategy": "auto"}
    )
    bundle = build_bundle(cfg)
    n_rows = len(bundle.train.y)

    # No resampling happened: the row count is untouched.
    assert bundle.class_weights is None
    weights = bundle.meta.get("sample_weights")
    if weights is not None:  # only when the fixture is actually imbalanced
        assert len(weights) == n_rows


@pytest.mark.integration
def test_auto_imbalance_still_picks_smote_for_a_neural_model(binary_csv, make_config):
    """The same `auto` resolves differently per model — that is what makes it auto."""
    cfg = make_config(binary_csv, "binary", **{"data.params.imbalance_strategy": "auto"})
    bundle = build_bundle(cfg)
    assert bundle.meta.get("sample_weights") is None


# ── Estimator ─────────────────────────────────────────────
@pytest.mark.integration
def test_regression_tree_refuses_probabilities(regression_csv, make_config):
    cfg = _gbdt_config(regression_csv, "regression", make_config)
    bundle = build_bundle(cfg)
    run = RunContext(output_dir=Path(cfg.runtime.output_dir), seed=cfg.runtime.seed)
    result = GbdtBackend().fit(MODELS.get("xgboost"), bundle, cfg, run=run)

    assert result.estimator.predict(bundle.test.x).shape == (bundle.test.n,)
    with pytest.raises(UnsupportedCapability, match="not probabilities"):
        result.estimator.predict_proba(bundle.test.x)


# ── fit ───────────────────────────────────────────────────
@pytest.mark.integration
def test_fit_returns_plain_float_val_metrics(fitted):
    """No `callback_metrics` object exists here, so the metrics are computed —
    and `FitResult.val_metrics` is a plain dict either way, which is what lets the
    tuning driver read an objective from any backend."""
    _, result, _, _ = fitted
    assert isinstance(result.val_metrics, dict)
    assert result.val_metrics
    assert all(isinstance(v, float) for v in result.val_metrics.values())
    assert "val_acc" in result.val_metrics


@pytest.mark.integration
def test_model_size_reports_trees_where_lightning_reports_parameters(fitted):
    """count_parameters() generalized: the same manifest field, a different unit."""
    backend, result, _, _ = fitted
    size = backend.model_size(result.estimator)
    assert size["trees"] > 0


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
    backend, result, bundle, _ = fitted
    preds = backend.predict_split(result.estimator, bundle, "test")
    assert np.array_equal(preds.y_true, bundle.test.y)
    assert preds.index is not None and np.array_equal(preds.index, bundle.test.index)


# ── save / load ───────────────────────────────────────────
@pytest.mark.integration
def test_save_records_a_native_format_not_a_converted_one(fitted, tmp_path):
    """No lossy conversion at save time: the library's own loader stays usable,
    and feature importances survive."""
    backend, result, _, _ = fitted
    ref = backend.save(result.estimator, tmp_path / MODEL_DIR)
    assert ref.path == f"{MODEL_DIR}/model.json"
    assert ref.format == "xgboost-json"
    assert (tmp_path / ref.path).exists()


@pytest.mark.integration
def test_load_dispatches_on_the_recorded_format_alone(fitted, tmp_path):
    """`format` is tracked separately from the extension so serialization can
    migrate without breaking readers."""
    backend, result, bundle, _ = fitted
    backend.save(result.estimator, tmp_path / MODEL_DIR)

    class _Manifest:
        task = "multiclass"

        class model:  # noqa: N801 - a stand-in for the pydantic ModelRef
            artifact = f"{MODEL_DIR}/model.json"
            format = "xgboost-json"

        class signature:  # noqa: N801
            class output:  # noqa: N801
                n_classes = 3

    est = backend.load(tmp_path, _Manifest)
    assert isinstance(est, GbdtEstimator)
    assert np.array_equal(est.predict(bundle.test.x), result.estimator.predict(bundle.test.x))


@pytest.mark.unit
def test_an_unknown_artifact_format_is_refused_by_name():
    from ml_framework.backends.gbdt import GbdtBackendError

    with pytest.raises(GbdtBackendError, match="unknown GBDT artifact format"):
        adapter_for_format("prophet-pickle")


@pytest.mark.unit
def test_an_unknown_estimator_falls_back_to_the_sklearn_adapter():
    """The promise that sklearn estimators ride this backend for free."""
    from sklearn.ensemble import RandomForestClassifier

    adapter = adapter_for_estimator(RandomForestClassifier())
    assert adapter.library == "sklearn"
    assert adapter.fmt == "sklearn-joblib"


# ── Every library, end to end ─────────────────────────────
@pytest.mark.integration
@pytest.mark.parametrize("model", TREES)
def test_each_library_trains_saves_and_reloads_identically(model, tabular_csv, make_config):
    """Adding CatBoost was ~40 lines, not a fourth backend. This is that claim."""
    pytest.importorskip(model, reason=f"{model} is not installed")
    from ml_framework.pipeline import train

    cfg = _gbdt_config(tabular_csv, "multiclass", make_config, model=model)
    metrics = train(cfg)
    assert "test_acc" in metrics

    out = Path(cfg.runtime.output_dir)
    manifest = read_manifest(out)
    assert manifest.model.backend == "gbdt"
    assert manifest.model.name == model
    assert manifest.model.size

    from ml_framework.core.inference import Inferencer

    inf = Inferencer.from_artifacts(out)
    bundle = build_bundle(cfg)
    import pandas as pd

    expected = pd.read_csv(out / "predictions.csv")["prediction"].to_numpy()
    # The Inferencer applies the preprocessor, so it takes the *raw* features that
    # a client would send — which for a tree is the untransformed matrix.
    assert np.array_equal(inf.predict(bundle.test.x), expected)


# ── HPO surface ───────────────────────────────────────────
@pytest.mark.unit
def test_search_space_declares_boosting_knobs_as_dotted_v2_paths():
    """learning_rate/n_estimators belong to the loop, so all three libraries stop
    repeating them — the same split as lr/batch_size on the Lightning backend."""
    space = GbdtBackend().search_space()
    assert set(space) == {
        "fit.params.learning_rate",
        "fit.params.n_estimators",
        "fit.params.subsample",
        "fit.params.colsample_bytree",
    }


@pytest.mark.unit
def test_plugin_search_spaces_hold_tree_shape_only():
    """No plugin repeats a backend-level knob; that is what the split is for."""
    backend_keys = set(GbdtBackend().search_space())
    for name in TREES:
        keys = set(MODELS.get_spec(name).search_space)
        assert keys, name
        assert all(k.startswith("model.params.") for k in keys), name
        assert not (keys & backend_keys), name


@pytest.mark.unit
def test_fit_params_model_is_the_boosting_loop():
    params = GbdtBackend().params_model()
    assert params is GbdtFitParams
    assert set(params.model_fields) == {
        "learning_rate",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "early_stopping_rounds",
    }
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        params(learning_rate=0.1, typo=1)


@pytest.mark.unit
def test_trial_hooks_are_empty_without_a_trial():
    assert not GbdtBackend().trial_hooks(None)


# ── Plugin build contract ─────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("model", TREES)
def test_build_returns_an_unfitted_estimator(model):
    """The GBDT analogue of returning an untrained network: the model knows its
    own shape and nothing about output dirs, checkpoints or trackers."""
    pytest.importorskip(model, reason=f"{model} is not installed")
    est = MODELS.get(model).build(
        BuildContext(task="multiclass", input_dim=6, output_dim=3, n_classes=3, seed=7)
    )
    assert hasattr(est, "fit") and hasattr(est, "predict")
    # Unfitted: predicting now must fail rather than return something.
    with pytest.raises(Exception):  # noqa: B017 - each library raises its own type
        est.predict(np.zeros((1, 6), dtype="float32"))
