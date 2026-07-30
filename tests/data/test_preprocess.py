"""Preprocessors own all fitted state, and round-trip through their own directory."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml_framework.data.preprocess import (
    MANIFEST_NAME,
    IdentityPreprocessor,
    PreprocessorError,
    load_preprocessor,
)
from ml_framework.data.preprocess.tabular import (
    SCALER_FILE,
    TabularPreprocessor,
    apply_smote,
    class_weights,
    detect_imbalance,
    resolve_imbalance,
)
from ml_framework.data.types import Split


def _train_split(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    # Deliberately off-centre and wide, so an unfitted transform is detectable.
    x = (rng.normal(size=(n, 3)) * 5.0 + 10.0).astype("float32")
    y = np.array([i % 2 for i in range(n)])
    return Split(x=x, y=y)


# ── Fitting ───────────────────────────────────────────────
@pytest.mark.unit
def test_scaler_standardizes_the_split_it_was_fitted_on():
    split = _train_split()
    pre = TabularPreprocessor()
    out = pre.fit_transform(split)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-5)


@pytest.mark.unit
def test_transform_uses_the_training_statistics_not_the_new_data():
    """Fitting on val/test is leakage; the check is that a shifted holdout does
    *not* come out centred."""
    pre = TabularPreprocessor()
    pre.fit(_train_split())
    shifted = np.full((50, 3), 100.0, dtype="float32")
    assert pre.transform(shifted).mean() > 10.0


@pytest.mark.unit
def test_needs_scaling_false_is_a_fitted_no_op():
    """Trees gain nothing from standardization and it destroys the meaning of
    their split thresholds — but the bundle layout must not change shape."""
    split = _train_split()
    pre = TabularPreprocessor(needs_scaling=False)
    out = pre.fit_transform(split)
    assert pre.fitted and pre.scaler is None
    assert np.array_equal(out, split.x)


@pytest.mark.unit
def test_fit_without_features_fails_loudly():
    with pytest.raises(PreprocessorError, match="needs the training features"):
        TabularPreprocessor().fit(Split(x=None))


# ── Round-trip ────────────────────────────────────────────
@pytest.mark.unit
def test_save_writes_a_self_describing_directory(tmp_path):
    pre = TabularPreprocessor()
    pre.fit(_train_split())
    fragment = pre.save(tmp_path / "preprocessor")

    assert fragment["class"].endswith(":TabularPreprocessor")
    assert fragment["files"] == [SCALER_FILE]
    assert fragment["dir"] == "preprocessor"
    on_disk = json.loads((tmp_path / "preprocessor" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == fragment


@pytest.mark.unit
def test_load_preprocessor_resolves_the_dotted_class_path(tmp_path):
    """Nothing outside the preprocessor knows what its files are — the loader
    resolves a class path and hands the directory back."""
    pre = TabularPreprocessor()
    pre.fit(_train_split())
    fragment = pre.save(tmp_path / "preprocessor")

    restored = load_preprocessor(tmp_path / "preprocessor", fragment)
    assert isinstance(restored, TabularPreprocessor)
    probe = np.full((4, 3), 7.0, dtype="float32")
    assert np.allclose(restored.transform(probe), pre.transform(probe))


@pytest.mark.unit
def test_a_preprocessor_directory_loads_without_its_bundle(tmp_path):
    pre = TabularPreprocessor(needs_scaling=False)
    pre.fit(_train_split())
    pre.save(tmp_path / "preprocessor")

    restored = TabularPreprocessor.load(tmp_path / "preprocessor")
    assert restored.needs_scaling is False  # params round-tripped


@pytest.mark.unit
def test_load_preprocessor_reports_an_unimportable_class(tmp_path):
    with pytest.raises(PreprocessorError, match="Cannot import preprocessor"):
        load_preprocessor(tmp_path, {"class": "no.such.module:Thing"})


@pytest.mark.unit
def test_load_preprocessor_requires_a_class_key(tmp_path):
    with pytest.raises(PreprocessorError, match="no 'class' key"):
        load_preprocessor(tmp_path, {"files": []})


@pytest.mark.unit
def test_v1_bundle_with_a_bare_scaler_pkl_still_loads(tmp_path):
    """The clean break was authorized for configs and tests, not for bundles that
    are already deployed."""
    pre = TabularPreprocessor()
    pre.fit(_train_split())
    import joblib

    joblib.dump(pre.scaler, tmp_path / SCALER_FILE)  # v1 layout: no preprocessor/ dir

    legacy = TabularPreprocessor.from_legacy_bundle(tmp_path)
    probe = np.full((4, 3), 7.0, dtype="float32")
    assert np.allclose(legacy.transform(probe), pre.transform(probe))


@pytest.mark.unit
def test_identity_preprocessor_round_trips_and_changes_nothing(tmp_path):
    pre = IdentityPreprocessor()
    fragment = pre.save(tmp_path / "preprocessor")
    assert fragment["files"] == []
    x = np.arange(6).reshape(3, 2)
    assert np.array_equal(load_preprocessor(tmp_path / "preprocessor", fragment).transform(x), x)


# ── Imbalance ─────────────────────────────────────────────
@pytest.mark.unit
def test_numpy_class_weights_match_the_torch_facing_helper():
    """One implementation, two faces: `core.lit_data.compute_class_weights` wraps
    this in a tensor rather than repeating the arithmetic."""
    from ml_framework.core.lit_data import compute_class_weights

    y = np.array([0] * 80 + [1] * 20)
    assert np.allclose(class_weights(y, "binary"), compute_class_weights(y, "binary").numpy())


@pytest.mark.unit
def test_binary_class_weight_is_a_scalar_pos_weight():
    y = np.array([0] * 80 + [1] * 20)  # n_neg / n_pos = 4.0
    weights = class_weights(y, "binary")
    assert weights.shape == (1,) and np.isclose(weights[0], 4.0)


@pytest.mark.unit
def test_resolve_imbalance_applies_exactly_one_correction():
    """Weighting *and* oversampling corrects the same skew twice."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3)).astype("float32")
    y = np.array([0] * 90 + [1] * 10)

    x_w, y_w, weights = resolve_imbalance(
        x, y, task="binary", strategy="class_weights", threshold=0.3, seed=0
    )
    assert weights is not None and len(x_w) == len(x)  # weighted, not resampled

    x_s, y_s, no_weights = resolve_imbalance(
        x, y, task="binary", strategy="smote", threshold=0.3, seed=0
    )
    assert no_weights is None and len(x_s) > len(x)  # resampled, not weighted


@pytest.mark.unit
def test_resolve_imbalance_is_a_no_op_on_balanced_data():
    x = np.zeros((100, 2), dtype="float32")
    y = np.array([i % 2 for i in range(100)])
    x_out, y_out, weights = resolve_imbalance(
        x, y, task="binary", strategy="class_weights", threshold=0.3, seed=0
    )
    assert weights is None and len(x_out) == len(y_out) == 100


@pytest.mark.unit
def test_resolve_imbalance_never_touches_regression():
    x = np.zeros((10, 2), dtype="float32")
    y = np.linspace(0, 1, 10).astype("float32")
    x_out, y_out, weights = resolve_imbalance(
        x, y, task="regression", strategy="smote", threshold=0.3, seed=0
    )
    assert weights is None
    assert np.array_equal(x_out, x) and np.array_equal(y_out, y)


@pytest.mark.unit
def test_detect_imbalance_thresholds_on_the_minority_majority_ratio():
    assert detect_imbalance(np.array([0] * 90 + [1] * 10), 0.3) is True
    assert detect_imbalance(np.array([0] * 50 + [1] * 50), 0.3) is False
    assert detect_imbalance(np.array([0] * 50), 0.3) is False  # single class


@pytest.mark.unit
def test_apply_smote_balances_the_training_classes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3)).astype("float32")
    y = np.array([0] * 90 + [1] * 10)
    _, y_res = apply_smote(x, y, seed=0)
    assert (y_res == 0).sum() == (y_res == 1).sum()
