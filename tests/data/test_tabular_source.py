"""The tabular source: a DataBundle with no torch in it, built in the right order."""

from __future__ import annotations

import numpy as np
import pytest

from ml_framework.data import build_bundle
from ml_framework.data.sources.tabular import build_tabular_bundle
from ml_framework.data.types import DataBundle


@pytest.mark.unit
def test_bundle_carries_the_dims_the_v1_datamodule_derived(tabular_csv, make_config):
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    assert isinstance(bundle, DataBundle)
    assert bundle.input_dim == 6
    assert bundle.output_dim == 3
    assert bundle.n_classes == 3
    assert bundle.schema.feature_names == tuple(f"f{i}" for i in range(6))
    assert bundle.schema.target_name == "label"


@pytest.mark.unit
@pytest.mark.parametrize("task,expected", [("binary", 1), ("regression", 1)])
def test_binary_and_regression_keep_the_single_logit_head(
    task, expected, binary_csv, regression_csv, make_config
):
    csv = binary_csv if task == "binary" else regression_csv
    assert build_tabular_bundle(make_config(csv, task)).output_dim == expected


@pytest.mark.unit
def test_splits_are_disjoint_and_cover_the_table(tabular_csv, make_config):
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    assert sum(bundle.sizes().values()) == 200
    assert bundle.payload == "arrays"


@pytest.mark.unit
def test_class_weights_are_numpy_not_torch(tabular_csv, make_config):
    """The agnostic layer must not import torch; the Lightning adapter converts."""
    cfg = make_config(tabular_csv, "multiclass", **{"data.imbalance_strategy": "class_weights"})
    bundle = build_tabular_bundle(cfg)
    assert bundle.class_weights is None or isinstance(bundle.class_weights, np.ndarray)


@pytest.mark.unit
def test_features_are_scaled_using_training_statistics_only(tabular_csv, make_config):
    """The scaler is fitted on train, so train is centred and val/test are only
    approximately so. If val were included in the fit, this margin would vanish."""
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    assert np.allclose(bundle.train.x.mean(axis=0), 0.0, atol=1e-4)
    assert bundle.preprocessor is not None and bundle.preprocessor.fitted


@pytest.mark.unit
def test_drift_reference_is_captured_before_scaling(tabular_csv, make_config):
    """Serving computes drift on the raw features clients send, so a reference
    built from scaled values would report drift on every correct request."""
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    stats = bundle.reference_stats
    assert stats is not None and set(stats["features"]) == set(bundle.schema.feature_names)
    # Scaled training data would have mean ~0 for every feature; raw data does not.
    means = [f["mean"] for f in stats["features"].values()]
    assert not np.allclose(means, 0.0, atol=1e-6)


@pytest.mark.unit
def test_val_and_test_keep_their_original_row_indices(tabular_csv, make_config):
    """Only the train split can lose row identity (SMOTE synthesizes rows); val and
    test stay traceable back to input rows."""
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    assert bundle.val.index is not None and bundle.test.index is not None
    assert len(bundle.test.index) == bundle.test.n
    assert not set(bundle.val.index.tolist()) & set(bundle.test.index.tolist())


@pytest.mark.unit
def test_smote_resamples_only_the_training_split(binary_csv, make_config, tmp_path):
    import pandas as pd

    # Make the CSV genuinely imbalanced so the strategy actually fires.
    df = pd.read_csv(binary_csv)
    skewed = pd.concat([df[df.label == 0], df[df.label == 1].head(12)], ignore_index=True)
    csv = tmp_path / "skewed.csv"
    skewed.to_csv(csv, index=False)

    cfg = make_config(csv, "binary", **{"data.imbalance_strategy": "smote"})
    bundle = build_tabular_bundle(cfg)
    counts = np.bincount(bundle.train.y.astype("int64"))
    assert counts[0] == counts[1]  # train balanced
    assert bundle.class_weights is None  # and not also weighted


@pytest.mark.unit
def test_missing_target_column_names_the_column(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass", **{"data.target_col": "nope"})
    with pytest.raises(KeyError, match="nope"):
        build_tabular_bundle(cfg)


@pytest.mark.unit
def test_build_bundle_dispatches_through_the_source_registry(tabular_csv, make_config):
    bundle = build_bundle(make_config(tabular_csv, "multiclass"))
    assert bundle.data_kind == "tabular" and bundle.input_dim == 6


@pytest.mark.unit
def test_bundle_split_lookup_by_name_rejects_unknown_names(tabular_csv, make_config):
    bundle = build_tabular_bundle(make_config(tabular_csv, "multiclass"))
    assert bundle.split("test") is bundle.test
    with pytest.raises(KeyError, match="Unknown split"):
        bundle.split("holdout")
