"""train() writes an artifact bundle v2 whose manifest is the only file a loader needs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.bundle import (
    BUNDLE_VERSION,
    CONFIG_NAME,
    MANIFEST_NAME,
    MODEL_DIR,
    PREPROCESSOR_DIR,
    detect_bundle_version,
    read_manifest,
)
from ml_framework.pipeline import train

pytestmark = pytest.mark.integration


@pytest.fixture
def trained(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    metrics = train(cfg)
    return Path(cfg.runtime.output_dir), cfg, metrics


def test_bundle_has_the_v2_layout(trained):
    out, _, _ = trained
    assert (out / MANIFEST_NAME).exists()
    assert (out / CONFIG_NAME).exists()
    assert (out / MODEL_DIR / "model.ckpt").exists()
    assert (out / PREPROCESSOR_DIR / "scaler.pkl").exists()
    assert (out / PREPROCESSOR_DIR / "preprocessor.json").exists()
    for name in ("metrics.json", "report.txt", "confusion_matrix.txt", "predictions.csv"):
        assert (out / name).exists(), name
    assert detect_bundle_version(out) == BUNDLE_VERSION


def test_manifest_names_the_artifact_rather_than_fixing_its_format(trained):
    """The invariant that makes heterogeneous model files a non-problem: the
    manifest is the only uniform thing, and it names the non-uniform things."""
    out, _, _ = trained
    manifest = read_manifest(out)
    assert manifest.model.artifact == f"{MODEL_DIR}/model.ckpt"
    assert manifest.model.format == "lightning-checkpoint"
    assert manifest.artifact_path(out).exists()
    assert manifest.model.backend == "lightning"
    assert manifest.model.size["trainable_parameters"] > 0


def test_signature_replaces_reaching_into_a_torch_module_for_the_contract(trained):
    """v1 read `getattr(inf.model, "input_dim", None)` — code that asked a torch
    module for its own API contract and returned None for anything else."""
    out, _, _ = trained
    signature = read_manifest(out).signature
    assert signature.input.payload == "arrays"
    assert signature.input.n_features == 6
    assert signature.input.features == [f"f{i}" for i in range(6)]
    assert signature.output.kind == "probabilities"
    assert signature.output.n_classes == 3


def test_preprocessor_fragment_points_at_a_self_describing_directory(trained):
    out, _, _ = trained
    ref = read_manifest(out).preprocessor
    assert ref is not None
    assert ref.cls == "ml_framework.data.preprocess.tabular:TabularPreprocessor"
    assert ref.dir == PREPROCESSOR_DIR
    assert ref.files == ["scaler.pkl"]
    on_disk = json.loads((out / ref.dir / "preprocessor.json").read_text(encoding="utf-8"))
    assert on_disk["class"] == ref.cls


def test_the_preprocessor_round_trips_without_anything_naming_scaler_pkl(trained):
    """Loading goes through the dotted class path in the manifest, so a tokenizer
    or a per-series scaler would load by the same code."""
    from ml_framework.data.preprocess import load_preprocessor

    out, cfg, _ = trained
    ref = read_manifest(out).preprocessor
    restored = load_preprocessor(out / ref.dir, ref.model_dump(by_alias=True))

    from ml_framework.data import build_bundle

    bundle = build_bundle(cfg)
    raw = np.zeros((3, 6), dtype="float32")
    assert np.allclose(restored.transform(raw), bundle.preprocessor.transform(raw))


def test_manifest_does_not_embed_the_training_config(trained):
    """Reconstructing an ExperimentConfig at serving time would need a populated
    plugin registry and every training extra. config.json is the audit record."""
    out, _, _ = trained
    raw = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "config" not in raw
    assert json.loads((out / CONFIG_NAME).read_text(encoding="utf-8"))["task"] == "multiclass"


def test_metrics_json_matches_the_returned_metrics(trained):
    """DVC and the Airflow quality gate read this file."""
    out, _, metrics = trained
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8")) == metrics
    assert read_manifest(out).metrics == metrics


def test_the_v1_root_artifacts_are_gone(trained):
    """P1 mirrored `model.ckpt`/`scaler.pkl`/`metadata.json` at the bundle root
    because the loader still read them. The loader is manifest-driven now, so
    keeping them would be dead weight in every bundle. Already-written v1 bundles
    still load — see test_bundle.py's back-compat case."""
    out, _, _ = trained
    for name in ("model.ckpt", "scaler.pkl", "metadata.json"):
        assert not (out / name).exists(), f"{name} should live under model/ or preprocessor/"


def test_reference_stats_are_written_for_serving_side_drift(trained):
    out, _, _ = trained
    stats = json.loads((out / "reference_stats.json").read_text(encoding="utf-8"))
    assert set(stats["features"]) == {f"f{i}" for i in range(6)}


def test_regression_bundle_reports_values_not_probabilities(regression_csv, make_config):
    cfg = make_config(regression_csv, "regression")
    train(cfg)
    manifest = read_manifest(Path(cfg.runtime.output_dir))
    assert manifest.signature.output.kind == "values"
    assert manifest.signature.output.n_classes is None


def test_binary_bundle_keeps_two_classes_behind_a_single_logit(binary_csv, make_config):
    """The asymmetric case: n_classes is 2 but the head is one logit wide."""
    cfg = make_config(binary_csv, "binary")
    train(cfg)
    manifest = read_manifest(Path(cfg.runtime.output_dir))
    assert manifest.signature.output.n_classes == 2
    assert manifest.signature.output.kind == "probabilities"
