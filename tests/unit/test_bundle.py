"""core/bundle.py — the manifest, i.e. the only file a loader must understand."""

from __future__ import annotations

import json

import pytest

from ml_framework.core.bundle import (
    BUNDLE_VERSION,
    InputSignature,
    Manifest,
    ModelRef,
    OutputSignature,
    PreprocessorRef,
    RequirementRef,
    Signature,
    UnsupportedBundleVersionError,
    check_requirements,
    detect_bundle_version,
    is_v1_bundle,
    read_manifest,
    write_bundle,
    write_manifest,
)
from ml_framework.core.plugins import MissingExtraError
from ml_framework.core.types import Requirement


def _manifest(**overrides) -> Manifest:
    base: dict = {
        "task": "multiclass",
        "data_kind": "tabular",
        "model": ModelRef(
            name="xgboost",
            backend="gbdt",
            artifact="model/model.json",
            format="xgboost-json",
            params={"max_depth": 6},
        ),
        "signature": Signature(
            input=InputSignature(payload="arrays", features=["f0", "f1"], n_features=2),
            output=OutputSignature(kind="probabilities", n_classes=3, class_names=["a", "b", "c"]),
        ),
        "metrics": {"test_acc": 0.75},
    }
    base.update(overrides)
    return Manifest(**base)


# ── Round-trip ────────────────────────────────────────────
@pytest.mark.unit
def test_manifest_round_trips_through_disk(tmp_path):
    # Arrange
    manifest = _manifest()

    # Act
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)

    # Assert
    assert loaded.model.name == "xgboost"
    assert loaded.model.format == "xgboost-json"
    assert loaded.signature.input.features == ["f0", "f1"]
    assert loaded.signature.output.n_classes == 3
    assert loaded.metrics == {"test_acc": 0.75}
    assert loaded.bundle_version == BUNDLE_VERSION


@pytest.mark.unit
def test_preprocessor_class_is_serialized_under_the_key_class(tmp_path):
    """`class` is a Python keyword, so the field is aliased; the JSON must still
    read `class` — the loader resolves the dotted path from it."""
    manifest = _manifest(
        preprocessor=PreprocessorRef(
            **{"class": "ml_framework.data.preprocess.tabular:TabularPreprocessor"},
            files=["scaler.pkl"],
        )
    )
    write_manifest(tmp_path, manifest)

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw["preprocessor"]["class"].endswith("TabularPreprocessor")
    assert read_manifest(tmp_path).preprocessor.cls.endswith("TabularPreprocessor")


@pytest.mark.unit
def test_artifact_may_name_a_directory_and_is_never_interpreted(tmp_path):
    """`model/` may be a file or a directory — only the backend looks inside."""
    manifest = _manifest(
        model=ModelRef(
            name="hf_text", backend="lightning", artifact="model/hf_model", format="hf-directory"
        )
    )
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path).artifact_path(tmp_path) == tmp_path / "model" / "hf_model"


@pytest.mark.unit
def test_manifest_is_frozen():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _manifest().task = "binary"  # type: ignore[misc]


# ── Version gating ────────────────────────────────────────
@pytest.mark.unit
def test_a_newer_bundle_is_refused_rather_than_misread(tmp_path):
    # Arrange
    raw = json.loads(_manifest().to_json())
    raw["bundle_version"] = BUNDLE_VERSION + 1
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    # Act / Assert
    with pytest.raises(UnsupportedBundleVersionError, match="Upgrade ml-framework"):
        read_manifest(tmp_path)


@pytest.mark.unit
def test_unknown_manifest_fields_are_ignored_for_forward_compatibility(tmp_path):
    """Additive manifest fields must not break an older reader; incompatible
    changes are gated by bundle_version instead."""
    raw = json.loads(_manifest().to_json())
    raw["some_future_field"] = {"a": 1}
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    assert read_manifest(tmp_path).model.name == "xgboost"


@pytest.mark.unit
def test_missing_manifest_names_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        read_manifest(tmp_path)


# ── write_bundle ──────────────────────────────────────────
@pytest.mark.unit
def test_write_bundle_lays_out_manifest_config_metrics_and_model_files(tmp_path):
    # Arrange
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "bundle"

    # Act
    write_bundle(
        dest,
        _manifest(),
        config={"task": "multiclass", "model": {"name": "xgboost"}},
        metrics={"test_acc": 0.75},
        files={"model/model.json": src / "model.json"},
    )

    # Assert
    assert (dest / "manifest.json").exists()
    assert (dest / "model" / "model.json").exists()
    assert json.loads((dest / "config.json").read_text(encoding="utf-8"))["task"] == "multiclass"
    assert json.loads((dest / "metrics.json").read_text(encoding="utf-8")) == {"test_acc": 0.75}


@pytest.mark.unit
def test_write_bundle_copies_a_directory_artifact_whole(tmp_path):
    # Arrange
    src = tmp_path / "hf_model"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "weights.bin").write_bytes(b"\x00")

    # Act
    write_bundle(tmp_path / "bundle", _manifest(), files={"model/hf_model": src})

    # Assert
    assert (tmp_path / "bundle" / "model" / "hf_model" / "weights.bin").exists()


# ── Requirements ──────────────────────────────────────────
@pytest.mark.unit
def test_bundle_requirements_are_checked_before_any_import_is_attempted():
    """A GBDT serving container gets a pip command, not a ModuleNotFoundError from
    four frames deep inside a backend."""
    manifest = _manifest(
        requires=[
            RequirementRef.from_requirement(
                Requirement("definitely_not_installed_xyz", extra="gbdt", min_version="2.0")
            )
        ]
    )
    with pytest.raises(MissingExtraError) as excinfo:
        check_requirements(manifest)
    assert "pip install 'ml-framework[gbdt]'" in str(excinfo.value)


@pytest.mark.unit
def test_requirement_refs_round_trip_through_the_manifest(tmp_path):
    req = Requirement("torchvision", extra="image", min_version="0.15")
    write_manifest(tmp_path, _manifest(requires=[RequirementRef.from_requirement(req)]))
    assert read_manifest(tmp_path).requirements() == (req,)


@pytest.mark.unit
def test_satisfied_requirements_pass_silently():
    manifest = _manifest(requires=[RequirementRef(module="json")])
    check_requirements(manifest)


# ── v1 compatibility ──────────────────────────────────────
@pytest.mark.unit
def test_a_v1_bundle_is_detected_by_its_metadata_and_checkpoint(tmp_path):
    """The clean break was authorized for configs and tests, not for bundles
    already deployed."""
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.ckpt").write_bytes(b"\x00")

    assert is_v1_bundle(tmp_path)
    assert detect_bundle_version(tmp_path) == 1


@pytest.mark.unit
def test_a_v2_bundle_reports_version_two(tmp_path):
    write_manifest(tmp_path, _manifest())
    assert not is_v1_bundle(tmp_path)
    assert detect_bundle_version(tmp_path) == BUNDLE_VERSION


@pytest.mark.unit
def test_a_non_bundle_directory_reports_no_version(tmp_path):
    assert detect_bundle_version(tmp_path) is None


@pytest.mark.unit
def test_reading_a_v1_bundle_says_it_looks_like_v1(tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.ckpt").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="v1 bundle"):
        read_manifest(tmp_path)
