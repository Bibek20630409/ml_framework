"""
core/bundle.py
──────────────
The artifact bundle v2 — and its ``manifest.json``, **the only file a loader must
understand**.

    <bundle>/
      manifest.json        ← the serving contract
      config.json          ← effective config post-defaults/post-HPO. Audit record.
      model/               ← opaque to everything but the backend
          model.ckpt | model.json | model.cbm | hf_model/ | prophet.pkl
      preprocessor/        ← opaque to everything but the Preprocessor
      metrics.json · reference_stats.json · hpo.json
      report.txt · confusion_matrix.txt · predictions.csv · training.log

The invariant that makes heterogeneous model files a non-problem: **the manifest
is the only uniform thing, and it names the non-uniform things.** ``model/`` may
be a file or a directory — the loader never looks. It hands ``bundle_dir`` plus
the manifest to ``backend.load()``. ``format`` is tracked separately from the file
extension so serialization can migrate (xgboost json → ubj) without breaking
readers.

Two deliberate absences:

* **The manifest does not embed the full training config.** v1's
  ``metadata.json`` did, and ``inference.py`` reconstructed an
  ``ExperimentConfig`` from it — which requires a populated plugin registry *and*
  every training extra at serving time. ``config.json`` sits beside it as the
  audit record; the manifest alone is the serving contract.
* **No torch.** This module is imported by the serving path.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field

from .plugins import check_requirements as _check_requirements
from .types import DataKind, FrameworkError, OutputKind, Payload, Requirement, Task

BUNDLE_VERSION: Final[int] = 2
MANIFEST_NAME: Final[str] = "manifest.json"
CONFIG_NAME: Final[str] = "config.json"
METRICS_NAME: Final[str] = "metrics.json"
MODEL_DIR: Final[str] = "model"
PREPROCESSOR_DIR: Final[str] = "preprocessor"


class UnsupportedBundleVersionError(FrameworkError):
    """The bundle was written by a newer framework than this one can read."""


def framework_version() -> str:
    from .. import __version__

    return __version__


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Manifest models ───────────────────────────────────────
# `extra="ignore"` deviates from the repo-wide `extra="forbid"` convention on
# purpose: a manifest is read by code that may be *older* than the writer, and an
# additive field must not be a hard failure. Backwards-incompatible changes are
# gated by `bundle_version` instead, which is checked explicitly.
_MANIFEST_CONFIG = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)


class RequirementRef(PydanticModel):
    """A requirement recorded in the bundle, so serving can refuse *before* it
    tries to import something that is not there."""

    model_config = _MANIFEST_CONFIG

    module: str
    extra: str | None = None
    min_version: str | None = None
    dist: str | None = None

    def to_requirement(self) -> Requirement:
        return Requirement(
            module=self.module, extra=self.extra, min_version=self.min_version, dist=self.dist
        )

    @classmethod
    def from_requirement(cls, req: Requirement) -> RequirementRef:
        return cls(module=req.module, extra=req.extra, min_version=req.min_version, dist=req.dist)


class ModelRef(PydanticModel):
    model_config = _MANIFEST_CONFIG

    name: str
    backend: str
    # Path relative to the bundle root; may be a file or a directory.
    artifact: str
    format: str
    params: dict[str, Any] = Field(default_factory=dict)
    # Generalized `count_parameters()`: params for Lightning, tree/leaf counts for
    # GBDT. Reported by the orchestrator, informational only.
    size: dict[str, Any] | None = None


class PreprocessorRef(PydanticModel):
    model_config = _MANIFEST_CONFIG

    # Dotted class path, e.g. "ml_framework.data.preprocess.tabular:TabularPreprocessor".
    cls: str = Field(alias="class")
    dir: str = PREPROCESSOR_DIR
    files: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class InputSignature(PydanticModel):
    model_config = _MANIFEST_CONFIG

    payload: Payload = "arrays"
    features: list[str] = Field(default_factory=list)
    n_features: int | None = None


class OutputSignature(PydanticModel):
    model_config = _MANIFEST_CONFIG

    kind: OutputKind
    n_classes: int | None = None
    class_names: list[str] | None = None


class Signature(PydanticModel):
    """The model's API contract.

    Replaces ``getattr(inf.model, "input_dim", None)`` in the serving layer — code
    that reached into a torch module to learn its own contract and returned
    ``None`` for any non-torch estimator. Feature *names* live here so the serving
    layer can reject silently-reordered columns, the most common serving defect.
    """

    model_config = _MANIFEST_CONFIG

    input: InputSignature
    output: OutputSignature


class Manifest(PydanticModel):
    model_config = _MANIFEST_CONFIG

    bundle_version: int = BUNDLE_VERSION
    framework_version: str = Field(default_factory=framework_version)
    created_at: str = Field(default_factory=_utc_now)

    task: Task
    data_kind: DataKind
    model: ModelRef
    signature: Signature
    preprocessor: PreprocessorRef | None = None
    requires: list[RequirementRef] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    hpo: dict[str, Any] | None = None
    # The bake-off this model won, when one ran: every candidate's score, latency,
    # size and explainability, plus the rule that picked between them. `None` when
    # the model was named rather than chosen — which is the common case, and is
    # why this is nullable rather than an empty dict.
    selection: dict[str, Any] | None = None

    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(r.to_requirement() for r in self.requires)

    def artifact_path(self, bundle_dir: str | Path) -> Path:
        return Path(bundle_dir) / self.model.artifact

    def to_json(self) -> str:
        return json.dumps(self.model_dump(by_alias=True), indent=2)


# ── Read / write ──────────────────────────────────────────
def manifest_path(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / MANIFEST_NAME


def read_manifest(bundle_dir: str | Path) -> Manifest:
    """Load and version-check a bundle's manifest.

    Rejects a bundle newer than :data:`BUNDLE_VERSION` outright — reading it with
    older rules would produce plausible-looking wrong predictions, which is worse
    than refusing.
    """
    path = manifest_path(bundle_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {bundle_dir}"
            + (" (this looks like a v1 bundle)" if is_v1_bundle(bundle_dir) else "")
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = int(raw.get("bundle_version", 1))
    if version > BUNDLE_VERSION:
        raise UnsupportedBundleVersionError(
            f"Bundle at {bundle_dir} is version {version}; this framework "
            f"({framework_version()}) reads up to version {BUNDLE_VERSION}. Upgrade ml-framework."
        )
    return Manifest.model_validate(raw)


def write_manifest(bundle_dir: str | Path, manifest: Manifest) -> Path:
    dest = Path(bundle_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = manifest_path(dest)
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def write_bundle(
    dest: str | Path,
    manifest: Manifest,
    *,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Assemble a bundle directory: manifest, audit config, metrics, extra files.

    ``files`` maps bundle-relative destinations to source paths, so a backend can
    contribute artifacts (``model/model.json``, ``preprocessor/scaler.pkl``)
    without knowing the bundle layout rules.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    (out / MODEL_DIR).mkdir(exist_ok=True)

    for rel, src in (files or {}).items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(src)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target)

    if config is not None:
        (out / CONFIG_NAME).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    if metrics is not None:
        (out / METRICS_NAME).write_text(json.dumps(dict(metrics), indent=2), encoding="utf-8")

    write_manifest(out, manifest)
    return out


def check_requirements(manifest: Manifest, *, what: str | None = None) -> None:
    """Refuse to load a bundle whose libraries are missing, **before** importing.

    This is why a GBDT serving container gets a pip command instead of a
    ``ModuleNotFoundError`` from four frames deep inside a backend.
    """
    _check_requirements(
        manifest.requirements(),
        what=what or f"bundle for model '{manifest.model.name}'",
    )


def head_width(manifest: Manifest) -> int:
    """The model's output width, recovered from the signature.

    Binary is the asymmetric case: two classes, one logit. Reading ``n_classes``
    directly would build a two-logit head, and the weights would not load into it.

    Lives here rather than on a backend because more than one thing rebuilds a
    model from a manifest — the Lightning backend for a checkpoint, a
    self-serializing model for its own directory — and two copies of this rule
    would eventually disagree about binary.
    """
    if manifest.task in ("multiclass", "token_classification"):
        # A tagger's head is one logit per tag *at every position*, and the
        # one-logit binary convention does not reach it: even a two-tag corpus gets
        # a two-wide head, because the loss is cross-entropy over positions rather
        # than a single sigmoid.
        return int(manifest.signature.output.n_classes or 0)
    if manifest.task == "seq2seq":
        # The head is the checkpoint's vocabulary. Reporting a width here would be
        # reporting something the framework neither chose nor can check.
        return 0
    return 1


# ── v1 compatibility ──────────────────────────────────────
def is_v1_bundle(bundle_dir: str | Path) -> bool:
    """A v1 bundle: ``metadata.json`` + ``model.ckpt``, no manifest.

    The clean break was authorized for configs and tests — **not** for bundles
    already deployed, so detection stays in the loader path.
    """
    d = Path(bundle_dir)
    return (
        not manifest_path(d).exists()
        and (d / "metadata.json").exists()
        and (d / "model.ckpt").exists()
    )


def detect_bundle_version(bundle_dir: str | Path) -> int | None:
    """1, 2, or ``None`` when the directory is not a bundle at all."""
    d = Path(bundle_dir)
    if manifest_path(d).exists():
        raw = json.loads(manifest_path(d).read_text(encoding="utf-8"))
        return int(raw.get("bundle_version", 1))
    return 1 if is_v1_bundle(d) else None
