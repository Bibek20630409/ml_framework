"""
core/inference.py
─────────────────
Production inference over an artifact **bundle v2**. Imports neither torch nor
Lightning — that absence is the whole point of the module.

    inf = Inferencer.from_artifacts("outputs")
    preds = inf.predict(X_new)               # tabular: raw numpy (auto-transformed)
    probs = inf.predict_proba(X_new)         # when the model produces probabilities

The load path is five steps, and every one of them is driven by ``manifest.json``:

1. :func:`read_manifest` — rejects a bundle newer than this framework can read.
2. :func:`check_requirements` — refuses with a ``pip install`` command **before**
   attempting any import, so a missing library is an instruction rather than a
   ``ModuleNotFoundError`` four frames deep.
3. ``BACKENDS.get(manifest.model.backend).load(dir, manifest)`` — the backend is
   the only thing that knows what its own artifact is. ``model/`` may be a file or
   a directory; the loader never looks inside.
4. :func:`load_preprocessor` via the dotted class path in the manifest, so nothing
   here knows whether the fitted state is a scaler, a tokenizer or a per-series
   set of scalers.
5. ``estimator.predict(preprocessor.transform(x))``.

**Why this matters beyond tidiness:** v1 imported torch at module scope, hardcoded
``scaler.pkl``, reconstructed a whole ``ExperimentConfig`` from ``metadata.json``
(which needs a populated plugin registry *and* every training extra), and gated
``predict_proba`` on ``task == "regression"``. A GBDT serving container therefore
had to install ~2 GB of torch to run a 50 KB booster. It no longer does.

v1 bundles (``metadata.json`` + ``model.ckpt``, no manifest) still load: the clean
break was authorized for configs and tests, **not** for bundles already deployed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import Manifest, check_requirements, is_v1_bundle, read_manifest
from .task import get_task_spec
from .types import FrameworkError, UnsupportedCapability

log = logging.getLogger(__name__)


class BundleLoadError(FrameworkError):
    """A bundle could not be loaded (missing files, unusable manifest)."""


class Inferencer:
    """An estimator plus its preprocessor and the contract that describes them.

    Holds a :class:`~ml_framework.core.protocols.Estimator` — predict-only — not a
    torch module. That is what lets the same class serve a Lightning checkpoint, an
    XGBoost booster and (later) a Prophet model without branching.
    """

    def __init__(
        self,
        estimator: Any,
        manifest: Manifest,
        *,
        preprocessor: Any = None,
        reference_stats: dict | None = None,
    ):
        self.estimator = estimator
        self.manifest = manifest
        self.task = manifest.task
        self.data_kind = manifest.data_kind
        self.preprocessor = preprocessor
        self.reference_stats = reference_stats  # drift baseline (may be None)

    # ── Contract accessors ────────────────────────────────
    # These replace `getattr(inf.model, "input_dim", None)` in the serving layer —
    # code that reached into a torch module to learn its own API contract and
    # returned None for any non-torch estimator.
    @property
    def signature(self):
        return self.manifest.signature

    @property
    def feature_cols(self) -> list[str]:
        return list(self.manifest.signature.input.features)

    @property
    def n_features(self) -> int | None:
        return self.manifest.signature.input.n_features

    @property
    def class_names(self) -> list[str] | None:
        return self.manifest.signature.output.class_names

    @property
    def model_name(self) -> str:
        return self.manifest.model.name

    @property
    def backend_name(self) -> str:
        return self.manifest.model.backend

    @property
    def produces_proba(self) -> bool:
        """Whether ``predict_proba`` is available, **from the manifest**.

        The serving layer's 400 for ``/predict_proba`` now comes from here rather
        than from a hardcoded ``task == "regression"`` check, which was wrong for
        every task the framework had not been taught about yet.
        """
        return self.manifest.signature.output.kind == "probabilities"

    # ── Loaders ───────────────────────────────────────────
    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> Inferencer:
        """Load a bundle from disk. v2 by manifest; v1 by its legacy layout."""
        art = Path(artifact_dir)
        if is_v1_bundle(art):
            return cls._from_v1_bundle(art)

        manifest = read_manifest(art)
        # Before any import: a missing library becomes a pip command here, not an
        # ImportError from inside a backend factory.
        check_requirements(manifest)

        from .registry import get_backend

        backend = get_backend(manifest.model.backend)
        estimator = backend.load(art, manifest)

        preprocessor = cls._load_preprocessor(art, manifest)
        reference = cls._load_reference_stats(art)
        log.info(
            "loaded bundle: model=%s backend=%s task=%s",
            manifest.model.name,
            manifest.model.backend,
            manifest.task,
        )
        return cls(estimator, manifest, preprocessor=preprocessor, reference_stats=reference)

    @classmethod
    def from_registry(
        cls,
        name: str,
        stage_or_version: str = "Production",
        tracking_uri: str | None = None,
    ) -> Inferencer:
        """Load a model from the MLflow Model Registry.

        Downloads the registered bundle to a local dir and reuses
        :meth:`from_artifacts`, so registry-loaded and file-loaded models take an
        identical code path. Requires the ``[mlops]`` extra.
        """
        from ..tracking import download_bundle

        local_dir = download_bundle(name, stage_or_version, tracking_uri)
        return cls.from_artifacts(local_dir)

    # ── Load helpers ──────────────────────────────────────
    @staticmethod
    def _load_preprocessor(art: Path, manifest: Manifest) -> Any:
        if manifest.preprocessor is None:
            return None
        from ..data.preprocess import load_preprocessor

        spec = manifest.preprocessor.model_dump(by_alias=True)
        return load_preprocessor(art / manifest.preprocessor.dir, spec)

    @staticmethod
    def _load_reference_stats(art: Path) -> dict | None:
        path = art / "reference_stats.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _from_v1_bundle(cls, art: Path) -> Inferencer:
        """Load a pre-manifest bundle: ``metadata.json`` + ``model.ckpt`` + ``scaler.pkl``.

        Synthesizes a manifest from ``metadata.json`` so everything downstream —
        the signature accessors, the serving schemas, ``predict_proba`` gating —
        works against one shape. Lightning-only by construction: v1 could not
        produce any other kind of bundle.
        """
        from .bundle import InputSignature, ModelRef, OutputSignature, Signature

        meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
        task = meta["task"]
        task_spec = get_task_spec(task)
        n_classes = None
        if task == "binary":
            n_classes = 2
        elif task == "multiclass":
            n_classes = int(meta.get("output_dim") or 0) or None

        model_name = str((meta.get("config") or {}).get("model", {}).get("name", "mlp"))
        manifest = Manifest(
            bundle_version=1,
            task=task,
            data_kind=(meta.get("config") or {}).get("data", {}).get("kind", "tabular"),
            model=ModelRef(
                name=model_name,
                backend="lightning",
                artifact="model.ckpt",
                format="lightning-checkpoint",
                params=cls._v1_model_params(meta),
            ),
            signature=Signature(
                input=InputSignature(
                    payload="arrays",
                    features=list(meta.get("feature_cols") or []),
                    n_features=meta.get("input_dim"),
                ),
                output=OutputSignature(
                    kind=task_spec.output_kind,
                    n_classes=n_classes,
                    class_names=meta.get("class_names"),
                ),
            ),
        )

        from .registry import get_backend

        estimator = get_backend("lightning").load(art, manifest)

        preprocessor = None
        if (art / "scaler.pkl").exists():
            from ..data.preprocess.tabular import TabularPreprocessor

            preprocessor = TabularPreprocessor.from_legacy_bundle(art)
        log.info("loaded a v1 bundle from %s", art)
        return cls(
            estimator,
            manifest,
            preprocessor=preprocessor,
            reference_stats=cls._load_reference_stats(art),
        )

    @staticmethod
    def _v1_model_params(meta: dict) -> dict[str, Any]:
        """The architecture params out of a v1 ``metadata.json``.

        v1 stored the whole config, and its flat ``ModelConfig`` carried every
        model's knobs for every model (an MLP bundle recorded ``backbone`` and
        ``pretrained``). Only the keys the plugin actually declares are kept, so
        the rebuilt model validates against the v2 params schema.
        """
        from .plugins import UnknownPluginError
        from .registry import MODELS

        model_block = dict((meta.get("config") or {}).get("model", {}))
        name = str(model_block.pop("name", "mlp"))
        try:
            params_model = MODELS.get_spec(name).params_model
        except UnknownPluginError:
            # A v1 bundle naming a model this install does not have. Hand back what
            # was recorded and let the backend's load fail with a real message.
            return model_block
        if params_model is None:
            return model_block
        known = set(params_model.model_fields)
        return {k: v for k, v in model_block.items() if k in known}

    # ── Prediction ────────────────────────────────────────
    def _prepare(self, x: Any) -> Any:
        """Apply the fitted transform. The preprocessor owns what that means."""
        if self.preprocessor is None:
            return x
        return self.preprocessor.transform(x)

    def predict(self, x: Any) -> np.ndarray:
        """Hard predictions: labels for classification, values for regression."""
        return np.asarray(self.estimator.predict(self._prepare(x)))

    def predict_proba(self, x: Any) -> np.ndarray:
        """Class probabilities.

        Gated on the manifest's ``signature.output.kind`` rather than on the task
        name, so a model that genuinely cannot produce probabilities refuses for
        the right reason.
        """
        if not self.produces_proba:
            raise UnsupportedCapability(
                f"this model produces {self.signature.output.kind}, not probabilities"
            )
        return np.asarray(self.estimator.predict_proba(self._prepare(x)))

    def forecast_interval(self, horizon: Any) -> tuple[Any, Any]:
        """Prediction bounds for a forecast, or ``(None, None)``.

        Optional on purpose: the seasonal-naive baseline has no uncertainty model,
        and inventing one would be worse than reporting that it has none.
        """
        bounds = getattr(self.estimator, "interval", None)
        return bounds(horizon) if bounds is not None else (None, None)

    def predict_with_confidence(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(predicted class, max probability)`` per row."""
        probs = self.predict_proba(x)
        return probs.argmax(axis=1), probs.max(axis=1)


__all__ = ["BundleLoadError", "Inferencer"]
