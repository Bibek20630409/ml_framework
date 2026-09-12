"""
core/types.py
─────────────
The dependency-free vocabulary shared by ``config``, ``plugins``, ``backends``
and the data layer.

**Nothing in this module may import numpy, torch, pydantic or any optional
dependency.** Both ``config.schema`` and ``core.plugins`` import from here; a
heavier import would either create a cycle or eagerly pull an optional library
into a bare install — the exact failure mode the plugin design exists to avoid.
Availability of an optional library is therefore answered with
``importlib.util.find_spec`` (no import, no exception), never with a
``try: import … except ImportError`` probe.

Task/DataKind here are the **v2** vocabulary and are intentionally wider than
``config.schema.Task`` (still ``binary|multiclass|regression``). The config
schema switches over to these aliases in P2; until then a task listed here has
no ``TaskSpec`` row unless ``core.task`` registers one.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from typing import Final, Literal

# ── Task / data vocabulary ────────────────────────────────
# Task determines loss, metrics and head; DataKind determines ingestion. The two
# stay orthogonal on purpose: text classification is `multiclass` + `kind: text`,
# not a `text_classification` task, so the Literal grows additively rather than
# combinatorially.
Task = Literal[
    "binary",
    "multiclass",
    "multilabel",
    "regression",
    "forecasting",
    "token_classification",
    "seq2seq",
]
DataKind = Literal["tabular", "image", "text", "timeseries", "audio", "video"]

# What a split's `payload` physically is — the honest alternative to pretending
# every estimator consumes an (n, d) matrix. A backend declares which of these it
# can consume via `Capabilities.accepts`, which is what turns "xgboost cannot
# consume an image folder" into a build-time error instead of a shape error 200
# lines deep.
Payload = Literal["arrays", "frame", "dataset", "series"]

# ── Staged-read vocabulary (P13) ──────────────────────────
# Storage → a tensor the model can consume, named end to end. Seven stages, and
# the split is not pedantry — each name marks a place where the cost, the device,
# or the thing that can go wrong is genuinely different:
#
#   read             a positioned read. Bytes, from wherever they live.
#   demux            container → elementary-stream packets. Real for MP3/MP4,
#                    degenerate for a JPEG (one image, one packet).
#   decode           reverse a compression scheme. A pre-tokenized `.bin` shard
#                    has neither of the two above (a memmap slice is a page
#                    fault, not a call); an MP3 has both.
#   construct        attach a dtype, a shape and strides to a pointer. NOT decode:
#                    a token shard decodes to nothing and constructs a tensor, a
#                    device decoder decodes and constructs nothing at all.
#   transform        change dtype/layout/scale — the mel matmul, the THWC→CTHW
#                    permute + normalize. A copy, and usually the 4x one.
#   h2d              host → device. Async only from a page-locked buffer.
#   gpu_transform    the same `transform` work, run AFTER h2d. Not a different
#                    operation: a different *placement*, which is why one
#                    implementation serves both and only the seam moves.
#
# The last four are the tail. Nothing about them belongs to a decoder — they are
# declared by a preprocessor (`BasePreprocessor.stages`) and executed by the
# transport layer, which is what keeps decode and tensor construction the two
# independent axes `data/streaming/stages.py` argues they are.
Stage = Literal["read", "demux", "decode", "construct", "transform", "h2d", "gpu_transform"]

# What a *decoder* can claim. Narrower than `Stage` on purpose: `DecoderSpec` is
# typed against this, so a decoder declaring `h2d` is a type error rather than a
# claim nothing would ever execute.
DecoderStage = Literal["read", "demux", "decode"]

# The physical arrangement of a decoded buffer, before any transform. `nv12` is
# listed because it is what a hardware video decoder actually emits; calling that
# `chw` would be a lie that costs a wrong reshape later.
Layout = Literal["hwc", "chw", "nv12", "pcm", "mel", "tokens", "thwc"]

# Where a decoder left its output. `device` means the bytes never touched host
# memory: no array to wrap, no pinned staging buffer, no H2D copy. Consumer:
# `BundleDataModule`, which skips pinning and forces `num_workers=0` for one.
LandsIn = Literal["host", "device"]

# How much a decoder's output can be trusted — what a failure *surfaces*, not
# whether one can occur:
#   checked  a verified integrity primitive; damage raises (FLAC CRC-16, PNG CRC-32)
#   loud     no checksum, but structural damage raises (missing `moov`, bad header)
#   silent   damage yields a valid-shaped wrong result and NO exception (MP3 resync,
#            mid-stream H.264 artifacts) — invisible at training time
#   none     no integrity information exists (a bit flip in a uint16 token shard is
#            a valid token id); only a recorded digest defends it
# Consumer: the corrupt-sample policy in `data/streaming/integrity.py`.
Integrity = Literal["checked", "loud", "silent", "none"]

STAGES: Final[tuple[Stage, ...]] = (
    "read",
    "demux",
    "decode",
    "construct",
    "transform",
    "h2d",
    "gpu_transform",
)
# The two halves, because every consumer wants one or the other and neither wants
# the union. `mlf decoders --show` renders the head; `BasePreprocessor.stages` and
# the tail probe range over the tail.
DECODER_STAGES: Final[tuple[DecoderStage, ...]] = ("read", "demux", "decode")
TAIL_STAGES: Final[tuple[Stage, ...]] = ("construct", "transform", "h2d", "gpu_transform")
LAYOUTS: Final[tuple[Layout, ...]] = ("hwc", "chw", "nv12", "pcm", "mel", "tokens", "thwc")
LANDS_IN: Final[tuple[LandsIn, ...]] = ("host", "device")
INTEGRITIES: Final[tuple[Integrity, ...]] = ("checked", "loud", "silent", "none")


# The shape of what a model emits, recorded in the bundle manifest's signature.
#
# `token_labels` and `text` are not decoration on `labels`/`values`: a per-token
# tagger emits one label per *position* in a variable-length sequence, and a
# generative model emits a string. Both break the "one row in, one number out"
# assumption that `labels` carries, and the serving layer, `evaluate()` and
# `predictions.csv` each have to know which they are holding.
OutputKind = Literal["labels", "probabilities", "values", "series", "token_labels", "text"]

# Canonical head postprocessing. One value per task, applied in exactly one place
# (the estimator) rather than being re-derived at every call site.
#
# `generate` is the odd one and says so honestly: the others are functions of the
# logits alone, while generation is an autoregressive loop that needs the *inputs*.
# Naming it here keeps the estimator's routing table complete rather than pushing
# a per-model special case into the backend.
Postprocess = Literal["sigmoid", "softmax", "identity", "generate"]

# Positions the loss and the metrics must ignore: padding in a token-tagged
# sequence, and pad tokens in a generation target. -100 is torch's
# `CrossEntropyLoss` default `ignore_index` and HuggingFace's convention for the
# same thing, so using it means neither has to be told.
IGNORE_INDEX: Final[int] = -100

# Optimisation direction for a metric.
Direction = Literal["min", "max"]

TASKS: Final[tuple[Task, ...]] = (
    "binary",
    "multiclass",
    "multilabel",
    "regression",
    "forecasting",
    "token_classification",
    "seq2seq",
)
DATA_KINDS: Final[tuple[DataKind, ...]] = (
    "tabular",
    "image",
    "text",
    "timeseries",
    "audio",
    "video",
)
PAYLOADS: Final[tuple[Payload, ...]] = ("arrays", "frame", "dataset", "series")

# The distribution name used in every `pip install …` hint we emit.
DIST_NAME: Final[str] = "ml-framework"


# ── Errors ────────────────────────────────────────────────
class FrameworkError(Exception):
    """Base class for every error this framework raises deliberately.

    Callers that want to distinguish "the framework told me no" from "something
    blew up" can catch this one class.
    """


class UnsupportedCapability(FrameworkError):
    """Raised when a capability is requested that the plugin does not declare.

    Example: ``predict_proba`` on a regression estimator. Companion to
    :class:`Capabilities` — the flag is the declaration, this is the refusal.
    """


# ── Optional-dependency requirements ──────────────────────
def _version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric components of a version string, for ordering only.

    ``"2.0.3.post1"`` → ``(2, 0, 3)``; ``"1.1"`` → ``(1, 1)``. Deliberately
    naive: this module may not import ``packaging``, and every comparison we make
    is a ``>=`` floor against a release version. Non-numeric segments stop the
    parse rather than raising, so a dev/rc version compares by its release part.
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class Requirement:
    """One optional import a plugin needs, and the extra that provides it.

    ``extra`` must match a key in ``[project.optional-dependencies]`` **verbatim**
    — it is interpolated straight into the ``pip install`` hint, and that hint is
    the whole point of the class.
    """

    module: str
    extra: str | None = None
    min_version: str | None = None
    # Distribution name when it differs from the import name (e.g. import `PIL`,
    # install `Pillow`). Used for version lookup and in the requirement string.
    dist: str | None = None

    @property
    def package(self) -> str:
        return self.dist or self.module

    def is_installed(self) -> bool:
        """True if the module can be found, **without importing it**.

        Note that ``find_spec`` does import the *parent* of a dotted name, so
        prefer top-level module names in requirements.
        """
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            # ValueError: module exists in sys.modules but has no spec.
            # ImportError: a parent package of a dotted name is missing.
            return False

    def installed_version(self) -> str | None:
        try:
            return importlib.metadata.version(self.package)
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_satisfied(self) -> bool:
        return self.unmet_reason() is None

    def unmet_reason(self) -> str | None:
        """Why this requirement is not met, or ``None`` if it is."""
        if not self.is_installed():
            return f"{self.package} is not installed"
        if self.min_version is None:
            return None
        found = self.installed_version()
        if found is None:
            # Importable but no distribution metadata (namespace package, vendored
            # copy, editable oddity). Trust the import rather than block the user.
            return None
        if _version_tuple(found) < _version_tuple(self.min_version):
            return f"{self.package} {found} is older than {self.min_version}"
        return None

    def spec(self) -> str:
        """PEP 508-ish requirement string, e.g. ``xgboost>=2.0``."""
        return f"{self.package}>={self.min_version}" if self.min_version else self.package

    def pip_hint(self, dist: str = DIST_NAME) -> str:
        """The command that fixes this requirement."""
        if self.extra:
            return f"pip install '{dist}[{self.extra}]'"
        return f"pip install '{self.spec()}'"


def unmet_requirements(requires: tuple[Requirement, ...]) -> tuple[Requirement, ...]:
    """The subset of ``requires`` that is not satisfied, in declaration order."""
    return tuple(r for r in requires if not r.is_satisfied())


# ── Capabilities ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a plugin or backend can do.

    Every flag here has **exactly one named consumer** — a flag nothing reads is
    decoration, and decoration drifts out of sync with reality. The consumer is
    named in each comment; adding a flag without one is a review failure.
    """

    # Which split payloads this can consume. Checked at build time by
    # `registry.validate_combination`.
    accepts: frozenset[Payload] = frozenset({"arrays"})

    # Preprocessor: skip StandardScaler when False (pointless for trees, and it
    # destroys feature interpretability).
    needs_scaling: bool = True
    # Preprocessor: pass pandas `category` dtype through instead of one-hot.
    native_categorical: bool = False
    # Preprocessor: skip imputation — XGB/LGBM route NaN natively and imputing
    # measurably hurts them.
    native_missing: bool = False
    # Imbalance resolver: prefer sample weights over SMOTE (SMOTE is nonsense for
    # GBDT).
    supports_sample_weight: bool = False

    # Serving: `/predict_proba` returns 400 from the manifest instead of relying
    # on a hardcoded task check.
    produces_proba: bool = True

    # `core.explain.feature_importance`: try the estimator's own
    # `feature_importances_`/`coef_` before falling back to permutation
    # importance, which costs a full predict pass per feature. Declaring it is
    # cheaper than probing, and the probe still runs — the flag decides the
    # *order* of the attempts, so a wrong declaration costs time, never
    # correctness.
    native_feature_importance: bool = False

    # HPO driver: install a pruning hook, or fall back to a non-pruning pruner.
    supports_pruning: bool = False

    # Device/precision resolution: warn instead of silently ignoring
    # `runtime.precision: 16` on a backend that cannot honour it.
    supports_gpu: bool = False
    supports_mixed_precision: bool = False

    # `mlf lr`: refuse politely instead of crashing inside torch_lr_finder.
    supports_lr_range_test: bool = False

    # `train(resume=...)`: warn and continue from scratch rather than silently
    # ignoring the flag. A one-shot `fit(X, y)` has no partial state to resume
    # from, so this is False for everything but an epoch loop.
    supports_resume: bool = False

    def can_accept(self, payload: Payload) -> bool:
        return payload in self.accepts
