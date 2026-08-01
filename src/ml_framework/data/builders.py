"""
data/builders.py
────────────────
Config → concrete objects, via the registries. Importing this module guarantees
the built-in sources, datamodules and models are registered.

``build_bundle`` is the entry point the training orchestrator uses: it returns a
:class:`~ml_framework.data.types.DataBundle`, which is framework-agnostic, rather
than a ``LightningDataModule``, which is not. ``build_datamodule`` remains for the
Lightning path (and for the LR finder), now as a thin wrapper.

For custom data, register a source:

    from ml_framework.core.registry import register_source
    from ml_framework.core.plugins import SourceSpec

    register_source(SourceSpec(name="my_source", data_kind="tabular",
                               build=my_build_fn, payload="arrays"))

...then set ``data.kind: my_source`` in the YAML.

**Import discipline:** everything below imports core *submodules*
(``..core.protocols``, ``..core.registry``) rather than the ``..core`` package
surface. ``core`` re-exports the data layer's helpers, so reaching for
``from ..core import X`` here would be a circular import that only shows up in
whichever module happens to be imported first. ``ExperimentConfig`` is imported
under ``TYPE_CHECKING`` for the same reason from the other direction: the config
validator resolves plugins, so a module-scope import here would make validating a
config depend on the data package being importable first.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

# Side-effect import: registers mlp/cnn in both registries.
from .. import plugins as _plugins  # noqa: F401
from ..core.plugins import SourceSpec
from ..core.protocols import BuildContext
from ..core.registry import MODELS, register_source
from ..core.types import Requirement
from .sources import (
    build_image_bundle,
    build_tabular_bundle,
    build_text_bundle,
    build_timeseries_bundle,
)
from .types import DataBundle

if TYPE_CHECKING:
    from ..config import ExperimentConfig

_BUNDLE_BUILDERS = {
    "tabular": build_tabular_bundle,
    "image": build_image_bundle,
    "text": build_text_bundle,
    "timeseries": build_timeseries_bundle,
}

# The sources that accept an injected partition. A kind absent from this table
# cannot be cross-validated, and `build_cv_bundles` says so by name rather than
# folding something it does not understand.
_CV_BUILDERS = {
    "tabular": build_tabular_bundle,
    "timeseries": build_timeseries_bundle,
    "image": build_image_bundle,
    "text": build_text_bundle,
}


def build_bundle(config: ExperimentConfig) -> DataBundle:
    """Materialize the configured data source as a :class:`DataBundle`.

    Dispatches through ``SOURCES`` so a third-party source is reachable by the
    same ``data.kind`` mechanism as the built-ins.
    """
    from ..core.registry import SOURCES

    if config.data.kind in SOURCES:
        spec = SOURCES.get(config.data.kind)
        return spec.build(config)
    builder = _BUNDLE_BUILDERS.get(config.data.kind)
    if builder is None:
        raise KeyError(f"Unknown data kind '{config.data.kind}'. Known: {SOURCES.names()}")
    return builder(config)


def build_cv_bundles(config: ExperimentConfig) -> Iterator[DataBundle]:
    """One :class:`DataBundle` per cross-validation fold.

    Each fold re-runs the *whole* source pipeline against its own partition, so
    the scaler, the drift reference and the imbalance correction are all fitted on
    that fold's training rows. Fitting them once and reusing them across folds
    would leak every fold's test set into every other fold's preprocessing —
    which produces a CV estimate that looks better than the model is.

    **Image folds are carved from the training folder only.** ``params.test_dir``
    is an explicit statement about which images are held back; pooling it in would
    override a decision made on disk. So under cross-validation "test" means a
    held-out slice of the training folder, and the final bundle's ``test_acc``
    still comes from ``test_dir`` — two numbers answering different questions.
    """
    from .splitters import CrossValidationSplitter, RollingOriginSplitter

    kind = config.data.kind
    if kind not in _CV_BUILDERS:
        raise NotImplementedError(
            f"cross-validation is implemented for {sorted(_CV_BUILDERS)}; got '{kind}'. "
            f"Set data.split.folds to 0 for a single holdout split."
        )

    split_cfg = config.data.split
    build = _CV_BUILDERS[kind]
    n, labels = _cv_population(config)

    if kind == "timeseries" or split_cfg.resolved_strategy(kind) == "temporal":
        # **Not** k-fold. Shuffled folds put future rows in training and past rows
        # in test, which is the leakage the config validator refuses elsewhere;
        # doing it here under the name "cross-validation" would be the same bug
        # wearing a different hat.
        folds = RollingOriginSplitter(
            folds=split_cfg.folds,
            horizon=split_cfg.horizon,
            gap=split_cfg.gap,
            expanding=split_cfg.expanding,
        ).split(n)
    else:
        folds = CrossValidationSplitter(
            folds=split_cfg.folds,
            seed=config.runtime.seed,
            task=config.task,
            val_size=split_cfg.val_size,
        ).split(n, y=labels)

    for indices in folds:
        yield build(config, indices=indices)


def _cv_population(config: ExperimentConfig) -> tuple[int, Any]:
    """``(row count, labels)`` for the pool being folded.

    Image data has no table to read: the pool is the **training folder**, and its
    labels come from an ``ImageFolder`` directory scan rather than from decoding
    pixels. Keeping that difference here rather than in ``build_cv_bundles`` is
    what stops the folding logic from growing a per-kind branch.
    """
    from .sources.image import train_labels
    from .sources.tabular import read_table
    from .sources.text import text_labels

    if config.data.kind == "image":
        labels = np.asarray(train_labels(config), dtype="int64")
        return len(labels), labels

    if config.data.kind == "text":
        # Read through the source's own encoder rather than off the raw column, so
        # folds stratify on exactly the integers training will see. String labels
        # encoded twice by two rules is a way to stratify on the wrong thing.
        labels = text_labels(config)
        return len(labels), labels

    frame = read_table(str(config.data.path))
    labels = frame[config.data.target].to_numpy()
    if config.task != "regression":
        labels = labels.astype("int64")
    return len(frame), labels


def build_datamodule(config: ExperimentConfig) -> Any:
    """The Lightning adapter for the configured data kind.

    The adapter is imported *here* rather than at module scope: it subclasses
    ``pl.LightningDataModule``, and ``build_bundle`` — which the orchestrator uses
    for every backend — lives in this same module. A module-scope import would put
    Lightning into the import path of a GBDT training run. Importing it also
    registers the v1 datamodules, which is what ``get_datamodule_class`` reads.
    """
    from ..core.registry import get_datamodule_class
    from . import lightning_adapter  # noqa: F401  (registers tabular/image datamodules)

    return get_datamodule_class(config.data.kind)(config)


def build_model(
    config: ExperimentConfig,
    *,
    input_dim: int,
    output_dim: int,
    class_weights: Any = None,
) -> Any:
    """Build the configured model through its plugin spec.

    The config is unpacked into a :class:`BuildContext` *here* rather than handed
    to the model, which is the point of the v2 change: a model knows its
    architecture and its forward pass, not the shape of the experiment around it.
    """
    spec = MODELS.get(config.model.name)
    return spec.build(
        BuildContext(
            task=config.task,
            input_dim=input_dim,
            output_dim=output_dim,
            class_weights=class_weights,
            params=config.model.params,
            optim=config.fit.params,
            seed=config.runtime.seed,
        )
    )


# ── v2 source specs ───────────────────────────────────────
# `build` returns a DataBundle, which is what `SourceSpec.build`'s docstring
# promises. Source-specific knobs come from `data.params` and are validated by
# the source's own frozen params model.
register_source(
    SourceSpec(
        name="tabular",
        data_kind="tabular",
        build=build_tabular_bundle,
        payload="arrays",
        requires=(),
        description="CSV/Parquet table with a target column.",
    )
)
register_source(
    SourceSpec(
        name="timeseries",
        data_kind="timeseries",
        build=build_timeseries_bundle,
        # The *declared* payload is the one a forecaster takes. The source emits
        # windowed arrays instead when the selected model asks for them, which is a
        # per-model decision rather than a property of the source.
        payload="series",
        requires=(),
        description="Time-ordered table with a value column; ordered by split.time_col.",
    )
)
register_source(
    SourceSpec(
        name="text",
        data_kind="text",
        build=build_text_bundle,
        # Raw strings, lazily; the tokenizer runs per batch in the preprocessor's
        # collate_fn so padding is to the batch rather than to the corpus.
        payload="dataset",
        # Reading and folding a text corpus needs neither transformers nor torch.
        # The tokenizer's requirements are declared by the *model*, which is what
        # makes them a `pip install` hint at model-selection time.
        requires=(),
        description="CSV/Parquet/JSONL with a text column and a label column.",
    )
)
register_source(
    SourceSpec(
        name="image",
        data_kind="image",
        build=build_image_bundle,
        payload="dataset",
        requires=(
            Requirement("torchvision", extra="image", min_version="0.15"),
            Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),
        ),
        description="Directory-of-class-directories image folders (ImageFolder layout).",
    )
)
