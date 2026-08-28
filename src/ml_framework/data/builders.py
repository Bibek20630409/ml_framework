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

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

# Side-effect import: registers mlp/cnn in both registries.
from .. import plugins as _plugins  # noqa: F401
from ..core.plugins import SourceSpec
from ..core.protocols import BuildContext
from ..core.registry import MODELS, register_source
from ..core.types import FrameworkError, Requirement
from .sources import (
    build_image_bundle,
    build_tabular_bundle,
    build_text_bundle,
    build_timeseries_bundle,
)
from .types import DataBundle

if TYPE_CHECKING:
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)

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


# The data kinds whose source honours `data.backend`. A kind absent from this
# table reads through pandas regardless, so selecting a distributed engine for it
# would be a promise the source does not keep — `_check_backend_supported` refuses
# by name instead, the same way `build_cv_bundles` refuses an unfoldable kind.
_BACKEND_AWARE_KINDS = frozenset({"tabular"})


def _check_backend_supported(config: ExperimentConfig) -> None:
    backend = config.data.backend
    if backend != "local" and config.data.kind not in _BACKEND_AWARE_KINDS:
        raise FrameworkError(
            f"data.backend '{backend}' is implemented for "
            f"{sorted(_BACKEND_AWARE_KINDS)} data; got kind '{config.data.kind}'. "
            f"Use data.backend: local."
        )


def build_bundle(config: ExperimentConfig) -> DataBundle:
    """Materialize the configured data source as a :class:`DataBundle`.

    Dispatches through ``SOURCES`` so a third-party source is reachable by the
    same ``data.kind`` mechanism as the built-ins.
    """
    from ..core.registry import SOURCES

    _check_backend_supported(config)
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
    kind = config.data.kind
    if kind not in _CV_BUILDERS:
        raise NotImplementedError(
            f"cross-validation is implemented for {sorted(_CV_BUILDERS)}; got '{kind}'. "
            f"Set data.split.folds to 0 for a single holdout split."
        )
    # Also checked here, not only in `build_bundle`: the CV path calls the source
    # builders directly, so a fold would otherwise slip past the refusal that the
    # single-holdout path enforces.
    _check_backend_supported(config)

    build = _CV_BUILDERS[kind]
    for indices in cv_folds(config):
        yield build(config, indices=indices)


def cv_folds(config: ExperimentConfig, *, folds: int | None = None) -> list[Any]:
    """The cross-validation partitions for ``config``, as a list of index sets.

    Split out from :func:`build_cv_bundles` because the inner loop of a
    CV-objective hyperparameter search needs the *partitions* without paying to
    materialize a bundle per fold up front, and because a partition is the thing
    worth testing directly.

    ``folds`` overrides ``data.split.folds`` so the inner folds of a nested search
    (``tune.cv_folds``) can differ from the outer estimate, which is the whole
    point of nesting them.
    """
    from .splitters import (
        CombinatorialPurgedSplitter,
        CrossValidationSplitter,
        PurgedKFoldSplitter,
        RollingOriginSplitter,
    )

    split_cfg = config.data.split
    kind = config.data.kind
    k = split_cfg.folds if folds is None else folds
    n, labels = _cv_population(config)
    strategy = split_cfg.resolved_cv_strategy(kind, config.task)

    if strategy == "rolling_origin":
        # **Not** k-fold. Shuffled folds put future rows in training and past rows
        # in test, which is the leakage the config validator refuses elsewhere;
        # doing it here under the name "cross-validation" would be the same bug
        # wearing a different hat.
        return RollingOriginSplitter(
            folds=k,
            horizon=split_cfg.horizon,
            gap=split_cfg.gap,
            expanding=split_cfg.expanding,
        ).split(n)

    if strategy == "purged":
        ends, times = _label_columns(config, n)
        return PurgedKFoldSplitter(
            folds=k,
            embargo=split_cfg.embargo,
            label_horizon=split_cfg.label_horizon,
            val_size=split_cfg.val_size,
        ).split(n, y=labels, label_end=ends, time=times)

    if strategy == "cpcv":
        # `folds` is not a fold count here — CPCV's count is C(groups, test_groups).
        # Saying so beats silently ignoring the number the user set.
        if folds is not None and folds != split_cfg.cpcv_groups:
            log.info(
                "cv_strategy=cpcv: fold count comes from cpcv_groups/cpcv_test_groups, "
                "not from folds=%d",
                folds,
            )
        ends, times = _label_columns(config, n)
        return CombinatorialPurgedSplitter(
            groups=split_cfg.cpcv_groups,
            test_groups=split_cfg.cpcv_test_groups,
            embargo=split_cfg.embargo,
            label_horizon=split_cfg.label_horizon,
            val_size=split_cfg.val_size,
            max_folds=split_cfg.cpcv_max_folds,
        ).split(n, y=labels, label_end=ends, time=times)

    # `stratified` and `kfold` are one splitter: it already picks between
    # StratifiedKFold and KFold from the task, and stratifying a continuous
    # target is what crashed the original framework. An explicit `stratified` on
    # a regression task is therefore a request the splitter must refuse rather
    # than quietly honour.
    if strategy == "stratified" and config.task not in ("binary", "multiclass"):
        raise ValueError(
            f"cv_strategy 'stratified' needs one class label per row, and task "
            f"'{config.task}' has none. Use cv_strategy: kfold."
        )
    return CrossValidationSplitter(
        folds=k,
        seed=config.runtime.seed,
        # `kfold` forces the unstratified branch even for a classification task,
        # which is how you ask for it.
        task=config.task if strategy == "stratified" else "regression",
        val_size=split_cfg.val_size,
    ).split(n, y=labels)


def _label_columns(config: ExperimentConfig, n: int) -> tuple[Any, Any]:
    """``(label_end, observation_times)``, or ``(None, None)`` when not configured.

    Read straight off the source table rather than off the bundle: both columns
    are metadata about *when* a row was observed and when its label became known,
    not features, and threading them through preprocessing would put them in the
    model's input matrix.

    Both are returned together because neither is usable alone —
    :func:`~ml_framework.data.splitters.label_spans` needs the observation times
    to place a label end time on a row, and returning them from one function is
    what stops a caller from supplying half the pair.

    Rows are sorted by ``time_col`` here, matching what the temporal sources do,
    so the positions the splitter computes line up with the rows it is splitting.
    """
    split_cfg = config.data.split
    col = split_cfg.label_end_col
    if col is None:
        return None, None
    if config.data.kind not in ("tabular", "timeseries"):
        raise ValueError(
            f"split.label_end_col is only readable for tabular/timeseries data; "
            f"got kind '{config.data.kind}'. Use split.label_horizon instead."
        )
    if not split_cfg.time_col:
        raise ValueError(
            "split.label_end_col needs split.time_col: a label end *time* can only be "
            "mapped onto a row position if the rows' own observation times are known. "
            "Set time_col, or state the span in rows with split.label_horizon."
        )

    from .backends import engine_for

    engine = engine_for(config)
    table = engine.read_table(str(config.data.path))

    # Both checks run against the *schema and row count*, before anything is
    # collected: under `spark` a mistyped column name should cost a metadata
    # lookup, not a full materialization that then fails.
    available = engine.columns(table)
    for name in (col, split_cfg.time_col):
        if name not in available:
            raise ValueError(
                f"split column '{name}' is not in {config.data.path}. "
                f"Columns: {sorted(available)[:20]}"
            )
    rows = engine.n_rows(table)
    if rows != n:
        raise ValueError(f"'{col}' has {rows} rows for {n} data rows")

    # Project first, then collect **once**. Two `engine.column` calls would be two
    # collects, and these two arrays have to line up row-for-row — `label_spans`
    # pairs each label-end time with the observation time on the same row. A
    # distributed engine re-executing a sorted plan twice need not agree on row
    # order (pyspark documents `monotonically_increasing_id`, which `sort_by`
    # leans on, as non-deterministic), so a second collect could silently pair the
    # wrong two values. One materialization, sliced locally, cannot.
    ordered = engine.select(engine.sort_by(table, split_cfg.time_col), [col, split_cfg.time_col])
    frame = engine.to_pandas(ordered)
    return frame[col].to_numpy(), frame[split_cfg.time_col].to_numpy()


def _cv_population(config: ExperimentConfig) -> tuple[int, Any]:
    """``(row count, labels)`` for the pool being folded.

    Image data has no table to read: the pool is the **training folder**, and its
    labels come from an ``ImageFolder`` directory scan rather than from decoding
    pixels. Keeping that difference here rather than in ``build_cv_bundles`` is
    what stops the folding logic from growing a per-kind branch.
    """
    from .backends import engine_for
    from .sources.image import train_labels
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

    # The one place a distributed engine genuinely earns its keep: planning folds
    # needs a row count and one label column, and neither requires the feature
    # matrix. Under `spark` this is a `count()` plus a one-column collect, so a
    # table far too large for the driver can still have its folds cut.
    #
    # `column` rather than a `to_pandas` slice because the labels stand alone here
    # — the row count is checked against the bundle separately, so there is no
    # second array that has to come out of the same materialization.
    engine = engine_for(config)
    table = engine.read_table(str(config.data.path))
    dtype = None if config.task == "regression" else "int64"
    return engine.n_rows(table), engine.column(table, config.data.target, dtype=dtype)


def build_datamodule(config: ExperimentConfig) -> Any:
    """The Lightning adapter for the configured data kind.

    The adapter is imported *here* rather than at module scope: it subclasses
    ``pl.LightningDataModule``, and ``build_bundle`` — which the orchestrator uses
    for every backend — lives in this same module. A module-scope import would put
    Lightning into the import path of a GBDT training run. Importing it also
    registers the v1 datamodules, which is what ``get_datamodule_class`` reads.
    """
    from ..core.registry import _DATAMODULE_REGISTRY, get_datamodule_class
    from . import lightning_adapter  # noqa: F401  (registers tabular/image datamodules)

    if config.data.kind not in _DATAMODULE_REGISTRY:
        # Only `tabular` and `image` ever got a v1 compat class, so this raised
        # `KeyError: Unknown datamodule 'text'` for every other kind -- `mlf lr` on
        # a text or time-series config has been broken since those sources landed.
        #
        # Fixed by falling through rather than by adding two more compat classes:
        # datamodules stopped being an extension point in P1, and every one of
        # these is now a config-bound bundle factory and nothing else. The v1
        # classes survive only because `data.kind` still selects between them.
        return lightning_adapter.BundleDataModule.from_bundle(build_bundle(config), config)
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
