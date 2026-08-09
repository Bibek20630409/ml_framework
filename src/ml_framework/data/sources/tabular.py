"""
data/sources/tabular.py
───────────────────────
CSV/Parquet → :class:`~ml_framework.data.types.DataBundle`.

This is ``TabularDataModule.setup`` with the DataLoader construction removed and
the order of operations preserved exactly:

    read → target check → split → drift reference (RAW features)
         → scale (fit on train only) → imbalance → dims

The order is not incidental. The drift reference is captured *before* scaling
because serving computes drift on the raw features clients send. SMOTE runs
*after* scaling because synthesizing neighbours in unscaled space lets a
high-variance column dominate the distance metric. Both were deliberate in v1.

No Lightning here, and no DataLoader: what comes out is arrays plus a schema.

:class:`TabularSourceParams` validates ``data.params`` — frozen and
``extra="forbid"``, so ``imbalance_strategy: smoate`` is an error rather than a
silently ignored key. It is run here rather than in the config validator because
resolving a source's schema from there would mean importing the whole data layer
to validate a YAML file (see ``config/schema.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.types import Capabilities, FrameworkError
from ..backends import engine_for
from ..preprocess.tabular import (
    TabularPreprocessor,
    balanced_sample_weights,
    resolve_imbalance,
)
from ..splitters import GroupSplitter, RandomSplitter, SplitError, TemporalSplitter
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)

# `PARQUET_REQUIREMENT` used to live here. It moved to `data/backends/local.py`
# with the pandas read it guards, and is not re-exported: nothing imported it from
# this module, and a compatibility alias for a name with no importers is a shim
# that only ever costs.


class TabularSourceParams(PydanticModel):
    """``data.params`` for the tabular source."""

    model_config = {"frozen": True, "extra": "forbid"}

    # `auto` resolves against the selected model's capabilities: sample weights
    # where the model consumes them natively (trees), SMOTE otherwise. The default
    # stays `smote` so an existing tabular config trains exactly as it did.
    imbalance_strategy: Literal["auto", "smote", "class_weights", "none"] = "smote"
    imbalance_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    # n >= this → plain holdout; below it a 5-fold cut yields a larger, more
    # stable training set than carving 30% off the top.
    holdout_threshold: int = Field(default=5000, gt=0)


def read_table(path: str, *, backend: str = "local") -> pd.DataFrame:
    """Read a tabular dataset as CSV or Parquet.

    Supports a ``.parquet``/``.pq`` file, or a directory of parquet part-files
    (what Spark writes), or a ``.csv`` file. This is what lets the Spark
    preprocessing stage hand off to training transparently.

    ``backend`` chooses *who does the reading* — under ``spark`` the read is
    distributed and this function is the collect point. The return type is
    **pandas under every backend**: this is public API, re-exported from ``core``
    and consumed by ``pipeline/contracts.py``, which hands the frame to pandera.
    Code that wants to stay lazy should go through
    :func:`~ml_framework.core.registry.get_data_backend` directly.
    """
    from ...core.registry import get_data_backend

    engine = get_data_backend(backend)
    return engine.to_pandas(engine.read_table(path))


def build_splitter(config, params: TabularSourceParams) -> Any:
    """The splitter ``data.split`` asks for, with ``auto`` already resolved.

    Existing only for ``random`` would make ``strategy`` decoration: the temporal
    and group splitters were written in P1 with no config path to reach them, and
    this is that path. Their *guard* — refusing an explicitly shuffled split on
    time-series data — belongs with the forecasting work that gives it something
    to guard.
    """
    split = config.data.split
    strategy = split.resolved_strategy(config.data.kind)
    log.info("split strategy=%s (declared: %s)", strategy, split.strategy)

    if strategy == "random":
        return RandomSplitter(
            seed=config.runtime.seed,
            task=config.task,
            val_size=split.val_size,
            test_size=split.test_size,
            holdout_threshold=params.holdout_threshold,
        )
    if strategy == "temporal":
        return TemporalSplitter(val_size=split.val_size, test_size=split.test_size, gap=split.gap)
    if strategy == "group":
        return GroupSplitter(
            seed=config.runtime.seed, val_size=split.val_size, test_size=split.test_size
        )
    raise SplitError(f"Unknown split strategy '{strategy}'")


def model_capabilities(config) -> Capabilities:
    """The capability flags of the *selected model*, defaulting conservatively.

    The source asking the model what it needs looks like a layering inversion and
    is not: scaling and imbalance correction are not properties of a CSV, they are
    properties of what will consume it. Standardizing for a tree buys nothing and
    destroys the interpretability of its split thresholds; SMOTE for a tree is
    nonsense where native sample weights exist. Those decisions have to be made
    somewhere, and the alternative — deciding in the backend, after the arrays are
    already built — would mean fitting the scaler twice or throwing one away.

    Falls back to neural defaults when the model is unknown, which keeps this
    usable in tests that build a bundle without a registered plugin.
    """
    from ...core.plugins import UnknownPluginError
    from ...core.registry import MODELS

    try:
        return MODELS.get_spec(config.model.name).capabilities
    except UnknownPluginError:
        log.debug("model '%s' is not registered; assuming default capabilities", config.model.name)
        return Capabilities()


def _apply_imbalance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    config,
    params: TabularSourceParams,
    caps: Capabilities,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Resolve the imbalance strategy. Returns ``(x, y, class_weights, sample_weights)``.

    Exactly one of the three corrections is ever applied — resampling, per-class
    loss weights, or per-row sample weights. Applying two would correct twice.
    """
    strategy: str = params.imbalance_strategy
    if strategy == "auto":
        strategy = "sample_weights" if caps.supports_sample_weight else "smote"
        log.info("imbalance strategy auto → %s", strategy)
    elif strategy == "smote" and caps.supports_sample_weight:
        # Honoured, because it was asked for explicitly — but flagged, because
        # interpolating synthetic neighbours for an axis-aligned splitter is a
        # known-poor choice when the model takes weights natively.
        log.warning(
            "model '%s' supports sample weights; 'smote' is a poor fit for it. "
            "Use data.params.imbalance_strategy: auto",
            config.model.name,
        )

    if strategy == "sample_weights":
        return x, y, None, balanced_sample_weights(y, config.task, params.imbalance_threshold)

    x_res, y_res, class_weights = resolve_imbalance(
        x,
        y,
        task=config.task,
        strategy=strategy,
        threshold=params.imbalance_threshold,
        seed=config.runtime.seed,
    )
    return x_res, y_res, class_weights, None


def build_tabular_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize a tabular :class:`DataBundle` from a validated config.

    ``indices`` overrides the configured splitter with a ready-made partition,
    which is how cross-validation gets one bundle per fold. Everything after the
    split — the drift reference, the scaler, the imbalance correction — is then
    recomputed *for that fold*, which is the entire point: a preprocessor fitted
    once on the full data and reused across folds leaks the test set into every
    one of them.
    """
    if config.data.path is None:
        raise ValueError("tabular data requires data.path")
    params = TabularSourceParams.model_validate(dict(config.data.params))

    engine = engine_for(config)
    table = engine.read_table(config.data.path)

    # Every check below runs against the **schema**, before a byte is collected.
    # Under `spark` that is the difference between "data.target 'labl' not in the
    # table's columns" arriving in a second and arriving after a full
    # materialization of a table that was never going to work.
    available = engine.columns(table)
    target = config.data.target
    if target not in available:
        raise KeyError(f"data.target '{target}' not in the table's columns")

    split_cfg = config.data.split
    # The ordering/grouping columns are inputs to the *split*, not features.
    # Leaving them in would let the model read the timestamp it is supposed to be
    # generalizing across.
    reserved = {target, split_cfg.time_col, split_cfg.group_col} - {None}
    for name in (split_cfg.time_col, split_cfg.group_col):
        if name is not None and name not in available:
            raise KeyError(f"split column '{name}' not in the table's columns")

    feature_cols = [c for c in available if c not in reserved]
    if not feature_cols:
        raise FrameworkError(f"'{config.data.path}' has no feature columns besides '{target}'")

    # Dtypes off the schema, normalized by the engine, so `FeatureSchema.dtypes` —
    # which reaches the bundle manifest and the serving signature — cannot record
    # which engine happened to build the bundle.
    dtypes = engine.dtypes(table, feature_cols)

    # **The collect point**, and deliberately a single one. x, y, the time column
    # and the group column must all index the same rows; separate collects would
    # be separate executions of the plan, and a distributed engine does not
    # promise two executions agree on row order. Pairing feature rows with the
    # wrong labels is the worst failure this module could have, and it would be
    # silent — so everything downstream slices one materialized frame.
    #
    # No projection: features + target + time + group *is* every column here, so
    # there is nothing for `select` to push down.
    df = engine.to_pandas(table)

    x = df[feature_cols].values.astype("float32")
    y = df[target].values
    y = y.astype("int64") if config.task != "regression" else y.astype("float32")

    parts = indices or build_splitter(config, params).split(
        len(x),
        y=y,
        time=df[split_cfg.time_col].values if split_cfg.time_col else None,
        groups=df[split_cfg.group_col].values if split_cfg.group_col else None,
    )

    x_train, y_train = x[parts.train], y[parts.train]
    x_val, y_val = x[parts.val], y[parts.val]
    x_test, y_test = x[parts.test], y[parts.test]

    # Drift baseline: the RAW (pre-scale) train feature distribution, since serving
    # computes drift on the raw features clients send.
    from ...monitoring.drift import build_reference

    reference_stats = build_reference(x_train, feature_cols)

    caps = model_capabilities(config)
    preprocessor = TabularPreprocessor(needs_scaling=caps.needs_scaling)
    x_train = preprocessor.fit_transform(Split(x=x_train, y=y_train))
    x_val = preprocessor.transform(x_val)
    x_test = preprocessor.transform(x_test)

    x_train, y_train, weights, sample_weights = _apply_imbalance(
        x_train, y_train, config=config, params=params, caps=caps
    )

    # Head convention, unchanged from v1: binary is a single logit
    # (BCEWithLogitsLoss), regression a single output, multiclass one per class.
    input_dim = int(x_train.shape[1])
    if config.task == "multiclass":
        output_dim = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    else:
        output_dim = 1

    schema = FeatureSchema(
        feature_names=tuple(feature_cols),
        dtypes=dict(dtypes),
        target_name=target,
        class_names=tuple(config.data.class_names) if config.data.class_names else None,
        time_col=split_cfg.time_col,
    )
    log.info("input_dim=%d output_dim=%d", input_dim, output_dim)

    return DataBundle(
        train=Split(payload="arrays", x=x_train, y=y_train, index=None),
        # Only the resampled train split loses its row identity; val/test keep
        # theirs so predictions stay traceable to input rows.
        val=Split(payload="arrays", x=x_val, y=y_val, index=parts.val),
        test=Split(payload="arrays", x=x_test, y=y_test, index=parts.test),
        schema=schema,
        task=config.task,
        data_kind="tabular",
        input_dim=input_dim,
        output_dim=output_dim,
        class_weights=weights,
        preprocessor=preprocessor,
        reference_stats=reference_stats,
        # Per-row weights when the model consumes them natively. `meta` rather than
        # a field because it is source-specific plumbing the DataBundle contract
        # should not name — the same slot the image source uses for its sampler
        # weights.
        meta={} if sample_weights is None else {"sample_weights": sample_weights},
    )
