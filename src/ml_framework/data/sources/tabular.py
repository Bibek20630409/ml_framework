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
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...core.plugins import check_requirements
from ...core.types import FrameworkError, Requirement
from ..preprocess.tabular import TabularPreprocessor, resolve_imbalance
from ..splitters import RandomSplitter
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)


# pandas needs an engine to read parquet, and it is not a pandas dependency.
# Declaring it here rather than leaning on `mlflow` (which happens to require
# pyarrow) keeps the coupling visible: the parquet path must fail with a pip
# command, not with a pandas ImportError, in exactly the environment where the
# dependency is most likely absent — a serving image built without the mlops extra.
PARQUET_REQUIREMENT = Requirement("pyarrow", extra="parquet", min_version="10.0.1")


def read_table(path: str) -> pd.DataFrame:
    """Read a tabular dataset as CSV or Parquet.

    Supports a ``.parquet``/``.pq`` file, or a directory of parquet part-files
    (what Spark writes), or a ``.csv`` file. This is what lets the Spark
    preprocessing stage hand off to training transparently.
    """
    p = Path(path)
    if p.is_dir() or p.suffix.lower() in (".parquet", ".pq"):
        check_requirements((PARQUET_REQUIREMENT,), what=f"reading parquet from '{path}'")
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_tabular_bundle(config) -> DataBundle:
    """Materialize a tabular :class:`DataBundle` from a validated config."""
    if config.data.csv_path is None:
        raise ValueError("tabular data requires data.csv_path")
    df = read_table(config.data.csv_path)
    target = config.data.target_col
    if target not in df.columns:
        raise KeyError(f"target_col '{target}' not in CSV columns")

    feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        raise FrameworkError(f"'{config.data.csv_path}' has no feature columns besides '{target}'")
    x = df[feature_cols].values.astype("float32")
    y = df[target].values
    y = y.astype("int64") if config.task != "regression" else y.astype("float32")

    parts = RandomSplitter(
        seed=config.seed,
        task=config.task,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        holdout_threshold=config.data.holdout_threshold,
    ).split(len(x), y=y)

    x_train, y_train = x[parts.train], y[parts.train]
    x_val, y_val = x[parts.val], y[parts.val]
    x_test, y_test = x[parts.test], y[parts.test]

    # Drift baseline: the RAW (pre-scale) train feature distribution, since serving
    # computes drift on the raw features clients send.
    from ...monitoring.drift import build_reference

    reference_stats = build_reference(x_train, feature_cols)

    preprocessor = TabularPreprocessor(needs_scaling=True)
    x_train = preprocessor.fit_transform(Split(x=x_train, y=y_train))
    x_val = preprocessor.transform(x_val)
    x_test = preprocessor.transform(x_test)

    x_train, y_train, weights = resolve_imbalance(
        x_train,
        y_train,
        task=config.task,
        strategy=config.data.imbalance_strategy,
        threshold=config.data.imbalance_threshold,
        seed=config.seed,
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
        dtypes={c: str(df[c].dtype) for c in feature_cols},
        target_name=target,
        class_names=tuple(config.data.class_names) if config.data.class_names else None,
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
    )
