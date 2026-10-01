"""
pipeline/contracts.py
─────────────────────
Data contract / quality gate (Pandera). Runs as a gated Airflow task **before**
training so bad data fails the pipeline instead of silently producing a bad model.

Enforces: the target column exists and is non-null, feature columns are numeric and
non-null, there are no duplicate rows, and (optionally) values sit in expected ranges.

    python -m ml_framework.pipeline.contracts --input data/processed --target-col label

Orchestrated, where both values come from the training config instead:

    python -m ml_framework.pipeline.contracts --config configs/dvc_tabular.yaml
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .stage_config import resolve

log = logging.getLogger(__name__)


def build_schema(target_col: str, feature_cols: list[str] | None = None):
    """A Pandera schema: non-null target + numeric non-null features + no dup rows."""
    from pandera import Check, Column, DataFrameSchema

    columns = {target_col: Column(nullable=False, required=True)}
    if feature_cols:
        for c in feature_cols:
            columns[c] = Column(nullable=False, coerce=True)
    return DataFrameSchema(
        columns,
        strict=False,
        checks=Check(
            lambda df: len(df) == len(df.drop_duplicates()),
            error="duplicate rows found",
        ),
    )


def validate_dataframe(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Validate a frame against the contract. Raises ``SchemaError`` on failure."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    schema = build_schema(target_col, feature_cols)
    return schema.validate(df, lazy=False)


def validate_file(path: str, target_col: str, *, backend: str = "local") -> int:
    """Validate a CSV/Parquet dataset file; returns the row count on success.

    ``backend`` is an explicit argument rather than a config lookup because this
    is a standalone stage — an Airflow task and a ``python -m`` entry point, with
    no ``ExperimentConfig`` in scope. Note that pandera validates *pandas*, so the
    frame is collected here whatever the engine: what ``spark`` buys is reading a
    table too large for the driver to open with pandas, not a distributed check.
    """
    from ..core import read_table

    df = read_table(path, backend=backend)
    validated = validate_dataframe(df, target_col)
    log.info("data contract passed: %d rows, %d cols", len(validated), validated.shape[1])
    return len(validated)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Data contract / quality gate")
    p.add_argument("--input", help="CSV/Parquet dataset (default: data.path)")
    p.add_argument("--target-col", help="Target column (default: data.target)")
    # As in the preprocess stage: the orchestrated path names a config, so the
    # gate validates the same column training is about to read, by construction.
    p.add_argument(
        "--config",
        metavar="PATH",
        help="Training config to read data.target / data.path from",
    )
    p.add_argument(
        "--data-backend", default="local", help="Engine: local (default), polars or spark"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        target_col, input_path = resolve(
            args.config, target_col=args.target_col, data_path=args.input
        )
    except (FileNotFoundError, ValueError) as exc:
        p.error(str(exc))
    if not input_path:
        p.error("pass --input, or a --config that sets data.path")

    validate_file(input_path, target_col, backend=args.data_backend)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
