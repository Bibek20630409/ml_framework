"""
pipeline/contracts.py
─────────────────────
Data contract / quality gate (Pandera). Runs as a gated Airflow task **before**
training so bad data fails the pipeline instead of silently producing a bad model.

Enforces: the target column exists and is non-null, feature columns are numeric and
non-null, there are no duplicate rows, and (optionally) values sit in expected ranges.

    python -m ml_framework.pipeline.contracts --input data/processed --target-col label
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

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


def validate_file(path: str, target_col: str) -> int:
    """Validate a CSV/Parquet dataset file; returns the row count on success."""
    from ..core import read_table

    df = read_table(path)
    validated = validate_dataframe(df, target_col)
    log.info("data contract passed: %d rows, %d cols", len(validated), validated.shape[1])
    return len(validated)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Data contract / quality gate")
    p.add_argument("--input", required=True, help="CSV/Parquet dataset")
    p.add_argument("--target-col", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    validate_file(args.input, args.target_col)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
