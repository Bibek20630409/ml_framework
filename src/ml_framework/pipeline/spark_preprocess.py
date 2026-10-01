"""
pipeline/spark_preprocess.py
────────────────────────────
The data-preprocessing stage. Reads raw data, cleans it, and writes a processed
Parquet dataset that training consumes.

This is the "data pipeline" stage — orchestrated by Airflow, versioned by DVC. It
scales to data far larger than memory; on small demo data it still runs. Fitted
transforms (scaling/encoding, fit on train only) are deliberately left to the
training source to avoid train/serve skew.

**It no longer speaks pyspark.** Every step goes through the selected
:class:`~ml_framework.core.protocols.DataBackend`, which is what collapsed the
two independent Spark codebases this repo used to carry — this stage and
``data/backends/spark.py`` — into one, with a single session configuration. The
side effect worth having: the same cleaning now runs under ``local`` on a laptop
with no JVM, which was impossible while this module imported pyspark directly.

The module keeps its name because DVC, the Airflow DAG and CI all reference the
path; ``--data-backend`` is what changes the engine.

Run standalone:
    python -m ml_framework.pipeline.spark_preprocess \\
        --input data/raw/dataset.csv --output data/processed --target-col label

Orchestrated (DVC/Airflow), where the target and output come from the training
config so no caller has to repeat them:
    python -m ml_framework.pipeline.spark_preprocess \\
        --input data/raw/dataset.csv --config configs/dvc_tabular.yaml

The default engine is ``spark``, which needs a JVM (Java 11/17) and
``pip install -e ".[mlops]"``. Add ``--data-backend local`` for neither.
"""

from __future__ import annotations

import argparse
import logging

from .stage_config import resolve

log = logging.getLogger(__name__)


def preprocess(
    input_path: str,
    output_path: str,
    target_col: str,
    dropna: bool = True,
    deduplicate: bool = True,
    backend: str = "spark",
) -> None:
    """Clean raw data → a Parquet dataset training can read.

    Steps, in order:
      1. read raw CSV/Parquet
      2. drop rows with a null target; optionally drop fully-null feature rows
      3. de-duplicate
      4. cast the target to a stable type
      5. write a deterministic Parquet dataset

    Every step runs through the selected :class:`DataBackend`, so ``backend`` is
    the whole difference between this being a cluster job and a laptop one. The
    default stays ``spark``: this is the distributed stage in the DVC/Airflow
    pipeline and changing what it runs on by default would change what that
    pipeline does. ``--data-backend local`` makes the *same* cleaning runnable
    without a JVM, which was impossible while this module spoke pyspark directly.
    """
    from ..core.registry import get_data_backend

    engine = get_data_backend(backend)

    table = engine.read_table(input_path)
    log.info("%s: read %d rows from %s", backend, engine.n_rows(table), input_path)

    table = engine.filter_notnull(table, target_col)
    if dropna:
        feature_cols = [c for c in engine.columns(table) if c != target_col]
        table = engine.drop_all_null_rows(table, feature_cols)
    if deduplicate:
        table = engine.drop_duplicates(table)

    # Example feature engineering hook: keep the schema stable and typed.
    # (Add domain transforms here — under `spark` they run across the cluster.)
    table = engine.cast(table, target_col, "float64")

    engine.write_parquet(table, output_path)
    log.info("%s: wrote %d rows → %s", backend, engine.n_rows(table), output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data preprocessing stage")
    parser.add_argument("--input", required=True, help="Raw CSV/Parquet path")
    parser.add_argument("--output", help="Processed Parquet output dir (default: data.path)")
    parser.add_argument("--target-col", help="Target column (default: data.target)")
    # The orchestrated path passes this instead of --output/--target-col, so the
    # train config is the one place those two values are written down. Explicit
    # flags still win, per the precedence the `mlf` CLI documents.
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Training config to read data.target / data.path from",
    )
    parser.add_argument("--no-dropna", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    # Additive and defaulted, so the DVC stage and the Airflow BashOperator --
    # which both invoke `python -m ml_framework.pipeline.spark_preprocess
    # --input X --output Y --target-col Z` -- keep working unchanged.
    parser.add_argument(
        "--data-backend",
        default="spark",
        metavar="NAME",
        help="Engine to clean with: spark (default), local or polars",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # A missing flag is a usage error, not a crash: argparse prints the usage
    # line and exits 2, which is what a DVC/Airflow log is readable with.
    try:
        target_col, output = resolve(args.config, target_col=args.target_col, data_path=args.output)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if not output:
        parser.error("pass --output, or a --config that sets data.path")

    preprocess(
        args.input,
        output,
        target_col,
        dropna=not args.no_dropna,
        deduplicate=not args.no_dedup,
        backend=args.data_backend,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
