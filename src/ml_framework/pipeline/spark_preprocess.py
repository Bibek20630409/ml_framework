"""
pipeline/spark_preprocess.py
────────────────────────────
Apache Spark data-preprocessing stage. Reads raw data, cleans + feature-engineers
it at scale, and writes a processed Parquet dataset that training consumes.

This is the "data pipeline" stage — orchestrated by Airflow, versioned by DVC. It
scales to data far larger than memory; on small demo data it still runs (Spark
local mode). Scaling-specific transforms (scaling/encoding fit on train only) are
deliberately left to the training DataModule to avoid train/serve skew.

Run standalone:
    spark-submit -m ml_framework.pipeline.spark_preprocess \\
        --input data/raw/dataset.csv --output data/processed --target-col label

Requires a JVM (Java 11/17) and ``pip install -e ".[mlops]"`` (pyspark).
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def preprocess(
    input_path: str,
    output_path: str,
    target_col: str,
    dropna: bool = True,
    deduplicate: bool = True,
) -> None:
    """Clean + feature-engineer raw data with Spark → Parquet.

    Steps (all distributed):
      1. read raw CSV/Parquet
      2. drop rows with a null target; optionally drop fully-null feature rows
      3. de-duplicate
      4. cast the target to a stable type
      5. write a single deterministic Parquet dataset
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("ml_framework-preprocess")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    try:
        reader = spark.read.option("header", True).option("inferSchema", True)
        df = (
            reader.parquet(input_path)
            if input_path.endswith((".parquet", ".pq"))
            else reader.csv(input_path)
        )
        log.info("spark: read %d rows from %s", df.count(), input_path)

        df = df.where(F.col(target_col).isNotNull())
        if dropna:
            feature_cols = [c for c in df.columns if c != target_col]
            df = df.dropna(how="all", subset=feature_cols)
        if deduplicate:
            df = df.dropDuplicates()

        # Example feature engineering hook: keep the schema stable and typed.
        # (Add domain transforms here — they run distributed across the cluster.)
        df = df.withColumn(target_col, F.col(target_col).cast("double"))

        # coalesce(1) → a single, deterministic output partition for downstream
        # training (drop this for very large data and read the parquet dir directly).
        df.coalesce(1).write.mode("overwrite").parquet(output_path)
        log.info("spark: wrote %d rows → %s", df.count(), output_path)
    finally:
        spark.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spark preprocessing stage")
    parser.add_argument("--input", required=True, help="Raw CSV/Parquet path")
    parser.add_argument("--output", required=True, help="Processed Parquet output dir")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--no-dropna", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    preprocess(
        args.input,
        args.output,
        args.target_col,
        dropna=not args.no_dropna,
        deduplicate=not args.no_dedup,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
