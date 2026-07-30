"""
monitoring/model_quality.py
───────────────────────────
Delayed-label model-quality monitoring. In production, ground-truth labels arrive
*after* predictions (label lag). This batch job joins logged predictions with the
labels once they're available and reports live accuracy/MAE — logged to MLflow so
you can watch quality decay over time and trigger retraining.

Prediction and label logs are CSVs with an ``id`` join key:
  predictions.csv : id, prediction
  labels.csv      : id, label

    python -m ml_framework.monitoring.model_quality \\
        --predictions preds.csv --labels labels.csv --task multiclass \\
        --tracking-uri http://mlflow:5000
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def evaluate_against_labels(
    predictions_csv: str,
    labels_csv: str,
    task: str,
) -> dict:
    """Join predictions↔labels on ``id`` and compute live quality metrics."""
    import numpy as np
    import pandas as pd

    preds = pd.read_csv(predictions_csv)
    labels = pd.read_csv(labels_csv)
    merged = preds.merge(labels, on="id", how="inner")
    if merged.empty:
        raise ValueError("no overlapping ids between predictions and labels")

    y_pred = merged["prediction"].to_numpy()
    y_true = merged["label"].to_numpy()
    n = int(len(merged))

    if task == "regression":
        mae = float(np.abs(y_pred - y_true).mean())
        rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
        return {"n": n, "live_mae": mae, "live_rmse": rmse}
    acc = float((y_pred == y_true).mean())
    return {"n": n, "live_accuracy": acc}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Delayed-label model-quality monitor")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment", default="ml-framework-monitoring")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    metrics = evaluate_against_labels(args.predictions, args.labels, args.task)
    log.info("live quality: %s", metrics)

    try:
        import mlflow

        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name="model-quality"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n"})
            mlflow.log_metric("n_labeled", metrics["n"])
    except Exception as exc:  # pragma: no cover - mlflow optional/offline
        log.warning("mlflow logging skipped: %s", exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
