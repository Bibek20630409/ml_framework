"""
core/evaluate.py
────────────────
Held-out test-set evaluation. Writes report.txt, predictions.csv, and (for
classification) confusion_matrix.txt. Returns a metrics dict.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report, confusion_matrix

from ..config import ExperimentConfig

log = logging.getLogger(__name__)


def evaluate(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    config: ExperimentConfig,
    output_dir: str | None = None,
) -> dict:
    out = Path(output_dir or config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_preds: list = []
    all_labels: list = []
    all_probs: list = []
    with torch.no_grad():
        for x, y in datamodule.test_dataloader():
            out_logits = model(x.to(device))
            if config.task == "binary":
                probs = torch.sigmoid(out_logits.squeeze(1))
                preds = (probs > 0.5).long()
            elif config.task == "multiclass":
                probs = torch.softmax(out_logits, dim=1)
                preds = out_logits.argmax(dim=1)
            else:  # regression
                preds = out_logits.squeeze(1)
                probs = preds
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    if config.task in ("binary", "multiclass"):
        report = classification_report(
            labels_arr, preds_arr, target_names=config.data.class_names, digits=4, zero_division=0
        )
        cm = confusion_matrix(labels_arr, preds_arr)
        acc = float(np.mean(preds_arr == labels_arr))
        log.info("Test accuracy: %.4f\n%s", acc, report)
        (out / "report.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")
        (out / "confusion_matrix.txt").write_text(str(cm), encoding="utf-8")
        metrics = {"test_acc": acc}
    else:
        mae = float(np.abs(preds_arr - labels_arr).mean())
        rmse = float(np.sqrt(((preds_arr - labels_arr) ** 2).mean()))
        log.info("Test MAE: %.4f | RMSE: %.4f", mae, rmse)
        (out / "report.txt").write_text(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", encoding="utf-8")
        metrics = {"test_mae": mae, "test_rmse": rmse}

    df_out = pd.DataFrame({"label": labels_arr, "prediction": preds_arr})
    if config.task == "multiclass" and probs_arr.ndim == 2:
        for i in range(probs_arr.shape[1]):
            df_out[f"prob_class_{i}"] = probs_arr[:, i]
    elif config.task == "binary":
        df_out["probability"] = probs_arr
    df_out.to_csv(out / "predictions.csv", index=False)
    log.info("predictions saved → %s", out / "predictions.csv")

    return metrics
