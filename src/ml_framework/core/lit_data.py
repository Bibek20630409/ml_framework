"""
core/lit_data.py
────────────────
LightningDataModules for tabular (CSV) and image (folder) data.

Both handle split, scaling/transforms, imbalance handling, and DataLoaders.
Derived values (``input_dim``, ``output_dim``, ``class_weights``, ``feature_cols``)
are stored on the instance after ``setup()`` — never written back onto config.

Key fixes vs. the original framework:
  * Regression + small dataset no longer crashes: plain ``KFold`` is used for
    regression, ``StratifiedKFold`` only for classification.
  * Imbalance handling is a single explicit choice (``smote`` | ``class_weights``
    | ``none``) computed on the pre-SMOTE distribution — no double correction.
  * Image label/output_dim extraction handles both ``ImageFolder`` and the
    ``random_split`` ``Subset`` case correctly.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..config import ExperimentConfig
from ..utils.seed import resolve_num_workers
from .registry import register_datamodule

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────


def read_table(path: str) -> pd.DataFrame:
    """Read a tabular dataset as CSV or Parquet.

    Supports a ``.parquet``/``.pq`` file, or a directory of parquet part-files
    (what Spark writes), or a ``.csv`` file. This is what lets the Spark
    preprocessing stage hand off to training transparently.
    """
    p = Path(path)
    if p.is_dir() or p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def detect_imbalance(y: np.ndarray, threshold: float = 0.3) -> bool:
    """True if the minority class is < ``threshold`` fraction of the majority."""
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return False
    mn, mx = min(counts.values()), max(counts.values())
    return (mn / mx) < threshold


def apply_smote(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE  # local import: optional dep

    before = dict(Counter(y.tolist()))
    x_res, y_res = SMOTE(random_state=seed).fit_resample(x, y)
    log.info("SMOTE: %s → %s", before, dict(Counter(y_res.tolist())))
    return x_res, y_res


def compute_class_weights(y: np.ndarray, task: str) -> torch.Tensor:
    """Balanced weights on the *pre-SMOTE* distribution.

    binary     → 1-element tensor ``[n_neg / n_pos]`` (BCE ``pos_weight``)
    multiclass → per-class balanced weights ``n / (n_classes * count_c)``
    """
    counts = np.bincount(y.astype("int64"))
    counts = np.clip(counts, 1, None)
    if task == "binary":
        n_neg, n_pos = counts[0], counts[1] if len(counts) > 1 else 1
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)
    n, nc = counts.sum(), len(counts)
    return torch.tensor(n / (nc * counts), dtype=torch.float32)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    task: str,
    val_size: float,
    test_size: float,
    holdout_threshold: int,
):
    """Train/val/test split. Auto-selects holdout vs. fold-derived split by size."""
    n = len(x)
    is_reg = task == "regression"
    stratify = None if is_reg else y

    if n >= holdout_threshold:
        log.info("n=%d >= %d → stratified holdout", n, holdout_threshold)
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        strat2 = None if is_reg else y_tmp
        rel_val = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_tmp, y_tmp, test_size=rel_val, random_state=seed, stratify=strat2
        )
    else:
        log.info("n=%d < %d → fold-derived split", n, holdout_threshold)
        # Regression cannot be stratified; use plain KFold to avoid a crash.
        splitter = (
            KFold(n_splits=5, shuffle=True, random_state=seed)
            if is_reg
            else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        )
        tr_idx, hold_idx = next(iter(splitter.split(x, y)))
        x_train, y_train = x[tr_idx], y[tr_idx]
        x_hold, y_hold = x[hold_idx], y[hold_idx]
        strat3 = None if is_reg else y_hold
        x_val, x_test, y_val, y_test = train_test_split(
            x_hold, y_hold, test_size=0.5, random_state=seed, stratify=strat3
        )

    log.info("split → train:%d val:%d test:%d", len(x_train), len(x_val), len(x_test))
    return x_train, x_val, x_test, y_train, y_val, y_test


# ── Base ──────────────────────────────────────────────────


class FrameworkDataModule(pl.LightningDataModule):
    """Base with the derived attributes the pipeline reads after ``setup()``.

    Declaring them here (typed) is what lets the training pipeline access
    ``dm.input_dim`` / ``dm.output_dim`` / ``dm.class_weights`` with type safety.
    """

    input_dim: int
    output_dim: int
    class_weights: torch.Tensor | None
    feature_cols: list[str]
    reference_stats: dict | None  # drift baseline (raw training feature distribution)

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.input_dim = 0
        self.output_dim = 0
        self.class_weights = None
        self.feature_cols = []
        self.reference_stats = None

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover
        raise NotImplementedError


# ── Tabular ───────────────────────────────────────────────


@register_datamodule("tabular")
class TabularDataModule(FrameworkDataModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.scaler_path = str(Path(config.output_dir) / "scaler.pkl")

    def prepare_data(self) -> None:  # noqa: D401 - Lightning hook
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config
        if cfg.data.csv_path is None:
            raise ValueError("tabular data requires data.csv_path")
        df = read_table(cfg.data.csv_path)
        if cfg.data.target_col not in df.columns:
            raise KeyError(f"target_col '{cfg.data.target_col}' not in CSV columns")

        self.feature_cols = [c for c in df.columns if c != cfg.data.target_col]
        x = df[self.feature_cols].values.astype("float32")
        y = df[cfg.data.target_col].values
        y = y.astype("int64") if cfg.task != "regression" else y.astype("float32")

        x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
            x,
            y,
            seed=cfg.seed,
            task=cfg.task,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            holdout_threshold=cfg.data.holdout_threshold,
        )

        # drift baseline: capture the RAW (pre-scale) train feature distribution,
        # since serving computes drift on the raw features clients send.
        from ..monitoring.drift import build_reference

        self.reference_stats = build_reference(x_train, self.feature_cols)

        # scale — fit on train only
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, self.scaler_path)
        log.info("scaler saved → %s", self.scaler_path)

        # imbalance — a single explicit strategy, computed on pre-SMOTE data
        if cfg.task in ("binary", "multiclass"):
            strategy = cfg.data.imbalance_strategy
            imbalanced = detect_imbalance(y_train, cfg.data.imbalance_threshold)
            if strategy == "class_weights" and imbalanced:
                self.class_weights = compute_class_weights(y_train, cfg.task)
                log.info("class weights: %s", self.class_weights.tolist())
            elif strategy == "smote" and imbalanced:
                x_train, y_train = apply_smote(x_train, y_train, cfg.seed)
            else:
                log.info("imbalance strategy=%s applied=%s", strategy, False)

        self.input_dim = x_train.shape[1]
        # binary uses a single-logit head (BCEWithLogitsLoss); regression a single
        # output; multiclass one logit per class.
        if cfg.task == "multiclass":
            self.output_dim = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
        else:
            self.output_dim = 1

        def make(a: np.ndarray, b: np.ndarray) -> TensorDataset:
            return TensorDataset(torch.tensor(a, dtype=torch.float32), torch.tensor(b))

        self._train = make(x_train, y_train)
        self._val = make(x_val, y_val)
        self._test = make(x_test, y_test)

    @property
    def _workers(self) -> int:
        return resolve_num_workers(self.config.train.num_workers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            drop_last=len(self._train) > self.config.train.batch_size,
            num_workers=self._workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self.config.train.batch_size, num_workers=self._workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test, batch_size=self.config.train.batch_size, num_workers=self._workers
        )


# ── Image ─────────────────────────────────────────────────


@register_datamodule("image")
class ImageDataModule(FrameworkDataModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # imbalance handled via WeightedRandomSampler, so no loss class weights
        self._sampler: WeightedRandomSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        from torchvision import datasets, transforms

        cfg = self.config
        size = cfg.data.img_size
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                norm,
            ]
        )
        eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])

        full_train = datasets.ImageFolder(cfg.data.train_dir, transform=train_tf)
        self._test = datasets.ImageFolder(cfg.data.test_dir, transform=eval_tf)
        classes = full_train.classes

        if cfg.data.val_dir:
            self._train = full_train
            self._val = datasets.ImageFolder(cfg.data.val_dir, transform=eval_tf)
            labels = list(full_train.targets)
        else:
            n = len(full_train)
            n_val = max(1, int(cfg.data.val_size * n))
            gen = torch.Generator().manual_seed(cfg.seed)
            self._train, self._val = torch.utils.data.random_split(
                full_train, [n - n_val, n_val], generator=gen
            )
            # Subset → recover labels via parent .targets and .indices
            labels = [full_train.targets[i] for i in self._train.indices]

        counts = Counter(labels)
        total = sum(counts.values())
        w_map = {c: total / cnt for c, cnt in counts.items()}
        self._sampler = WeightedRandomSampler([w_map[c] for c in labels], len(labels))

        self.input_dim = 3 * size * size
        # binary → single-logit head; multiclass → one logit per class.
        self.output_dim = 1 if cfg.task == "binary" else len(classes)
        log.info("image classes=%d train=%d", len(classes), len(labels))

    @property
    def _workers(self) -> int:
        return resolve_num_workers(self.config.train.num_workers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.config.train.batch_size,
            sampler=self._sampler,
            num_workers=self._workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self.config.train.batch_size, num_workers=self._workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test, batch_size=self.config.train.batch_size, num_workers=self._workers
        )
