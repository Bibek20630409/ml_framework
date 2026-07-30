import numpy as np
import pytest
import torch

from ml_framework.core.lit_data import (
    compute_class_weights,
    detect_imbalance,
    split_dataset,
)


@pytest.mark.unit
def test_detect_imbalance_true_when_skewed():
    y = np.array([0] * 90 + [1] * 10)
    assert detect_imbalance(y, threshold=0.3) is True


@pytest.mark.unit
def test_detect_imbalance_false_when_balanced():
    y = np.array([0] * 50 + [1] * 50)
    assert detect_imbalance(y, threshold=0.3) is False


@pytest.mark.unit
def test_binary_class_weight_is_scalar_pos_weight():
    y = np.array([0] * 80 + [1] * 20)  # n_neg/n_pos = 4.0
    w = compute_class_weights(y, "binary")
    assert w.numel() == 1
    assert torch.isclose(w[0], torch.tensor(4.0))


@pytest.mark.unit
def test_multiclass_class_weight_vector_length():
    y = np.array([0] * 10 + [1] * 20 + [2] * 70)
    w = compute_class_weights(y, "multiclass")
    assert w.numel() == 3


@pytest.mark.unit
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_split_small_dataset_all_tasks(task):
    # n < holdout_threshold forces the KFold-derived branch.
    n = 100
    x = np.random.default_rng(0).normal(size=(n, 4)).astype("float32")
    if task == "regression":
        y = np.random.default_rng(1).normal(size=n).astype("float32")
    else:
        n_classes = 2 if task == "binary" else 3
        y = np.array([i % n_classes for i in range(n)])
    parts = split_dataset(
        x, y, seed=0, task=task, val_size=0.15, test_size=0.15, holdout_threshold=5000
    )
    x_train, x_val, x_test = parts[0], parts[1], parts[2]
    assert len(x_train) > 0 and len(x_val) > 0 and len(x_test) > 0
    assert len(x_train) + len(x_val) + len(x_test) == n


@pytest.mark.unit
def test_split_large_dataset_holdout():
    n = 6000
    x = np.random.default_rng(0).normal(size=(n, 3)).astype("float32")
    y = np.array([i % 2 for i in range(n)])
    parts = split_dataset(
        x, y, seed=0, task="binary", val_size=0.15, test_size=0.15, holdout_threshold=5000
    )
    x_val, x_test = parts[1], parts[2]
    assert abs(len(x_test) / n - 0.15) < 0.02
    assert abs(len(x_val) / n - 0.15) < 0.02
