import pytest
from pydantic import ValidationError

from ml_framework.config import ExperimentConfig


@pytest.mark.unit
def test_tabular_requires_csv_and_target():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"task": "binary", "data": {"kind": "tabular"}})


@pytest.mark.unit
def test_image_requires_dirs():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"task": "multiclass", "data": {"kind": "image"}})


@pytest.mark.unit
def test_invalid_task_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "task": "clustering",
                "data": {"kind": "tabular", "csv_path": "d.csv", "target_col": "y"},
            }
        )


@pytest.mark.unit
def test_config_is_frozen(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        cfg.seed = 7  # frozen


@pytest.mark.unit
def test_with_overrides_returns_new_copy(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    new = cfg.with_overrides({"optim.lr": 0.05, "train.epochs": 9})
    assert new.optim.lr == 0.05
    assert new.train.epochs == 9
    assert cfg.optim.lr != 0.05  # original untouched


@pytest.mark.unit
def test_with_overrides_unknown_key_raises(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    with pytest.raises(KeyError):
        cfg.with_overrides({"optim.nonexistent": 1})


@pytest.mark.unit
def test_val_test_size_sum_validated():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "task": "binary",
                "data": {
                    "kind": "tabular",
                    "csv_path": "d.csv",
                    "target_col": "y",
                    "val_size": 0.6,
                    "test_size": 0.5,
                },
            }
        )
