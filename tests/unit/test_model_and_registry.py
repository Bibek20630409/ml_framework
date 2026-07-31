import pytest
import torch

from ml_framework.config import ExperimentConfig
from ml_framework.core import available_models
from ml_framework.data import build_datamodule, build_model


def _cfg(csv, task, tmp_path):
    return ExperimentConfig.model_validate(
        {
            "task": task,
            "runtime": {"output_dir": str(tmp_path / "out"), "num_workers": 0},
            "data": {"kind": "tabular", "path": str(csv), "target": "label"},
            "model": {"name": "mlp", "params": {"hidden_dims": [16, 8]}},
            "fit": {"batch_size": 16},
            "logging": {"backend": "none"},
        }
    )


@pytest.mark.unit
def test_mlp_registered():
    assert "mlp" in available_models()


@pytest.mark.unit
def test_binary_criterion_pos_weight_is_scalar(binary_csv, tmp_path):
    cfg = _cfg(binary_csv, "binary", tmp_path)
    dm = build_datamodule(cfg)
    dm.setup()
    weights = torch.tensor([3.0])  # simulate 1-element pos_weight
    model = build_model(
        cfg, input_dim=dm.input_dim, output_dim=dm.output_dim, class_weights=weights
    )
    pos_weight = model.criterion.pos_weight
    assert pos_weight is not None
    assert pos_weight.numel() == 1  # the core bug fix


@pytest.mark.unit
def test_multiclass_criterion_weight_vector(tabular_csv, tmp_path):
    cfg = _cfg(tabular_csv, "multiclass", tmp_path)
    dm = build_datamodule(cfg)
    dm.setup()
    w = torch.ones(dm.output_dim)
    model = build_model(cfg, input_dim=dm.input_dim, output_dim=dm.output_dim, class_weights=w)
    assert model.criterion.weight.numel() == dm.output_dim


@pytest.mark.unit
def test_datamodule_sets_dims(tabular_csv, tmp_path):
    cfg = _cfg(tabular_csv, "multiclass", tmp_path)
    dm = build_datamodule(cfg)
    dm.setup()
    assert dm.input_dim == 6
    assert dm.output_dim == 3
    assert dm.feature_cols == [f"f{i}" for i in range(6)]


@pytest.mark.unit
def test_forward_shape(tabular_csv, tmp_path):
    cfg = _cfg(tabular_csv, "multiclass", tmp_path)
    dm = build_datamodule(cfg)
    dm.setup()
    model = build_model(cfg, input_dim=dm.input_dim, output_dim=dm.output_dim)
    out = model(torch.randn(4, dm.input_dim))
    assert out.shape == (4, dm.output_dim)


@pytest.mark.unit
def test_the_model_takes_its_params_not_the_whole_config(tabular_csv, tmp_path):
    """v1's BaseModel read `config.model.hidden_dims` and `config.optim.lr`, so the
    model knew the shape of the entire experiment — and could not be rebuilt from a
    bundle without reconstructing a config first."""
    cfg = _cfg(tabular_csv, "multiclass", tmp_path)
    model = build_model(cfg, input_dim=6, output_dim=3)

    assert not hasattr(model, "config")
    assert model.params.hidden_dims == [16, 8]
    assert model.optim.lr == pytest.approx(1e-3)  # LightningFitParams default


@pytest.mark.unit
def test_a_model_built_from_a_plain_params_dict_matches_one_built_from_a_config():
    """The manifest hands `load_from_checkpoint` a dict, so both paths must agree."""
    from ml_framework.core.protocols import BuildContext
    from ml_framework.core.registry import MODELS

    ctx = BuildContext(
        task="multiclass",
        input_dim=6,
        output_dim=3,
        params={"hidden_dims": [16, 8], "dropout": 0.25},
        optim={"lr": 0.01},
    )
    model = MODELS.get("mlp").build(ctx)
    assert model.params.hidden_dims == [16, 8]
    assert model.params.dropout == pytest.approx(0.25)
    assert model.optim.lr == pytest.approx(0.01)
