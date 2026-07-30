"""
config/schema.py
────────────────
Validated, immutable experiment configuration (Pydantic v2).

Replaces the old mutable `class config` global. A frozen model prevents the
hidden-side-effect anti-pattern where training code wrote derived values
(class weights, dims) back onto a shared global. Derived values now live on the
DataModule instance instead.

Load from YAML:
    cfg = ExperimentConfig.from_yaml("configs/example_tabular.yaml")

Override programmatically (returns a new copy — frozen):
    cfg = cfg.with_overrides({"optim.lr": 3e-4, "train.epochs": 5})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Task = Literal["binary", "multiclass", "regression"]
DataKind = Literal["tabular", "image"]
ImbalanceStrategy = Literal["smote", "class_weights", "none"]


class DataConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: DataKind = "tabular"

    # tabular
    csv_path: str | None = None
    target_col: str | None = None

    # image
    train_dir: str | None = None
    val_dir: str | None = None
    test_dir: str | None = None
    img_size: int = Field(default=224, gt=0)

    # shared
    class_names: list[str] | None = None
    val_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    holdout_threshold: int = Field(default=5000, gt=0)
    imbalance_strategy: ImbalanceStrategy = "smote"
    imbalance_threshold: float = Field(default=0.3, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_required_by_kind(self) -> DataConfig:
        if self.kind == "tabular":
            if not self.csv_path or not self.target_col:
                raise ValueError("tabular data requires 'csv_path' and 'target_col'")
        elif self.kind == "image":
            if not self.train_dir or not self.test_dir:
                raise ValueError("image data requires 'train_dir' and 'test_dir'")
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")
        return self


class ModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    name: str = "mlp"  # key in the model registry
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    # image models
    backbone: str = "resnet18"
    pretrained: bool = True

    @model_validator(mode="after")
    def _check_dims(self) -> ModelConfig:
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be positive")
        return self


class OptimConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    lr_patience: int = Field(default=10, ge=1)
    lr_factor: float = Field(default=0.5, gt=0.0, lt=1.0)


class TrainConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=-1)  # -1 → auto (0 on Windows, else 4)
    patience: int = Field(default=20, ge=1)  # early stopping
    gradient_clip_val: float = Field(default=1.0, ge=0.0)
    deterministic: bool = True


class LoggingConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    backend: Literal["wandb", "csv", "mlflow", "none"] = "csv"
    wandb_project: str = "ml-framework"
    wandb_run: str | None = None
    log_model: bool = False

    # ── MLflow (tracking + model registry) ────────────────
    # tracking_uri: None → local "sqlite:///mlflow.db" (registry-capable; the file
    # store is deprecated in MLflow 3.x). In production point at "http://mlflow:5000".
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "ml-framework"
    # Where run/model artifacts are stored (None → MLflow default, e.g. ./mlartifacts
    # locally or an s3://… bucket in production).
    mlflow_artifact_location: str | None = None
    # If set, the trained bundle is registered as a version of this model.
    registered_model_name: str | None = None


class ExperimentConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    task: Task
    seed: int = 42
    output_dir: str = "outputs"

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # HPO extras
    hpo_n_trials: int = Field(default=50, ge=1)
    hpo_timeout: int = Field(default=3600, ge=1)

    # ── Loaders ───────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def with_overrides(self, overrides: dict[str, Any]) -> ExperimentConfig:
        """Return a new config with dotted-key overrides applied.

        Example: {"optim.lr": 3e-4, "train.epochs": 5}
        """
        data = self.model_dump()
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            node = data
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    raise KeyError(f"Unknown config path: {dotted}")
                node = node[k]
            if keys[-1] not in node:
                raise KeyError(f"Unknown config key: {dotted}")
            node[keys[-1]] = value
        return self.__class__.model_validate(data)
