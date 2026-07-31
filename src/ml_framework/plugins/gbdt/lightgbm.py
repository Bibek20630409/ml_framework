"""
plugins/gbdt/lightgbm.py
────────────────────────
LightGBM, registered as ``"lightgbm"``. Wins on wide and large tabular data.

LightGBM grows trees leaf-wise, so ``num_leaves`` — not ``max_depth`` — is the
capacity knob that matters; ``max_depth`` is available as a guard rail and ``-1``
means unbounded. That difference from XGBoost is exactly the kind of thing a
per-plugin params schema exists to express, and exactly what a shared
``ModelConfig`` could not.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class LightGBMParams(PydanticModel):
    """``model.params`` for LightGBM: leaf-wise growth knobs."""

    model_config = {"frozen": True, "extra": "forbid"}

    num_leaves: int = Field(default=31, ge=2)
    # -1 = no limit. Leaf-wise growth makes this a guard rail, not the main knob.
    max_depth: int = Field(default=-1, ge=-1)
    min_child_samples: int = Field(default=20, ge=1)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=0.0, ge=0.0)


def build(ctx: BuildContext) -> Any:
    """An **unfitted** LightGBM estimator. The backend owns the fit call."""
    import lightgbm as lgb

    params = LightGBMParams.model_validate(dict(ctx.params))
    loop = dict(ctx.optim)
    common: dict[str, Any] = {
        "num_leaves": params.num_leaves,
        "max_depth": params.max_depth,
        "min_child_samples": params.min_child_samples,
        "reg_alpha": params.reg_alpha,
        "reg_lambda": params.reg_lambda,
        "learning_rate": loop.get("learning_rate", 0.1),
        "n_estimators": loop.get("n_estimators", 500),
        "subsample": loop.get("subsample", 1.0),
        "colsample_bytree": loop.get("colsample_bytree", 1.0),
        "random_state": ctx.seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    if ctx.task == "regression":
        return lgb.LGBMRegressor(objective="regression", **common)
    if ctx.task == "binary":
        return lgb.LGBMClassifier(objective="binary", **common)
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=ctx.n_classes or ctx.output_dim, **common
    )


__all__ = ["LightGBMParams", "build"]
