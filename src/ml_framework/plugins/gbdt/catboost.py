"""
plugins/gbdt/catboost.py
────────────────────────
CatBoost, registered as ``"catboost"``. The one to reach for on high-cardinality
categorical features, where its ordered target statistics beat one-hot encoding
and beat naive target encoding's leakage.

``depth`` is symmetric-tree depth here, which is a different quantity from
XGBoost's ``max_depth`` despite the similar name — another difference the
per-plugin schema records rather than papers over.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class CatBoostParams(PydanticModel):
    """``model.params`` for CatBoost: symmetric-tree knobs."""

    model_config = {"frozen": True, "extra": "forbid"}

    # CatBoost builds oblivious (symmetric) trees; 6-10 is the useful band and
    # >16 is refused by the library itself.
    depth: int = Field(default=6, ge=1, le=16)
    l2_leaf_reg: float = Field(default=3.0, gt=0.0)
    border_count: int = Field(default=254, ge=1, le=65535)
    random_strength: float = Field(default=1.0, ge=0.0)


def build(ctx: BuildContext) -> Any:
    """An **unfitted** CatBoost estimator. The backend owns the fit call."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = CatBoostParams.model_validate(dict(ctx.params))
    loop = dict(ctx.optim)
    common: dict[str, Any] = {
        "depth": params.depth,
        "l2_leaf_reg": params.l2_leaf_reg,
        "border_count": params.border_count,
        "random_strength": params.random_strength,
        "learning_rate": loop.get("learning_rate", 0.1),
        "iterations": loop.get("n_estimators", 500),
        "random_seed": ctx.seed,
        # CatBoost writes a `catboost_info/` directory next to the CWD otherwise,
        # which would litter the bundle with training telemetry nobody asked for.
        "allow_writing_files": False,
        "verbose": False,
    }
    # `subsample` is only valid with a bootstrap type that supports it; passing it
    # unconditionally makes CatBoost raise on its default Bayesian bootstrap.
    subsample = loop.get("subsample", 1.0)
    if subsample < 1.0:
        common["bootstrap_type"] = "Bernoulli"
        common["subsample"] = subsample

    if ctx.task == "regression":
        return CatBoostRegressor(loss_function="RMSE", **common)
    if ctx.task == "binary":
        return CatBoostClassifier(loss_function="Logloss", **common)
    return CatBoostClassifier(loss_function="MultiClass", **common)


__all__ = ["CatBoostParams", "build"]
