"""
plugins/gbdt/xgboost.py
───────────────────────
XGBoost, registered as ``"xgboost"``. The default tabular model.

The plugin declares **tree shape**; the boosting loop's knobs
(``learning_rate``, ``n_estimators``, ``subsample``, ``colsample_bytree``,
``early_stopping_rounds``) live on the GBDT backend, so all three libraries stop
repeating them.

``import xgboost`` happens inside :func:`build`, never at module scope — the rule
that lets ``mlf models`` list this plugin, honestly marked unavailable, on an
install without the ``gbdt`` extra.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class XGBoostParams(PydanticModel):
    """``model.params`` for XGBoost: the tree, not the loop."""

    model_config = {"frozen": True, "extra": "forbid"}

    max_depth: int = Field(default=6, ge=1, le=64)
    min_child_weight: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=0.0, ge=0.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    # "hist" is the modern default and the only one with native categorical
    # support; "exact" is kept reachable for small-data reproducibility work.
    tree_method: str = "hist"


def build(ctx: BuildContext) -> Any:
    """An **unfitted** XGBoost estimator. The backend owns the fit call.

    Returning the configured-but-unfitted booster is the GBDT analogue of
    returning an untrained network: the model knows its own shape and nothing
    about output dirs, checkpoints or trackers.
    """
    import xgboost as xgb

    params = XGBoostParams.model_validate(dict(ctx.params))
    loop = dict(ctx.optim)
    common: dict[str, Any] = {
        "max_depth": params.max_depth,
        "min_child_weight": params.min_child_weight,
        "gamma": params.gamma,
        "reg_alpha": params.reg_alpha,
        "reg_lambda": params.reg_lambda,
        "tree_method": params.tree_method,
        "learning_rate": loop.get("learning_rate", 0.1),
        "n_estimators": loop.get("n_estimators", 500),
        "subsample": loop.get("subsample", 1.0),
        "colsample_bytree": loop.get("colsample_bytree", 1.0),
        "random_state": ctx.seed,
        "n_jobs": -1,
        # NaN is routed down a learned default branch rather than imputed. This is
        # what `Capabilities.native_missing` promises, and imputing first
        # measurably hurts these models.
        "missing": float("nan"),
    }

    if ctx.task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", **common)
    if ctx.task == "binary":
        return xgb.XGBClassifier(objective="binary:logistic", **common)
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=ctx.n_classes or ctx.output_dim, **common
    )


__all__ = ["XGBoostParams", "build"]
