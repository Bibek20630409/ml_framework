"""The gradient-boosted-tree plugins: xgboost, lightgbm, catboost.

All three ride one backend, because their fit loops are the same eight lines. What
differs — tree shape, serialization format, the spelling of early stopping — is
split between each plugin's ``params_model`` (shape) and the backend's adapter
table (the rest).

Every spec here declares ``requires``, so on an install without the ``gbdt`` extra
these appear in ``mlf models`` marked unavailable and refuse selection with
``pip install 'ml-framework[gbdt]'`` — rather than vanishing, which is what a
``try/except ImportError`` around the import would have done.

``auto_priority`` ranks them for the zero-config chooser: XGBoost is the default
tabular pick, LightGBM the one that wins on wide/large data, CatBoost the one for
high-cardinality categoricals. The chooser that reads these arrives with the
zero-config work; the ordering is declared here because this is where the
knowledge lives.
"""

from __future__ import annotations

from ...core.plugins import ModelSpec
from ...core.protocols import Float, Int
from ...core.registry import register_model_spec
from ...core.types import Capabilities, DataKind, Payload, Requirement, Task
from .catboost import CatBoostParams
from .catboost import build as _build_catboost
from .lightgbm import LightGBMParams
from .lightgbm import build as _build_lightgbm
from .xgboost import XGBoostParams
from .xgboost import build as _build_xgboost

BUILTINS: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")

# Shared by all three. The four flags that are not defaults each have a named
# consumer, per the rule that a capability nothing reads is decoration:
#   needs_scaling=False        → TabularPreprocessor skips StandardScaler
#   native_categorical=True    → pandas `category` dtype passes through
#   native_missing=True        → imputation is skipped; NaN is routed natively
#   supports_sample_weight=True→ imbalance resolver prefers weights over SMOTE
#   native_feature_importance=True → core.explain reads `feature_importances_`
#       instead of paying for a permutation pass, and select's a-priori gate
#       keeps trees eligible under a high `min_explainability`
_TREE_CAPS: dict[str, object] = {
    "needs_scaling": False,
    "native_categorical": True,
    "native_missing": True,
    "supports_sample_weight": True,
    "produces_proba": True,
    "supports_pruning": True,
    "native_feature_importance": True,
    "supports_gpu": True,
    "supports_mixed_precision": False,
    "supports_lr_range_test": False,
}

_TASKS: frozenset[Task] = frozenset({"binary", "multiclass", "regression"})
_KINDS: frozenset[DataKind] = frozenset({"tabular"})
_ACCEPTS: frozenset[Payload] = frozenset({"arrays", "frame"})


def _requirement(module: str, min_version: str) -> tuple[Requirement, ...]:
    return (Requirement(module, extra="gbdt", min_version=min_version),)


register_model_spec(
    ModelSpec(
        name="xgboost",
        backend="gbdt",
        build=_build_xgboost,
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=_requirement("xgboost", "2.0"),
        capabilities=Capabilities(accepts=_ACCEPTS, **_TREE_CAPS),  # type: ignore[arg-type]
        # Tree shape only — learning_rate/n_estimators/subsample/colsample come
        # from the backend's space, declared once for all three.
        search_space={
            "model.params.max_depth": Int(3, 12),
            "model.params.min_child_weight": Float(0.5, 10.0, log=True),
            "model.params.reg_lambda": Float(0.1, 10.0, log=True),
        },
        params_model=XGBoostParams,
        auto_priority=30,
        description="Gradient-boosted trees (XGBoost). The default tabular model.",
    )
)

register_model_spec(
    ModelSpec(
        name="lightgbm",
        backend="gbdt",
        build=_build_lightgbm,
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=_requirement("lightgbm", "4.0"),
        capabilities=Capabilities(accepts=_ACCEPTS, **_TREE_CAPS),  # type: ignore[arg-type]
        search_space={
            "model.params.num_leaves": Int(15, 255, log=True),
            "model.params.min_child_samples": Int(5, 100, log=True),
            "model.params.reg_lambda": Float(1e-3, 10.0, log=True),
        },
        params_model=LightGBMParams,
        auto_priority=25,
        description="Gradient-boosted trees (LightGBM). Leaf-wise; best on wide/large data.",
    )
)

register_model_spec(
    ModelSpec(
        name="catboost",
        backend="gbdt",
        build=_build_catboost,
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=_requirement("catboost", "1.2"),
        capabilities=Capabilities(accepts=_ACCEPTS, **_TREE_CAPS),  # type: ignore[arg-type]
        search_space={
            "model.params.depth": Int(4, 10),
            "model.params.l2_leaf_reg": Float(1.0, 10.0, log=True),
        },
        params_model=CatBoostParams,
        auto_priority=20,
        description="Gradient-boosted trees (CatBoost). Best on high-cardinality categoricals.",
    )
)

__all__ = ["BUILTINS", "CatBoostParams", "LightGBMParams", "XGBoostParams"]
