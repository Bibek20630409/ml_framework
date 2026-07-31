"""Training backends: one per fit-loop **shape**, not per library.

Registration is a spec plus a lazy factory, never an import of the backend
module. That is what lets ``mlf backends`` list ``lightning`` alongside a
not-yet-installed ``gbdt`` on a bare install, and what keeps
``pipeline/train.py`` — which imports this package — free of torch.

    lightning   MLP, CNN, LSTM, TFT, HF transformer   (iterative, mini-batch)
    gbdt        XGBoost, LightGBM, CatBoost, sklearn  (one-shot fit(X, y))
    forecast    Prophet, statsmodels, seasonal-naive  (fit-per-series)

``forecast`` arrives with the model family that needs it.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..core.plugins import BackendSpec
from ..core.registry import register_backend
from ..core.types import Capabilities, Requirement


def _lazy_factory(module: str, attr: str = "build_backend"):
    """Defer the import to call time, so registering never imports torch."""

    def _build() -> Any:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)()

    return _build


# Capabilities are duplicated from `LightningBackend.capabilities` rather than
# imported, because reading them off the class would require importing it — the
# exact thing this registration exists to avoid. The pair is covered by a test
# asserting they stay in agreement.
register_backend(
    BackendSpec(
        name="lightning",
        factory=_lazy_factory(".lightning"),
        capabilities=Capabilities(
            accepts=frozenset({"arrays", "dataset"}),
            needs_scaling=True,
            produces_proba=True,
            supports_pruning=True,
            supports_gpu=True,
            supports_mixed_precision=True,
            supports_lr_range_test=True,
            supports_resume=True,
            supports_sample_weight=False,
        ),
        requires=(
            Requirement("torch", extra="lightning", min_version="2.0"),
            Requirement(
                "pytorch_lightning", extra="lightning", min_version="2.0", dist="pytorch-lightning"
            ),
        ),
        description="Iterative mini-batch training: epoch loop + validation callbacks.",
    )
)

# No `requires`: the backend itself is pure-python dispatch, and *which* library it
# needs depends on the model selected. Each GBDT plugin declares its own
# requirement, so `mlf models` can say "xgboost needs xgboost>=2.0" while still
# listing the backend as available. Declaring the union here would refuse a
# lightgbm run on an install that has lightgbm but not catboost.
register_backend(
    BackendSpec(
        name="gbdt",
        factory=_lazy_factory(".gbdt"),
        capabilities=Capabilities(
            accepts=frozenset({"arrays", "frame"}),
            needs_scaling=False,
            native_categorical=True,
            native_missing=True,
            supports_sample_weight=True,
            produces_proba=True,
            supports_pruning=True,
            supports_gpu=True,
            supports_mixed_precision=False,
            supports_lr_range_test=False,
        ),
        requires=(),
        description="One-shot fit(X, y, eval_set=) with library-native early stopping.",
    )
)

__all__ = ["BackendSpec"]
