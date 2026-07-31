"""
plugins/params.py
─────────────────
The neural plugins' ``model.params`` schemas, **separated from their networks**.

A ``ModelSpec`` carries ``params_model`` because the config validator runs it at
load time — on any install, including one without the plugin's runtime. But
``MLP`` is a ``LightningModule`` subclass, so *defining the class* imports torch
at module scope, which a plugin module is not allowed to do for an optional
dependency. Both things cannot live in one file.

So the split follows the dependency: schemas here (pydantic only), networks in
``mlp.py``/``cnn.py`` behind a lazy ``build``. That is what lets ``mlf models``
list ``mlp`` — marked unavailable, with ``pip install 'ml-framework[lightning]'``
— on a GBDT-only install, and what lets a tree config validate its ``model.params``
in a process where torch does not exist.

The GBDT plugins need no equivalent split: their modules define only pydantic
schemas at import time and reach for xgboost inside ``build()``.
"""

from __future__ import annotations

from pydantic import BaseModel as PydanticModel
from pydantic import Field, model_validator


class MLPParams(PydanticModel):
    """``model.params`` for the MLP.

    v1's ``ModelConfig._check_dims`` moves here with the field it guards: "all
    hidden_dims must be positive" is a fact about this model, not about every
    model in the framework.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_dims(self) -> MLPParams:
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be positive")
        return self


class CNNParams(PydanticModel):
    """``model.params`` for the transfer-learning CNN.

    Deliberately has no ``dropout``. v1's shared ``ModelConfig`` carried one for
    every model and the CNN never read it; ``extra="forbid"`` now says so out loud
    instead of accepting a knob that does nothing.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    backbone: str = "resnet18"
    pretrained: bool = True


__all__ = ["CNNParams", "MLPParams"]
