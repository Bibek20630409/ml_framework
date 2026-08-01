"""
plugins/nlp/params.py
─────────────────────
``model.params`` for the HuggingFace text classifier — pydantic only.

Split from ``hf_text.py`` for the reason every neural plugin is:
:class:`~ml_framework.core.lit_model.BaseModel` is a ``LightningModule``, so
*defining* the class imports torch, and a spec's ``params_model`` has to be
importable on an install that has neither torch nor transformers. This module has
no import that is not always present.
"""

from __future__ import annotations

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.types import Requirement
from ...data.preprocess.text import DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME, NLP_REQUIREMENTS

# torch arrives through the Lightning backend; transformers through `[nlp]`. Both
# are needed before this plugin can build anything, and both are answered by
# `find_spec` so `mlf models` can list `nlp.hf_text` as unavailable-with-a-fix
# rather than hiding it.
TORCH_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("torch", extra="lightning", min_version="2.0"),
    Requirement(
        "pytorch_lightning", extra="lightning", min_version="2.0", dist="pytorch-lightning"
    ),
)
HF_TEXT_REQUIREMENTS: tuple[Requirement, ...] = (*TORCH_REQUIREMENTS, *NLP_REQUIREMENTS)


class HFTextParams(PydanticModel):
    """``model.params`` for ``nlp.hf_text``.

    ``model_name`` and ``max_length`` are read by the **text source** as well as by
    the model, and that is deliberate rather than a leak: the tokenizer must come
    from the same checkpoint as the weights, so there is exactly one place to say
    which checkpoint that is. Putting a second copy under ``data.params`` would
    create a way to set them inconsistently, and the resulting model would score
    badly rather than fail.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, gt=0)
    # Head-only training: the encoder is left at its pretrained weights and only
    # the classifier learns. Much faster and much weaker — worth having for a
    # first pass on a small corpus, not worth defaulting to.
    freeze_encoder: bool = False


__all__ = ["HF_TEXT_REQUIREMENTS", "TORCH_REQUIREMENTS", "HFTextParams"]
