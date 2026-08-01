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
from ...data.preprocess.text import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_LENGTH,
    NLP_REQUIREMENTS,
)

# BART rather than T5 as the seq2seq default, for one unglamorous reason: T5's
# tokenizer needs `sentencepiece`, which is a separate install the `[nlp]` extra
# does not pull. A default that fails on a correctly-installed extra is not a
# default. Name a T5 checkpoint explicitly and it works, once sentencepiece is
# present.
DEFAULT_SEQ2SEQ_MODEL = "facebook/bart-base"

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


class HFTokenParams(PydanticModel):
    """``model.params`` for ``nlp.hf_token``.

    No ``max_target_length`` and no ``num_beams``: a tagger emits exactly one
    decision per input token, so there is nothing to decode and nothing to search
    over. ``extra="forbid"`` means setting one is an error rather than a knob that
    does nothing.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    model_name: str = DEFAULT_MODEL_NAME
    # Truncation cuts *words* off the end of a sentence, and their tags go with
    # them. A tagging corpus with long sentences wants this raised.
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, gt=0)
    freeze_encoder: bool = False


class HFSeq2SeqParams(PydanticModel):
    """``model.params`` for ``nlp.hf_seq2seq``.

    ``max_target_length`` is read by the source as well as the model, like
    ``model_name`` and for the same reason: one place to set one thing. It bounds
    both the training target and the decode loop, so raising it costs time twice.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    model_name: str = DEFAULT_SEQ2SEQ_MODEL
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, gt=0)
    max_target_length: int = Field(default=DEFAULT_TARGET_LENGTH, gt=0)
    # 1 is greedy decoding. Beams improve output and cost linearly in time, which
    # is paid on every evaluation pass — hence a default that does not surprise.
    num_beams: int = Field(default=1, ge=1)
    freeze_encoder: bool = False


__all__ = [
    "DEFAULT_SEQ2SEQ_MODEL",
    "HF_TEXT_REQUIREMENTS",
    "TORCH_REQUIREMENTS",
    "HFSeq2SeqParams",
    "HFTextParams",
    "HFTokenParams",
]
