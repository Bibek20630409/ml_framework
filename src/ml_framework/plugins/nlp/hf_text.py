"""
plugins/nlp/hf_text.py
──────────────────────
Fine-tuning a HuggingFace encoder for text classification. Registered as
``"nlp.hf_text"``.

It reuses the Lightning backend rather than adding a fourth one, which is the
whole claim of the backend-per-fit-loop-shape design: fine-tuning a transformer is
an epoch loop with validation callbacks, exactly like training a CNN. What differs
is a batch's *shape* and the model's *file format*, and both are handled here
rather than by a branch in the backend.

**The batch.** A collated text batch is ``(encoding, labels)`` where ``encoding``
is a dict of ``input_ids``/``attention_mask``. :meth:`HFTextClassifier.forward`
unpacks it; ``BaseModel._shared_step`` is untouched and still just does
``x, y = batch; self(x)``.

**The file format.** A checkpoint *directory*, not a Lightning ``.ckpt`` — see
:class:`~ml_framework.plugins.nlp.base.HFPretrainedModel`, which owns that half
for all three HuggingFace plugins.

The paired tokenizer lives under ``preprocessor/tokenizer/`` and is owned by
:class:`~ml_framework.data.preprocess.text.TextPreprocessor`. Between the two, a
served bundle is self-contained: no network, no cache, no ``model_name`` lookup.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import torch

from ...core.protocols import BuildContext
from ...core.registry import register_model
from .base import HF_FORMAT, HF_MODEL_DIR, HFPretrainedModel
from .params import HFTextParams

log = logging.getLogger(__name__)


@register_model("nlp.hf_text")
class HFTextClassifier(HFPretrainedModel):
    """An ``AutoModelForSequenceClassification`` wearing the framework's contract."""

    AUTO_CLASS: ClassVar[str] = "AutoModelForSequenceClassification"

    @classmethod
    def params_model(cls) -> type[HFTextParams]:
        return HFTextParams

    def from_pretrained_kwargs(self) -> dict[str, Any]:
        return {
            "num_labels": self.output_dim,
            # Replacing the classification head is the entire point of fine-tuning
            # for a new label set, so a head-shaped mismatch against the checkpoint
            # is expected rather than an error. Without this, any checkpoint that
            # already carries a head refuses to load for a different class count.
            "ignore_mismatched_sizes": True,
        }

    def forward(self, x: Any) -> torch.Tensor:
        """Tokenized batch → logits.

        ``output_dim`` is 1 for binary and n for multiclass, matching the
        framework's head convention exactly — which is also what ``num_labels``
        means to transformers, so the two agree without a translation step.
        """
        return cast("torch.Tensor", self.hf_model(**x).logits)


def build(ctx: BuildContext) -> HFTextClassifier:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return HFTextClassifier(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        # Imbalance is corrected by the text source's sampler, never also by loss
        # weights — correcting twice overshoots.
        class_weights=None,
    )


__all__ = ["HF_FORMAT", "HF_MODEL_DIR", "HFTextClassifier", "build"]
