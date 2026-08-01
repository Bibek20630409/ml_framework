"""
plugins/nlp/hf_token.py
───────────────────────
Per-token tagging — named-entity recognition, part-of-speech. Registered as
``"nlp.hf_token"``.

The only thing that distinguishes this from ``nlp.hf_text`` at the model layer is
where the head sits: a sequence classifier pools the encoder's output into one
vector and predicts once, a token classifier predicts at **every position**. So
the logits are ``(B, T, C)`` rather than ``(B, C)``, and everything downstream —
the loss, the metrics, ``predict_split``, ``predictions.csv`` — has to know that
one input row produces a variable number of predictions.

The genuinely hard part is not here. It is the word-to-sub-word label alignment in
:class:`~ml_framework.data.preprocess.text.TokenTextPreprocessor`, which decides
what "Acme" tagged ``B-ORG`` means once the tokenizer has turned it into
``["ac", "##me"]``.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch

from ...core.protocols import BuildContext
from ...core.registry import register_model
from .base import HFPretrainedModel
from .params import HFTokenParams


@register_model("nlp.hf_token")
class HFTokenClassifier(HFPretrainedModel):
    AUTO_CLASS: ClassVar[str] = "AutoModelForTokenClassification"

    @classmethod
    def params_model(cls) -> type[HFTokenParams]:
        return HFTokenParams

    def from_pretrained_kwargs(self) -> dict[str, Any]:
        return {
            "num_labels": self.output_dim,
            # Replacing the head for a new tag set is the point of fine-tuning, so
            # a head-shaped mismatch against the checkpoint is expected rather than
            # an error.
            "ignore_mismatched_sizes": True,
        }

    def forward(self, x: Any) -> torch.Tensor:
        """Tokenized batch → ``(batch, positions, tags)`` logits.

        Note the head width: ``output_dim`` is the tag count even when there are
        only two tags. The framework's one-logit binary convention does not reach
        here, because the loss is cross-entropy over positions rather than a single
        sigmoid — and a two-tag corpus is still a `multiclass`-shaped decision made
        many times.
        """
        return cast("torch.Tensor", self.hf_model(**x).logits)


def build(ctx: BuildContext) -> HFTokenClassifier:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return HFTokenClassifier(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        # Tag imbalance is real and severe (`O` dominates), but class weights in
        # the loss are the wrong lever for it by default: they trade precision for
        # recall at a ratio nobody chose. The honest answer is macro-F1 as the
        # primary metric, which is what the TaskSpec sets.
        class_weights=None,
    )


__all__ = ["HFTokenClassifier", "build"]
