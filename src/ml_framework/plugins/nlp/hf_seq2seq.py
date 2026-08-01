"""
plugins/nlp/hf_seq2seq.py
─────────────────────────
Encoder-decoder fine-tuning: summarization, translation, text normalization.
Registered as ``"nlp.hf_seq2seq"``.

This is the model that does not fit the ``logits → argmax`` shape the rest of the
framework is built around, and rather than bending it, the two halves are kept
apart and both are named:

* **Training is teacher forcing.** The decoder is fed the *reference* prefix at
  every position, so one forward pass produces logits over the whole target at
  once and the loss is ordinary cross-entropy. That is what
  :meth:`HFSeq2Seq.forward` returns, and it is why the fit loop needs no changes.
* **Prediction is generation.** An autoregressive loop, one decoder step per
  token, with no reference available. It cannot be expressed as a function of the
  logits — which is why ``Postprocess`` grew a fourth value, ``generate``, instead
  of this task pretending to be ``identity``.

The consequence worth stating plainly: **``val/loss`` is teacher-forced and the
reported metrics are not.** A model can improve on the loss while getting worse at
generating, because the loss never asks it to survive its own mistakes. The loss
remains the early-stopping monitor because generating on every validation epoch
would multiply epoch time by the decode length; the reported ROUGE-L/token-F1/
exact-match come from real generation on the test split, once.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch

from ...core.protocols import BuildContext
from ...core.registry import register_model
from .base import HFPretrainedModel
from .params import HFSeq2SeqParams

# Keys the encoder needs at generation time. `labels` is deliberately absent: it
# is in the batch during evaluation (the collate function puts it there so the
# loss can be computed), and passing it to `generate` is an error.
GENERATE_INPUTS = ("input_ids", "attention_mask")


@register_model("nlp.hf_seq2seq")
class HFSeq2Seq(HFPretrainedModel):
    AUTO_CLASS: ClassVar[str] = "AutoModelForSeq2SeqLM"

    @classmethod
    def params_model(cls) -> type[HFSeq2SeqParams]:
        return HFSeq2SeqParams

    def from_pretrained_kwargs(self) -> dict[str, Any]:
        # No `num_labels`: the head is the vocabulary and the checkpoint owns it.
        # Passing a label count here would silently build a classifier.
        return {}

    def forward(self, x: Any) -> torch.Tensor:
        """Teacher-forced logits over the target, ``(batch, positions, vocab)``.

        ``labels`` arrives *inside* ``x`` because transformers derives
        ``decoder_input_ids`` from it — the right-shift that makes teacher forcing
        work. Doing that shift by hand is a well-known way to train a model that
        predicts the token it was just handed, and it produces a beautiful loss
        curve while doing so.
        """
        return cast("torch.Tensor", self.hf_model(**x).logits)

    # ── generation ──
    def generate_ids(self, x: Any) -> torch.Tensor:
        """Autoregressive decode → token ids.

        Called by the estimator, which then asks the *preprocessor* to turn the ids
        back into strings. Splitting it there rather than returning text here is
        deliberate: the tokenizer that produced the ids belongs to the
        preprocessor, and a second copy living on the model is a second thing that
        can disagree with the first.
        """
        inputs = {k: v for k, v in dict(x).items() if k in GENERATE_INPUTS}
        return cast(
            "torch.Tensor",
            self.hf_model.generate(
                **inputs,
                max_new_tokens=self.params.max_target_length,
                num_beams=self.params.num_beams,
            ),
        )


def build(ctx: BuildContext) -> HFSeq2Seq:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return HFSeq2Seq(
        input_dim=ctx.input_dim,
        # 0: the head is the checkpoint's vocabulary, not a width this framework
        # chooses. `BaseModel` uses it only to size metrics, and seq2seq builds
        # none — see `_build_metrics`.
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=None,
    )


__all__ = ["GENERATE_INPUTS", "HFSeq2Seq", "build"]
