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

**The file format.** A Lightning checkpoint would round-trip the *weights*, but
rebuilding the architecture to put them in calls ``from_pretrained(model_name)``
— which needs the hub, or a warm HF cache, at load time. A bundle that cannot be
loaded on an offline machine is not a bundle. So this model owns its own
serialization: ``save_pretrained`` into ``model/hf_model/``, and
``from_pretrained`` back out of that directory. The backend discovers the two
hooks by name (:meth:`save_artifact` / :meth:`load_artifact`) and uses the
checkpoint path for every model that does not define them.

The paired tokenizer lives under ``preprocessor/tokenizer/`` and is owned by
:class:`~ml_framework.data.preprocess.text.TextPreprocessor`. Between the two, a
served bundle is self-contained: no network, no cache, no ``model_name`` lookup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch

from ...core.bundle import head_width
from ...core.lit_model import BaseModel
from ...core.protocols import ArtifactRef, BuildContext
from ...core.registry import register_model
from .params import HFTextParams

log = logging.getLogger(__name__)

# The subdirectory inside `model/` holding a `save_pretrained` dump.
HF_MODEL_DIR = "hf_model"
# Tracked separately from the directory name so the layout can change without
# breaking readers — the same reason every other artifact records a format.
HF_FORMAT = "huggingface-pretrained"


@register_model("nlp.hf_text")
class HFTextClassifier(BaseModel):
    """A ``AutoModelForSequenceClassification`` wearing the framework's contract."""

    @classmethod
    def params_model(cls) -> type[HFTextParams]:
        return HFTextParams

    def build_network(self) -> Any:
        from transformers import AutoModelForSequenceClassification

        net = AutoModelForSequenceClassification.from_pretrained(
            self.params.model_name,
            num_labels=self.output_dim,
            # Replacing the classification head is the entire point of fine-tuning
            # for a new label set, so a head-shaped mismatch against the checkpoint
            # is expected rather than an error. Without this, any checkpoint that
            # already carries a head refuses to load for a different class count.
            ignore_mismatched_sizes=True,
        )
        if self.params.freeze_encoder:
            for p in net.base_model.parameters():
                p.requires_grad = False
            log.info("encoder frozen: training the classification head only")
        return net

    @property
    def hf_model(self) -> Any:
        """``self.network``, with the type it actually has.

        torch types a module attribute as ``Tensor | Module``, so calling a
        transformers method on one is a type error even though that is precisely
        what the attribute is. Narrowed once here rather than at each use.
        """
        return cast("Any", self.network)

    def forward(self, x: Any) -> torch.Tensor:
        """Tokenized batch → logits.

        ``output_dim`` is 1 for binary and n for multiclass, matching the
        framework's head convention exactly — which is also what ``num_labels``
        means to transformers, so the two agree without a translation step.
        """
        return cast("torch.Tensor", self.hf_model(**x).logits)

    # ── self-serialization ──
    def save_artifact(self, dest: str | Path) -> ArtifactRef:
        """Write the HF directory and return its bundle-relative path.

        ``dest`` is the bundle's ``model/`` directory, so the returned path is
        relative to its parent — the bundle root, which is what the manifest
        records.
        """
        target_dir = Path(dest)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.hf_model.save_pretrained(target_dir / HF_MODEL_DIR)
        return ArtifactRef(path=f"{target_dir.name}/{HF_MODEL_DIR}", format=HF_FORMAT)

    @classmethod
    def load_artifact(cls, path: str | Path, manifest: Any) -> HFTextClassifier:
        """Rebuild from ``model/hf_model/`` — no hub, no cache, no network.

        ``model_name`` is redirected at the bundle's own directory. The manifest
        keeps the original checkpoint name in ``model.params``, so the audit trail
        still says what was fine-tuned; this only changes where the weights are
        read from.
        """
        params = dict(manifest.model.params)
        params["model_name"] = str(path)
        return cls(
            # A token sequence has no fixed input width; the encoder's own config
            # carries the shapes that matter and is loaded from `path`.
            input_dim=0,
            output_dim=head_width(manifest),
            task=manifest.task,
            params=params,
        )


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
