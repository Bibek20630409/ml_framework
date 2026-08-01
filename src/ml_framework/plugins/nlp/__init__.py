"""NLP plugins: transformer fine-tuning on the Lightning backend.

One model so far. The package exists rather than a flat ``plugins/hf_text.py``
because the split this family needs is the same one ``gbdt/`` and ``ts/`` needed:
a pydantic-only ``params.py`` that any install can import, and a module reached
lazily that imports torch and transformers.

**Fine-tuning defaults are the load-bearing part of this file.** The framework's
default learning rate is 1e-3 with Adam, which is right for a network trained from
scratch and catastrophic for a pretrained encoder — at that rate the first few
steps destroy the representations the checkpoint exists to provide, and the run
scores near chance while looking entirely healthy. So the spec carries
``fit_defaults``, applied by the config validator to keys the user did not write,
and a ``search_space`` that narrows the backend's learning-rate range to the
fine-tuning band rather than inheriting one that would spend most trials wrecking
the model.
"""

from __future__ import annotations

import importlib
from typing import Any

from ...core.plugins import ModelSpec
from ...core.protocols import Float
from ...core.registry import register_model_spec
from ...core.types import Capabilities
from .params import HF_TEXT_REQUIREMENTS, HFTextParams

BUILTINS: tuple[str, ...] = ("nlp.hf_text",)

__all__ = ["BUILTINS", "HFTextParams"]


def _lazy_build(module: str) -> Any:
    def _build(ctx: Any) -> Any:
        return importlib.import_module(module, __name__).build(ctx)

    return _build


register_model_spec(
    ModelSpec(
        name="nlp.hf_text",
        # Not a fourth backend: fine-tuning a transformer is an epoch loop with
        # validation callbacks, which is what `lightning` already is.
        backend="lightning",
        build=_lazy_build(".hf_text"),
        tasks=frozenset({"binary", "multiclass"}),
        data_kinds=frozenset({"text"}),
        requires=HF_TEXT_REQUIREMENTS,
        capabilities=Capabilities(
            accepts=frozenset({"dataset"}),
            # Scaling a token id would be meaningless; the embedding table is the
            # normalization.
            needs_scaling=False,
            produces_proba=True,
            supports_pruning=True,
            supports_gpu=True,
            supports_mixed_precision=True,
            # An LR range test over a pretrained encoder is a slow way to be told
            # what the literature already settled: 2e-5 to 5e-5.
            supports_lr_range_test=False,
            # The source corrects imbalance by sampling, not by per-row weights.
            supports_sample_weight=False,
        ),
        # Narrower than the backend's own lr range on purpose, and it wins because
        # the more specific declaration does. The backend proposes 1e-4..1e-2,
        # which for a pretrained encoder is a range in which most trials are
        # damage.
        search_space={"fit.params.lr": Float(1e-5, 5e-5, log=True)},
        # Keys of `fit.params`; only those the user did not set are applied. These
        # are the standard BERT fine-tuning settings, and they are defaults rather
        # than constants precisely so a user can disagree.
        fit_defaults={
            "lr": 2e-5,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            # Linear-ish decay to zero over the run. `plateau` needs several epochs
            # of patience to react, and a fine-tune is over in three.
            "scheduler": "cosine",
        },
        params_model=HFTextParams,
        auto_priority=10,
        description="Fine-tunes a HuggingFace encoder (default distilbert-base-uncased) for text.",
    )
)
