"""NLP plugins: transformer fine-tuning on the Lightning backend.

Three models, three head shapes, one fit loop:

===================  ==========================  =====================================
model                task                        what one input row produces
===================  ==========================  =====================================
``nlp.hf_text``      binary / multiclass         one class
``nlp.hf_token``     token_classification        one class **per token**
``nlp.hf_seq2seq``   seq2seq                     a **string**
===================  ==========================  =====================================

All three ride the ``lightning`` backend, because all three are an epoch loop with
validation callbacks. What differs is the shape of a batch and the shape of a
prediction, and both belong to the model rather than to the loop.

The package rather than a flat ``plugins/hf_text.py`` because the split this
family needs is the same one ``gbdt/`` and ``ts/`` needed: a pydantic-only
``params.py`` that any install can import, and modules reached lazily that import
torch and transformers.

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
from .params import HF_TEXT_REQUIREMENTS, HFSeq2SeqParams, HFTextParams, HFTokenParams

BUILTINS: tuple[str, ...] = ("nlp.hf_text", "nlp.hf_token", "nlp.hf_seq2seq")

__all__ = ["BUILTINS", "HFSeq2SeqParams", "HFTextParams", "HFTokenParams"]

# Shared by all three: the same optimizer settings, the same reasoning. See the
# module docstring on why the framework default would destroy a pretrained encoder.
_FINE_TUNING_DEFAULTS: dict[str, Any] = {
    "lr": 2e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "scheduler": "cosine",
}

# Also shared: what a transformer can and cannot do on this backend. Spelled once
# rather than three times, so a capability cannot end up true for one head shape
# and false for another by transcription accident.
_HF_CAPABILITIES: dict[str, Any] = {
    "accepts": frozenset({"dataset"}),
    # Scaling a token id would be meaningless; the embedding table is the
    # normalization.
    "needs_scaling": False,
    "supports_pruning": True,
    "supports_gpu": True,
    "supports_mixed_precision": True,
    # An LR range test over a pretrained encoder is a slow way to be told what the
    # literature already settled: 2e-5 to 5e-5.
    "supports_lr_range_test": False,
    "supports_sample_weight": False,
}

# Narrower than the backend's own lr range on purpose, and it wins because the
# more specific declaration does. The backend proposes 1e-4..1e-2, which for a
# pretrained encoder is a range in which most trials are damage.
_FINE_TUNING_SPACE = {"fit.params.lr": Float(1e-5, 5e-5, log=True)}


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
        capabilities=Capabilities(produces_proba=True, **_HF_CAPABILITIES),
        search_space=_FINE_TUNING_SPACE,
        # Keys of `fit.params`; only those the user did not set are applied. These
        # are the standard BERT fine-tuning settings, and they are defaults rather
        # than constants precisely so a user can disagree.
        fit_defaults=_FINE_TUNING_DEFAULTS,
        params_model=HFTextParams,
        auto_priority=10,
        description="Fine-tunes a HuggingFace encoder (default distilbert-base-uncased) for text.",
    )
)

register_model_spec(
    ModelSpec(
        name="nlp.hf_token",
        backend="lightning",
        build=_lazy_build(".hf_token"),
        tasks=frozenset({"token_classification"}),
        data_kinds=frozenset({"text"}),
        requires=HF_TEXT_REQUIREMENTS,
        capabilities=Capabilities(
            # True, and it means something different here: a tagger emits a
            # distribution per *position*, so `/predict_proba` would return a
            # (tokens x tags) matrix per row. The serving layer reports labels and
            # per-token confidence instead.
            produces_proba=True,
            **_HF_CAPABILITIES,
        ),
        search_space=_FINE_TUNING_SPACE,
        fit_defaults=_FINE_TUNING_DEFAULTS,
        params_model=HFTokenParams,
        auto_priority=10,
        description="Fine-tunes a HuggingFace encoder for per-token tagging (NER, POS).",
    )
)

register_model_spec(
    ModelSpec(
        name="nlp.hf_seq2seq",
        backend="lightning",
        build=_lazy_build(".hf_seq2seq"),
        tasks=frozenset({"seq2seq"}),
        data_kinds=frozenset({"text"}),
        requires=HF_TEXT_REQUIREMENTS,
        capabilities=Capabilities(
            # A generated string has no class distribution behind it — the model
            # emits a token distribution per decoder step, which is not what
            # `/predict_proba` means. Declared False so serving answers 400 from
            # the manifest rather than producing something shaped like an answer.
            produces_proba=False,
            **_HF_CAPABILITIES,
        ),
        search_space=_FINE_TUNING_SPACE,
        fit_defaults=_FINE_TUNING_DEFAULTS,
        params_model=HFSeq2SeqParams,
        auto_priority=10,
        description="Fine-tunes an encoder-decoder to generate text (summarize, translate).",
    )
)
