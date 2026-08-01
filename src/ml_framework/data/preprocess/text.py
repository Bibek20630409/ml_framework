"""
data/preprocess/text.py
───────────────────────
The tokenizer, as a preprocessor — which is the only place it can correctly live.

A tokenizer is fitted state in every sense that matters. It maps strings to
integer ids through a vocabulary, and a model trained against one vocabulary
produces confident nonsense when fed ids from another. That failure is
**completely silent**: no shape error, no exception, just a degraded model that
looks like it merely trained badly. Treating the tokenizer as configuration
("serving re-downloads ``distilbert-base-uncased``") is how train/serve skew of
this kind happens, so it round-trips through the bundle like a fitted scaler:

    bundle/preprocessor/
        preprocessor.json          {"class": "…:TextPreprocessor", "files": ["tokenizer"], …}
        tokenizer/                 vocab, merges, special tokens, the added tokens

:meth:`TextPreprocessor._read` loads from **that directory**, never from the hub.
The consequence worth stating: a serving container needs no network and no HF
cache, and a tokenizer the training run modified (an added domain token, a resized
vocabulary) is the one serving uses.

Two jobs, one class, because they must agree:

* ``collate_fn`` — batches of raw strings → padded token ids, at DataLoader time.
  Padding to the longest sequence *in the batch* rather than to ``max_length``
  means short batches stay short; on typical text that is most of them.
* ``transform`` — the serving path, one list of strings → the same encoding.

``transformers`` is imported inside the methods, so this module stays importable
on an install without the ``[nlp]`` extra — the rule every optional-runtime module
in the framework follows, and the reason the serving path can import the whole
data layer without pulling a deep-learning stack in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.plugins import check_requirements
from ...core.types import IGNORE_INDEX, Requirement
from .base import BasePreprocessor, PreprocessorError

# Declared here rather than in the plugin because this is the lowest module that
# needs them: the *source* builds a tokenizer before any model exists. The plugin
# imports this tuple rather than restating it, so one edit changes both the
# availability check and the `pip install` hint the user is told to run.
NLP_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("transformers", extra="nlp", min_version="4.40"),
    Requirement("tokenizers", extra="nlp"),
)

# The subdirectory the tokenizer occupies inside `bundle/preprocessor/`.
TOKENIZER_DIR = "tokenizer"

# A small, fast, widely-available encoder. Chosen as the default because a text
# run that has to name a checkpoint before it can start is not a default at all.
DEFAULT_MODEL_NAME = "distilbert-base-uncased"

# 128 tokens covers a sentence, a review title, a support ticket subject. Cost
# grows quadratically with length in an attention stack, so the default is the
# short end and lengthening it is a deliberate act.
DEFAULT_MAX_LENGTH = 128

# Generation targets are short -- a summary, a translation, a normalized field --
# and the decode loop runs one step per token, so this is the axis that decides
# how long evaluating a seq2seq model takes.
DEFAULT_TARGET_LENGTH = 64


class TextPreprocessor(BasePreprocessor):
    """An HF tokenizer plus the two lengths that decide what it emits.

    ``model_name`` is the *source* of the vocabulary, not merely a label: it has to
    be the checkpoint the model itself was built from. The text source enforces
    that by reading both from ``model.params`` — see
    :func:`~ml_framework.data.sources.text.build_text_bundle`.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        # Populated lazily, or by `_read` from a bundle. `_read` winning is the
        # whole point: see the module docstring.
        self._tokenizer: Any = None

    def params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "max_length": self.max_length}

    # ── the tokenizer ──
    @property
    def tokenizer(self) -> Any:
        """The tokenizer, loaded on first use.

        Lazy because constructing a ``TextPreprocessor`` happens in places where
        the tokenizer is not needed — listing a model, validating a config — and
        the load touches disk and possibly the network.
        """
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer(self.model_name)
        return self._tokenizer

    @staticmethod
    def _load_tokenizer(source: str | Path) -> Any:
        """``AutoTokenizer.from_pretrained``, with a pip command instead of an
        ``ImportError`` from four frames deep."""
        check_requirements(NLP_REQUIREMENTS, what="text data")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(source))

    def encode(self, texts: Sequence[str]) -> Any:
        """Strings → a batch encoding of torch tensors.

        ``padding=True`` pads to the longest sequence in *this* batch rather than
        to ``max_length``; ``truncation`` then caps it. The two together are what
        keep a batch of tweets from being padded out to 128 tokens of nothing.
        """
        return self.tokenizer(
            [str(t) for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """The serving path: raw text in, model input out.

        ``Inferencer.predict`` calls this before handing the result to the
        estimator, which is why ``/predict`` can accept ``{"inputs": ["…"]}`` and
        the estimator never learns what a string is.
        """
        if isinstance(x, str):
            x = [x]
        return self.encode(list(x))

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any]:
        """Batch ``(text, label)`` pairs into ``(encoding, labels)``.

        A 2-tuple specifically, because that is what ``BaseModel._shared_step``
        unpacks. The first element being a *dict* of tensors rather than a single
        tensor is the only text-shaped thing about it, and the model's ``forward``
        is where that is understood.
        """
        return self._collate

    def _collate(self, batch: Sequence[Any]) -> Any:
        import torch

        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encoding = self.encode(texts)
        # int64 for classification's CrossEntropy/BCE label handling, which is what
        # every task this plugin declares uses. A regression head would need
        # float32 here; `nlp.hf_text` does not declare `regression`, so the day it
        # does this line is part of that change.
        return encoding, torch.as_tensor(labels, dtype=torch.long)

    # ── round-trip ──
    def _write(self, dest: Path) -> list[str]:
        self.tokenizer.save_pretrained(dest / TOKENIZER_DIR)
        return [TOKENIZER_DIR]

    def _read(self, src: Path, spec: Mapping[str, Any]) -> None:
        """Restore from the bundle's own directory — **never** from the hub.

        A missing directory is an error rather than a quiet fall back to
        ``from_pretrained(self.model_name)``: that fallback would silently swap in
        a different vocabulary than the one the model was trained against, which is
        precisely the failure this class exists to prevent. Failing loudly at load
        time beats scoring wrong at request time.
        """
        path = Path(src) / TOKENIZER_DIR
        if not path.is_dir():
            raise PreprocessorError(
                f"the bundle has no tokenizer at {path}. Refusing to fall back to "
                f"'{self.model_name}' from the hub: a different vocabulary would "
                f"produce token ids this model was never trained on, silently."
            )
        self._tokenizer = self._load_tokenizer(path)


class TokenTextPreprocessor(TextPreprocessor):
    """Tokenizer plus **word to sub-word label alignment**, for per-token tagging.

    The alignment is the entire content of this class, and the reason token
    classification is not just classification with a bigger output. A corpus is
    tagged per *word* (``["Ada", "works", "at", "Acme"]`` becomes
    ``["B-PER", "O", "O", "B-ORG"]``), but the model consumes *sub-words*, and
    "Acme" may arrive as ``["ac", "##me"]``. Something has to decide what the
    second piece is labelled.

    **The first piece gets the word's tag; every continuation gets
    ``IGNORE_INDEX``.** The tempting alternative -- repeat the tag across all
    pieces -- is wrong in a way that flatters the score: one word's single
    decision then counts once per piece, so the corpus is silently re-weighted
    toward whichever words the tokenizer splits most, which is exactly the rare
    proper nouns entity recognition is about. It also makes the reported figure
    incomparable with anything published.

    Truncation is the other trap: a sentence cut at ``max_length`` loses its tail
    words, and their tags must go with them. Alignment reads ``word_ids()`` off
    the encoding rather than counting the original list, so truncation is handled
    by construction rather than by a length check that can drift out of step.
    """

    def _require_fast(self) -> Any:
        """Fast tokenizers only -- ``word_ids()`` exists nowhere else.

        A clear refusal here beats an ``AttributeError`` inside the collate
        function on the first batch, which is where this would otherwise surface.
        """
        tokenizer = self.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise PreprocessorError(
                f"token classification needs a fast tokenizer to align word tags to "
                f"sub-words, and '{self.model_name}' resolved to a slow one. Pick a "
                f"checkpoint that ships a tokenizer.json."
            )
        return tokenizer

    def encode_words(self, word_lists: Sequence[Sequence[str]]) -> Any:
        """Pre-split words to a batch encoding that remembers word boundaries."""
        return self._require_fast()(
            [[str(w) for w in words] for words in word_lists],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def align_tags(self, encoding: Any, tag_lists: Sequence[Sequence[int]]) -> Any:
        """Word-level tags to one label per sub-word, padded with ``IGNORE_INDEX``."""
        import torch

        aligned: list[list[int]] = []
        for row, tags in enumerate(tag_lists):
            previous: int | None = None
            labels: list[int] = []
            for word_id in encoding.word_ids(batch_index=row):
                if word_id is None:
                    # [CLS], [SEP], padding: not a word, so not a decision.
                    labels.append(IGNORE_INDEX)
                elif word_id != previous:
                    labels.append(int(tags[word_id]))
                else:
                    labels.append(IGNORE_INDEX)
                previous = word_id
            aligned.append(labels)
        return torch.tensor(aligned, dtype=torch.long)

    def transform(self, x: Any) -> Any:
        """Serving: one sentence or several.

        Raw strings are split on whitespace so a caller need not pre-tokenize;
        pre-split word lists pass through. Both shapes are accepted because both
        are unambiguous, which a bare list of strings would not be if the class
        also took a single pre-split sentence.
        """
        if isinstance(x, str):
            return self.encode_words([x.split()])
        rows = list(x)
        if rows and isinstance(rows[0], str):
            return self.encode_words([str(r).split() for r in rows])
        return self.encode_words(rows)

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any]:
        return self._collate

    def _collate(self, batch: Sequence[Any]) -> Any:
        encoding = self.encode_words([item[0] for item in batch])
        return encoding, self.align_tags(encoding, [item[1] for item in batch])


class Seq2SeqPreprocessor(TextPreprocessor):
    """Source and target text, encoded as one batch.

    Two lengths, not one: a summary is far shorter than its input, and truncating
    both at ``max_length`` would either waste most of the target budget or cut the
    source in half. ``max_target_length`` exists for that reason, and it is also
    the knob that decides how long evaluation takes -- generation runs one decoder
    step per token.

    Pad positions in the target become ``IGNORE_INDEX`` before the loss sees them.
    Left as the pad id they are *learnable*, and since padding is most of a short
    target in a mixed batch, a model could score well on the loss by learning to
    predict nothing.

    ``labels`` rides **inside** the encoding rather than beside it, because
    transformers builds ``decoder_input_ids`` from it -- the right-shift that makes
    teacher forcing work. Performing that shift by hand is a well-known way to
    train a model that predicts the token it was just handed.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_target_length: int = DEFAULT_TARGET_LENGTH,
    ) -> None:
        super().__init__(model_name=model_name, max_length=max_length)
        self.max_target_length = int(max_target_length)

    def params(self) -> dict[str, Any]:
        return {**super().params(), "max_target_length": self.max_target_length}

    def encode_targets(self, targets: Sequence[str]) -> Any:
        import torch

        labels = self.tokenizer(
            text_target=[str(t) for t in targets],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return labels
        return torch.where(labels == pad_id, torch.full_like(labels, IGNORE_INDEX), labels)

    def decode(self, token_ids: Any) -> list[str]:
        """Generated -- or reference -- ids back to strings.

        ``IGNORE_INDEX`` is not a token id, so reference labels must be put back to
        the pad id before they can be decoded. That is the inverse of what
        :meth:`encode_targets` did, and it is easy to forget until the references
        come out of the tokenizer as an index error.
        """
        import torch

        ids = torch.as_tensor(token_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        ids = torch.where(ids == IGNORE_INDEX, torch.full_like(ids, pad_id), ids)
        return [str(t) for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any]:
        return self._collate

    def _collate(self, batch: Sequence[Any]) -> Any:
        encoding = self.encode([item[0] for item in batch])
        labels = self.encode_targets([item[1] for item in batch])
        encoding["labels"] = labels
        return encoding, labels


__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TARGET_LENGTH",
    "NLP_REQUIREMENTS",
    "TOKENIZER_DIR",
    "Seq2SeqPreprocessor",
    "TextPreprocessor",
    "TokenTextPreprocessor",
]
