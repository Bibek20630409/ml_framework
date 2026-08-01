"""
data/sources/text.py
────────────────────
A labelled text file → :class:`~ml_framework.data.types.DataBundle`.

CSV, Parquet or JSONL with one column of raw text and one of labels. What comes
out holds **strings, not token ids** (``payload="dataset"``, a lazy
:class:`TextDataset`), and tokenization happens per batch in the preprocessor's
``collate_fn``. That ordering is the design, for two reasons:

* **Padding to the batch, not to the corpus.** Tokenizing up front means padding
  every sequence to a single length; tokenizing per batch pads to the longest
  sequence *in that batch*. On text with a long tail — which is all text — the
  difference is most of the compute.
* **The bundle stays cheap to fold.** Cross-validation builds one bundle per fold,
  and re-tokenizing the whole corpus k times to produce k partitions of it would
  be pure waste.

Two things this source refuses to guess at, because guessing wrong is silent:

* **Which column is the text.** With one obvious candidate it is used; with
  several it raises and names them. A framework that picks the first string column
  will one day train on a column of user IDs.
* **Which tokenizer.** It comes from ``model.params.model_name`` — the checkpoint
  the model itself is built from — never from a separate ``data.params`` knob.
  Tokenizing with one checkpoint's vocabulary and running another checkpoint's
  weights produces ids the model was never trained on, and nothing about the
  failure looks like a failure. See :mod:`ml_framework.data.preprocess.text`.

No torch at module scope, and none in :class:`TextDataset` either: a dataset is
``__len__`` plus ``__getitem__``, and torch's ``DataLoader`` is happy with any
object that has them. That is what lets a bare install read a text corpus, encode
its labels and lay out its folds without a deep-learning stack present.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticModel

from ...core.types import FrameworkError
from ..preprocess.text import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_LENGTH,
    Seq2SeqPreprocessor,
    TextPreprocessor,
    TokenTextPreprocessor,
)
from ..splitters import RandomSplitter
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)

# Column names that mean "this is the text" in practically every dataset that
# ships with one. Consulted only to break a tie; an explicit `params.text_col`
# always wins.
TEXT_COLUMN_HINTS: tuple[str, ...] = (
    "text",
    "tokens",
    "sentence",
    "content",
    "review",
    "body",
    "source",
)


class TextSourceParams(PydanticModel):
    """``data.params`` for the text source.

    Deliberately small. ``model_name`` and ``max_length`` are **not** here: they
    belong to the model, and duplicating them would create two places to set one
    thing — with the losing one failing silently.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Absent → sniffed, and the sniff refuses to guess between candidates.
    text_col: str | None = None


class TextDataset:
    """``(text, label)`` pairs, indexable and no more.

    Not a ``torch.utils.data.Dataset`` subclass: ``DataLoader`` duck-types on
    ``__len__``/``__getitem__``, and subclassing would put torch in the import path
    of every text config load — including the ones that only wanted to check a
    column name.
    """

    __slots__ = ("labels", "texts")

    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels disagree: {len(texts)} vs {len(labels)}")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[str, Any]:
        return self.texts[i], self.labels[i]


class PairDataset:
    """``(first, second)`` pairs, indexable and no more.

    The same shape as :class:`TextDataset` and for the same reason -- what the two
    members *are* differs by task, and the collate function is what knows. For
    token classification they are a word list and a tag list; for seq2seq, a source
    string and a target string. Making them separate classes would have produced
    three near-identical containers whose only difference was a type annotation
    the DataLoader never reads.
    """

    __slots__ = ("firsts", "seconds")

    def __init__(self, firsts: list[Any], seconds: list[Any]) -> None:
        if len(firsts) != len(seconds):
            raise ValueError(f"pair halves disagree: {len(firsts)} vs {len(seconds)}")
        self.firsts = firsts
        self.seconds = seconds

    def __len__(self) -> int:
        return len(self.firsts)

    def __getitem__(self, i: int) -> tuple[Any, Any]:
        return self.firsts[i], self.seconds[i]


def read_text_table(path: str | Path) -> Any:
    """CSV, Parquet or JSONL as a DataFrame.

    JSONL is handled here rather than in ``read_table`` because it is a text
    convention: one JSON object per line is how labelled text corpora are
    distributed, and the tabular reader has no reason to grow the case.
    """
    import pandas as pd

    from .tabular import read_table

    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    return read_table(str(path))


def _pick_text_column(frame: Any, target: str, declared: str | None) -> str:
    """Which column holds the text — or a refusal that names the candidates."""
    if declared is not None:
        if declared not in frame.columns:
            raise KeyError(f"data.params.text_col '{declared}' is not a column of the file")
        return declared

    candidates = [
        c for c in frame.columns if c != target and frame[c].dtype == object  # noqa: E721
    ]
    if not candidates:
        raise FrameworkError(
            f"no text column found beside the target '{target}'. "
            f"Set data.params.text_col explicitly."
        )
    if len(candidates) == 1:
        return str(candidates[0])
    for hint in TEXT_COLUMN_HINTS:
        if hint in candidates:
            return hint
    # Several string columns and no conventional name: refuse. Picking the first
    # would eventually train a sentiment model on a column of usernames, and it
    # would score badly rather than fail.
    raise FrameworkError(
        f"several columns could be the text ({sorted(map(str, candidates))}). "
        f"Set data.params.text_col to say which."
    )


def _encode_labels(values: Any, task: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Labels as ints, plus the class names when they had to be derived.

    Text labels arrive as strings far more often than tabular ones do
    (``positive``/``negative``, not ``0``/``1``), so encoding belongs here rather
    than being pushed onto the user. **Sorted** unique order, not order of
    appearance: a shuffled input file must not silently renumber the classes, or
    two runs over the same data produce models whose class 0 means different
    things.
    """
    import pandas as pd

    series = pd.Series(values)
    if task == "regression":
        return series.to_numpy(dtype="float32"), None
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype="int64"), None

    classes = tuple(str(c) for c in sorted(series.astype(str).unique()))
    lookup = {name: i for i, name in enumerate(classes)}
    return series.astype(str).map(lookup).to_numpy(dtype="int64"), classes


def read_text_corpus(config) -> tuple[list[str], np.ndarray, tuple[str, ...] | None]:
    """``(texts, encoded labels, class names)`` — the one place the file is parsed.

    Shared by :func:`build_text_bundle` and :func:`text_labels` so cross-validation
    stratifies on exactly the labels training will see, rather than on a second
    reading that could encode them differently.
    """
    frame, text_col, target = _read_and_locate(config)
    texts = [str(t) for t in frame[text_col].tolist()]
    labels, class_names = _encode_labels(frame[target], config.task)
    log.info("text corpus: %d rows, text column '%s'", len(texts), text_col)
    return texts, labels, class_names


def _as_sequence(cell: Any, *, column: str, row: int) -> list[str]:
    """One cell holding a sequence, whether the file stored it as a list or a string.

    JSONL keeps ``["Ada", "works"]`` a list; a CSV export of the same corpus
    flattens it to ``"Ada works"``. Both are common enough that refusing either
    would be picking a favourite file format rather than a data model.
    """
    if isinstance(cell, str):
        return cell.split()
    try:
        return [str(v) for v in cell]
    except TypeError:
        raise FrameworkError(
            f"row {row} of column '{column}' is {type(cell).__name__}, not a sequence. "
            f"Token classification needs one list of words and one list of tags per row."
        ) from None


def read_token_corpus(config) -> tuple[list[list[str]], list[list[int]], tuple[str, ...]]:
    """``(word lists, encoded tag lists, tag names)`` for token classification.

    The tag vocabulary is built across the **whole file**, sorted, before any
    split. Deriving it per split would let a tag that appears only in the test set
    get a different integer there than in training -- or no integer at all -- which
    is a silent relabelling rather than an error.
    """
    frame, text_col, target = _read_and_locate(config)

    words = [_as_sequence(cell, column=text_col, row=i) for i, cell in enumerate(frame[text_col])]
    raw_tags = [
        _as_sequence(cell, column=str(target), row=i) for i, cell in enumerate(frame[target])
    ]
    for i, (row_words, row_tags) in enumerate(zip(words, raw_tags, strict=True)):
        if len(row_words) != len(row_tags):
            raise FrameworkError(
                f"row {i} has {len(row_words)} words but {len(row_tags)} tags. "
                f"Token classification needs one tag per word."
            )

    names = tuple(sorted({tag for row in raw_tags for tag in row}))
    lookup = {name: i for i, name in enumerate(names)}
    tags = [[lookup[t] for t in row] for row in raw_tags]
    log.info("token corpus: %d sentences, %d tags %s", len(words), len(names), names)
    return words, tags, names


def read_seq2seq_corpus(config) -> tuple[list[str], list[str]]:
    """``(sources, targets)`` -- both strings, neither encoded.

    No label encoding at all, which is the point: the target of a generative model
    is text, and turning it into a class index is the mistake that makes seq2seq
    look like classification with a very large number of classes.
    """
    frame, text_col, target = _read_and_locate(config)
    sources = [str(v) for v in frame[text_col].tolist()]
    targets = [str(v) for v in frame[target].tolist()]
    log.info("seq2seq corpus: %d pairs, '%s' -> '%s'", len(sources), text_col, target)
    return sources, targets


def _read_and_locate(config) -> tuple[Any, str, str]:
    """``(frame, text column, target column)`` -- the checks all three tasks share."""
    if config.data.path is None:
        raise ValueError("text data requires data.path")
    target = config.data.target
    if not target:
        raise ValueError("text data requires data.target")

    params = TextSourceParams.model_validate(dict(config.data.params))
    frame = read_text_table(config.data.path)
    if target not in frame.columns:
        raise KeyError(f"data.target '{target}' not in the file's columns")
    return frame, _pick_text_column(frame, target, params.text_col), str(target)


def text_labels(config) -> np.ndarray:
    """The corpus labels, for stratifying cross-validation folds."""
    _, labels, _ = read_text_corpus(config)
    return labels


def _tokenizer_settings(config) -> tuple[str, int]:
    """The checkpoint and truncation length, read off ``model.params``.

    The single source of truth for both. A text config that names no model falls
    back to the preprocessor's defaults, which is what makes ``kind: text`` usable
    before a model has been chosen (``mlf train`` with zero-config selection, and
    every test that only wants to look at the data).
    """
    model_params = dict(getattr(config.model, "params", {}) or {})
    return (
        str(model_params.get("model_name") or DEFAULT_MODEL_NAME),
        int(model_params.get("max_length") or DEFAULT_MAX_LENGTH),
    )


def _partition(config, n: int, labels: Any, indices: Any) -> Any:
    """The configured split, or the one cross-validation injected."""
    if indices is not None:
        return indices
    split_cfg = config.data.split
    return RandomSplitter(
        seed=config.runtime.seed,
        task=config.task,
        val_size=split_cfg.val_size,
        test_size=split_cfg.test_size,
    ).split(n, y=labels)


def build_text_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize a text :class:`DataBundle` from a validated config.

    Three shapes behind one ``data.kind``, chosen by ``config.task``, because the
    *kind* is a statement about the data being text and the *task* is what decides
    what a label is: one class per document, one tag per token, or another string.
    Splitting these into three data kinds would have made ``kind`` and ``task``
    encode the same fact twice.

    ``indices`` replaces the configured split with a ready-made partition, which
    is how cross-validation gets one bundle per fold.
    """
    if config.task == "token_classification":
        return _build_token_bundle(config, indices=indices)
    if config.task == "seq2seq":
        return _build_seq2seq_bundle(config, indices=indices)

    texts, labels, derived_names = read_text_corpus(config)
    model_name, max_length = _tokenizer_settings(config)

    parts = _partition(config, len(texts), labels, indices)

    def subset(idx: Any) -> TextDataset:
        rows = np.asarray(idx, dtype="int64")
        return TextDataset([texts[i] for i in rows], labels[rows])

    train_ds = subset(parts.train)
    preprocessor = TextPreprocessor(model_name=model_name, max_length=max_length)

    # Imbalance is corrected by *sampling*, exactly as the image source does it,
    # and for the same reason: both hand the loop a lazy dataset, so a
    # WeightedRandomSampler is the correction that does not require materializing
    # anything. Corrected once — hence `class_weights=None` below.
    train_labels = train_ds.labels
    class_weights = None
    sample_weights = None
    if config.task != "regression":
        counts = Counter(int(c) for c in train_labels)
        total = sum(counts.values())
        w_map = {c: total / n for c, n in counts.items()}
        sample_weights = np.asarray([w_map[int(c)] for c in train_labels], dtype="float64")

    if config.task == "multiclass":
        output_dim = int(len(np.unique(labels)))
    else:
        output_dim = 1

    class_names = tuple(config.data.class_names) if config.data.class_names else derived_names
    schema = FeatureSchema(
        target_name=config.data.target,
        class_names=class_names,
    )
    log.info(
        "text splits: train=%d val=%d test=%d output_dim=%d tokenizer=%s",
        len(parts.train),
        len(parts.val),
        len(parts.test),
        output_dim,
        model_name,
    )

    return DataBundle(
        train=Split(payload="dataset", x=train_ds, y=train_ds.labels),
        val=Split(payload="dataset", x=subset(parts.val), y=labels[parts.val], index=parts.val),
        test=Split(payload="dataset", x=subset(parts.test), y=labels[parts.test], index=parts.test),
        schema=schema,
        task=config.task,
        data_kind="text",
        # A token sequence has no fixed width, so there is no feature count to
        # report. Recorded as 0 rather than as `max_length`, which would be a
        # number the serving layer could check requests against and be wrong.
        input_dim=0,
        output_dim=output_dim,
        class_weights=class_weights,
        preprocessor=preprocessor,
        # Drift over token ids is a number without a meaning; /drift answers 501
        # for text rather than computing one.
        reference_stats=None,
        meta={} if sample_weights is None else {"sample_weights": sample_weights},
    )


def _build_token_bundle(config, *, indices: Any = None) -> DataBundle:
    """One label per token: NER, POS tagging.

    ``output_dim`` is the size of the **tag** vocabulary, and the head is one
    logit per tag at every position -- which is why the binary head convention
    (one logit, two classes) does not apply here even for a two-tag scheme. A
    2-tag corpus gets a 2-wide head, because the loss is cross-entropy over
    positions rather than a single sigmoid.
    """
    words, tags, tag_names = read_token_corpus(config)
    model_name, max_length = _tokenizer_settings(config)
    # Unstratified: a sentence has no single label to balance folds against. The
    # splitter knows this from the task rather than being told here.
    parts = _partition(config, len(words), None, indices)

    def subset(idx: Any) -> PairDataset:
        rows = np.asarray(idx, dtype="int64")
        return PairDataset([words[i] for i in rows], [tags[i] for i in rows])

    preprocessor = TokenTextPreprocessor(model_name=model_name, max_length=max_length)
    schema = FeatureSchema(target_name=config.data.target, class_names=tag_names)
    log.info(
        "token splits: train=%d val=%d test=%d tags=%d",
        len(parts.train),
        len(parts.val),
        len(parts.test),
        len(tag_names),
    )

    return DataBundle(
        train=Split(payload="dataset", x=subset(parts.train), index=parts.train),
        val=Split(payload="dataset", x=subset(parts.val), index=parts.val),
        test=Split(payload="dataset", x=subset(parts.test), index=parts.test),
        schema=schema,
        task=config.task,
        data_kind="text",
        input_dim=0,
        output_dim=len(tag_names),
        # No sampler either: a WeightedRandomSampler weights whole *sentences*,
        # and the imbalance in token tagging is between tags within a sentence.
        # Reweighting sentences would not touch it, so nothing pretends to.
        class_weights=None,
        preprocessor=preprocessor,
        reference_stats=None,
        meta={"tag_names": tag_names},
    )


def _build_seq2seq_bundle(config, *, indices: Any = None) -> DataBundle:
    """Source string in, target string out.

    ``output_dim`` is 0, not the vocabulary size. The head *is* the vocabulary and
    the checkpoint owns it; recording a number here would be recording something
    the framework does not choose and cannot check.
    """
    sources, targets = read_seq2seq_corpus(config)
    model_name, max_length = _tokenizer_settings(config)
    target_length = int(
        dict(getattr(config.model, "params", {}) or {}).get("max_target_length")
        or DEFAULT_TARGET_LENGTH
    )
    parts = _partition(config, len(sources), None, indices)

    def subset(idx: Any) -> PairDataset:
        rows = np.asarray(idx, dtype="int64")
        return PairDataset([sources[i] for i in rows], [targets[i] for i in rows])

    preprocessor = Seq2SeqPreprocessor(
        model_name=model_name, max_length=max_length, max_target_length=target_length
    )
    log.info(
        "seq2seq splits: train=%d val=%d test=%d", len(parts.train), len(parts.val), len(parts.test)
    )

    return DataBundle(
        train=Split(payload="dataset", x=subset(parts.train), index=parts.train),
        val=Split(payload="dataset", x=subset(parts.val), index=parts.val),
        test=Split(payload="dataset", x=subset(parts.test), index=parts.test),
        schema=FeatureSchema(target_name=config.data.target),
        task=config.task,
        data_kind="text",
        input_dim=0,
        output_dim=0,
        class_weights=None,
        preprocessor=preprocessor,
        reference_stats=None,
        meta={},
    )


__all__ = [
    "TEXT_COLUMN_HINTS",
    "PairDataset",
    "read_seq2seq_corpus",
    "read_token_corpus",
    "TextDataset",
    "TextSourceParams",
    "build_text_bundle",
    "read_text_corpus",
    "read_text_table",
    "text_labels",
]
