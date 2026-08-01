"""The tokenizer round-trip — P7's first exit gate.

A tokenizer is fitted state. It maps strings to integer ids through a vocabulary,
and a model trained against one vocabulary produces confident nonsense when fed
ids from another. That failure is completely silent: no shape error, no exception,
just a model that looks like it trained badly.

So it is not enough for a bundle to *have* a tokenizer directory. What has to be
true is that loading the bundle uses **that directory** rather than re-fetching
``model_name`` from the hub, and the test below proves it the only way that
distinguishes the two: by modifying the tokenizer before saving and checking the
modification survives.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="the nlp extra is not installed")

from ml_framework.data.preprocess import load_preprocessor  # noqa: E402
from ml_framework.data.preprocess.base import PreprocessorError  # noqa: E402
from ml_framework.data.preprocess.text import TOKENIZER_DIR, TextPreprocessor  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
MARKER = "[[a_token_the_base_vocabulary_does_not_have]]"


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor(model_name=TINY_MODEL, max_length=16)


# ── The exit gate ─────────────────────────────────────────
@pytest.mark.unit
def test_the_saved_tokenizer_is_the_one_that_comes_back(preprocessor, tmp_path):
    """The bundle's tokenizer, not a fresh download of ``model_name``.

    A vocabulary the training run modified — an added domain token, a resized
    vocabulary — is the one serving must use. Asserting only that *a* tokenizer
    loads would pass just as happily if the loader silently went back to the hub,
    which is the bug this gate exists to catch.
    """
    preprocessor.tokenizer.add_tokens([MARKER])
    assert len(preprocessor.tokenizer.tokenize(MARKER)) == 1  # known before saving

    fragment = preprocessor.save(tmp_path / "preprocessor")
    restored = load_preprocessor(tmp_path / "preprocessor", fragment)

    assert restored.tokenizer.tokenize(MARKER) == [MARKER]
    # And the base checkpoint does *not* know it — which is what makes the
    # assertion above evidence rather than a coincidence.
    fresh = TextPreprocessor(model_name=TINY_MODEL)
    assert fresh.tokenizer.tokenize(MARKER) != [MARKER]


@pytest.mark.unit
def test_the_manifest_fragment_names_the_class_and_its_directory(preprocessor, tmp_path):
    """Nothing outside a preprocessor reads a preprocessor's files, so the
    fragment is the whole interface."""
    fragment = preprocessor.save(tmp_path / "preprocessor")

    assert fragment["class"].endswith(":TextPreprocessor")
    assert fragment["files"] == [TOKENIZER_DIR]
    assert fragment["params"] == {"model_name": TINY_MODEL, "max_length": 16}
    assert (tmp_path / "preprocessor" / TOKENIZER_DIR).is_dir()


@pytest.mark.unit
def test_a_bundle_with_no_tokenizer_directory_refuses(preprocessor, tmp_path):
    """Loudly at load time beats scoring wrong at request time.

    Falling back to ``from_pretrained(model_name)`` here would swap in a different
    vocabulary than the model was trained against — precisely the failure this
    class exists to prevent, and one that produces no error at all.
    """
    fragment = preprocessor.save(tmp_path / "preprocessor")
    for path in sorted((tmp_path / "preprocessor" / TOKENIZER_DIR).iterdir(), key=lambda p: p.name):
        path.unlink()
    (tmp_path / "preprocessor" / TOKENIZER_DIR).rmdir()

    with pytest.raises(PreprocessorError, match="no tokenizer"):
        load_preprocessor(tmp_path / "preprocessor", fragment)


# ── Batching ──────────────────────────────────────────────
@pytest.mark.unit
def test_collate_pads_to_the_batch_not_to_max_length():
    """A short batch stays short.

    Stated as a comparison between two batches rather than as "narrower than
    ``max_length``", because that would also pass for a tokenizer whose padding
    happened to be capped by truncation. What matters is that the width *follows
    the batch*: attention cost grows with the padded length, so padding every
    batch of tweets out to the corpus maximum is most of the compute.
    """
    pre = TextPreprocessor(model_name=TINY_MODEL, max_length=128)

    short, labels = pre.collate_fn([("hi", 0), ("ok", 1)])
    long, _ = pre.collate_fn([("hi", 0), ("a considerably longer document here", 1)])

    assert short["input_ids"].shape[0] == 2
    assert short["input_ids"].shape[1] < long["input_ids"].shape[1] <= pre.max_length
    assert labels.tolist() == [0, 1]


@pytest.mark.unit
def test_collate_returns_a_two_tuple_the_shared_step_can_unpack(preprocessor):
    """``BaseModel._shared_step`` does ``x, y = batch``, unchanged for text. Only
    ``x`` being a dict is new, and the model's ``forward`` is where that lands."""
    from collections.abc import Mapping

    x, y = preprocessor.collate_fn([("a", 0), ("b", 1)])
    assert isinstance(x, Mapping)
    assert {"input_ids", "attention_mask"} <= set(x)
    assert y.dtype.is_signed and y.numel() == 2


@pytest.mark.unit
def test_longer_than_max_length_is_truncated(preprocessor):
    encoding = preprocessor.transform(["word " * 200])
    assert encoding["input_ids"].shape[1] == preprocessor.max_length


@pytest.mark.unit
def test_a_bare_string_is_treated_as_one_row(preprocessor):
    """The serving path hands over a list, but a caller with one document should
    not have to remember to wrap it."""
    assert preprocessor.transform("hello").input_ids.shape[0] == 1


@pytest.mark.unit
def test_constructing_one_does_not_load_a_tokenizer(preprocessor):
    """Lazy on purpose: listing a model or validating a config constructs these,
    and neither should touch the disk or the network."""
    assert preprocessor._tokenizer is None
