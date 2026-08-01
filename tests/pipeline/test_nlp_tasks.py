"""Token classification and seq2seq, end to end.

Both tasks were in the ``Task`` literal with no ``TaskSpec`` row, so a config
naming either was refused at load. The fix is not the row — a row alone would
trade an honest refusal for a confusing failure somewhere inside the fit loop.
What makes the row *true* is a source, a model, a loss, an evaluation path and a
response shape for each, and that is what this file exercises.

The two hard parts, which most of these tests are about:

* **Word-to-sub-word alignment.** A corpus is tagged per word; a model consumes
  sub-words. Getting this wrong does not raise — it inflates the score.
* **Teacher forcing versus generation.** A seq2seq model is trained one way and
  evaluated another, and the two numbers can move in opposite directions.

Everything runs against tiny random checkpoints. They learn nothing, which is the
point: these tests are about plumbing, and a real checkpoint would buy nothing but
a download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers", reason="the nlp extra is not installed")
pytest.importorskip("pytorch_lightning", reason="the lightning extra is not installed")

from ml_framework.config import ExperimentConfig  # noqa: E402
from ml_framework.core.inference import Inferencer  # noqa: E402
from ml_framework.core.task import get_task_spec, has_task_spec  # noqa: E402
from ml_framework.core.types import IGNORE_INDEX  # noqa: E402
from ml_framework.pipeline import train  # noqa: E402

TOKEN_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
SEQ_MODEL = "hf-internal-testing/tiny-random-BartForConditionalGeneration"

SENTENCES = [
    (["Ada", "works", "at", "Acme"], ["B-PER", "O", "O", "B-ORG"]),
    (["Bob", "left", "Globex"], ["B-PER", "O", "B-ORG"]),
    (["the", "meeting", "is", "today"], ["O", "O", "O", "O"]),
    (["Carol", "joined", "Initech"], ["B-PER", "O", "B-ORG"]),
]
PAIRS = [
    ("summarize: the cat sat on the mat", "cat on mat"),
    ("summarize: the dog ran in the park", "dog in park"),
    ("summarize: a bird flew over the lake", "bird over lake"),
    ("summarize: the fish swam in the bowl", "fish in bowl"),
]


@pytest.fixture
def ner_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "ner.jsonl"
    rows = [{"tokens": t, "tags": g} for t, g in SENTENCES * 8]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def pairs_csv(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.csv"
    pd.DataFrame([{"source": s, "target": t} for s, t in PAIRS * 8]).to_csv(path, index=False)
    return path


def ner_config(path: Path, out: Path, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig.model_validate(
        {
            "task": "token_classification",
            "runtime": {
                "output_dir": str(out),
                "seed": 0,
                "num_workers": 0,
                "accelerator": "cpu",
            },
            "data": {
                "kind": "text",
                "path": str(path),
                "target": "tags",
                "split": {"val_size": 0.2, "test_size": 0.2},
            },
            "model": {
                "name": "nlp.hf_token",
                "params": {"model_name": TOKEN_MODEL, "max_length": 32},
            },
            "fit": {"budget": {"max_epochs": 1}, "batch_size": 4},
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
    )
    return cfg.with_overrides(overrides) if overrides else cfg


def seq_config(path: Path, out: Path, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig.model_validate(
        {
            "task": "seq2seq",
            "runtime": {
                "output_dir": str(out),
                "seed": 0,
                "num_workers": 0,
                "accelerator": "cpu",
            },
            "data": {
                "kind": "text",
                "path": str(path),
                "target": "target",
                "split": {"val_size": 0.2, "test_size": 0.2},
            },
            "model": {
                "name": "nlp.hf_seq2seq",
                "params": {
                    "model_name": SEQ_MODEL,
                    "max_length": 32,
                    "max_target_length": 12,
                },
            },
            "fit": {"budget": {"max_epochs": 1}, "batch_size": 4},
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
    )
    return cfg.with_overrides(overrides) if overrides else cfg


# ── The fix itself ────────────────────────────────────────
@pytest.mark.unit
def test_both_tasks_are_registered_and_no_longer_refused(ner_jsonl, pairs_csv, tmp_path):
    """The reported defect: in the `Task` literal, absent from the table."""
    assert has_task_spec("token_classification")
    assert has_task_spec("seq2seq")

    # Loading a config is where the refusal used to happen.
    assert ner_config(ner_jsonl, tmp_path / "a").task == "token_classification"
    assert seq_config(pairs_csv, tmp_path / "b").task == "seq2seq"


@pytest.mark.unit
def test_multilabel_is_still_refused_and_that_is_the_rule():
    """A row means the framework can run the task.

    `multilabel` has no source, no model and no loss, so it has no row — an honest
    refusal at load time rather than a confusing failure inside the fit loop. This
    test exists so the rule is not quietly abandoned the next time somebody wants
    a config to validate.
    """
    from ml_framework.core.task import UnknownTaskError

    assert not has_task_spec("multilabel")
    with pytest.raises(UnknownTaskError, match="multilabel"):
        get_task_spec("multilabel")


@pytest.mark.unit
def test_a_tagger_is_not_reported_as_ordinary_classification():
    """`is_classification` stays False for token tagging even though it predicts
    classes: every consumer that branches on it — the confusion matrix,
    predictions.csv, the serving response — would produce the wrong shape."""
    assert get_task_spec("token_classification").is_classification is False
    assert get_task_spec("multiclass").is_classification is True


@pytest.mark.unit
def test_macro_f1_leads_for_tagging_not_accuracy():
    """`O` dominates a tagging corpus, so a model that predicts "not an entity"
    everywhere scores ~90% accuracy and is worth nothing."""
    assert get_task_spec("token_classification").primary_metric == "f1"


# ── Word to sub-word alignment ────────────────────────────
@pytest.mark.unit
def test_only_the_first_sub_word_of_a_word_carries_its_tag():
    """The alignment rule, checked against the tokenizer's own pieces."""
    from ml_framework.data.preprocess.text import TokenTextPreprocessor

    pre = TokenTextPreprocessor(model_name=TOKEN_MODEL, max_length=32)
    words, tags = ["Ada", "works"], [1, 0]
    encoding, labels = pre.collate_fn([(words, tags)])

    word_ids = encoding.word_ids(batch_index=0)
    row = labels[0].tolist()
    for position, (word_id, label) in enumerate(zip(word_ids, row, strict=True)):
        if word_id is None:
            assert label == IGNORE_INDEX, f"special token at {position} was labelled"
        elif word_id != word_ids[position - 1]:
            assert label == tags[word_id]
        else:
            assert label == IGNORE_INDEX, "a sub-word continuation was labelled"


@pytest.mark.unit
def test_a_words_tag_is_not_repeated_across_its_pieces():
    """The bug this rule exists to prevent, stated as a count.

    Repeating the tag would make one word's single decision count once per piece,
    re-weighting the corpus toward whichever words the tokenizer splits most —
    which is exactly the rare proper nouns entity recognition is about. The
    resulting score is higher and comparable with nothing.
    """
    from ml_framework.data.preprocess.text import TokenTextPreprocessor

    pre = TokenTextPreprocessor(model_name=TOKEN_MODEL, max_length=32)
    words = ["Acme", "Globex"]
    encoding, labels = pre.collate_fn([(words, [1, 2])])

    scored = int((labels[0] != IGNORE_INDEX).sum())
    pieces = int((encoding["attention_mask"][0] == 1).sum())
    assert scored == len(words), "one label per word, not per piece"
    assert pieces > len(words), "this tokenizer really does split these words"


@pytest.mark.unit
def test_a_word_past_max_length_takes_its_tag_with_it():
    """Truncation is handled by reading `word_ids()` rather than by a separate
    length check that could drift out of step with the encoder."""
    from ml_framework.data.preprocess.text import TokenTextPreprocessor

    pre = TokenTextPreprocessor(model_name=TOKEN_MODEL, max_length=8)
    words = [f"word{i}" for i in range(40)]
    _, labels = pre.collate_fn([(words, [1] * 40)])

    assert labels.shape[1] == 8
    assert int((labels[0] != IGNORE_INDEX).sum()) < len(words)


@pytest.mark.unit
def test_a_slow_tokenizer_is_refused_with_a_reason():
    """`word_ids()` exists only on fast tokenizers, and an AttributeError on the
    first batch is a worse way to learn that."""
    from ml_framework.data.preprocess.base import PreprocessorError
    from ml_framework.data.preprocess.text import TokenTextPreprocessor

    pre = TokenTextPreprocessor(model_name=TOKEN_MODEL)

    class _Slow:
        is_fast = False

    pre._tokenizer = _Slow()
    with pytest.raises(PreprocessorError, match="fast tokenizer"):
        pre.encode_words([["a"]])


# ── The token corpus ──────────────────────────────────────
@pytest.mark.unit
def test_the_tag_vocabulary_is_built_over_the_whole_corpus(ner_jsonl, tmp_path):
    """Per-split vocabularies would give a tag a different integer in test than in
    training — a silent relabelling rather than an error."""
    from ml_framework.data.sources.text import read_token_corpus

    words, tags, names = read_token_corpus(ner_config(ner_jsonl, tmp_path / "o"))

    assert names == ("B-ORG", "B-PER", "O")  # sorted, not order of appearance
    assert len(words) == len(tags) == len(SENTENCES) * 8
    assert all(len(w) == len(t) for w, t in zip(words, tags, strict=True))


@pytest.mark.unit
def test_a_tag_count_that_disagrees_with_the_word_count_raises(tmp_path):
    """Silently zipping short would drop the tail of every long sentence."""
    from ml_framework.core.types import FrameworkError
    from ml_framework.data.sources.text import read_token_corpus

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"tokens": ["a", "b", "c"], "tags": ["O", "O"]}), encoding="utf-8")

    with pytest.raises(FrameworkError, match="3 words but 2 tags"):
        read_token_corpus(ner_config(path, tmp_path / "o"))


@pytest.mark.unit
def test_a_csv_may_store_the_sequences_as_strings(tmp_path):
    """JSONL keeps lists; a CSV export of the same corpus flattens them. Refusing
    either would be picking a favourite file format rather than a data model."""
    from ml_framework.data.sources.text import read_token_corpus

    path = tmp_path / "flat.csv"
    pd.DataFrame({"tokens": ["Ada works"], "tags": ["B-PER O"]}).to_csv(path, index=False)

    words, tags, names = read_token_corpus(ner_config(path, tmp_path / "o"))
    assert words == [["Ada", "works"]]
    assert names == ("B-PER", "O")
    assert tags == [[0, 1]]


@pytest.mark.unit
def test_the_head_is_the_tag_count_even_for_two_tags(tmp_path):
    """The one-logit binary convention does not reach a tagger: the loss is
    cross-entropy over positions, not a single sigmoid."""
    from ml_framework.data.builders import build_bundle

    path = tmp_path / "two.jsonl"
    rows = [{"tokens": ["a", "b"], "tags": ["O", "X"]} for _ in range(20)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    bundle = build_bundle(ner_config(path, tmp_path / "o"))
    assert bundle.output_dim == 2
    assert bundle.n_classes == 2


# ── The seq2seq corpus ────────────────────────────────────
@pytest.mark.unit
def test_seq2seq_targets_are_never_label_encoded(pairs_csv, tmp_path):
    """Turning a target string into a class index is the mistake that makes
    seq2seq look like classification with a very large number of classes."""
    from ml_framework.data.sources.text import read_seq2seq_corpus

    sources, targets = read_seq2seq_corpus(seq_config(pairs_csv, tmp_path / "o"))

    assert all(isinstance(t, str) for t in targets)
    assert targets[0] == PAIRS[0][1]
    assert sources[0] == PAIRS[0][0]


@pytest.mark.unit
def test_no_head_width_is_claimed_for_a_vocabulary(pairs_csv, tmp_path):
    """The head is the checkpoint's vocabulary. Recording a number here would be
    recording something the framework neither chose nor can check."""
    from ml_framework.data.builders import build_bundle

    bundle = build_bundle(seq_config(pairs_csv, tmp_path / "o"))
    assert bundle.output_dim == 0
    assert bundle.n_classes is None


@pytest.mark.unit
def test_labels_ride_inside_the_encoding_for_teacher_forcing(pairs_csv, tmp_path):
    """transformers derives `decoder_input_ids` from `labels` — the right-shift
    that makes teacher forcing work. Building the shift by hand is a well-known
    way to train a model that predicts the token it was just handed."""
    from ml_framework.data.preprocess.text import Seq2SeqPreprocessor

    pre = Seq2SeqPreprocessor(model_name=SEQ_MODEL, max_length=32, max_target_length=12)
    encoding, labels = pre.collate_fn([PAIRS[0], PAIRS[1]])

    assert "labels" in encoding
    assert np.array_equal(encoding["labels"].numpy(), labels.numpy())


@pytest.mark.unit
def test_target_padding_becomes_ignore_index(pairs_csv, tmp_path):
    """Left as the pad id it is learnable, and padding is most of a short target
    in a mixed batch — so a model could score well by predicting nothing."""
    from ml_framework.data.preprocess.text import Seq2SeqPreprocessor

    pre = Seq2SeqPreprocessor(model_name=SEQ_MODEL, max_length=32, max_target_length=12)
    _, labels = pre.collate_fn([("a", "one"), ("b", "a much longer target string here")])

    assert (labels == IGNORE_INDEX).any()
    assert (labels[0] == IGNORE_INDEX).sum() > (labels[1] == IGNORE_INDEX).sum()


@pytest.mark.unit
def test_references_decode_back_through_the_same_tokenizer():
    """`IGNORE_INDEX` is not a token id, so it has to be put back before decoding —
    the inverse of what encoding did, and easy to forget until the references come
    out of the tokenizer as an index error."""
    from ml_framework.data.preprocess.text import Seq2SeqPreprocessor

    pre = Seq2SeqPreprocessor(model_name=SEQ_MODEL, max_length=32, max_target_length=12)
    _, labels = pre.collate_fn([("a", "cat on mat"), ("b", "dog")])

    decoded = pre.decode(labels)
    assert decoded[0].strip() == "cat on mat"
    assert decoded[1].strip() == "dog"


# ── Training, evaluation, the bundle ──────────────────────
@pytest.fixture
def ner_bundle(ner_jsonl, tmp_path) -> Path:
    out = tmp_path / "ner_run"
    train(ner_config(ner_jsonl, out))
    return out


@pytest.fixture
def seq_bundle(pairs_csv, tmp_path) -> Path:
    out = tmp_path / "seq_run"
    train(seq_config(pairs_csv, out))
    return out


@pytest.mark.integration
def test_tagging_scores_over_real_tokens_only(ner_bundle):
    """Padding, specials and sub-word continuations are excluded, so the token
    count in the report is the number of *words* the model was actually asked
    about."""
    report = (ner_bundle / "report.txt").read_text(encoding="utf-8")
    metrics = json.loads((ner_bundle / "metrics.json").read_text(encoding="utf-8"))

    assert "Macro-F1 (token-level)" in report
    assert 0.0 <= metrics["test_f1"] <= 1.0
    scored = int(report.split("Tokens scored:")[1].split()[0])
    # 4 test sentences of 3-4 words each; anything near the padded width would
    # mean the ignored positions leaked into the score.
    assert 0 < scored <= 4 * 4


@pytest.mark.integration
def test_the_tagging_report_says_it_is_not_entity_level(ner_bundle):
    """Token-level figures run several points above the entity-level ones papers
    quote. Printing this under the bare name "F1" would invite that comparison."""
    report = (ner_bundle / "report.txt").read_text(encoding="utf-8")
    assert "NOT entity-level" in report


@pytest.mark.integration
def test_ragged_batches_are_flattened_rather_than_padded_together(ner_bundle):
    """Batches are padded to their own longest sequence, so (B, T) arrays from
    different batches cannot be concatenated. Both sides flatten to 1-D over real
    tokens, which is also the only granularity the metrics mean anything at."""
    predictions = pd.read_csv(ner_bundle / "predictions.csv")

    assert set(predictions.columns) >= {"label", "prediction"}
    assert (predictions["label"] != IGNORE_INDEX).all()


@pytest.mark.integration
def test_the_tagger_bundle_records_its_tags(ner_bundle):
    manifest = json.loads((ner_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["signature"]["output"]["kind"] == "token_labels"
    assert manifest["signature"]["output"]["class_names"] == ["B-ORG", "B-PER", "O"]
    assert manifest["model"]["artifact"] == "model/hf_model"


@pytest.mark.integration
def test_generation_is_scored_by_overlap_and_says_what_it_misses(seq_bundle):
    manifest = json.loads((seq_bundle / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((seq_bundle / "metrics.json").read_text(encoding="utf-8"))
    report = (seq_bundle / "report.txt").read_text(encoding="utf-8")

    assert manifest["signature"]["output"]["kind"] == "text"
    assert set(metrics) == {"test_rouge_l", "test_token_f1", "test_exact_match"}
    assert "paraphrase scores near zero" in report
    # The gap that would otherwise mislead: the monitor and the metrics disagree.
    assert "teacher-forced" in report


@pytest.mark.integration
def test_a_reloaded_generator_produces_strings(seq_bundle):
    """The round-trip that matters for a model whose output is not a number."""
    inf = Inferencer.from_artifacts(seq_bundle)
    out = inf.predict(["summarize: the cat sat on the mat"])

    assert len(out) == 1
    assert isinstance(out[0], str)


@pytest.mark.integration
def test_a_generator_refuses_predict_proba(seq_bundle):
    """A generated string has no class distribution behind it."""
    from ml_framework.core.types import UnsupportedCapability

    inf = Inferencer.from_artifacts(seq_bundle)
    assert inf.produces_proba is False
    with pytest.raises(UnsupportedCapability):
        inf.predict_proba(["anything"])


@pytest.mark.integration
def test_generation_decodes_with_the_bundles_own_tokenizer(seq_bundle):
    """One tokenizer in the bundle, not two. The estimator borrows the
    preprocessor's rather than the model carrying a second copy that could
    disagree with it about the vocabulary."""
    inf = Inferencer.from_artifacts(seq_bundle)

    assert inf.estimator.preprocessor is inf.preprocessor
    assert not (seq_bundle / "model" / "hf_model" / "tokenizer.json").exists()


# ── Serving ───────────────────────────────────────────────
@pytest.mark.serving
def test_tagging_returns_one_label_per_word(ner_bundle):
    """Per *word*, not per sub-word. Re-deriving the alignment client-side would
    require the caller to own a copy of the tokenizer, which is the coupling
    shipping it in the bundle removed."""
    pytest.importorskip("fastapi", reason="the serve extra is not installed")
    from fastapi.testclient import TestClient

    from ml_framework.serving.api import create_app

    client = TestClient(create_app(str(ner_bundle)))
    response = client.post("/predict", json={"inputs": ["Ada works at Acme"]})

    assert response.status_code == 200
    body = response.json()
    assert body["tokens"] == [["Ada", "works", "at", "Acme"]]
    assert len(body["labels"][0]) == 4
    assert set(body["labels"][0]) <= {"B-ORG", "B-PER", "O", "<truncated>"}


@pytest.mark.serving
def test_generation_returns_strings_over_http(seq_bundle):
    pytest.importorskip("fastapi", reason="the serve extra is not installed")
    from fastapi.testclient import TestClient

    from ml_framework.serving.api import create_app

    client = TestClient(create_app(str(seq_bundle)))
    response = client.post("/predict", json={"inputs": ["summarize: the cat sat"]})

    assert response.status_code == 200
    assert len(response.json()["generated"]) == 1
    # And the shape that would be wrong for it is refused from the manifest.
    assert client.post("/predict_proba", json={"inputs": ["x"]}).status_code == 400
