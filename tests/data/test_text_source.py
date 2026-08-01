"""The text source: what it reads, what it refuses to guess, and what it hands on.

Most of this file is about the two guesses that are *not* made. A framework that
picks the first string column will eventually train a sentiment model on a column
of usernames, and a framework that tokenizes with a checkpoint other than the
model's produces ids the model has never seen. Neither failure raises; both score
badly and look like an unlucky dataset. So both are pinned here.

Reading a corpus needs neither torch nor transformers — but *naming* a text model
in a config does, because `validate_combination` checks a plugin's requirements at
config-load time. So the file skips without the extra, and the import-discipline
claim is pinned separately, in a subprocess, where it can actually be observed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers", reason="the nlp extra is not installed")

from ml_framework.config import ExperimentConfig  # noqa: E402
from ml_framework.core.types import FrameworkError  # noqa: E402
from ml_framework.data.builders import build_bundle, build_cv_bundles  # noqa: E402
from ml_framework.data.sources.text import (  # noqa: E402
    TextDataset,
    build_text_bundle,
    read_text_corpus,
    read_text_table,
)

# Small enough to be a rounding error next to the framework's own install, and
# real enough that the tokenizer round-trip is a real tokenizer.
TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"

N_ROWS = 40


@pytest.fixture
def reviews_csv(tmp_path: Path) -> Path:
    """String labels, deliberately: text corpora rarely arrive pre-encoded."""
    path = tmp_path / "reviews.csv"
    rows = [
        {"text": f"review number {i}", "label": "pos" if i % 2 else "neg"} for i in range(N_ROWS)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def text_config(path: Path, tmp_path: Path, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig.model_validate(
        {
            "task": "binary",
            "runtime": {"output_dir": str(tmp_path / "out"), "seed": 0, "num_workers": 0},
            "data": {
                "kind": "text",
                "path": str(path),
                "target": "label",
                "split": {"val_size": 0.2, "test_size": 0.2},
            },
            # Named because `validate_combination` refuses `mlp` on text at
            # config-load time — correctly. Never actually built here: these tests
            # are about the source.
            "model": {"name": "nlp.hf_text", "params": {"model_name": TINY_MODEL}},
            "fit": {"budget": {"max_epochs": 1}, "batch_size": 8},
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
    )
    return cfg.with_overrides(overrides) if overrides else cfg


# ── Reading ───────────────────────────────────────────────
@pytest.mark.unit
def test_jsonl_is_read_as_one_object_per_line(tmp_path):
    """One JSON object per line is how labelled text corpora are distributed."""
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "a", "label": 0}\n{"text": "b", "label": 1}\n', encoding="utf-8")

    frame = read_text_table(path)
    assert list(frame.columns) == ["text", "label"]
    assert frame["text"].tolist() == ["a", "b"]


@pytest.mark.unit
def test_string_labels_are_encoded_in_sorted_order(reviews_csv, tmp_path):
    """**Sorted** unique, not order of appearance.

    Order of appearance means shuffling the input file renumbers the classes, so
    two runs over the same data produce models whose class 0 means different
    things — and the class-name list in the bundle would be wrong for one of them.
    """
    _, labels, class_names = read_text_corpus(text_config(reviews_csv, tmp_path))

    assert class_names == ("neg", "pos")
    assert labels.dtype == np.int64
    # Row 0 is "neg" and row 1 is "pos"; sorted order puts neg first.
    assert labels[0] == 0 and labels[1] == 1


@pytest.mark.unit
def test_shuffling_the_file_does_not_renumber_the_classes(reviews_csv, tmp_path):
    """The property the sorted order exists to guarantee."""
    frame = pd.read_csv(reviews_csv).iloc[::-1].reset_index(drop=True)
    shuffled = tmp_path / "shuffled.csv"
    frame.to_csv(shuffled, index=False)

    _, _, original_names = read_text_corpus(text_config(reviews_csv, tmp_path))
    _, _, shuffled_names = read_text_corpus(text_config(shuffled, tmp_path))
    assert original_names == shuffled_names


@pytest.mark.unit
def test_numeric_labels_are_left_alone(tmp_path):
    """Already-encoded labels keep their values, and record no class names."""
    path = tmp_path / "c.csv"
    pd.DataFrame({"text": ["a", "b", "c"], "label": [2, 0, 1]}).to_csv(path, index=False)

    _, labels, class_names = read_text_corpus(text_config(path, tmp_path))
    assert labels.tolist() == [2, 0, 1]
    assert class_names is None


# ── The guesses it refuses to make ────────────────────────
@pytest.mark.unit
def test_several_string_columns_is_a_refusal_naming_them(tmp_path):
    """Picking the first would train on the wrong column and merely score badly."""
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"author": ["ann", "bob"], "comment": ["hi", "yo"], "label": ["a", "b"]}).to_csv(
        path, index=False
    )

    with pytest.raises(FrameworkError, match="several columns could be the text"):
        read_text_corpus(text_config(path, tmp_path))


@pytest.mark.unit
def test_an_ambiguous_file_is_resolved_by_naming_the_column(tmp_path):
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"author": ["ann", "bob"], "comment": ["hi", "yo"], "label": ["a", "b"]}).to_csv(
        path, index=False
    )

    cfg = text_config(path, tmp_path, **{"data.params.text_col": "comment"})
    texts, _, _ = read_text_corpus(cfg)
    assert texts == ["hi", "yo"]


@pytest.mark.unit
def test_a_conventional_name_breaks_the_tie(tmp_path):
    """`text` beside another string column is not ambiguous in practice."""
    path = tmp_path / "conventional.csv"
    pd.DataFrame({"author": ["ann", "bob"], "text": ["hi", "yo"], "label": ["a", "b"]}).to_csv(
        path, index=False
    )

    texts, _, _ = read_text_corpus(text_config(path, tmp_path))
    assert texts == ["hi", "yo"]


@pytest.mark.unit
def test_a_named_column_that_does_not_exist_is_an_error(reviews_csv, tmp_path):
    cfg = text_config(reviews_csv, tmp_path, **{"data.params.text_col": "nope"})
    with pytest.raises(KeyError, match="text_col"):
        read_text_corpus(cfg)


# ── The bundle ────────────────────────────────────────────
@pytest.mark.unit
def test_splits_hold_strings_not_token_ids(reviews_csv, tmp_path):
    """Tokenizing up front would pad every sequence to one length and make each
    cross-validation fold re-tokenize the whole corpus."""
    bundle = build_text_bundle(text_config(reviews_csv, tmp_path))

    assert bundle.train.payload == "dataset"
    assert isinstance(bundle.train.x, TextDataset)
    text, label = bundle.train.x[0]
    assert isinstance(text, str)
    assert int(label) in (0, 1)


@pytest.mark.unit
def test_the_three_splits_partition_the_corpus(reviews_csv, tmp_path):
    bundle = build_text_bundle(text_config(reviews_csv, tmp_path))

    assert bundle.train.n + bundle.val.n + bundle.test.n == N_ROWS
    val, test = set(bundle.val.index), set(bundle.test.index)
    assert not val & test


@pytest.mark.unit
def test_the_tokenizer_comes_from_the_models_checkpoint(reviews_csv, tmp_path):
    """One place names the checkpoint, and it is ``model.params``.

    A separate ``data.params`` knob would be a way to tokenize with one
    vocabulary and run another checkpoint's weights — which produces ids the model
    was never trained on, silently.
    """
    cfg = text_config(reviews_csv, tmp_path)
    bundle = build_text_bundle(cfg)
    assert bundle.preprocessor.model_name == TINY_MODEL
    assert bundle.preprocessor.max_length == cfg.model.params["max_length"]


@pytest.mark.unit
def test_max_length_is_not_a_second_data_params_knob(reviews_csv, tmp_path):
    """``data.params`` is frozen + extra=forbid, so the duplicate is an error
    rather than a value that silently loses."""
    with pytest.raises(Exception, match="max_length"):
        build_text_bundle(text_config(reviews_csv, tmp_path, **{"data.params.max_length": 8}))


@pytest.mark.unit
def test_imbalance_is_corrected_once_by_sampling(tmp_path):
    """Sample weights for the sampler and **no** loss weights.

    The same choice the image source makes, for the same reason: both hand the
    loop a lazy dataset, and applying both corrections would overshoot.
    """
    path = tmp_path / "skewed.csv"
    rows = [{"text": f"t{i}", "label": "rare" if i < 12 else "common"} for i in range(60)]
    pd.DataFrame(rows).to_csv(path, index=False)

    bundle = build_text_bundle(text_config(path, tmp_path))
    weights = bundle.meta["sample_weights"]
    assert len(weights) == bundle.train.n
    assert bundle.class_weights is None

    # Sorted encoding: "common" is 0, "rare" is 1. The rare class is weighted up,
    # which is the entire point of computing them.
    train_labels = bundle.train.y
    assert min(weights[train_labels == 1]) > max(weights[train_labels == 0])


@pytest.mark.unit
def test_no_feature_count_is_claimed_for_a_token_sequence(reviews_csv, tmp_path):
    """`input_dim` 0, not `max_length`: a number there would be one the serving
    layer checks incoming requests against, and be wrong."""
    bundle = build_text_bundle(text_config(reviews_csv, tmp_path))
    assert bundle.input_dim == 0
    assert bundle.schema.feature_names == ()


# ── Cross-validation ──────────────────────────────────────
@pytest.mark.unit
def test_text_folds_partition_the_corpus_exactly_once(reviews_csv, tmp_path):
    cfg = text_config(reviews_csv, tmp_path, **{"data.split.folds": 4})
    bundles = list(build_cv_bundles(cfg))

    assert len(bundles) == 4
    seen = np.concatenate([b.test.index for b in bundles])
    assert np.array_equal(np.sort(seen), np.arange(N_ROWS))


@pytest.mark.unit
def test_text_folds_stratify_on_the_encoded_labels(reviews_csv, tmp_path):
    """Stratifying on a second, independently-encoded reading of the label column
    would stratify on the wrong integers."""
    cfg = text_config(reviews_csv, tmp_path, **{"data.split.folds": 4})
    for bundle in build_cv_bundles(cfg):
        assert set(bundle.train.y.tolist()) == {0, 1}


@pytest.mark.unit
def test_build_bundle_routes_text_through_the_source(reviews_csv, tmp_path):
    """`data.kind: text` reaches the text source through the registry, not a
    hardcoded branch in the orchestrator."""
    bundle = build_bundle(text_config(reviews_csv, tmp_path))
    assert bundle.data_kind == "text"


# ── Import discipline ─────────────────────────────────────
@pytest.mark.unit
def test_reading_a_corpus_pulls_in_neither_torch_nor_transformers():
    """The claim the source's design rests on, checked where it is observable.

    Cross-validation reads labels and lays out folds through this module. If
    importing it dragged in a deep-learning stack, every one of those operations
    would require the `[nlp]` and `[lightning]` extras to be installed — and the
    serving path, which imports the data layer, would inherit them too.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import ml_framework.data.sources.text as m; "
        "print(sorted(k for k in ('torch', 'transformers') if k in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout
