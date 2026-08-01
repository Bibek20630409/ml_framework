"""Cross-validation for image data: folds carved from the training folder.

The semantics are the load-bearing part. `params.test_dir` is an explicit
statement about which images are held back, so cross-validation does **not** pool
it in — folds come from `data.path` alone, and the final bundle's `test_acc`
still comes from `test_dir`. The two numbers answer different questions, and
these tests pin that rather than just "it runs".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchvision", reason="the image extra is not installed")
pytest.importorskip("PIL", reason="the image extra is not installed")

from ml_framework.config import ExperimentConfig  # noqa: E402
from ml_framework.data.builders import build_bundle, build_cv_bundles  # noqa: E402

CLASSES = ("cat", "dog", "bird")
PER_CLASS_TRAIN = 8
PER_CLASS_TEST = 3


@pytest.fixture
def image_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny ImageFolder corpus: a train tree and a *separate* test tree."""
    from PIL import Image

    rng = np.random.default_rng(0)

    def make_tree(root: Path, per_class: int) -> None:
        for label, name in enumerate(CLASSES):
            folder = root / name
            folder.mkdir(parents=True)
            for i in range(per_class):
                # Distinct per class so a model could in principle learn something.
                pixels = rng.integers(label * 60, label * 60 + 60, (8, 8, 3), dtype="uint8")
                Image.fromarray(pixels).save(folder / f"{i}.png")

    train_dir, test_dir = tmp_path / "train", tmp_path / "test"
    make_tree(train_dir, PER_CLASS_TRAIN)
    make_tree(test_dir, PER_CLASS_TEST)
    return train_dir, test_dir


def image_config(dirs: tuple[Path, Path], tmp_path: Path, **overrides) -> ExperimentConfig:
    train_dir, test_dir = dirs
    cfg = ExperimentConfig.model_validate(
        {
            "task": "multiclass",
            "runtime": {"output_dir": str(tmp_path / "out"), "seed": 0, "num_workers": 0},
            "data": {
                "kind": "image",
                "path": str(train_dir),
                "split": {"val_size": 0.2, "test_size": 0.2},
                "params": {"test_dir": str(test_dir), "img_size": 8},
            },
            "model": {"name": "cnn"},
            "fit": {"budget": {"max_epochs": 1}, "batch_size": 4},
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
    )
    return cfg.with_overrides(overrides) if overrides else cfg


# ── The semantics ─────────────────────────────────────────
@pytest.mark.integration
def test_folds_partition_the_training_folder_exactly_once(image_dirs, tmp_path):
    """Every training image is in exactly one fold's test slice — otherwise the
    estimate double-counts some images and never sees others."""
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    bundles = list(build_cv_bundles(cfg))

    assert len(bundles) == 3
    seen = np.concatenate([b.test.x.indices for b in bundles])
    assert np.array_equal(np.sort(seen), np.arange(len(CLASSES) * PER_CLASS_TRAIN))


@pytest.mark.integration
def test_the_configured_test_dir_is_left_out_of_the_folds(image_dirs, tmp_path):
    """`test_dir` is a decision made on disk; cross-validation does not override it.

    Every fold's indices address the *training* tree, so the fold sizes add up to
    the training corpus and never touch the 9 held-back images.
    """
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    bundles = list(build_cv_bundles(cfg))

    n_train_images = len(CLASSES) * PER_CLASS_TRAIN
    for bundle in bundles:
        total = bundle.train.n + bundle.val.n + bundle.test.n
        assert total == n_train_images
        # Recorded so a reader of cv.json knows what "test" meant in these folds.
        assert bundle.meta["cv_test_source"] == "train_dir"

    # The holdout path still uses test_dir, and it is a different size.
    holdout = build_bundle(image_config(image_dirs, tmp_path))
    assert holdout.test.n == len(CLASSES) * PER_CLASS_TEST


@pytest.mark.integration
def test_no_image_appears_in_two_parts_of_one_fold(image_dirs, tmp_path):
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    for bundle in build_cv_bundles(cfg):
        train, val, test = (
            set(bundle.train.x.indices),
            set(bundle.val.x.indices),
            set(bundle.test.x.indices),
        )
        assert not train & val
        assert not train & test
        assert not val & test


# ── The two per-fold details ──────────────────────────────
@pytest.mark.integration
def test_validation_and_test_images_are_not_augmented(image_dirs, tmp_path):
    """Augmentation exists to make *training* harder. Measuring on augmented
    images measures the augmentation, so val/test get the eval transforms — which
    needs a second ImageFolder over the same directory, because a transform
    belongs to the dataset rather than to the subset."""
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    bundle = next(iter(build_cv_bundles(cfg)))

    # Same underlying directory, different transform pipelines.
    assert bundle.train.x.dataset is not bundle.val.x.dataset
    assert bundle.val.x.dataset is bundle.test.x.dataset
    train_ops = {type(t).__name__ for t in bundle.train.x.dataset.transform.transforms}
    eval_ops = {type(t).__name__ for t in bundle.val.x.dataset.transform.transforms}
    assert train_ops - eval_ops, "the training view should carry extra augmentation"


@pytest.mark.integration
def test_sample_weights_are_recomputed_per_fold(image_dirs, tmp_path):
    """Reusing one weight vector across folds would weight each fold by another
    fold's class balance — the same category of mistake as sharing a fitted
    scaler, and just as invisible in the result."""
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    bundles = list(build_cv_bundles(cfg))

    for bundle in bundles:
        weights = bundle.meta["sample_weights"]
        assert len(weights) == bundle.train.n  # this fold's rows, not the corpus
    # Imbalance is corrected by the sampler, never also by loss weights.
    assert all(b.class_weights is None for b in bundles)


@pytest.mark.integration
def test_fold_labels_come_from_the_parent_not_the_subset(image_dirs, tmp_path):
    """`Subset` has no `.targets`; reading it returns the *parent's* full label
    list and mis-weights the sampler. The same v1 bug, in a new place."""
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    bundle = next(iter(build_cv_bundles(cfg)))

    assert len(bundle.train.y) == bundle.train.n < len(CLASSES) * PER_CLASS_TRAIN
    expected = [bundle.train.x.dataset.targets[i] for i in bundle.train.x.indices]
    assert list(bundle.train.y) == expected


@pytest.mark.integration
def test_folds_are_stratified_across_classes(image_dirs, tmp_path):
    """A fold missing a class entirely would score it as zero and drag the mean."""
    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    for bundle in build_cv_bundles(cfg):
        assert len(set(bundle.train.y)) == len(CLASSES)


# ── Labels without decoding ───────────────────────────────
@pytest.mark.unit
def test_labels_are_read_from_the_directory_tree(image_dirs, tmp_path):
    """Stratifying needs the labels up front; paying for pixel decoding to get
    them would be absurd."""
    from ml_framework.data.sources.image import train_labels

    labels = train_labels(image_config(image_dirs, tmp_path))
    assert len(labels) == len(CLASSES) * PER_CLASS_TRAIN
    assert sorted(set(labels)) == list(range(len(CLASSES)))


# ── End to end ────────────────────────────────────────────
@pytest.mark.integration
def test_training_with_folds_reports_a_cv_estimate_beside_the_holdout_score(image_dirs, tmp_path):
    """The CV estimate and `test_acc` answer different questions, so both appear."""
    import json

    from ml_framework.pipeline import train

    cfg = image_config(image_dirs, tmp_path, **{"data.split.folds": 3})
    metrics = train(cfg)

    assert "cv_acc_mean" in metrics and "cv_acc_std" in metrics
    assert "test_acc" in metrics  # still from test_dir

    cv = json.loads((Path(cfg.runtime.output_dir) / "cv.json").read_text(encoding="utf-8"))
    assert cv["folds"] == 3 and len(cv["per_fold"]) == 3
