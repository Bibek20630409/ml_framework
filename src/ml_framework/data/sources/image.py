"""
data/sources/image.py
─────────────────────
``ImageFolder`` directories → :class:`~ml_framework.data.types.DataBundle`.

``ImageDataModule.setup``, minus the DataLoaders. Two v1 bug fixes survive
unchanged, and one v1 bug is fixed here:

* **``Subset`` label recovery.** When there is no explicit ``val_dir`` the train
  split comes from ``random_split``, which yields a ``Subset`` with no
  ``.targets``. Labels are recovered through the parent's ``targets`` and the
  subset's ``indices``; reading ``.targets`` off the ``Subset`` silently returns
  the *full* label list and mis-weights the sampler.
* **Imbalance is handled by sampling, not by loss weights.** The bundle therefore
  carries ``meta["sample_weights"]`` and no ``class_weights``; applying both
  would correct twice.
* **Validation images are no longer augmented.** v1 carved the validation set out
  of the *augmented* training dataset, so every validation image arrived randomly
  cropped and flipped. A transform belongs to the dataset rather than to a
  ``Subset`` of it, so the fix is a second un-augmented view — see
  :func:`eval_view`. The partition at a given seed is unchanged; only the
  pipeline each side goes through is. **Expect validation metrics on image runs to
  differ from before, and to be slightly better**: they were previously measured
  on deliberately degraded inputs, and early stopping and checkpoint selection
  both read them.

The sampler itself is not built here — ``meta`` carries the per-row weights and
the Lightning adapter constructs the ``WeightedRandomSampler``. Keeping torch's
sampler out of the source is what lets a non-Lightning backend consume an image
bundle later without inheriting a DataLoader concept.

Splits hold lazy ``Dataset`` objects (``payload="dataset"``), never decoded pixel
arrays: an image corpus does not fit in memory and should not pretend to.

:class:`ImageSourceParams` validates ``data.params``; ``data.path`` is the
training folder. v1 spelled these ``train_dir``/``val_dir``/``test_dir`` on the
shared ``DataConfig``, where every non-image run also validated against them.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..preprocess.image import ImagePreprocessor
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)


class ImageSourceParams(PydanticModel):
    """``data.params`` for the image source. ``data.path`` is the train folder."""

    model_config = {"frozen": True, "extra": "forbid"}

    img_size: int = Field(default=224, gt=0)
    # Optional: absent → the validation split is carved out of the train folder.
    val_dir: str | None = None
    test_dir: str | None = None


def eval_view(dataset: Any, eval_tf: Any) -> Any:
    """The same corpus, without augmentation.

    A transform belongs to the *dataset*, not to a ``Subset`` of it, so carving a
    validation set out of an augmented dataset leaves it augmented. Augmentation
    exists to make training harder; measuring on augmented images measures the
    augmentation, and the resulting validation score is pessimistic and noisy for
    no reason — early stopping and checkpoint selection both read it.

    A shallow copy rather than a second ``ImageFolder``: the file list, the class
    map and the loader are read-only and shared, so this rebinds one attribute
    instead of walking the directory tree again. On a large corpus that walk is
    the expensive part.
    """
    import copy

    view = copy.copy(dataset)
    view.transform = eval_tf
    return view


def train_labels(config) -> list[int]:
    """The training folder's labels, without decoding a single image.

    ``ImageFolder`` builds its ``targets`` list by walking the directory tree, so
    this is a directory scan rather than a load. Cross-validation needs the labels
    up front to stratify its folds, and paying for pixel decoding to get them would
    be absurd.
    """
    from torchvision import datasets

    return list(datasets.ImageFolder(config.data.path).targets)


def build_image_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize an image :class:`DataBundle` from a validated config.

    ``indices`` partitions the **training folder** for cross-validation. The
    configured ``params.test_dir`` is deliberately left out of that partition: it
    is an explicit statement about which images are held back, and silently folding
    it into the pool would override a decision the user made on disk.

    The consequence is worth stating plainly, and ``cv.json`` is where a reader
    will look for it: under cross-validation, "test" means *a held-out slice of the
    training folder*, not ``test_dir``. The final bundle's ``test_acc`` still comes
    from ``test_dir``, so the two numbers answer different questions.
    """
    import torch
    from torchvision import datasets

    params = ImageSourceParams.model_validate(dict(config.data.params))
    preprocessor = ImagePreprocessor(img_size=params.img_size, augment=True)
    train_tf = preprocessor.train_transform()
    eval_tf = preprocessor.eval_transform()

    full_train = datasets.ImageFolder(config.data.path, transform=train_tf)
    classes = list(full_train.classes)

    if indices is not None:
        return _fold_bundle(config, params, preprocessor, full_train, eval_tf, classes, indices)

    test_ds = datasets.ImageFolder(params.test_dir, transform=eval_tf)

    if params.val_dir:
        train_ds = full_train
        val_ds = datasets.ImageFolder(params.val_dir, transform=eval_tf)
        labels = list(full_train.targets)
    else:
        from torch.utils.data import Subset

        n = len(full_train)
        n_val = max(1, int(config.data.split.val_size * n))
        gen = torch.Generator().manual_seed(config.runtime.seed)
        # The same generator and the same call, so the *partition* is unchanged at
        # a given seed — only which transform pipeline each side goes through.
        train_ds, val_part = torch.utils.data.random_split(full_train, [n - n_val, n_val], gen)
        val_ds = Subset(eval_view(full_train, eval_tf), val_part.indices)
        # Subset → recover labels via the parent's .targets and the subset .indices.
        labels = [full_train.targets[i] for i in train_ds.indices]

    counts = Counter(labels)
    total = sum(counts.values())
    w_map = {c: total / cnt for c, cnt in counts.items()}
    sample_weights = np.asarray([w_map[c] for c in labels], dtype="float64")

    size = params.img_size
    input_dim = 3 * size * size
    # binary → single-logit head; multiclass → one logit per class.
    output_dim = 1 if config.task == "binary" else len(classes)
    log.info("image classes=%d train=%d", len(classes), len(labels))

    schema = FeatureSchema(
        target_name=config.data.target,
        class_names=tuple(config.data.class_names) if config.data.class_names else tuple(classes),
    )

    return DataBundle(
        train=Split(payload="dataset", x=train_ds, y=np.asarray(labels)),
        val=Split(payload="dataset", x=val_ds),
        test=Split(payload="dataset", x=test_ds),
        schema=schema,
        task=config.task,
        data_kind="image",
        input_dim=input_dim,
        output_dim=output_dim,
        # Deliberately None: imbalance is corrected by the sampler below.
        class_weights=None,
        preprocessor=preprocessor,
        reference_stats=None,
        meta={"sample_weights": sample_weights, "classes": tuple(classes)},
    )


def _fold_bundle(
    config,
    params: ImageSourceParams,
    preprocessor: ImagePreprocessor,
    full_train: Any,
    eval_tf: Any,
    classes: list[str],
    indices: Any,
) -> DataBundle:
    """One cross-validation fold, carved from the training folder.

    Two details matter and neither is cosmetic:

    * **Validation and test get the *eval* transforms**, via :func:`eval_view` —
      see there for why a ``Subset`` cannot carry its own.
    * **Sample weights are recomputed per fold**, from this fold's training labels.
      Reusing one weight vector across folds would weight each fold by another
      fold's class balance — the same category of mistake as sharing a fitted
      scaler, and just as invisible in the result.
    """
    from torch.utils.data import Subset

    # One un-augmented view of the same corpus, shared by val and test.
    unaugmented = eval_view(full_train, eval_tf)

    train_ds = Subset(full_train, list(indices.train))
    val_ds = Subset(unaugmented, list(indices.val))
    test_ds = Subset(unaugmented, list(indices.test))

    # `Subset` has no `.targets`; reading it would silently return the *parent's*
    # full label list. Same bug the non-CV path documents, same recovery.
    labels = [full_train.targets[i] for i in indices.train]
    counts = Counter(labels)
    total = sum(counts.values())
    w_map = {c: total / cnt for c, cnt in counts.items()}

    size = params.img_size
    log.info(
        "image fold: train=%d val=%d test=%d (carved from %s; test_dir untouched)",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        config.data.path,
    )
    return DataBundle(
        train=Split(payload="dataset", x=train_ds, y=np.asarray(labels)),
        val=Split(payload="dataset", x=val_ds),
        test=Split(
            payload="dataset",
            x=test_ds,
            y=np.asarray([full_train.targets[i] for i in indices.test]),
        ),
        schema=FeatureSchema(
            target_name=config.data.target,
            class_names=(
                tuple(config.data.class_names) if config.data.class_names else tuple(classes)
            ),
        ),
        task=config.task,
        data_kind="image",
        input_dim=3 * size * size,
        output_dim=1 if config.task == "binary" else len(classes),
        class_weights=None,
        preprocessor=preprocessor,
        reference_stats=None,
        meta={
            "sample_weights": np.asarray([w_map[c] for c in labels], dtype="float64"),
            "classes": tuple(classes),
            # Recorded so a reader of cv.json knows what "test" meant here.
            "cv_test_source": "train_dir",
        },
    )
