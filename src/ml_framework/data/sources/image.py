"""
data/sources/image.py
─────────────────────
``ImageFolder`` directories → :class:`~ml_framework.data.types.DataBundle`.

``ImageDataModule.setup`` verbatim, minus the DataLoaders. Two details survive
unchanged because both were bug fixes:

* **``Subset`` label recovery.** When there is no explicit ``val_dir`` the train
  split comes from ``random_split``, which yields a ``Subset`` with no
  ``.targets``. Labels are recovered through the parent's ``targets`` and the
  subset's ``indices``; reading ``.targets`` off the ``Subset`` silently returns
  the *full* label list and mis-weights the sampler.
* **Imbalance is handled by sampling, not by loss weights.** The bundle therefore
  carries ``meta["sample_weights"]`` and no ``class_weights``; applying both
  would correct twice.

The sampler itself is not built here — ``meta`` carries the per-row weights and
the Lightning adapter constructs the ``WeightedRandomSampler``. Keeping torch's
sampler out of the source is what lets a non-Lightning backend consume an image
bundle later without inheriting a DataLoader concept.

Splits hold lazy ``Dataset`` objects (``payload="dataset"``), never decoded pixel
arrays: an image corpus does not fit in memory and should not pretend to.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from ..preprocess.image import ImagePreprocessor
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)


def build_image_bundle(config) -> DataBundle:
    """Materialize an image :class:`DataBundle` from a validated config."""
    import torch
    from torchvision import datasets

    preprocessor = ImagePreprocessor(img_size=config.data.img_size, augment=True)
    train_tf = preprocessor.train_transform()
    eval_tf = preprocessor.eval_transform()

    full_train = datasets.ImageFolder(config.data.train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(config.data.test_dir, transform=eval_tf)
    classes = list(full_train.classes)

    if config.data.val_dir:
        train_ds = full_train
        val_ds = datasets.ImageFolder(config.data.val_dir, transform=eval_tf)
        labels = list(full_train.targets)
    else:
        n = len(full_train)
        n_val = max(1, int(config.data.val_size * n))
        gen = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = torch.utils.data.random_split(full_train, [n - n_val, n_val], gen)
        # Subset → recover labels via the parent's .targets and the subset .indices.
        labels = [full_train.targets[i] for i in train_ds.indices]

    counts = Counter(labels)
    total = sum(counts.values())
    w_map = {c: total / cnt for c, cnt in counts.items()}
    sample_weights = np.asarray([w_map[c] for c in labels], dtype="float64")

    size = config.data.img_size
    input_dim = 3 * size * size
    # binary → single-logit head; multiclass → one logit per class.
    output_dim = 1 if config.task == "binary" else len(classes)
    log.info("image classes=%d train=%d", len(classes), len(labels))

    schema = FeatureSchema(
        target_name=config.data.target_col,
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
