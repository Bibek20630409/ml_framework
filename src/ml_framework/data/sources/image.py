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

:class:`ImageSourceParams` validates ``data.params``; ``data.path`` is the
training folder. v1 spelled these ``train_dir``/``val_dir``/``test_dir`` on the
shared ``DataConfig``, where every non-image run also validated against them.
"""

from __future__ import annotations

import logging
from collections import Counter

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


def build_image_bundle(config) -> DataBundle:
    """Materialize an image :class:`DataBundle` from a validated config."""
    import torch
    from torchvision import datasets

    params = ImageSourceParams.model_validate(dict(config.data.params))
    preprocessor = ImagePreprocessor(img_size=params.img_size, augment=True)
    train_tf = preprocessor.train_transform()
    eval_tf = preprocessor.eval_transform()

    full_train = datasets.ImageFolder(config.data.path, transform=train_tf)
    test_ds = datasets.ImageFolder(params.test_dir, transform=eval_tf)
    classes = list(full_train.classes)

    if params.val_dir:
        train_ds = full_train
        val_ds = datasets.ImageFolder(params.val_dir, transform=eval_tf)
        labels = list(full_train.targets)
    else:
        n = len(full_train)
        n_val = max(1, int(config.data.split.val_size * n))
        gen = torch.Generator().manual_seed(config.runtime.seed)
        train_ds, val_ds = torch.utils.data.random_split(full_train, [n - n_val, n_val], gen)
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
