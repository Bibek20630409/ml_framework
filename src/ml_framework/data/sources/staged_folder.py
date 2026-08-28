"""
data/sources/staged_folder.py
─────────────────────────────
The shared machinery behind the audio and video sources.

Both read a class-directory corpus through the staged pipeline, and they differ in
exactly four things: their params model, their preprocessor, the
:class:`DecodeContext` they want, and how they compute a decorative ``input_dim``.
Everything else — resolving the index, refusing an unmaterialized corpus, carving
the validation split, renumbering a fold's subset, computing class weights — is
identical, so it lives here once.

Extracted rather than copied because the parts most worth getting right are the
parts a copy would drift on: ``_subset_index`` has to renumber, and class weights
have to be recomputed per fold. Both are silent when wrong.

## Two deliberate differences from the image source

**Labels come from the shard index, not from decoding.** The index records a label
per entry at materialization, so fold planning and class-balance counting are a
JSON scan — the analogue of reading ``ImageFolder.targets``. Decoding four seconds
of audio, or sixteen frames of video, per sample to learn its label would be
absurd.

**Imbalance is corrected by loss weights, not by sampling.** The image source
emits ``meta["sample_weights"]`` for a ``WeightedRandomSampler``; these two emit
``class_weights`` and deliberately no sample weights. Two reasons, and the second
is load-bearing:

* a weighted random sampler over sharded storage is one seek per sample, which
  destroys the read locality the whole staged pipeline is built on;
* ``WeightedRandomSampler`` and :class:`ShardShuffleSampler` would both claim
  ownership of the visit order, and the adapter cannot honour both.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ...core.types import DataKind, FrameworkError
from ..streaming.dataset import StagedDataset
from ..streaming.shards import ShardEntry, ShardIndex
from ..streaming.stages import DecodeContext
from ..types import DataBundle, FeatureSchema, Split

log = logging.getLogger(__name__)


def index_for(path: str | Path, config: Any) -> ShardIndex:
    """The shard index for a corpus, refusing clearly rather than building one.

    Materialization is **not** done implicitly. It walks and decodes the whole
    corpus — minutes to hours of work that a training command should not start by
    surprise — and its whole value is being a deliberate, reportable pass with a
    fault ceiling and a non-zero exit. Doing it silently here would throw that away
    and make ``mlf train`` occasionally take an hour for no stated reason.
    """
    index_dir = ShardIndex.location(path, config.data.shards.index_dir)
    if not ShardIndex.exists(index_dir):
        raise FrameworkError(
            f"no shard index for {path}. Run `mlf materialize --data {path}` first: it "
            "decode-probes every sample, records the labels and digests this source "
            "reads, and reports what is unreadable before a run starts depending on it."
        )
    return ShardIndex.read(index_dir)


def staged_dataset(
    index: ShardIndex,
    root: Path,
    config: Any,
    *,
    decoder_name: str | None,
    ctx: DecodeContext,
) -> StagedDataset:
    """One :class:`StagedDataset` over ``index``, wired to this run's policy."""
    from ...core.registry import DECODERS
    from ..streaming import decoder_for, suffix_of
    from ..streaming.sources_io import source_for

    first = index.entries[0] if index.entries else None
    name = decoder_name or index.decoder_hint or None
    decoder = decoder_for(
        explicit=name,
        media_type=first.media_type if first else "",
        suffix=suffix_of(first.key) if first else "",
        params=dict(config.data.decoder_params),
    )
    spec = DECODERS.get_spec(decoder.name)

    return StagedDataset(
        index,
        source=source_for(root),
        decoder=decoder,
        ctx=ctx,
        on_corrupt=config.data.integrity.on_corrupt,
        substitute=config.data.integrity.substitute,
        verify_checksums=config.data.integrity.verify_checksums,
        integrity=spec.integrity,
        allow_unverified=config.data.integrity.allow_unverified,
        seed=config.runtime.seed,
        fault_dir=Path(config.runtime.output_dir) / "faults",
    )


def subset_index(index: ShardIndex, keep: Any) -> ShardIndex:
    """A new index over a subset of entries, **renumbered from zero**.

    Renumbered rather than filtered in place, because ``n_samples`` and the entry
    positions have to stay consistent: ``StagedDataset.__len__`` and the
    distributed sampler both derive from that pair, and a sparse index would break
    the identity the whole batch-count-parity argument rests on.
    """
    entries = tuple(
        ShardEntry(
            i=new,
            shard=index.entry(int(old)).shard,
            key=index.entry(int(old)).key,
            offset=index.entry(int(old)).offset,
            nbytes=index.entry(int(old)).nbytes,
            media_type=index.entry(int(old)).media_type,
            label=index.entry(int(old)).label,
            digest=index.entry(int(old)).digest,
            n_units=index.entry(int(old)).n_units,
        )
        for new, old in enumerate(keep)
    )
    return ShardIndex(
        n_samples=len(entries),
        entries=entries,
        shards=index.shards,
        data_kind=index.data_kind,
        decoder_hint=index.decoder_hint,
        # Carried through: a fold is a view of the same corpus, and a resume must
        # still be able to tell that corpus from a different one.
        index_digest=index.index_digest,
        materialized_at=index.materialized_at,
        class_names=index.class_names,
    )


def class_weights(labels: list[int], n_classes: int, task: str) -> np.ndarray | None:
    """Balanced loss weights, or ``None`` when there is nothing to correct.

    numpy, never torch: the agnostic data layer must not import it, and the
    Lightning adapter converts.
    """
    if not labels:
        return None
    counts = Counter(labels)
    if len(counts) < 2:
        return None
    if task == "binary":
        # One element: BCEWithLogitsLoss' `pos_weight`, negatives over positives.
        negatives, positives = counts.get(0, 0), counts.get(1, 0)
        return None if not positives else np.asarray([negatives / positives], dtype="float32")
    total = len(labels)
    return np.asarray(
        [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)], dtype="float32"
    )


def labels_of(index: ShardIndex, where: str) -> list[int]:
    """Every label in ``index``, refusing a partially-labelled corpus by name."""
    values = index.labels()
    if values is None:
        raise ValueError(
            f"the shard index at {where} has no labels, so folds cannot be stratified and "
            "imbalance cannot be corrected. Re-run `mlf materialize` over a "
            "class-directory layout."
        )
    return [int(v) for v in values]


def classes_of(index: ShardIndex) -> list[str]:
    """Class names, from the manifest or from the shard (= directory) names."""
    return list(index.class_names) or sorted({entry.shard for entry in index.entries})


def build_staged_bundle(
    config,
    *,
    data_kind: DataKind,
    preprocessor: Any,
    ctx: DecodeContext,
    decoder_name: str | None,
    input_dim: int,
    val_dir: str | None,
    test_dir: str | None,
    indices: Any = None,
) -> DataBundle:
    """The shared bundle-building flow for a staged class-directory corpus.

    ``indices`` partitions the **training folder** for cross-validation, on the
    same terms the image source documents: ``test_dir`` is an explicit statement
    about what is held back, so folding it into the pool would override a decision
    the user made on disk. Under CV, "test" therefore means a held-out slice of the
    *training* folder — recorded in ``meta["cv_test_source"]`` so a reader of
    ``cv.json`` is not left guessing.
    """
    root = Path(config.data.path)
    index = index_for(root, config)
    classes = classes_of(index)

    def dataset_for(idx: ShardIndex, at: Path) -> StagedDataset:
        return staged_dataset(idx, at, config, decoder_name=decoder_name, ctx=ctx)

    def bundle(
        *,
        train_index: ShardIndex,
        train_ds: Any,
        val_ds: Any,
        test_ds: Any,
        test_labels: list[int] | None,
        extra_meta: dict[str, Any],
    ) -> DataBundle:
        train_labels = labels_of(train_index, str(root))
        return DataBundle(
            train=Split(payload="dataset", x=train_ds, y=np.asarray(train_labels)),
            val=Split(payload="dataset", x=val_ds),
            test=Split(
                payload="dataset",
                x=test_ds,
                y=None if test_labels is None else np.asarray(test_labels),
            ),
            schema=FeatureSchema(
                target_name=config.data.target,
                class_names=(
                    tuple(config.data.class_names) if config.data.class_names else tuple(classes)
                ),
            ),
            task=config.task,
            data_kind=data_kind,
            input_dim=input_dim,
            output_dim=1 if config.task == "binary" else len(classes),
            # Loss weights, NOT sample weights -- see the module docstring.
            class_weights=class_weights(train_labels, len(classes), config.task),
            preprocessor=preprocessor,
            reference_stats=None,
            meta={
                "classes": tuple(classes),
                # Read by the transport layer to decide pinning and worker count.
                "lands_in": train_ds.decoder_lands_in,
                # So a served artifact can say which corpus version it trained on.
                "index_digest": index.index_digest,
                **extra_meta,
            },
        )

    if indices is not None:
        train_index = subset_index(index, list(indices.train))
        val_index = subset_index(index, list(indices.val))
        test_index = subset_index(index, list(indices.test))
        log.info(
            "%s fold: train=%d val=%d test=%d (carved from %s; test_dir untouched)",
            data_kind,
            train_index.n_samples,
            val_index.n_samples,
            test_index.n_samples,
            root,
        )
        return bundle(
            train_index=train_index,
            train_ds=dataset_for(train_index, root),
            val_ds=dataset_for(val_index, root),
            test_ds=dataset_for(test_index, root),
            test_labels=labels_of(test_index, str(root)),
            extra_meta={"cv_test_source": "train_dir"},
        )

    if not test_dir:
        raise ValueError(f"{data_kind} data requires 'params.test_dir' (the held-out folder)")
    test_root = Path(test_dir)
    test_ds = dataset_for(index_for(test_root, config), test_root)

    if val_dir:
        val_root = Path(val_dir)
        val_ds = dataset_for(index_for(val_root, config), val_root)
        train_index = index
    else:
        # Carve validation out of the training corpus: a deterministic permutation
        # at the run's seed, so the partition is reproducible without importing
        # torch into the agnostic layer to get a generator.
        rng = np.random.default_rng(config.runtime.seed)
        order = rng.permutation(index.n_samples)
        n_val = max(1, int(config.data.split.val_size * index.n_samples))
        val_ds = dataset_for(subset_index(index, order[:n_val]), root)
        train_index = subset_index(index, order[n_val:])

    train_ds = dataset_for(train_index, root)
    log.info(
        "%s classes=%d train=%d val=%d (decoder=%s)",
        data_kind,
        len(classes),
        len(train_ds),
        len(val_ds),
        train_ds.decoder.name,
    )
    return bundle(
        train_index=train_index,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        test_labels=None,
        extra_meta={},
    )
