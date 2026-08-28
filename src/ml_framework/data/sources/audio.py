"""
data/sources/audio.py
─────────────────────
A directory of audio clips → :class:`~ml_framework.data.types.DataBundle`, read
through the staged pipeline.

This is the first source that consumes P13's machinery end to end: a shard index
supplies the sample list and the labels, a :class:`BlobSource` does the positioned
read, a registered decoder does the demux and decode, and a corrupt sample is
substituted rather than skipped.

Layout is ``ImageFolder``-shaped — one directory per class — because it is what
people already have, and it makes the class map a directory listing rather than a
sidecar file to keep in sync.

Everything structural lives in :mod:`staged_folder`, which the video source shares.
What is genuinely audio-specific is here: the params schema, the mel front-end, and
the one :class:`DecodeContext` field that matters most in this whole package —
``dtype="float32"``, pinned here because ``soundfile.read()`` defaults to
**float64** and a corpus read at that width is a silent 4x blowup that nothing
reports.

Splits hold a lazy :class:`StagedDataset` (``payload="dataset"``), never decoded
waveforms: an audio corpus does not fit in memory and should not pretend to.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..preprocess.audio import AudioPreprocessor
from ..streaming.stages import DecodeContext
from ..types import DataBundle
from .staged_folder import build_staged_bundle, index_for, labels_of

log = logging.getLogger(__name__)


class AudioSourceParams(PydanticModel):
    """``data.params`` for the audio source. ``data.path`` is the train folder."""

    model_config = {"frozen": True, "extra": "forbid"}

    sample_rate: int = Field(default=16_000, gt=0)
    clip_seconds: float = Field(default=4.0, gt=0.0)
    n_mels: int = Field(default=64, gt=0)
    # Explicit decoder choice. None → resolved from the index's media types, which
    # is right for a homogeneous corpus and ambiguous for a mixed one.
    decoder: str | None = None
    # Optional: absent → the validation split is carved out of the train folder.
    val_dir: str | None = None
    test_dir: str | None = None


def audio_labels(config) -> list[int]:
    """The training corpus's labels, without decoding a single clip.

    A JSON scan of the shard index — the analogue of reading
    ``ImageFolder.targets``. Cross-validation needs the labels up front to
    stratify its folds, and decoding four seconds of audio per sample to get them
    would be absurd.
    """
    return labels_of(index_for(config.data.path, config), str(config.data.path))


def build_audio_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize an audio :class:`DataBundle` from a validated config."""
    params = AudioSourceParams.model_validate(dict(config.data.params))
    preprocessor = AudioPreprocessor(
        sample_rate=params.sample_rate,
        clip_seconds=params.clip_seconds,
        n_mels=params.n_mels,
    )
    return build_staged_bundle(
        config,
        data_kind="audio",
        preprocessor=preprocessor,
        # float32 pinned in ONE place. `soundfile.read()` defaults to float64,
        # which is a 4x blowup over every sample in the corpus and produces
        # perfectly valid audio, so nothing anywhere would report it.
        ctx=DecodeContext(dtype="float32", target_rate=params.sample_rate),
        decoder_name=params.decoder,
        # Decorative but consistent with the image source: the flattened
        # spectrogram size, which is what a dense head would see.
        input_dim=params.n_mels * preprocessor.n_frames,
        val_dir=params.val_dir,
        test_dir=params.test_dir,
        indices=indices,
    )
