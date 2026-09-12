"""
data/sources/video.py
─────────────────────
A directory of video clips → :class:`~ml_framework.data.types.DataBundle`, read
through the staged pipeline.

The sibling of the audio source, sharing :mod:`staged_folder` for everything
structural. What is video-specific is here: the clip geometry, and the fact that
the decoder's own params carry it.

**The clip geometry lives in ``decoder_params``, not only in the preprocessor.**
That is the one real asymmetry with audio, and it follows from where the work
happens: a video decoder should stop after ``clip_len`` frames rather than decode
the whole file and throw most of it away, so it has to be *told* the clip length
before it starts. An audio decoder has no equivalent early exit — it decodes the
file and the preprocessor crops. So this source forwards ``clip_len`` and
``frame_stride`` into the decoder's params, and the preprocessor's copy is the
fallback that fixes up a clip which came back short.

Splits hold a lazy :class:`StagedDataset` (``payload="dataset"``): a video corpus
does not fit in memory by an even wider margin than an audio one.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..preprocess.video import VideoPreprocessor
from ..streaming.stages import DecodeContext
from ..types import DataBundle
from .staged_folder import build_staged_bundle, index_for, labels_of

log = logging.getLogger(__name__)


class VideoSourceParams(PydanticModel):
    """``data.params`` for the video source. ``data.path`` is the train folder."""

    model_config = {"frozen": True, "extra": "forbid"}

    # Frames per clip and the stride between kept frames. Defaults match r3d_18's
    # expected input (16 frames at 112x112) -- feeding a video backbone the wrong
    # clip geometry is a shape error several layers deep.
    clip_len: int = Field(default=16, gt=0)
    frame_stride: int = Field(default=2, ge=1)
    img_size: int = Field(default=112, gt=0)
    decoder: str | None = None
    val_dir: str | None = None
    test_dir: str | None = None


def video_labels(config) -> list[int]:
    """The training corpus's labels, without decoding a single frame."""
    return labels_of(index_for(config.data.path, config), str(config.data.path))


def video_preprocessor(config) -> VideoPreprocessor:
    """The clip geometry this config implies, without touching the corpus.

    Sibling of ``audio_preprocessor``, and for the same caller: the tail probe in
    ``mlf materialize --probe-full``.
    """
    params = VideoSourceParams.model_validate(dict(config.data.params))
    return VideoPreprocessor(clip_len=params.clip_len, img_size=params.img_size)


def video_decoder_params(config) -> dict[str, Any]:
    """The decoder params a video run implies, geometry included.

    The decoder needs the geometry **up front** so it can stop after ``clip_len``
    frames instead of decoding a whole file and discarding most of it. The user's
    own ``decoder_params`` still win, because an explicit setting is a decision.

    Split out from :func:`build_video_bundle` because ``mlf materialize`` needs the
    same dict and was passing ``data.decoder_params`` raw — so every video
    materialization decoded at the decoder's default geometry rather than the
    configured one, and the shard index recorded ``n_units`` for clips nobody asked
    for.
    """
    params = VideoSourceParams.model_validate(dict(config.data.params))
    return {
        "clip_len": params.clip_len,
        "frame_stride": params.frame_stride,
        **dict(config.data.decoder_params),
    }


def build_video_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize a video :class:`DataBundle` from a validated config."""
    params = VideoSourceParams.model_validate(dict(config.data.params))
    preprocessor = video_preprocessor(config)

    decoder_params = video_decoder_params(config)
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"decoder_params": decoder_params})}
    )

    return build_staged_bundle(
        config,
        data_kind="video",
        preprocessor=preprocessor,
        # No dtype: a video decoder's output width is uint8 and there is no
        # float64 trap to close, unlike audio.
        ctx=DecodeContext(layout="thwc"),
        decoder_name=params.decoder,
        input_dim=3 * params.clip_len * params.img_size * params.img_size,
        val_dir=params.val_dir,
        test_dir=params.test_dir,
        indices=indices,
    )
