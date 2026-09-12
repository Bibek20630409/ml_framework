"""Data sources: ingestion → :class:`~ml_framework.data.types.DataBundle`.

A *source* is the extension point that replaces datamodules. From P1 there is
exactly one datamodule (the Lightning adapter over a bundle), so registering
datamodules stopped making sense while registering sources started to.

``build_image_bundle`` is imported here for symmetry with ``build_tabular_bundle``
— safe because its torchvision imports live inside the function, per the rule that
a module must be importable with zero optional dependencies installed.
"""

from typing import Any

from .audio import audio_labels, audio_preprocessor, build_audio_bundle
from .image import build_image_bundle, train_labels
from .tabular import build_tabular_bundle, read_table
from .text import build_text_bundle, text_labels
from .timeseries import build_timeseries_bundle
from .video import (
    build_video_bundle,
    video_decoder_params,
    video_labels,
    video_preprocessor,
)

# The staged kinds, and the only ones with a tail worth probing offline: their
# preprocessors declare `construct`/`transform`/`gpu_transform`. A kind absent
# here has no staged tail, and `preprocessor_for` says so by name rather than
# returning something that would probe nothing.
_STAGED_PREPROCESSORS = {"audio": audio_preprocessor, "video": video_preprocessor}


def preprocessor_for(config) -> Any:
    """The staged preprocessor for ``config``'s data kind, without building a bundle.

    Single consumer: ``mlf materialize --probe-full``. Raises for a kind that has
    no staged tail, because "probed the tail and found nothing" and "this kind has
    no tail" are different answers and the flag should not conflate them.
    """
    build = _STAGED_PREPROCESSORS.get(config.data.kind)
    if build is None:
        raise ValueError(
            f"data kind '{config.data.kind}' has no staged tail to probe. "
            f"--probe-full applies to {sorted(_STAGED_PREPROCESSORS)} corpora."
        )
    return build(config)


# Kinds whose decoder needs more than `data.decoder_params` states. Only video so
# far: its geometry has to reach the decoder so it can stop early, and it lives in
# `data.params` rather than `data.decoder_params` because it is also the
# preprocessor's geometry.
_DECODER_PARAMS = {"video": video_decoder_params}


def decoder_params_for(config) -> dict[str, Any]:
    """The decoder params for ``config``'s kind, including anything implied.

    Unlike :func:`preprocessor_for` this never raises: a kind that implies nothing
    extra has a correct answer — ``data.decoder_params`` as written — and refusing
    would make the caller branch on the kind, which is the branch this function
    exists to absorb.

    Consumer: the ``mlf materialize`` CLI path, which passed the raw dict and so
    probed video at the decoder's default geometry rather than the configured one.
    """
    build = _DECODER_PARAMS.get(config.data.kind)
    return build(config) if build is not None else dict(config.data.decoder_params)


__all__ = [
    "train_labels",
    "audio_labels",
    "video_labels",
    "audio_preprocessor",
    "video_preprocessor",
    "video_decoder_params",
    "preprocessor_for",
    "decoder_params_for",
    "build_audio_bundle",
    "build_video_bundle",
    "text_labels",
    "build_text_bundle",
    "build_tabular_bundle",
    "build_image_bundle",
    "build_timeseries_bundle",
    "read_table",
]
