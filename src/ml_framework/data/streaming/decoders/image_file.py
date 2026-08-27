"""
data/streaming/decoders/image_file.py
─────────────────────────────────────
JPEG and PNG through Pillow (libjpeg-turbo / libpng).

Neither format has a demux stage — a JFIF file is one image and the bytes go
straight to the codec. Their **integrity** stories are what separate them, and the
JPEG one is the reason this module does something non-default:

* **PNG is ``checked``.** zlib carries an Adler-32 per block and PNG a CRC-32 per
  chunk. libpng raises. There is no GPU decode path for it, and it runs roughly
  5-10x slower than JPEG — a real reason to prefer JPEG for a large corpus, and a
  real reason not to be surprised when a PNG pipeline is IO-bound at the CPU.

* **JPEG is ``loud`` only because we make it so.** By default libjpeg treats a
  truncated file as a *warning*: it returns a partial image with the missing
  scanlines filled grey, and Pillow honours that. Nothing raises. You train on
  grey-bottomed images and the only symptom is a model that is slightly worse than
  it should be.

  :func:`_strict_image_decode` promotes that warning to an error. It is the single
  most valuable line in this module, and it is a deviation from the library
  default rather than a use of it — which is exactly why it needs saying out loud.

**Decode output is ``uint8`` HWC, deliberately.** The permute to CHW, the cast to
float32 and the ``/255`` are all a copy with a 4x blowup, and they belong in the
collate function where a *batch* pays for them once — not here, where every worker
would pay per image. Decode and tensor construction are separate axes; this module
only does the first.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from ..stages import DecodeContext, Decoded, Packet
from .base import BaseDecoder, as_bytes

# Pillow mode -> channel count. RGB is forced for everything else so a corpus with
# mixed palette/greyscale/CMYK sources still stacks into one batch.
_TARGET_MODE = "RGB"


class PillowDecoder(BaseDecoder):
    """JFIF or PNG bytes → HWC uint8. No demux; decode is the whole path."""

    name: ClassVar[str] = "image.pillow"

    def __init__(self, *, fmt: str, mode: str = _TARGET_MODE) -> None:
        self.fmt = fmt
        self.mode = mode

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        payload = as_bytes(next(iter(packets)).data)
        array = _strict_image_decode(payload, mode=self.mode)
        return Decoded(
            array=array,
            layout="hwc",
            dtype="uint8",
            lands_in="host",
            rate=None,
            meta={
                "height": int(array.shape[0]),
                "width": int(array.shape[1]),
                "channels": int(array.shape[2]),
                # The conversion this decoder deliberately does NOT do.
                "to_chw_float_in": "collate_fn",
            },
        )


def _strict_image_decode(payload: bytes, *, mode: str = _TARGET_MODE) -> np.ndarray:
    """Decode, refusing the partial image a truncated file would otherwise yield.

    Two defences, because Pillow has two ways of being lenient:

    1. ``ImageFile.LOAD_TRUNCATED_IMAGES`` is forced ``False`` for the duration.
       Some libraries set it globally to ``True`` at import; leaving that in place
       would silently re-enable exactly the behaviour we are refusing, so it is set
       and restored rather than assumed.
    2. Warnings are promoted to errors around ``load()``. libjpeg reports a
       truncated scan as a warning, and ``DecompressionBombWarning`` arrives the
       same way — both should stop a sample, not colour it grey.
    """
    from PIL import Image, ImageFile

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(io.BytesIO(payload)) as opened:
                # `load()` is where the scan actually happens — `open()` only reads
                # the header, so a truncation would not surface until here.
                opened.load()
                # A separate name rather than a rebind: `open()` returns an
                # ImageFile bound to the buffer, `convert()` a detached Image, and
                # conflating them hides which one owns the bytes.
                image = opened if opened.mode == mode else opened.convert(mode)
                # `np.asarray` on a loaded Pillow image copies once, into a
                # C-contiguous HWC buffer. That copy is unavoidable: the decoder's
                # own buffer is not ours to keep.
                return np.asarray(image, dtype="uint8")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _build(fmt: str, params: dict[str, Any]) -> PillowDecoder:
    unknown = set(params) - {"mode"}
    if unknown:
        raise ValueError(f"image.{fmt} got unknown decoder_params {sorted(unknown)}; accepts: mode")
    return PillowDecoder(fmt=fmt, mode=str(params.get("mode", _TARGET_MODE)))


def build_jpeg_decoder(**params: Any) -> PillowDecoder:
    return _build("jpeg", params)


def build_png_decoder(**params: Any) -> PillowDecoder:
    return _build("png", params)
