"""
data/streaming/decoders/video_file.py
─────────────────────────────────────
MP4 / H.264 through PyAV (libavformat + libavcodec on CPU threads).

**This is the row where demux is unarguably a stage of its own.** An MP4 is a tree
of boxes; getting from a byte range to decodable packets means traversing them,
reading the ``stss`` keyframe index, and seeking — none of which is decoding, and
all of which is required before decoding can start. The proof that the seam is
real rather than tidy: on the GPU path, **demux stays on the CPU** while decode
moves to fixed-function silicon. A design that folded the two together could not
express that at all.

**Integrity: ``loud``, with a large caveat.** A missing ``moov`` box means the
container will not open, and that fails immediately and clearly. But mid-stream
corruption does *not* raise — libavcodec logs a complaint and emits **artifacted
frames**. The tensor has the right shape and dtype, the batch collates, the loss
is finite, and the model quietly learns from garbage. Detecting it means comparing
the decoded frame count against what the container's index claims, which is
``mlf materialize``'s job.

**Output is ``uint8`` THWC.** A hardware decoder would hand back NV12 and the
YUV→RGB conversion would be a CUDA kernel; here libswscale does it inside
``to_ndarray(format="rgb24")``, and that conversion frequently costs about as much
as the decode itself. It is not broken out as a separate stage because a timer in
that loop would cost more than it reports — the stall profiler gives the
actionable aggregate instead.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import numpy as np

from ..stages import Blob, DecodeContext, Decoded, Packet
from .base import BaseDecoder, as_bytes

# Frames per clip, and how many source frames to advance between kept frames.
# Defaults match r3d_18's expected input (16 frames at 112x112).
_DEFAULT_CLIP_LEN = 16
_DEFAULT_STRIDE = 2


class VideoDecoder(BaseDecoder):
    """MP4/H.264 → a (T, H, W, C) uint8 clip.

    Holds the open container between ``demux`` and ``decode``: an H.264 slice is
    meaningless without the SPS/PPS in its stream context, so the packets carry a
    native handle and the context has to outlive the demux call. :meth:`close`
    releases it.
    """

    name: ClassVar[str] = "video.h264"

    def __init__(self, *, clip_len: int = _DEFAULT_CLIP_LEN, stride: int = _DEFAULT_STRIDE) -> None:
        if clip_len < 1:
            raise ValueError(f"clip_len must be >= 1, got {clip_len}")
        if stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {stride}")
        self.clip_len = clip_len
        self.stride = stride
        self._container: Any = None
        self._declared_frames: int | None = None

    # ── stage 2: demux ──
    def demux(self, blob: Blob) -> Iterator[Packet]:
        """Box traversal and packet extraction. Stays on the CPU, always."""
        import av

        self.close()
        payload = as_bytes(blob.data)
        # A missing `moov` raises here, which is the loud half of this format's
        # integrity story.
        self._container = av.open(io.BytesIO(payload))

        streams = [s for s in self._container.streams if s.type == "video"]
        if not streams:
            raise ValueError("video.h264: no video stream in container")
        stream = streams[0]
        # What the container *claims*. Materialization compares this against what
        # actually decodes — the only way mid-stream corruption is ever seen.
        self._declared_frames = int(stream.frames or 0)

        for packet in self._container.demux(stream):
            if packet.size == 0:  # libav's flush packet
                continue
            yield Packet(
                data=bytes(packet),
                stream=stream.index,
                keyframe=bool(packet.is_keyframe),
                pts=packet.pts,
                time_base=(
                    (stream.time_base.numerator, stream.time_base.denominator)
                    if stream.time_base
                    else None
                ),
                opaque=packet,
            )

    # ── stage 3: decode ──
    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        frames: list[np.ndarray] = []
        seen = 0
        n_packets = 0
        want = self.clip_len * self.stride

        for packet in packets:
            n_packets += 1
            native = packet.opaque
            if native is None:
                raise ValueError(
                    "video.h264 decode() needs the demuxer's native packet; call "
                    "demux() first rather than synthesizing packets"
                )
            for frame in native.decode():
                if seen % self.stride == 0:
                    # `format="rgb24"` makes libswscale do YUV->RGB inside this
                    # call. A hardware decoder would return NV12 and defer this to
                    # a CUDA kernel; the layout string is what records which
                    # happened.
                    frames.append(frame.to_ndarray(format="rgb24"))
                seen += 1
                if len(frames) >= self.clip_len:
                    break
            if len(frames) >= self.clip_len:
                break

        if not frames:
            raise ValueError("video.h264: decoded no frames")

        clip = _fit_clip(frames, self.clip_len)
        return Decoded(
            array=clip,
            layout="thwc",
            dtype="uint8",
            lands_in="host",
            rate=None,
            meta={
                "n_frames": int(clip.shape[0]),
                "n_packets": n_packets,
                "frames_seen": seen,
                "frames_wanted": want,
                # The pair materialization compares. A gap between them is how
                # mid-stream H.264 damage becomes visible.
                "declared_frames": self._declared_frames,
                "to_chw_float_in": "collate_fn",
            },
        )

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


def _fit_clip(frames: list[np.ndarray], clip_len: int) -> np.ndarray:
    """Stack to exactly ``clip_len`` frames, repeating the last one if short.

    Padding rather than raising, because a clip shorter than the window is a
    property of short source material, not damage — and a dataset must return a
    fixed shape for every index or the batch will not collate. Genuine damage is
    caught by the declared-vs-decoded frame comparison, not here.
    """
    if len(frames) < clip_len:
        frames = frames + [frames[-1]] * (clip_len - len(frames))
    return np.ascontiguousarray(np.stack(frames[:clip_len], axis=0), dtype="uint8")


def build_decoder(**params: Any) -> VideoDecoder:
    """Factory named by :data:`DecoderSpec.factory`. Unknown keys raise."""
    unknown = set(params) - {"clip_len", "frame_stride"}
    if unknown:
        raise ValueError(
            f"video.h264 got unknown decoder_params {sorted(unknown)}; "
            "accepts: clip_len, frame_stride"
        )
    return VideoDecoder(
        clip_len=int(params.get("clip_len", _DEFAULT_CLIP_LEN)),
        stride=int(params.get("frame_stride", _DEFAULT_STRIDE)),
    )
