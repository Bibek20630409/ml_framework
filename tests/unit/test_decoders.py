"""The decoder registry, and what each decoder promises about itself.

Three halves, deliberately separated:

* **Registry tests** need no codec at all. They cover the half a user on a bare
  install actually hits — that ``video.h264`` is *listed* rather than hidden, that
  selecting it without the extra names the pip command, and that importing the
  package pulls in no codec.
* **Contract tests** check that a spec does not lie about itself: the stages it
  declares are the ones it overrides, its ``oracle`` is a registered host decoder,
  and its integrity value comes from the closed set.
* **Decode tests** run the two zero-dependency decoders — ``audio.pcm`` (the
  oracle) and ``text.tokens`` — end to end. Everything gated on an extra is
  ``importorskip``-ed, never hard-imported.
"""

from __future__ import annotations

import io
import subprocess
import sys
import wave

import numpy as np
import pytest

import ml_framework.data.streaming as streaming  # noqa: F401  (populates DECODERS)
from ml_framework.core.plugins import MissingExtraError, UnknownPluginError
from ml_framework.core.registry import DECODERS, get_decoder
from ml_framework.core.types import DATA_KINDS, INTEGRITIES, LANDS_IN, LAYOUTS, STAGES
from ml_framework.data.streaming.decoders.base import BaseDecoder, Decoder
from ml_framework.data.streaming.stages import DecodeContext, Packet, SampleRef

pytestmark = pytest.mark.unit


def _packet(payload: bytes) -> list[Packet]:
    return [Packet(data=payload)]


def _wav_bytes(pcm: np.ndarray, *, rate: int = 16_000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


# ── Registry: the bare-install half ───────────────────────────────────
def test_every_decoder_is_listed_on_a_bare_install_even_without_its_codec():
    """A registry that hid what it could not run would be useless for choosing."""
    names = set(DECODERS.names())
    assert {"audio.pcm", "audio.flac", "audio.mp3", "image.jpeg", "video.h264"} <= names


def test_importing_the_streaming_package_imports_no_codec():
    """The property the whole lazy-factory design protects.

    A subprocess, not this one: pytest's own plugins may well have imported
    Pillow already, and asserting against a dirty ``sys.modules`` would pass for
    the wrong reason.
    """
    code = (
        "import sys; import ml_framework.data.streaming; "
        "leaked = {'av', 'soundfile', 'PIL', 'torch', 'torchaudio'} & set(sys.modules); "
        "assert not leaked, leaked; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_selecting_a_decoder_without_its_extra_names_the_pip_command():
    spec = DECODERS.get_spec("video.h264")
    if all(req.is_satisfied() for req in spec.requires):
        pytest.skip("av is installed, so there is no missing-extra message to check")
    with pytest.raises(MissingExtraError) as excinfo:
        get_decoder("video.h264")
    message = str(excinfo.value)
    assert "decoder 'video.h264'" in message
    assert "pip install 'ml-framework[video]'" in message


def test_an_unknown_decoder_lists_what_is_registered():
    with pytest.raises(UnknownPluginError) as excinfo:
        get_decoder("audio.wma")
    assert "audio.pcm" in str(excinfo.value)


def test_the_registry_kind_is_one_word_so_its_prose_reads_correctly():
    """`kind` is interpolated into `check_requirements`' sentence.

    "data backend" needs its space; "decoder" must not have one, or the message
    would read "decoder s" after the CLI's verb derivation.
    """
    assert DECODERS.kind == "decoder"
    assert DECODERS.kind.replace(" ", "-") + "s" == "decoders"


# ── Contract: a spec must not lie about itself ────────────────────────
@pytest.mark.parametrize(
    "spec", sorted(DECODERS.specs(), key=lambda s: s.name), ids=lambda s: s.name
)
def test_every_spec_field_comes_from_its_closed_set(spec):
    assert spec.data_kind in DATA_KINDS
    assert spec.integrity in INTEGRITIES
    assert spec.lands_in in LANDS_IN
    assert spec.output_layout in LAYOUTS
    assert set(spec.stages) <= set(STAGES)
    assert spec.stages, f"{spec.name} declares no stages at all"
    assert spec.output_dtype, f"{spec.name} does not say what dtype it decodes to"


@pytest.mark.parametrize(
    "spec", sorted(DECODERS.specs(), key=lambda s: s.name), ids=lambda s: s.name
)
def test_a_spec_that_cannot_report_its_own_damage_documents_how_it_fails(spec):
    """`silent` and `none` are the classifications that need prose most."""
    if spec.integrity in ("silent", "none"):
        assert spec.integrity_note, (
            f"{spec.name} fails invisibly and says nothing about it; "
            "integrity_note is the only warning a user gets"
        )


@pytest.mark.parametrize(
    "spec", sorted(DECODERS.specs(), key=lambda s: s.name), ids=lambda s: s.name
)
def test_an_oracle_is_a_registered_host_decoder(spec):
    """Cross-checking against a device decoder would prove nothing."""
    if spec.oracle is None:
        return
    assert spec.oracle in DECODERS.names()
    assert DECODERS.get_spec(spec.oracle).lands_in == "host"
    assert spec.oracle != spec.name


def test_the_device_seam_is_registered_and_lands_in_device():
    spec = DECODERS.get_spec("fake.device")
    assert spec.lands_in == "device"
    assert spec.requires == (), "the seam must be exercisable on a bare install"
    assert spec.oracle is not None, "a silent device decoder needs a host cross-check"


def test_every_decoder_declares_a_read_stage():
    """Bytes always have to come off a device; no format is exempt.

    What varies is demux and decode, which is why `text.tokens` declaring
    ``{"read"}`` *alone* is the informative case rather than a degenerate one.
    """
    for spec in DECODERS.specs():
        assert "read" in spec.stages, f"{spec.name} claims not to read anything"


def test_a_declared_demux_stage_means_the_decoder_actually_overrides_demux():
    """`stages` is a declaration, and a declaration that drifts is worse than none.

    ``BaseDecoder.demux`` yields one packet wrapping the whole blob, which is not
    work. So for this stage — unlike read, which the source always performs, and
    decode, whose cost is not visible from the method table — declaring it and
    overriding it must agree exactly.
    """
    for spec in DECODERS.specs():
        if any(not req.is_satisfied() for req in spec.requires):
            continue
        decoder = get_decoder(spec.name)
        overrides = any(
            "demux" in klass.__dict__
            for klass in type(decoder).__mro__
            if klass not in (BaseDecoder, object)
        )
        if "demux" in spec.stages:
            assert overrides, f"{spec.name} declares a demux stage but uses the default"
        else:
            assert not overrides, f"{spec.name} overrides demux without declaring it"


def test_only_the_token_shard_claims_decoding_costs_nothing():
    """The one row in the table with no compression to reverse.

    Pinned because it is the claim most likely to be copied onto a new spec by
    mistake — every other format pays for a decode.
    """
    free = {spec.name for spec in DECODERS.specs() if "decode" not in spec.stages}
    assert free == {"text.tokens"}


def test_every_decoder_satisfies_the_protocol():
    for spec in DECODERS.specs():
        if any(not req.is_satisfied() for req in spec.requires):
            continue
        assert isinstance(get_decoder(spec.name), Decoder)


# ── audio.pcm: the zero-dependency oracle ─────────────────────────────
def test_the_audio_oracle_needs_no_extra_at_all():
    """A cross-decoder equivalence claim is only worth making if the reference
    path is guaranteed present. That is why `audio.pcm` has no requirements."""
    assert DECODERS.get_spec("audio.pcm").requires == ()
    assert DECODERS.is_available("audio.pcm")


def test_wave_decodes_to_int16_pcm_at_the_declared_rate():
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype="int16")
    decoder = get_decoder("audio.pcm")
    decoded = decoder.decode(_packet(_wav_bytes(samples)), ctx=DecodeContext())

    assert decoded.layout == "pcm"
    assert decoded.dtype == "int16"
    assert decoded.lands_in == "host"
    assert decoded.rate == 16_000
    np.testing.assert_array_equal(decoded.array, samples)


def test_wave_honours_a_float32_request_without_ever_producing_float64():
    samples = np.array([0, 16384, -16384], dtype="int16")
    decoder = get_decoder("audio.pcm")
    decoded = decoder.decode(_packet(_wav_bytes(samples)), ctx=DecodeContext(dtype="float32"))

    assert decoded.array.dtype == np.float32
    # Divided by 32768, matching libsndfile, so the oracle and audio.flac agree
    # bit-for-bit rather than within a tolerance.
    np.testing.assert_allclose(decoded.array, samples.astype("float32") / 32768.0)


def test_multichannel_wave_comes_back_channels_first_and_c_contiguous():
    """A transposed view would knock every downstream from_numpy off zero-copy."""
    interleaved = np.array([1, -1, 2, -2, 3, -3], dtype="int16")
    decoder = get_decoder("audio.pcm")
    decoded = decoder.decode(_packet(_wav_bytes(interleaved, channels=2)), ctx=DecodeContext())

    assert decoded.array.shape == (2, 3)
    assert decoded.array.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(decoded.array[0], [1, 2, 3])
    np.testing.assert_array_equal(decoded.array[1], [-1, -2, -3])


def test_a_malformed_riff_header_raises_rather_than_returning_silence():
    """This is the whole of `audio.pcm`'s `loud` classification."""
    decoder = get_decoder("audio.pcm")
    with pytest.raises(wave.Error):
        decoder.decode(_packet(b"not a riff header at all"), ctx=DecodeContext())


# ── text.tokens: no demux, no decode, no integrity ────────────────────
def test_a_token_shard_decodes_to_a_uint16_view_and_is_never_widened():
    """The load-bearing dtype invariant.

    torch has no usable uint16 arithmetic, so the cast to int64 must happen — but
    per *batch*, in the collate. Doing it here would cast the whole corpus and
    quadruple its footprint to save a memcpy-bound operation over a few MB.
    """
    tokens = np.array([1, 2, 65535, 0], dtype="uint16")
    decoder = get_decoder("text.tokens")
    decoded = decoder.decode(_packet(tokens.tobytes()), ctx=DecodeContext())

    assert decoded.dtype == "uint16"
    assert decoded.array.dtype == np.uint16
    assert decoded.layout == "tokens"
    np.testing.assert_array_equal(decoded.array, tokens)


def test_the_token_decoder_ignores_a_dtype_request_that_would_widen_the_corpus():
    """`ctx.dtype` pins a codec's output width; it is not a licence to upcast."""
    tokens = np.array([7, 8, 9], dtype="uint16")
    decoded = get_decoder("text.tokens").decode(
        _packet(tokens.tobytes()), ctx=DecodeContext(dtype="int64")
    )
    assert decoded.array.dtype == np.uint16


def test_the_token_decoder_records_where_the_int64_cast_belongs():
    """Documentation that travels with the data, not just in a docstring."""
    tokens = np.array([1, 2], dtype="uint16")
    decoded = get_decoder("text.tokens").decode(_packet(tokens.tobytes()), ctx=DecodeContext())
    assert decoded.meta["cast_to_int64_in"] == "collate_fn"


def test_int64_is_not_an_offered_on_disk_token_width():
    """Storing int64 quadruples the corpus to save nothing. Refuse it by name."""
    with pytest.raises(ValueError, match="int64"):
        get_decoder("text.tokens", dtype="int64")


def test_a_token_shard_has_no_integrity_at_all_and_says_so():
    spec = DECODERS.get_spec("text.tokens")
    assert spec.integrity == "none"
    assert spec.stages == frozenset({"read"})
    assert "digest" in spec.integrity_note.lower()


def test_a_flipped_bit_in_a_token_shard_is_a_valid_token_id():
    """Why `integrity: none` is not pessimism.

    Nothing raises, the shape is right, and the value is a legal token. This test
    exists to pin the fact that the decoder *cannot* help here — the shard index
    digest is the only defence, which is what justifies its cost.
    """
    tokens = np.array([1000, 2000, 3000], dtype="uint16")
    raw = bytearray(tokens.tobytes())
    raw[0] ^= 0x01

    decoded = get_decoder("text.tokens").decode(_packet(bytes(raw)), ctx=DecodeContext())
    assert decoded.array.shape == tokens.shape
    assert decoded.array[0] == 1001
    assert decoded.array[0] != tokens[0]


# ── The device seam ───────────────────────────────────────────────────
def test_a_device_decoder_returns_something_that_is_not_an_array():
    """If the handle were array-like, every host branch would accidentally work
    and the seam test would prove nothing."""
    decoded = get_decoder("fake.device").decode(_packet(b"\x00" * 192), ctx=DecodeContext())

    assert decoded.lands_in == "device"
    assert not isinstance(decoded.array, np.ndarray)
    assert decoded.array.device.startswith("cuda")
    # It can still describe itself without a copy — what a real CUDA tensor offers.
    assert decoded.array.shape == (8, 8, 3)


def test_a_device_decoder_copies_to_host_only_through_an_explicit_call():
    """Every D2H copy should be a visible call, not a buffer-protocol accident."""
    decoded = get_decoder("fake.device").decode(_packet(b"\x01" * 192), ctx=DecodeContext())
    host = decoded.array.to_host()
    assert isinstance(host, np.ndarray)
    assert host.shape == (8, 8, 3)


# ── image: the promoted warning ───────────────────────────────────────
requires_pillow = pytest.mark.skipif(
    not DECODERS.is_available("image.jpeg"), reason="Pillow is not installed"
)


def _encode(fmt: str, size: tuple[int, int] = (64, 64)) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (*size, 3), dtype="uint8")
    buf = io.BytesIO()
    # quality=95 so the truncated half still carries real scan data rather than
    # compressing to almost nothing.
    Image.fromarray(pixels).save(buf, format=fmt, quality=95)
    return buf.getvalue()


@requires_pillow
def test_an_intact_jpeg_decodes_to_hwc_uint8_and_stays_that_way():
    """Decode stops at uint8 HWC on purpose.

    The permute to CHW, the float32 cast and the /255 are a copy with a 4x
    blowup, and they belong in the collate where a *batch* pays for them once.
    """
    decoded = get_decoder("image.jpeg").decode(_packet(_encode("JPEG")), ctx=DecodeContext())
    assert decoded.layout == "hwc"
    assert decoded.dtype == "uint8"
    assert decoded.array.shape == (64, 64, 3)
    assert decoded.meta["to_chw_float_in"] == "collate_fn"


@requires_pillow
def test_pillow_would_return_a_grey_bottomed_image_for_a_truncated_jpeg():
    """The behaviour this decoder exists to refuse — pinned so the claim is real.

    Left as an explicit test rather than a comment because if Pillow ever starts
    raising here on its own, the promotion below becomes redundant and we should
    find out from a failure rather than never.
    """
    from PIL import Image, ImageFile

    payload = _encode("JPEG")
    truncated = payload[: len(payload) // 2]

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = Image.open(io.BytesIO(truncated))
        image.load()  # no exception
        array = np.asarray(image)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

    flat_rows = int((array.reshape(array.shape[0], -1).std(axis=1) < 1e-6).sum())
    assert flat_rows > 0, "expected the filled-grey scanlines that make this dangerous"


@requires_pillow
def test_a_truncated_jpeg_raises_instead_of_decoding_to_grey():
    """The single most valuable line in the image decoder."""
    payload = _encode("JPEG")
    with pytest.raises(OSError, match="truncated"):
        get_decoder("image.jpeg").decode(_packet(payload[: len(payload) // 2]), ctx=DecodeContext())


@requires_pillow
def test_truncation_still_raises_when_something_else_enabled_the_global_override():
    """Some libraries set LOAD_TRUNCATED_IMAGES = True at import.

    Assuming it is False would silently re-enable exactly the behaviour we are
    refusing, so the decoder sets and restores it rather than trusting it.
    """
    from PIL import ImageFile

    payload = _encode("JPEG")
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with pytest.raises(OSError, match="truncated"):
            get_decoder("image.jpeg").decode(
                _packet(payload[: len(payload) // 2]), ctx=DecodeContext()
            )
        # ...and the caller's setting is left exactly as it was found.
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


@requires_pillow
def test_a_corrupted_png_raises_because_its_checksums_are_real():
    """PNG is `checked`: Adler-32 per zlib block, CRC-32 per chunk.

    The contrast with JPEG above is the point of the whole integrity column —
    one format detects damage itself, the other has to be made to.
    """
    payload = bytearray(_encode("PNG"))
    # Well past the 8-byte signature and IHDR, into compressed image data.
    payload[len(payload) // 2] ^= 0xFF
    with pytest.raises(Exception) as excinfo:
        get_decoder("image.png").decode(_packet(bytes(payload)), ctx=DecodeContext())
    assert excinfo.type is not AssertionError


# ── decoder_for: the single resolution point ──────────────────────────
def test_resolution_prefers_media_type_over_suffix():
    """The index records what the bytes ARE; a suffix records what they are named."""
    assert streaming._match(media_type="audio/flac", suffix=".wav") == "audio.flac"


def test_resolution_falls_back_to_suffix_before_a_corpus_is_materialized():
    assert streaming._match(suffix=".mp4") == "video.h264"
    assert streaming._match(suffix=".WAV") == "audio.pcm"


def test_an_explicit_decoder_choice_is_never_second_guessed():
    decoder = streaming.decoder_for(explicit="text.tokens", suffix=".wav")
    assert type(decoder).__name__ == "TokenDecoder"


def test_an_unresolvable_sample_names_what_is_registered():
    with pytest.raises(LookupError) as excinfo:
        streaming.decoder_for(suffix=".xyz")
    assert "audio.pcm" in str(excinfo.value)


def test_decoder_params_reach_the_decoder_and_unknown_keys_raise():
    """`decoder_params` has exactly one owner: the decoder's own factory.

    A silently dropped knob is how a run ends up not doing what the config says.
    """
    decoder = streaming.decoder_for(suffix=".bin", params={"dtype": "uint32"})
    assert decoder.dtype == "uint32"

    with pytest.raises(ValueError, match="unknown decoder_params"):
        streaming.decoder_for(suffix=".bin", params={"dtpye": "uint32"})


# ── Stage defaults ────────────────────────────────────────────────────
def test_the_default_demux_yields_one_packet_and_copies_nothing():
    """The degenerate stage must not become a copy for the formats that skip it."""
    array = np.arange(8, dtype="uint16")
    blob_source = _StubSource(array)
    decoder = get_decoder("text.tokens")
    blob = decoder.read(SampleRef(index=0, shard="s", key="k"), source=blob_source)

    packets = list(decoder.demux(blob))
    assert len(packets) == 1
    assert packets[0].data is array
    assert packets[0].opaque is None


def test_decode_ref_runs_all_three_stages_in_order():
    array = np.arange(4, dtype="uint16")
    decoded = get_decoder("text.tokens").decode_ref(
        SampleRef(index=0, shard="s", key="k"),
        source=_StubSource(array),
        ctx=DecodeContext(),
    )
    np.testing.assert_array_equal(decoded.array, array)


class _StubSource:
    """Minimal :class:`BlobSource`. The real ones arrive with the shard index."""

    def __init__(self, payload) -> None:
        self.payload = payload

    def read_range(self, ref: SampleRef):
        return self.payload

    def open(self, shard: str):  # pragma: no cover - unused by these decoders
        raise NotImplementedError

    def close(self) -> None:
        return None
