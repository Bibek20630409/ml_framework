"""Tensor construction: when it is free, when it copies, and why uint16 never gets here.

Decode and tensor construction are independent axes. Decode reverses a compression
scheme; tensor construction attaches a dtype, a shape and strides to a pointer —
and it only has to *copy* when the pointer is not already pointing at what torch
needs.

The previous `torch.tensor(np.asarray(x, dtype="float32"))` copied unconditionally,
and twice when the input was float64. These tests pin the three conditions under
which it is now free, and the one dtype that is deliberately never handled here.

**Everything is asserted on dtypes and buffer identity, never on RSS.** A memory
measurement is flaky by construction — the allocator, the GC and the page cache all
move underneath it — and a flaky test about a performance property is worse than no
test, because it gets muted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml_framework.data.lightning_adapter import _to_tensor

pytestmark = pytest.mark.unit


# ── Free ──────────────────────────────────────────────────────────────
def test_a_c_contiguous_float32_array_is_wrapped_without_a_copy():
    """The fast path, and the one every existing tabular bundle already takes."""
    array = np.arange(12, dtype="float32").reshape(3, 4)
    tensor = _to_tensor(array)

    assert np.shares_memory(array, tensor.numpy())
    assert tensor.dtype == torch.float32


def test_the_wrap_is_a_view_so_writing_through_one_is_visible_in_the_other():
    """The observable consequence of zero-copy, and the reason WRITEABLE matters."""
    array = np.zeros((2, 2), dtype="float32")
    tensor = _to_tensor(array)

    tensor[0, 0] = 5.0
    assert array[0, 0] == 5.0


# ── Copies, each for a stated reason ──────────────────────────────────
def test_float64_is_cast_rather_than_wrapped():
    """torch computes in float32 here; a float64 buffer is the wrong dtype and
    there is no way to attach strides that fixes that."""
    array = np.arange(6, dtype="float64")
    tensor = _to_tensor(array)

    assert tensor.dtype == torch.float32
    assert not np.shares_memory(array, tensor.numpy())


def test_a_fortran_ordered_array_is_made_contiguous():
    """The right dtype in the wrong layout. `from_numpy` would honour the strides
    and every downstream kernel would pay for them."""
    array = np.asfortranarray(np.arange(12, dtype="float32").reshape(3, 4))
    assert not array.flags["C_CONTIGUOUS"]

    tensor = _to_tensor(array)

    assert not np.shares_memory(array, tensor.numpy())
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_a_read_only_array_is_copied_rather_than_wrapped():
    """A memmap view is read-only, and `from_numpy` on one produces a tensor whose
    in-place operations are undefined behaviour. torch *warns* rather than raising,
    so relying on it to complain is not an option — hence the explicit check.
    """
    array = np.arange(6, dtype="float32")
    array.flags.writeable = False

    tensor = _to_tensor(array)

    assert not np.shares_memory(array, tensor.numpy())
    # And the result is writeable, so an in-place op downstream is well-defined.
    tensor[0] = 99.0
    assert array[0] == 0.0


def test_a_strided_view_is_copied():
    array = np.arange(20, dtype="float32")[::2]
    assert not array.flags["C_CONTIGUOUS"]

    tensor = _to_tensor(array)

    assert not np.shares_memory(array, tensor.numpy())
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_every_path_produces_the_same_values():
    """Zero-copy is an optimization, and an optimization that changes numbers is a
    bug. This is the property the byte-identical-artifacts gate rests on."""
    reference = np.arange(12, dtype="float64").reshape(3, 4)

    variants = [
        np.ascontiguousarray(reference, dtype="float32"),
        reference,
        np.asfortranarray(reference.astype("float32")),
    ]
    for array in variants:
        np.testing.assert_array_equal(_to_tensor(array).numpy(), reference.astype("float32"))


# ── The dtype that never arrives here ─────────────────────────────────
def test_uint16_tokens_are_not_something_this_function_widens():
    """torch has no usable uint16 arithmetic, so a token corpus must be cast — but
    the cast belongs in the collate, per BATCH over a few MB, not here, over the
    whole corpus with a 4x blowup.

    At 15T tokens that is 30 TB on disk versus 60 TB, against a memcpy-bound
    operation per step. This test exists so that "handle uint16 here" reads as the
    regression it would be.
    """
    tokens = np.array([1, 2, 65535], dtype="uint16")
    tensor = _to_tensor(tokens)

    # It falls to the copy path and lands as float32 -- which is exactly why a
    # token corpus is a `dataset` payload and never reaches this function.
    assert tensor.dtype == torch.float32
    assert not np.shares_memory(tokens, tensor.numpy())


def test_a_token_decoder_keeps_uint16_all_the_way_to_the_collate_boundary():
    """The corpus stays 2 bytes/token; only the batch is widened.

    Asserted on the decoder's own output rather than on a training run, because
    this is a property of the decode stage and it holds whether or not anything
    downstream ever runs.
    """
    from ml_framework.core.registry import get_decoder
    from ml_framework.data.streaming.stages import DecodeContext, Packet

    tokens = np.arange(1024, dtype="uint16")
    decoded = get_decoder("text.tokens").decode(
        [Packet(data=tokens.tobytes())], ctx=DecodeContext()
    )

    assert decoded.dtype == "uint16"
    assert decoded.array.dtype == np.uint16
    assert decoded.meta["cast_to_int64_in"] == "collate_fn"

    # What the collate does, once per batch: a real copy, and a cheap one.
    batch = decoded.array[:32].astype("int64")
    assert batch.dtype == np.int64
    assert batch.nbytes == 32 * 8
    assert decoded.array.nbytes == 1024 * 2, "the corpus itself is untouched"


def test_a_memmapped_token_shard_stays_a_view_of_the_page_cache(tmp_path):
    """A read here is a page fault, not a copy. `arr.base` being the memmap is the
    evidence — and it is also why the read-only branch above exists at all."""
    from ml_framework.core.registry import get_decoder
    from ml_framework.data.streaming.sources_io import MemmapSource
    from ml_framework.data.streaming.stages import DecodeContext, SampleRef

    path = tmp_path / "tokens.bin"
    path.write_bytes(np.arange(256, dtype="uint16").tobytes())

    source = MemmapSource(path, dtype="uint16")
    decoder = get_decoder("text.tokens")
    ref = SampleRef(index=0, shard="tokens.bin", key="0", offset=0, nbytes=128)

    decoded = decoder.decode_ref(ref, source=source, ctx=DecodeContext())

    assert decoded.array.dtype == np.uint16
    assert isinstance(decoded.array.base, np.memmap)
    assert not decoded.array.flags["WRITEABLE"], "a memmap view is read-only"
    np.testing.assert_array_equal(decoded.array, np.arange(64, dtype="uint16"))
