"""
data/streaming/materialize.py
─────────────────────────────
The offline pass: walk a corpus, decode-probe every sample, and write the index.

## Why this earns its cost

Two of the integrity classifications are invisible at training time by
construction. An MP3 resyncs past damage and returns *shorter audio with no
error*; a damaged H.264 stream emits *artifacted frames* while libavcodec merely
logs; a hardware decoder emits *green frames* with nothing surfaced at all. Every
one of those produces a correctly-shaped tensor, so no exception handler anywhere
in the training loop will ever see them. They do not crash the run — they quietly
make the model worse, and the only symptom is a number you cannot explain months
later.

The only way to catch that class of failure is to decode each sample once, offline,
and compare the result against what the container *claims*:

    MP3 resync        decoded duration vs the frame-count-derived expectation
    H.264 artifacts   decoded frame count vs the container's declared frame count
    device decoders   decode again through the spec's host ``oracle`` and compare
    integrity: none   nothing to compare -- record a digest, which is the whole
                      defence a flat token shard has

## Why the fault ceiling is enforced *here* and nowhere else

``max_fault_rate`` is a content-dependent abort. At training time under DDP, rank 0
could trip it while rank 1 does not — and a rank-divergent abort is exactly the
collective hang this phase exists to prevent. Here there is one process, no
collective exists yet, and failing is safe, deterministic and early. So this is the
only place the ceiling raises; at runtime it is counted and logged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.registry import DECODERS, get_decoder
from ...core.types import FrameworkError
from . import suffix_of
from .integrity import SampleFault
from .shards import ShardEntry, ShardIndex, ShardIndexWriter, digest_bytes
from .sources_io import DirSource, TarSource, scan_tar
from .stages import DecodeContext, SampleRef

log = logging.getLogger(__name__)

# A decoded duration may legitimately differ from the frame-count expectation by a
# partial final frame. 2% is comfortably above that and far below a resync, which
# drops whole seconds.
DURATION_TOLERANCE = 0.02

# How closely a device decoder's output must match its host oracle. Not exact:
# a hardware YUV->RGB and libswscale's differ by a rounding step in the colour
# conversion, and demanding equality would fail every real GPU decoder.
ORACLE_TOLERANCE = 2.0


class MaterializeError(FrameworkError):
    """The corpus cannot be materialized, or is too damaged to train on."""


@dataclass
class MaterializeReport:
    """What the pass found. Printed as a table and returned for the exit code."""

    n_seen: int = 0
    n_ok: int = 0
    faults: list[SampleFault] = field(default_factory=list)
    shards: list[str] = field(default_factory=list)
    decoder: str = ""
    sampled: bool = False

    @property
    def n_faults(self) -> int:
        return len(self.faults)

    @property
    def fault_rate(self) -> float:
        return self.n_faults / self.n_seen if self.n_seen else 0.0

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fault in self.faults:
            counts[fault.kind] = counts.get(fault.kind, 0) + 1
        return dict(sorted(counts.items()))

    def render(self) -> str:
        lines = [
            f"decoder      {self.decoder}",
            f"shards       {len(self.shards)}",
            f"samples      {self.n_seen}" + ("  (sampled)" if self.sampled else ""),
            f"decoded ok   {self.n_ok}",
            f"faults       {self.n_faults}  ({self.fault_rate:.2%})",
        ]
        for kind, count in self.by_kind().items():
            lines.append(f"  {kind:<14} {count}")
        # Enough to act on, not so many that the ceiling message scrolls away.
        for fault in self.faults[:10]:
            lines.append(f"  - {fault.summary()}")
        if self.n_faults > 10:
            lines.append(f"  ... and {self.n_faults - 10} more (see faults.jsonl)")
        return "\n".join(lines)


def discover(path: str | Path) -> tuple[list[dict[str, Any]], Any, list[str]]:
    """Every candidate sample under ``path``, plus the source that can read them.

    Two layouts, decided by what is actually there rather than by configuration:
    a directory containing ``.tar`` files is a sharded corpus, anything else is a
    directory of per-sample files.
    """
    root = Path(path)
    if not root.exists():
        raise MaterializeError(f"no such corpus: {root}")

    tars = sorted(p for p in root.glob("*.tar"))
    if tars:
        candidates: list[dict[str, Any]] = []
        for tar in tars:
            for member in scan_tar(tar):
                candidates.append({"shard": tar.name, **member})
        return candidates, TarSource(root), [t.name for t in tars]

    files = sorted(p for p in root.rglob("*") if p.is_file() and "_mlf_shards" not in p.parts)
    candidates = [
        {
            "shard": _shard_name_of(p, root),
            "key": p.relative_to(root).as_posix(),
            "offset": 0,
            "nbytes": p.stat().st_size,
        }
        for p in files
    ]
    shards = sorted({c["shard"] for c in candidates})
    return candidates, DirSource(root), shards


def _shard_name_of(path: Path, root: Path) -> str:
    """A loose-file corpus still has shards: its top-level directories.

    Substitution draws from within a shard and the block shuffle keeps reads
    inside one, so a corpus with no shard concept would make both degenerate.
    Using the first path component gives ``ImageFolder``-style class directories a
    sensible grouping for free.
    """
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "shard-0000"


def materialize(
    path: str | Path,
    *,
    index_dir: str | Path | None = None,
    decoder: str | None = None,
    decoder_params: dict[str, Any] | None = None,
    data_kind: str = "tabular",
    labels: dict[str, int] | None = None,
    class_names: Sequence[str] = (),
    max_fault_rate: float = 0.01,
    sample_rate: float = 1.0,
    seed: int = 42,
    probe: bool = True,
) -> tuple[ShardIndex, MaterializeReport]:
    """Walk ``path``, probe every sample, write the index, and report.

    Raises :class:`MaterializeError` when the fault rate exceeds
    ``max_fault_rate`` — the one place that ceiling is enforced.
    """
    candidates, source, shards = discover(path)
    if not candidates:
        raise MaterializeError(f"{path} holds no readable samples")

    decoder_name = decoder or _infer_decoder(candidates)
    spec = DECODERS.get_spec(decoder_name)
    engine = get_decoder(decoder_name, **(decoder_params or {}))
    ctx = DecodeContext()
    oracle = get_decoder(spec.oracle) if (probe and spec.oracle and _wants_oracle(spec)) else None

    # A class-directory corpus already states its labels in its layout, and the
    # shard name IS the top-level directory. Inferring them here is what lets fold
    # planning and class-balance counting be a JSON scan later rather than a decode
    # pass over the whole corpus. An explicit `labels` mapping still wins.
    if labels is None:
        labels, class_names = _labels_from_shards(candidates, class_names)
    chosen = _subsample(candidates, sample_rate=sample_rate, seed=seed)
    report = MaterializeReport(
        shards=shards, decoder=decoder_name, sampled=len(chosen) < len(candidates)
    )

    root = ShardIndex.location(path, index_dir)
    with ShardIndexWriter(root, data_kind=data_kind) as writer:
        for i, candidate in enumerate(chosen):
            entry, fault = _probe_one(
                i,
                candidate,
                source=source,
                decoder=engine,
                ctx=ctx,
                spec=spec,
                oracle=oracle,
                labels=labels or {},
                probe=probe,
            )
            report.n_seen += 1
            if fault is not None:
                report.faults.append(fault)
            else:
                report.n_ok += 1
            # The entry is written either way: the index declares what SHOULD
            # exist, and a sample that failed today is what substitution is for.
            # Dropping it would shrink `n_samples`, which is the one number that
            # must not depend on what happens to decode.
            writer.add(entry)

        index = writer.finalize(
            decoder_hint=decoder_name,
            class_names=class_names,
            faults=report.n_faults,
            materialized=probe,
        )

    _write_faults(root, report.faults)
    source.close()
    engine.close()

    if report.fault_rate > max_fault_rate:
        raise MaterializeError(
            f"{report.n_faults} of {report.n_seen} samples failed to decode "
            f"({report.fault_rate:.2%}), above data.integrity.max_fault_rate "
            f"({max_fault_rate:.2%}). Fix the corpus, or raise the ceiling deliberately.\n\n"
            + report.render()
        )
    return index, report


def _probe_one(
    i: int,
    candidate: dict[str, Any],
    *,
    source: Any,
    decoder: Any,
    ctx: DecodeContext,
    spec: Any,
    oracle: Any,
    labels: dict[str, int],
    probe: bool,
) -> tuple[ShardEntry, SampleFault | None]:
    key = candidate["key"]
    ref = SampleRef(
        index=i,
        shard=candidate["shard"],
        key=key,
        offset=int(candidate.get("offset", 0)),
        nbytes=int(candidate.get("nbytes", 0)),
    )
    entry = ShardEntry(
        i=i,
        shard=ref.shard,
        key=key,
        offset=ref.offset,
        nbytes=ref.nbytes,
        media_type=_media_type_for(spec, key),
        label=labels.get(key),
    )
    if not probe:
        return entry, None

    def fault(kind: str, detail: str) -> SampleFault:
        return SampleFault(
            index=i,
            shard=ref.shard,
            key=key,
            stage="decode",
            kind=kind,  # type: ignore[arg-type]
            detail=detail,
            decoder=spec.name,
        )

    try:
        blob = decoder.read(ref, source=source)
        digest = digest_bytes(blob.data)
        decoded = decoder.decode(decoder.demux(blob), ctx=ctx)
    except Exception as exc:  # noqa: BLE001 - any codec may raise anything
        return entry, fault("decode_error", f"{type(exc).__name__}: {exc}")

    problem = _cross_check(decoded, spec=spec, oracle=oracle, ref=ref, source=source, ctx=ctx)
    n_units = _n_units(decoded)
    entry = ShardEntry(
        i=i,
        shard=ref.shard,
        key=key,
        offset=ref.offset,
        nbytes=ref.nbytes,
        media_type=entry.media_type,
        label=entry.label,
        digest=digest,
        n_units=n_units,
    )
    if problem is not None:
        kind, detail = problem
        return entry, fault(kind, detail)
    return entry, None


def _cross_check(
    decoded: Any, *, spec: Any, oracle: Any, ref: SampleRef, source: Any, ctx: DecodeContext
) -> tuple[str, str] | None:
    """The checks that only an offline pass can make.

    Each targets one row the runtime cannot see. Returning ``None`` means the
    sample decoded to something consistent with what the container claimed.
    """
    meta = decoded.meta

    # MP3's silent resync: shorter audio, no error. The frame count is the only
    # independent statement of how long the sample should have been.
    expected = meta.get("expected_samples")
    actual = meta.get("n_samples")
    if expected and actual:
        drift = abs(actual - expected) / expected
        if drift > DURATION_TOLERANCE:
            return (
                "duration",
                f"decoded {actual} samples but the frame count implies {expected} "
                f"({drift:.1%} short) -- the hallmark of a silent resync past damage",
            )

    # Mid-stream H.264 damage: libavcodec logs and emits artifacted frames.
    declared = meta.get("declared_frames")
    got = meta.get("n_frames")
    if declared and got and got < declared:
        wanted = min(declared, meta.get("frames_wanted") or declared)
        if got < wanted:
            return (
                "shape",
                f"decoded {got} frames but the container declares {declared}; "
                "mid-stream corruption emits artifacted frames without raising",
            )

    # A device decoder cannot report its own damage at all. Decode again through a
    # host path and compare -- the practical answer for NVDEC vs CPU.
    if oracle is not None and decoded.lands_in == "device":
        try:
            reference = oracle.decode(oracle.demux(oracle.read(ref, source=source)), ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            return ("decode_error", f"oracle {spec.oracle} could not decode it either: {exc}")
        host = decoded.array.to_host()
        if host.shape != reference.array.shape:
            return (
                "shape",
                f"device decoder produced {host.shape}, oracle {spec.oracle} "
                f"produced {reference.array.shape}",
            )
        drift = float(np.abs(host.astype("float32") - reference.array.astype("float32")).max())
        if drift > ORACLE_TOLERANCE:
            return (
                "decode_error",
                f"device output differs from oracle {spec.oracle} by up to {drift:.1f} "
                "-- the green-frame signature, invisible at training time",
            )
    return None


def _n_units(decoded: Any) -> int | None:
    for key in ("n_samples", "n_frames", "n_tokens"):
        value = decoded.meta.get(key)
        if value is not None:
            return int(value)
    shape = decoded.shape
    return int(shape[0]) if shape else None


def _wants_oracle(spec: Any) -> bool:
    """Only cross-decode when the decoder genuinely cannot self-report.

    A ``checked`` format already raises on damage; decoding it twice would double
    the cost of materialization to learn nothing.
    """
    return spec.integrity in ("silent", "none") or spec.lands_in == "device"


def _media_type_for(spec: Any, key: str) -> str:
    """The media type the index records, preferring the decoder's own vocabulary."""
    suffix = suffix_of(key)
    for candidate in DECODERS.specs():
        if suffix in candidate.suffixes and candidate.media_types:
            return candidate.media_types[0]
    return spec.media_types[0] if spec.media_types else ""


def _labels_from_shards(
    candidates: list[dict[str, Any]], class_names: Sequence[str]
) -> tuple[dict[str, int], Sequence[str]]:
    """``key -> class index`` from the shard (= class directory) names.

    Sorted, so the mapping is stable across runs and machines — a class order that
    depended on filesystem iteration would silently relabel the corpus when the
    directory was copied, and every metric would still look plausible.

    Returns an empty mapping when there is only one shard: a single class is not a
    classification corpus, and labelling everything ``0`` would be a claim rather
    than an inference.
    """
    shards = sorted({str(c["shard"]) for c in candidates})
    if len(shards) < 2:
        return {}, class_names
    index = {name: i for i, name in enumerate(shards)}
    mapping = {str(c["key"]): index[str(c["shard"])] for c in candidates}
    return mapping, tuple(class_names) or tuple(shards)


def _infer_decoder(candidates: Iterable[dict[str, Any]]) -> str:
    """The decoder matching the most common suffix in the corpus.

    A majority vote rather than the first match: a directory of FLACs with one
    stray README should materialize as audio, and picking by first-seen makes the
    result depend on filesystem ordering.
    """
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[suffix_of(candidate["key"])] = counts.get(suffix_of(candidate["key"]), 0) + 1

    for suffix, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        for spec in DECODERS.specs():
            if suffix in spec.suffixes:
                return spec.name
    raise MaterializeError(
        f"no decoder matches the suffixes in this corpus ({sorted(counts)}). "
        f"Pass --decoder explicitly; registered: {sorted(DECODERS.names())}"
    )


def _subsample(
    candidates: list[dict[str, Any]], *, sample_rate: float, seed: int
) -> list[dict[str, Any]]:
    """A deterministic fraction of the corpus, for a quick check.

    Sorted back into corpus order after sampling so the written index is stable
    and its entries stay grouped by shard.
    """
    if sample_rate >= 1.0:
        return candidates
    if not 0.0 < sample_rate < 1.0:
        raise MaterializeError(f"--sample-rate must be in (0, 1], got {sample_rate}")
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(candidates) * sample_rate)))
    picked = sorted(rng.choice(len(candidates), size=n, replace=False))
    return [candidates[i] for i in picked]


def _write_faults(root: Path, faults: list[SampleFault]) -> None:
    """``faults.jsonl`` beside the index. Always written, empty when clean.

    An absent file would be indistinguishable from an index written before this
    check existed — the same rule ``hpo.json`` follows.
    """
    import json

    path = root / "faults.jsonl"
    path.write_text(
        "".join(json.dumps(f.to_dict(), sort_keys=True) + "\n" for f in faults),
        encoding="utf-8",
    )
