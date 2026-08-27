"""
data/streaming/shards.py
────────────────────────
The shard index: what samples exist, where their bytes are, and what they hashed
to when they were last known good.

Two files, because they are read completely differently.

``shards.json`` — the manifest. Small, fully parsed, and the source of the one
number the whole DDP-safety argument rests on: ``n_samples``. It is *declared*,
never derived from what happens to decode.

``entries.jsonl`` — one JSON object per sample. JSON Lines rather than a single
document because a large index is not something to ``json.loads`` whole: it
appends during materialization, streams during verification, and is greppable when
something has gone wrong at 3am.

## Why ``len()`` never touches the entries

Under DDP every rank must produce an identical number of batches or the next
collective hangs with no error message. That reduces to one property:

    len(dataset) is a constant read from this manifest, and __getitem__ is total.

If the length were derived from what decodes successfully, two ranks reading
different shards could disagree about it, and the disagreement would surface as a
hang rather than an error. So :attr:`ShardIndex.n_samples` comes from
``shards.json`` — a file byte-identical on every rank — and corrupt samples are
substituted inside ``__getitem__``, which changes *which* sample is served and
never *how many*.

## The digest, and why it is not optional for some formats

Each entry records a BLAKE2b-128 of its raw bytes. For a format that validates
itself (FLAC's CRC-16, PNG's CRC-32) this is belt-and-braces and
``verify_checksums: "auto"`` skips it. For a flat ``uint16`` token shard it is the
*only* thing standing between a flipped bit and a valid-looking token id, which is
why ``integrity: "none"`` turns verification on.

``index_digest`` covers the entries file as a whole. A checkpoint records it, and a
resume against a changed corpus is refused rather than silently replaying a
different dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...core.types import DataKind, FrameworkError

# The directory that holds an index, placed beside the data it describes. The
# index is a property of the *dataset*, not of a run: two runs over one corpus
# must share it, or the fault-detection cost is paid twice and the shuffle order
# silently changes between them.
INDEX_DIRNAME = "_mlf_shards"
MANIFEST_NAME = "shards.json"
ENTRIES_NAME = "entries.jsonl"

INDEX_VERSION = 1

# 16 bytes. Long enough that a collision is not a thing that happens, short enough
# that 15M of them are 480 MB of hex rather than a gigabyte. BLAKE2b rather than
# SHA-256 because it is roughly twice as fast and this runs over every byte of the
# corpus during materialization.
DIGEST_ALGORITHM = "blake2b-128"
_DIGEST_SIZE = 16

# Above this, holding every entry as a Python object stops being reasonable
# (~200 bytes each) and the index wants a binary offset table instead. Warned
# about rather than enforced: the ceiling is a property of this implementation,
# not of the format, and a user who knows that should not be blocked.
LARGE_INDEX_WARNING = 5_000_000


class ShardIndexError(FrameworkError):
    """The index is missing, malformed, or disagrees with itself."""


def digest_bytes(payload: bytes | memoryview | np.ndarray) -> str:
    """BLAKE2b-128 of a sample's raw bytes, as hex.

    Accepts an array so a memmap slice can be hashed without being copied into a
    ``bytes`` first — which for a token shard would defeat the point of memmapping
    it at all.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    if isinstance(payload, np.ndarray):
        hasher.update(memoryview(np.ascontiguousarray(payload)).cast("B"))
    else:
        hasher.update(payload)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class ShardEntry:
    """One sample: where its bytes are, and what they were.

    ``label`` lives here so fold planning and class-balance counting can read the
    targets **without decoding anything** — the analogue of reading
    ``ImageFolder.targets`` rather than opening every JPEG. For a corpus of 4-second
    clips that is the difference between a second and an hour.

    ``n_units`` is samples for audio, frames for video, tokens for a text shard:
    the count materialization compares against what actually decodes. That
    comparison is the only way an MP3's silent resync is ever detected.
    """

    i: int
    shard: str
    key: str
    offset: int = 0
    nbytes: int = 0
    media_type: str = ""
    label: int | None = None
    digest: str = ""
    n_units: int | None = None

    def to_dict(self) -> dict[str, Any]:
        # Drop the Nones: a 15M-line file should not carry "label": null 15M times.
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShardEntry:
        return cls(
            i=int(raw["i"]),
            shard=str(raw["shard"]),
            key=str(raw.get("key", "")),
            offset=int(raw.get("offset", 0)),
            nbytes=int(raw.get("nbytes", 0)),
            media_type=str(raw.get("media_type", "")),
            label=None if raw.get("label") is None else int(raw["label"]),
            digest=str(raw.get("digest", "")),
            n_units=None if raw.get("n_units") is None else int(raw["n_units"]),
        )


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """A materialized corpus: the manifest, plus the entries it declares.

    Construct through :meth:`read` or :class:`ShardIndexWriter`, not directly —
    both guarantee the manifest and the entries agree about ``n_samples``, which
    is the invariant everything downstream leans on.
    """

    n_samples: int
    entries: tuple[ShardEntry, ...]
    shards: tuple[str, ...]
    data_kind: DataKind | str = "tabular"
    decoder_hint: str = ""
    index_digest: str = ""
    materialized_at: str = ""
    class_names: tuple[str, ...] = ()
    faults: int = 0
    index_version: int = INDEX_VERSION
    # index -> position within `entries`. Not the identity map when an index was
    # written out of order, which a parallel materialization pass can do.
    _by_index: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.entries) != self.n_samples:
            raise ShardIndexError(
                f"manifest declares {self.n_samples} samples but entries.jsonl holds "
                f"{len(self.entries)}. The index is inconsistent; re-run `mlf materialize`."
            )
        object.__setattr__(self, "_by_index", {e.i: pos for pos, e in enumerate(self.entries)})

    def __len__(self) -> int:
        """The **declared** sample count.

        Deliberately not ``len(self.entries)`` even though a validated index makes
        them equal: this is the number that must be identical on every rank, and
        reading it from the manifest is what makes that true by construction
        rather than by coincidence.
        """
        return self.n_samples

    def entry(self, index: int) -> ShardEntry:
        try:
            return self.entries[self._by_index[index]]
        except KeyError:
            raise IndexError(f"index {index} is not in this shard index") from None

    @property
    def is_materialized(self) -> bool:
        """Whether an offline decode-probe pass has actually been run.

        The gate for a corpus whose decoder cannot report its own damage. An index
        can exist without this — writing one is cheap, probing every sample is not.
        """
        return bool(self.materialized_at)

    def labels(self) -> np.ndarray | None:
        """Every entry's label, in index order, **without decoding anything**.

        ``None`` when any entry lacks one, rather than a partially-filled array: a
        stratified split over half-known labels is worse than a refusal to
        stratify.
        """
        values = [self.entry(i).label for i in range(self.n_samples)]
        if any(v is None for v in values):
            return None
        return np.asarray(values, dtype="int64")

    def indices_in_shard(self, shard: str) -> tuple[int, ...]:
        """Every index whose bytes live in ``shard``.

        Substitution draws from here, which is what keeps a re-draw local to an
        already-open shard instead of seeking across the corpus.
        """
        return tuple(e.i for e in self.entries if e.shard == shard)

    def shard_groups(self) -> dict[str, tuple[int, ...]]:
        """``shard -> indices``, in declaration order. Built once by the sampler."""
        groups: dict[str, list[int]] = {name: [] for name in self.shards}
        for entry in self.entries:
            groups.setdefault(entry.shard, []).append(entry.i)
        return {name: tuple(idx) for name, idx in groups.items()}

    # ── IO ──
    @staticmethod
    def location(data_path: str | Path, index_dir: str | Path | None = None) -> Path:
        """Where an index for ``data_path`` lives.

        Beside the data by default; ``index_dir`` overrides it for a read-only
        mount, which is the common case for a shared corpus.
        """
        if index_dir is not None:
            return Path(index_dir)
        path = Path(data_path)
        base = path if path.is_dir() else path.parent
        return base / INDEX_DIRNAME

    @classmethod
    def read(cls, directory: str | Path) -> ShardIndex:
        root = Path(directory)
        manifest_path = root / MANIFEST_NAME
        entries_path = root / ENTRIES_NAME
        if not manifest_path.is_file():
            raise ShardIndexError(f"no shard index at {root}. Run `mlf materialize` to build one.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        version = int(manifest.get("index_version", 0))
        if version > INDEX_VERSION:
            # Refuse forward, like `read_manifest`: a newer index may carry fields
            # whose absence changes behaviour silently.
            raise ShardIndexError(
                f"shard index version {version} is newer than this build understands "
                f"({INDEX_VERSION}). Upgrade ml-framework, or re-materialize."
            )

        entries = tuple(_stream_entries(entries_path))
        return cls(
            n_samples=int(manifest["n_samples"]),
            entries=entries,
            shards=tuple(manifest.get("shards", ())),
            data_kind=manifest.get("data_kind", "tabular"),
            decoder_hint=manifest.get("decoder_hint", ""),
            index_digest=manifest.get("index_digest", ""),
            materialized_at=manifest.get("materialized_at", ""),
            class_names=tuple(manifest.get("class_names", ())),
            faults=int(manifest.get("faults", 0)),
            index_version=version,
        )

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        root = Path(directory)
        return (root / MANIFEST_NAME).is_file() and (root / ENTRIES_NAME).is_file()

    def manifest(self) -> dict[str, Any]:
        """The manifest as it is written. Also what the bundle records.

        Only this half goes into a trained bundle — never the entries — so a served
        artifact can say which corpus version it trained on without carrying a
        copy of the whole index.
        """
        return {
            "index_version": self.index_version,
            "data_kind": self.data_kind,
            "n_shards": len(self.shards),
            "n_samples": self.n_samples,
            "shards": list(self.shards),
            "digest_algorithm": DIGEST_ALGORITHM,
            "index_digest": self.index_digest,
            "materialized_at": self.materialized_at,
            "decoder_hint": self.decoder_hint,
            "class_names": list(self.class_names),
            "faults": self.faults,
        }


def _stream_entries(path: Path) -> Iterator[ShardEntry]:
    """Parse ``entries.jsonl`` a line at a time.

    Streamed rather than ``read_text().splitlines()`` so peak memory is one line,
    not the whole file — the property that makes JSON Lines the right shape here.
    """
    if not path.is_file():
        raise ShardIndexError(f"shard index at {path.parent} has no {ENTRIES_NAME}")
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ShardEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ShardIndexError(f"{path}:{lineno} is not a valid index entry: {exc}") from exc


class ShardIndexWriter:
    """Builds an index incrementally, then finalizes the manifest.

    Two-pass by construction: entries are appended as they are discovered (so a
    materialization pass over a 10 TB corpus does not hold them all), and
    ``n_samples`` plus ``index_digest`` are only known once the last one is in.
    Writing the manifest *last* is what guarantees a manifest never claims a count
    the entries file cannot satisfy — a half-written index is detectably
    half-written rather than quietly wrong.
    """

    def __init__(self, directory: str | Path, *, data_kind: str = "tabular") -> None:
        self.directory = Path(directory)
        self.data_kind = data_kind
        self._entries: list[ShardEntry] = []
        self._shards: list[str] = []
        self._handle: Any = None

    def __enter__(self) -> ShardIndexWriter:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = (self.directory / ENTRIES_NAME).open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def add(self, entry: ShardEntry) -> None:
        if self._handle is None:
            raise RuntimeError("use ShardIndexWriter as a context manager")
        self._handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        self._entries.append(entry)
        if entry.shard not in self._shards:
            self._shards.append(entry.shard)

    def finalize(
        self,
        *,
        decoder_hint: str = "",
        class_names: Sequence[str] = (),
        faults: int = 0,
        materialized: bool = True,
    ) -> ShardIndex:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

        entries_path = self.directory / ENTRIES_NAME
        index = ShardIndex(
            n_samples=len(self._entries),
            entries=tuple(self._entries),
            shards=tuple(self._shards),
            data_kind=self.data_kind,
            decoder_hint=decoder_hint,
            # Over the entries file as written: any change to any entry changes it,
            # which is exactly the property a resume needs to check.
            index_digest=digest_bytes(entries_path.read_bytes()),
            materialized_at=(
                datetime.now(timezone.utc).isoformat(timespec="seconds") if materialized else ""
            ),
            class_names=tuple(class_names),
            faults=faults,
        )
        (self.directory / MANIFEST_NAME).write_text(
            json.dumps(index.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return index
