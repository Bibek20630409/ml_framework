"""
utils/logging.py
────────────────
Central logging setup. Call ``setup_logging(output_dir)`` at the start of a run.

v1 tracked "is the file handler attached?" in a single process-global boolean, so
the *second* ``train()`` in one process logged into the *first* run's
``training.log`` — and the first run's file stayed open. That was harmless while
one process meant one run. It stops being harmless the moment tuning calls
``train()`` N times in a process, which is why the fix lands now rather than with
the tuning driver.

Handlers are tracked **per output directory** instead. Calling ``setup_logging``
for a new directory detaches the previous run's file handler and attaches (or
reuses) this one, so at most one file handler is live and each run's log contains
that run. Reusing a directory reattaches the same handler rather than opening a
second one — repeated calls with the same argument stay idempotent, which the CLI
depends on.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Marks the handlers this module owns, so detaching never touches a handler the
# host application attached to the root logger.
_OWNED = "_mlf_owned"

_stream_handler: logging.Handler | None = None
_file_handlers: dict[str, logging.FileHandler] = {}


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT)


def _ensure_stream(root: logging.Logger) -> None:
    global _stream_handler
    if _stream_handler is not None and _stream_handler in root.handlers:
        return
    if _stream_handler is None:
        # Log messages contain non-ASCII (→, ─); force UTF-8 so the Windows console
        # (cp1252 by default) doesn't raise UnicodeEncodeError.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass
        _stream_handler = logging.StreamHandler(sys.stdout)
        _stream_handler.setFormatter(_formatter())
        setattr(_stream_handler, _OWNED, True)
    root.addHandler(_stream_handler)


def _attach_file(root: logging.Logger, output_dir: Path) -> None:
    """Attach this run's file handler and detach any other run's."""
    key = str(output_dir.resolve())
    for other_key, other in _file_handlers.items():
        if other_key != key and other in root.handlers:
            root.removeHandler(other)

    handler = _file_handlers.get(key)
    if handler is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
        handler.setFormatter(_formatter())
        setattr(handler, _OWNED, True)
        _file_handlers[key] = handler
    if handler not in root.handlers:
        root.addHandler(handler)


def setup_logging(
    output_dir: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure root logging to stdout and (optionally) ``output_dir/training.log``.

    Idempotent per target: the stdout handler is attached once, and each output
    directory gets exactly one file handler for the life of the process.
    """
    root = logging.getLogger()
    root.setLevel(level)
    _ensure_stream(root)
    if output_dir is not None:
        _attach_file(root, Path(output_dir))
    return logging.getLogger("ml_framework")


def active_handlers() -> list[logging.Handler]:
    """The handlers this module currently has attached to the root logger.

    Exists because "how many handlers are attached?" is not answerable by counting
    ``root.handlers``: pytest, Lightning and any host application attach their own,
    and this module must be able to distinguish them — both to avoid detaching
    someone else's handler and to let tests assert on its own behaviour.
    """
    return [h for h in logging.getLogger().handlers if getattr(h, _OWNED, False)]


def close_logging() -> None:
    """Detach and close every handler this module owns.

    For tests and for long-lived hosts that run many trainings: without it, file
    descriptors accumulate one per output directory and Windows refuses to delete
    the still-open log files of a temporary directory.
    """
    global _stream_handler
    root = logging.getLogger()
    # Sweep by the ownership marker rather than by the bookkeeping dicts: a handler
    # can outlive them (a caller that re-attached a saved handler list, say), and a
    # cleanup that only closes what it still remembers is not a cleanup.
    tracked: set[logging.Handler] = set(_file_handlers.values())
    if _stream_handler is not None:
        tracked.add(_stream_handler)
    for handler in [*root.handlers]:
        if getattr(handler, _OWNED, False):
            root.removeHandler(handler)
            tracked.add(handler)
    for handler in tracked:
        handler.close()
    _file_handlers.clear()
    _stream_handler = None
