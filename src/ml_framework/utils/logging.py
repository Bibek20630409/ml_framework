"""
utils/logging.py
────────────────
Central logging setup. Call `setup_logging(output_dir)` once at process start.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_STREAM_READY = False
_FILE_READY = False


def setup_logging(
    output_dir: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure root logging to stdout and (optionally) a file.

    Idempotent *per handler*: the stdout handler is attached once, and the file
    handler is attached the first time an ``output_dir`` is supplied — even if an
    earlier call (e.g. from the CLI, before the output dir exists) already set up
    stdout logging. This is what guarantees ``outputs/training.log`` is written.
    """
    global _STREAM_READY, _FILE_READY
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not _STREAM_READY:
        # Log messages contain non-ASCII (→, ─); force UTF-8 so the Windows console
        # (cp1252 by default) doesn't raise UnicodeEncodeError.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        root.addHandler(stream)
        _STREAM_READY = True

    if output_dir is not None and not _FILE_READY:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out / "training.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        _FILE_READY = True

    return logging.getLogger("ml_framework")
