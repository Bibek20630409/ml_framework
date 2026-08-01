"""
core/export.py
──────────────
The export vocabulary: what formats exist, and what a backend says when it cannot
produce one.

**Export is a backend method, not a function with a switch in it.** Only the
backend knows what its estimator physically is — a checkpoint, a booster, a
pickled statsmodels object — so only the backend can turn it into something else.
A central `export()` with `if backend == "lightning"` in it would need editing
every time a backend is added, which is the coupling the whole plugin design
exists to remove.

The consequence worth stating: **an unsupported combination fails loudly.**
`mlf export --format onnx` on a Prophet bundle raises here rather than writing a
file that is not really ONNX, or silently falling back to pickle and leaving the
user to discover at deployment time that the artifact is not what they asked for.

No optional imports at module scope: this is imported by the CLI, which must stay
usable on an install with no export extra present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn

from .types import FrameworkError, Requirement

# What `mlf export --format` accepts.
#
# `native` is not a cop-out. A gradient-booster's own `.json`/`.cbm` is the format
# every serving runtime for that library already reads, loads faster than ONNX and
# preserves exact behaviour — converting it would trade all three for portability
# nobody asked for.
ExportFormat = Literal["onnx", "torchscript", "native", "pickle"]

EXPORT_FORMATS: Final[tuple[ExportFormat, ...]] = ("onnx", "torchscript", "native", "pickle")

# Declared here rather than in the backend so the refusal can name the pip command
# without importing anything.
ONNX_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("onnx", extra="export"),
    Requirement("onnxruntime", extra="export"),
    # The dynamo exporter's own dependency. Checked here rather than left to
    # torch, whose failure for a missing `onnxscript` arrives from inside the
    # exporter and does not name the pip extra that fixes it.
    Requirement("onnxscript", extra="export"),
)


class UnsupportedExportError(FrameworkError):
    """This backend cannot produce this format.

    A refusal, never a fallback. Writing *some* file when the user asked for ONNX
    would be discovered at deployment time by a runtime that cannot load it.
    """


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Where the exported artifact landed, and anything the user should know."""

    path: Path
    format: ExportFormat
    # Free text for caveats that are true but not errors: the opset used, the fact
    # that a traced graph fixes the batch dimension, and so on. Printed by the CLI.
    notes: str = ""


def unsupported(backend: str, fmt: str, supported: tuple[str, ...], why: str = "") -> NoReturn:
    """Raise a refusal that names what *is* possible.

    An error saying only "unsupported" makes the user guess; one that lists the
    formats this backend can produce turns a dead end into a next step.
    """
    detail = f" ({why})" if why else ""
    raise UnsupportedExportError(
        f"backend '{backend}' cannot export '{fmt}'{detail}. "
        f"It supports: {', '.join(supported) if supported else 'no export formats'}."
    )


def example_input_shape(manifest: Any) -> tuple[int, ...]:
    """The shape of one input row, for tracing a graph.

    ONNX and TorchScript both need a concrete example to record operations
    against. The shape comes from the **manifest** rather than from the live data,
    because export must work against a bundle alone — which is the situation it
    exists for.

    Refuses rather than guesses for inputs that are not a fixed-width float
    tensor. A tokenized text batch is a *dict* of variable-length integer tensors,
    and tracing one would bake this batch's sequence length into the graph — an
    artifact that silently truncates or pads every future request to whatever
    length happened to be traced.
    """
    signature = manifest.signature
    payload = signature.input.payload

    if payload == "arrays":
        n_features = int(signature.input.n_features or 0)
        if n_features <= 0:
            raise UnsupportedExportError(
                "the bundle records no feature count, so there is no shape to trace"
            )
        return (1, n_features)

    if payload == "dataset":
        params = dict(getattr(manifest.preprocessor, "params", None) or {})
        size = params.get("img_size")
        if size:
            # (batch, channels, height, width) — the layout every torchvision
            # backbone takes.
            return (1, 3, int(size), int(size))
        raise UnsupportedExportError(
            "this bundle's input is a tokenized text batch (a dict of variable-length "
            "integer tensors), not a fixed-width tensor. Tracing it would bake this "
            "batch's sequence length into the graph. Export the HuggingFace directory "
            "instead — it is already portable."
        )

    raise UnsupportedExportError(
        f"no example input can be built for a '{payload}' payload; "
        f"tracing needs a fixed-width tensor"
    )


__all__ = [
    "EXPORT_FORMATS",
    "ONNX_REQUIREMENTS",
    "ExportFormat",
    "ExportResult",
    "UnsupportedExportError",
    "example_input_shape",
    "unsupported",
]
