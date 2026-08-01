"""
data/sniff.py
─────────────
Look at a dataset and say what it is: ``data.kind``, ``data.target``, ``task``.

**Every inference carries the rule that produced it.** That is not decoration —
it is the difference between zero-config being a convenience and being a black
box. A framework that silently decides your integer column is a class label, and
is wrong, costs more than one that made you type three lines of YAML. So this
module returns :class:`Inference` records, the CLI logs each one, and ``mlf init``
writes them into the generated file as comments.

Two rules the whole module is built around:

* **Refuse rather than guess between equals.** Two columns named ``label`` and
  ``target`` is not a tie to be broken by column order — it is a question only the
  user can answer. Same for several plausible text columns.
* **Warn when the guess is weak.** Falling back to "the last column is the target"
  is right often enough to be worth doing and wrong often enough to say out loud.

No config import: this runs *before* there is a config, and returning plain data
is what lets :mod:`ml_framework.config.autoconfig` slot it into an ordinary
precedence chain rather than special-casing synthesized values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.types import DataKind, FrameworkError, Task

log = logging.getLogger(__name__)


class SniffError(FrameworkError):
    """The dataset could not be identified, or the answer was ambiguous."""


# Column names that mean "this is the label" in practically every dataset that
# ships with one. Order is *not* preference — several matches is an error, not a
# tie to be broken.
TARGET_NAMES: tuple[str, ...] = ("label", "target", "y")

# Mean whitespace tokens above which a string column is prose rather than a
# category. A sentiment corpus averages 15-30; a column of "red"/"blue"/"green"
# averages 1. Anything in between is genuinely ambiguous, and 4 sits in the gap
# rather than on either shoulder.
TEXT_TOKEN_THRESHOLD: float = 4.0

# At or below this many distinct non-float values, an integer column is a class
# label rather than a quantity. Above it, treating it as classification would
# build a head with hundreds of logits for what is obviously a count or an id.
MAX_CLASSES: int = 20

# File extensions that are text corpora regardless of what is inside them.
TEXT_SUFFIXES: frozenset[str] = frozenset({".jsonl", ".ndjson", ".txt"})
TABLE_SUFFIXES: frozenset[str] = frozenset({".csv", ".parquet", ".pq", ".json"})
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


@dataclass(frozen=True, slots=True)
class Inference:
    """One inferred value, and why.

    ``field`` is a dotted config path so the CLI can log it in the vocabulary the
    user would have typed, and ``mlf init`` can attach the comment to the right
    line.
    """

    field: str
    value: Any
    rule: str
    # True when the rule is a fallback rather than positive evidence. The CLI logs
    # these at WARNING; everything else at INFO.
    weak: bool = False

    def __str__(self) -> str:
        return f"{self.field}={self.value!r} ({self.rule})"


@dataclass(frozen=True, slots=True)
class Sniffed:
    """What the dataset appears to be, plus the reasoning."""

    path: str
    kind: DataKind
    task: Task
    target: str | None = None
    time_col: str | None = None
    text_col: str | None = None
    n_rows: int = 0
    n_features: int = 0
    inferences: tuple[Inference, ...] = ()

    def log(self) -> None:
        """Report every rule that fired, weak ones loudly."""
        for inference in self.inferences:
            (log.warning if inference.weak else log.info)("inferred %s", inference)


# ── Kind ──────────────────────────────────────────────────
def _looks_like_image_folder(path: Path) -> bool:
    """A directory of class directories of images (the ``ImageFolder`` layout).

    Checked positively — there must be an actual image file under a subdirectory —
    rather than by "it is a directory and nothing else matched". A directory of
    parquet part-files is also a directory, and it is a table.
    """
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir():
            continue
        for item in child.iterdir():
            if item.suffix.lower() in IMAGE_SUFFIXES:
                return True
    return False


def _mean_tokens(series: Any) -> float:
    """Average whitespace-token count over the first rows of a string column."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return 0.0
    return float(sample.map(lambda text: len(text.split())).mean())


def _datetime_column(frame: Any, declared: str | None) -> str | None:
    """A column that parses as a datetime **and is sorted**, or ``None``.

    Both halves matter. Parsing alone would call a table of birthdays a time
    series; monotonicity is what says the rows are *ordered by* time, which is the
    thing that makes forecasting meaningful and a shuffled split wrong.
    """
    import pandas as pd

    candidates = [declared] if declared else list(frame.columns)
    for name in candidates:
        if name not in frame.columns:
            raise SniffError(f"--time-col '{name}' is not a column of the file")
        column = frame[name]
        if column.dtype.kind in "if":
            # Numeric columns parse as epochs and would make every id column a
            # timestamp. A declared one is the user's call, so honour that.
            if declared is None:
                continue
        try:
            parsed = pd.to_datetime(column, errors="raise", format="mixed")
        except (ValueError, TypeError, OverflowError):
            continue
        if parsed.is_monotonic_increasing:
            return str(name)
        if declared:
            raise SniffError(
                f"--time-col '{name}' parses as a datetime but is not sorted. "
                f"Sort the file by it, or the split cannot be temporal."
            )
    return None


# ── Target ────────────────────────────────────────────────
def _pick_target(frame: Any, declared: str | None, reserved: set[str]) -> Inference:
    if declared is not None:
        if declared not in frame.columns:
            raise SniffError(f"--target '{declared}' is not a column of the file")
        return Inference("data.target", declared, "given explicitly")

    named = [c for c in frame.columns if str(c).lower() in TARGET_NAMES]
    if len(named) > 1:
        raise SniffError(
            f"several columns could be the target ({sorted(map(str, named))}). "
            f"Pass --target to say which."
        )
    if named:
        return Inference("data.target", str(named[0]), f"column is named '{named[0]}'")

    usable = [c for c in frame.columns if c not in reserved]
    if not usable:
        raise SniffError("the file has no column that could be a target")
    return Inference(
        "data.target",
        str(usable[-1]),
        "no column named label/target/y, so the last column was used",
        weak=True,
    )


# ── Task ──────────────────────────────────────────────────
def _infer_task(series: Any) -> Inference:
    """binary | multiclass | regression, from the target's own values."""
    import pandas as pd

    values = series.dropna()
    distinct = int(values.nunique())

    if distinct <= 1:
        raise SniffError(f"the target has {distinct} distinct value(s); there is nothing to learn")
    if distinct == 2:
        return Inference("task", "binary", "the target has exactly 2 distinct values")

    is_float = pd.api.types.is_float_dtype(values)
    if not is_float and distinct <= MAX_CLASSES:
        return Inference(
            "task", "multiclass", f"the target is non-float with {distinct} distinct values"
        )
    reason = "the target is floating-point" if is_float else f"the target has {distinct} values"
    return Inference("task", "regression", f"{reason} (> {MAX_CLASSES} means a quantity)")


# ── The entry point ───────────────────────────────────────
def sniff(
    path: str | Path,
    *,
    target: str | None = None,
    time_col: str | None = None,
    text_col: str | None = None,
) -> Sniffed:
    """Identify ``path``: what kind of data, which column is the target, what task.

    Explicit arguments always win and are recorded as "given explicitly", so the
    log reads the same whether a value was inferred or supplied — which is what
    makes the log trustworthy as a record of what actually happened.
    """
    source = Path(path)
    if not source.exists():
        raise SniffError(f"no such dataset: {source}")

    if _looks_like_image_folder(source):
        return _sniff_images(source)

    suffix = source.suffix.lower()
    if suffix in TEXT_SUFFIXES and suffix not in TABLE_SUFFIXES:
        return _sniff_table(source, target, time_col, text_col, forced_text=True)
    if source.is_dir() or suffix in TABLE_SUFFIXES:
        return _sniff_table(source, target, time_col, text_col, forced_text=False)

    raise SniffError(
        f"cannot tell what '{source}' is. Known: a directory of class directories "
        f"(images), {sorted(TABLE_SUFFIXES | TEXT_SUFFIXES)}."
    )


def _sniff_images(source: Path) -> Sniffed:
    classes = sorted(child.name for child in source.iterdir() if child.is_dir())
    task: Task = "binary" if len(classes) == 2 else "multiclass"
    inferences = (
        Inference("data.kind", "image", "the directory holds class directories of images"),
        Inference("task", task, f"the training folder has {len(classes)} class directories"),
    )
    return Sniffed(
        path=str(source),
        kind="image",
        task=task,
        n_features=len(classes),
        inferences=inferences,
    )


def _sniff_table(
    source: Path,
    target: str | None,
    time_col: str | None,
    text_col: str | None,
    *,
    forced_text: bool,
) -> Sniffed:
    """A file with columns: tabular, upgraded to timeseries or text on evidence."""
    from .sources.text import read_text_table

    try:
        frame = read_text_table(source)
    except SniffError:
        raise
    except Exception as exc:
        # Whatever pandas or pyarrow raised, the useful thing to say is that this
        # file is not a dataset this framework recognises. A directory reaches here
        # because a directory of parquet part-files *is* a supported table (Spark
        # writes them) -- so pointing `--data` at an arbitrary folder surfaces an
        # arrow schema error, which tells the user nothing about what to do.
        raise SniffError(
            f"could not read '{source}' as a dataset: {exc}. Expected a CSV, a "
            f"Parquet file or directory, a JSONL corpus, or a directory of class "
            f"directories of images."
        ) from exc
    if frame.empty:
        raise SniffError(f"'{source}' has no rows")

    inferences: list[Inference] = []
    time_name = None if forced_text else _datetime_column(frame, time_col)
    reserved = {time_name} - {None}

    target_inference = _pick_target(frame, target, reserved)  # type: ignore[arg-type]
    target_name = str(target_inference.value)
    reserved.add(target_name)

    # ── kind ──
    kind: DataKind
    if forced_text:
        kind = "text"
        inferences.append(
            Inference("data.kind", "text", f"'{source.suffix}' files are text corpora")
        )
    elif time_name is not None:
        kind = "timeseries"
        why = "given explicitly" if time_col else "parses as a datetime and is sorted"
        inferences.append(Inference("data.kind", "timeseries", f"column '{time_name}' {why}"))
    else:
        kind, text_name = _tabular_or_text(frame, target_name, text_col, inferences)
        text_col = text_name or text_col

    inferences.append(target_inference)

    # ── task ──
    if kind == "timeseries":
        inferences.append(
            Inference("task", "forecasting", "time-ordered data is forecast, not classified")
        )
        task: Task = "forecasting"
    else:
        task_inference = _infer_task(frame[target_name])
        inferences.append(task_inference)
        task = task_inference.value

    if kind == "text" and text_col is None:
        text_col = _text_column(frame, target_name)

    if kind == "timeseries":
        inferences.append(
            Inference("data.split.time_col", time_name, "rows are ordered by this column")
        )

    return Sniffed(
        path=str(source),
        kind=kind,
        task=task,
        target=target_name,
        time_col=time_name,
        text_col=text_col,
        n_rows=int(len(frame)),
        n_features=int(len(frame.columns) - len(reserved)),
        inferences=tuple(inferences),
    )


def _string_columns(frame: Any, target: str) -> list[str]:
    return [c for c in frame.columns if c != target and frame[c].dtype == object]  # noqa: E721


def _text_column(frame: Any, target: str) -> str | None:
    """The prose column, or ``None``. Ambiguity is resolved by the caller."""
    prose = [
        c for c in _string_columns(frame, target) if _mean_tokens(frame[c]) >= TEXT_TOKEN_THRESHOLD
    ]
    return str(prose[0]) if len(prose) == 1 else None


def _tabular_or_text(
    frame: Any, target: str, declared: str | None, inferences: list[Inference]
) -> tuple[DataKind, str | None]:
    """Tabular unless a column is clearly prose.

    The threshold is doing real work here: a column of category names is a
    *feature*, and treating it as the text a transformer should read would fine-tune
    a 66M-parameter encoder on the word "red".
    """
    if declared is not None:
        if declared not in frame.columns:
            raise SniffError(f"--text-col '{declared}' is not a column of the file")
        inferences.append(Inference("data.kind", "text", "given explicitly via --text-col"))
        return "text", declared

    prose = [
        (c, _mean_tokens(frame[c]))
        for c in _string_columns(frame, target)
        if _mean_tokens(frame[c]) >= TEXT_TOKEN_THRESHOLD
    ]
    if len(prose) > 1:
        raise SniffError(
            f"several columns look like prose ({sorted(str(c) for c, _ in prose)}). "
            f"Pass --text-col to say which, or --kind tabular."
        )
    if prose:
        name, mean = prose[0]
        inferences.append(
            Inference(
                "data.kind",
                "text",
                f"column '{name}' averages {mean:.1f} words (>= {TEXT_TOKEN_THRESHOLD})",
            )
        )
        return "text", str(name)

    inferences.append(Inference("data.kind", "tabular", "columns are numeric or short strings"))
    return "tabular", None


__all__ = [
    "IMAGE_SUFFIXES",
    "MAX_CLASSES",
    "TABLE_SUFFIXES",
    "TARGET_NAMES",
    "TEXT_SUFFIXES",
    "TEXT_TOKEN_THRESHOLD",
    "Inference",
    "Sniffed",
    "SniffError",
    "sniff",
]
