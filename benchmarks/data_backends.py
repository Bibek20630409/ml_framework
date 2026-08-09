"""
benchmarks/data_backends.py
───────────────────────────
Resolve the gate ``choose.md`` §7 put on the Polars backend.

That section argued Polars was not worth adding because *"the win is confined to
parse time, and it is unmeasured"*. This is the measurement. Run it before
trusting either the claim or its refutation — the absolute numbers are specific to
one machine, and only the ratios travel.

    python benchmarks/data_backends.py

Deliberately not a pytest module: it takes minutes, it measures rather than
asserts, and a benchmark that fails CI on a noisy runner teaches people to ignore
CI. Nothing here is imported by the framework.
"""

from __future__ import annotations

import logging
import pathlib
import statistics
import tempfile
import time
from collections.abc import Callable
from typing import Any

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_framework.config import ExperimentConfig  # noqa: E402
from ml_framework.core.registry import get_data_backend  # noqa: E402
from ml_framework.data.builders import build_bundle  # noqa: E402

REPS = 5
SHAPES = [(50_000, 20), (500_000, 20), (200_000, 50)]
ENGINES = ("local", "polars")


def make_dataset(tmp: pathlib.Path, n_rows: int, n_cols: int) -> tuple[pathlib.Path, pathlib.Path]:
    """The same table as CSV and as Parquet — the two read paths differ sharply."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            **{f"f{i}": rng.normal(size=n_rows) for i in range(n_cols)},
            "label": rng.integers(0, 3, size=n_rows),
        }
    )
    csv = tmp / f"d_{n_rows}_{n_cols}.csv"
    parquet = tmp / f"d_{n_rows}_{n_cols}.parquet"
    frame.to_csv(csv, index=False)
    frame.to_parquet(parquet, index=False)
    return csv, parquet


def median_seconds(fn: Callable[[], Any]) -> float:
    """Median of REPS runs, after one discarded warm-up.

    The warm-up matters more than the repetitions here: the first call pays for
    the engine's import and for pulling the file into the page cache, and charging
    those to whichever engine ran first would invent a difference.
    """
    fn()
    return statistics.median(_timed(fn) for _ in range(REPS))


def _timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def config_for(tmp: pathlib.Path, path: pathlib.Path, backend: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "task": "multiclass",
            "runtime": {"seed": 42, "output_dir": str(tmp / "out"), "num_workers": 0},
            "data": {
                "kind": "tabular",
                "path": str(path),
                "target": "label",
                "backend": backend,
            },
            "model": {"name": "xgboost"},
        }
    )


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    print(f"{'rows x cols':>14} {'op':<16} {'pandas':>10} {'polars':>10} {'speedup':>8}")
    print("-" * 64)

    for n_rows, n_cols in SHAPES:
        csv, parquet = make_dataset(tmp, n_rows, n_cols)
        rows: list[tuple[str, dict[str, float]]] = []

        for label, path in (("parse CSV", csv), ("parse Parquet", parquet)):
            timings = {}
            for name in ENGINES:
                engine = get_data_backend(name)
                timings[name] = median_seconds(lambda e=engine, p=path: e.read_table(str(p)))
            rows.append((label, timings))

        # The whole bundle: parse + collect + split + scale + imbalance. This is
        # the number that decides whether the parse win survives to the user.
        timings = {}
        for name in ENGINES:
            cfg = config_for(tmp, csv, name)
            timings[name] = median_seconds(lambda c=cfg: build_bundle(c))
        rows.append(("build_bundle", timings))

        for label, t in rows:
            speedup = t["local"] / t["polars"]
            print(
                f"{n_rows:>8}x{n_cols:<5} {label:<16} "
                f"{t['local'] * 1e3:>9.1f}ms {t['polars'] * 1e3:>9.1f}ms {speedup:>7.2f}x"
            )
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
