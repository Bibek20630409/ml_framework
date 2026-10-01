"""
pipeline/stage_config.py
────────────────────────
One answer to "what is the target column, and where does the processed data
live" — read from the training config, for the pipeline stages that run *outside*
a training process.

Why this exists: ``data.target`` and ``data.path`` used to be typed out in four
places that had to agree by hand — ``params.yaml``, the train config, and twice in
the Airflow DAG. Passing a value on a command line means every caller has to know
it; reading it from the config the stage is already pointed at means none of them
do. The stages keep their explicit flags for standalone/ad-hoc runs, but the
orchestrated path names a config file and nothing else.

This deliberately reads the YAML rather than building an
:class:`~ml_framework.config.schema.ExperimentConfig`. A preprocess stage should
not fail because ``model.name`` is missing: at that point in the pipeline the
processed data does not exist yet, so a config that cannot yet describe a valid
*training* run is still a perfectly good description of the *data*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class StageData:
    """The data-locating fields a standalone pipeline stage needs."""

    target: str
    path: str | None


def load_stage_data(config_path: str | Path) -> StageData:
    """Read ``data.target`` / ``data.path`` out of a training config.

    Accepts the v1 spelling ``data.target_col`` as well, matching the rename in
    :mod:`ml_framework.config.migrate`, so pointing a stage at an unmigrated
    config reports the *missing target* rather than a confusing KeyError.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"--config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data") or {}

    target = data.get("target") or data.get("target_col")
    if not target:
        raise ValueError(f"{path} names no data.target, so the stage has no target column")

    return StageData(target=str(target), path=data.get("path"))


def resolve(
    config_path: str | Path | None,
    *,
    target_col: str | None,
    data_path: str | None,
) -> tuple[str, str | None]:
    """Merge an optional ``--config`` with explicit flags; flags win.

    Explicit beats derived, matching the precedence the ``mlf`` CLI already
    documents (``YAML file < --set < explicit CLI flags``), so a one-off run can
    still override the committed config without editing it.
    """
    if config_path is None:
        if not target_col:
            raise ValueError("pass --target-col, or --config to read it from a training config")
        return target_col, data_path

    stage = load_stage_data(config_path)
    return target_col or stage.target, data_path or stage.path
