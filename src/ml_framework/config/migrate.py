"""
config/migrate.py
─────────────────
v1 → v2 YAML remapper. The clean break in :mod:`ml_framework.config.schema` was
authorized on the condition that migrating is mechanical, and this is the
mechanism that keeps that promise:

    mlf migrate-config -i configs/old.yaml -o configs/new.yaml

Two rules make it trustworthy rather than merely convenient:

1. **Every v1 key has an explicit destination.** A key that is not in
   :data:`V1_TO_V2` and is not a pass-through block raises
   :class:`MigrationError` naming it. A migrator that silently drops what it does
   not recognise is worse than no migrator — the config would validate and train
   something other than what the user wrote.
2. **The result is validated by default.** Migration is a text transformation;
   validation is what proves the output is a config. ``--no-validate`` exists for
   the one case where it legitimately fails: a model whose optional extra is not
   installed on the machine doing the migration.

One v1 key can legitimately fail validation after a faithful migration:
``model.dropout`` on a ``cnn``. The CNN never read it — v1's ``ModelConfig``
simply carried every model's knobs for every model. v2's per-plugin
``extra="forbid"`` params model reports it, which is the schema doing its job.
Delete the line.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..core.types import FrameworkError

log = logging.getLogger(__name__)


class MigrationError(FrameworkError):
    """A v1 config could not be mapped onto the v2 schema."""


# Dotted v1 path → dotted v2 path. This table *is* the §3.3 migration map; the
# names on the right are the only place the two schemas touch.
V1_TO_V2: Mapping[str, str] = {
    "task": "task",
    "seed": "runtime.seed",
    "output_dir": "runtime.output_dir",
    # ── data ──
    "data.kind": "data.kind",
    "data.csv_path": "data.path",
    "data.target_col": "data.target",
    "data.train_dir": "data.path",
    "data.val_dir": "data.params.val_dir",
    "data.test_dir": "data.params.test_dir",
    "data.img_size": "data.params.img_size",
    "data.class_names": "data.class_names",
    "data.val_size": "data.split.val_size",
    "data.test_size": "data.split.test_size",
    "data.holdout_threshold": "data.params.holdout_threshold",
    "data.imbalance_strategy": "data.params.imbalance_strategy",
    "data.imbalance_threshold": "data.params.imbalance_threshold",
    # ── model ──
    "model.name": "model.name",
    "model.hidden_dims": "model.params.hidden_dims",
    "model.dropout": "model.params.dropout",
    "model.backbone": "model.params.backbone",
    "model.pretrained": "model.params.pretrained",
    # ── optim → the fit loop's own params ──
    "optim.lr": "fit.params.lr",
    "optim.weight_decay": "fit.params.weight_decay",
    "optim.lr_patience": "fit.params.lr_patience",
    "optim.lr_factor": "fit.params.lr_factor",
    # ── train → split across runtime / fit ──
    "train.epochs": "fit.budget.max_epochs",
    "train.batch_size": "fit.batch_size",
    "train.patience": "fit.patience",
    # gradient clipping is a Lightning Trainer knob, so it is correctly a
    # *backend* param rather than a framework-level one.
    "train.gradient_clip_val": "fit.params.gradient_clip_val",
    "train.num_workers": "runtime.num_workers",
    "train.deterministic": "runtime.deterministic",
    # ── HPO ──
    "hpo_n_trials": "tune.max_trials",
    "hpo_timeout": "tune.max_seconds",
}

# Blocks copied across unchanged, key for key.
PASSTHROUGH_BLOCKS: tuple[str, ...] = ("logging",)

# Two v1 keys share one v2 destination: only one of them is ever set, because
# `data.kind` decides which. Setting both is a v1 config that was already
# ambiguous, and guessing would be the wrong kind of helpful.
_COLLIDING = ("data.csv_path", "data.train_dir")


def _flatten(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted-key view of a nested mapping. Leaf lists stay whole."""
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _assign(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def looks_like_v1(raw: Mapping[str, Any]) -> bool:
    """Whether ``raw`` is a v1 config.

    Keyed on blocks v2 removed outright (``optim``/``train``) and top-level
    scalars v2 moved into ``runtime``, so a v2 file is never mistaken for one.
    """
    return any(k in raw for k in ("optim", "train", "hpo_n_trials", "hpo_timeout")) or (
        "runtime" not in raw and any(k in raw for k in ("seed", "output_dir"))
    )


def migrate_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a parsed v1 config mapping to the v2 shape.

    Raises :class:`MigrationError` for any key with no destination, listing all of
    them at once so a hand-written config is fixed in one pass rather than N.
    """
    out: dict[str, Any] = {}
    unknown: list[str] = []

    for block in PASSTHROUGH_BLOCKS:
        if block in raw:
            value = raw[block]
            if not isinstance(value, Mapping):
                raise MigrationError(f"'{block}' must be a mapping, got {type(value).__name__}")
            out[block] = dict(value)

    flat = _flatten(raw)
    colliding = [k for k in _COLLIDING if k in flat]
    if len(colliding) > 1:
        raise MigrationError(
            f"{' and '.join(colliding)} both map to 'data.path'; keep only the one "
            f"matching data.kind"
        )

    for dotted, value in flat.items():
        if dotted.split(".", 1)[0] in PASSTHROUGH_BLOCKS:
            continue
        target = V1_TO_V2.get(dotted)
        if target is None:
            unknown.append(dotted)
            continue
        _assign(out, target, value)

    if unknown:
        raise MigrationError(
            "no v2 destination for: "
            + ", ".join(sorted(unknown))
            + ". Remove them, or map them by hand — this migrator never drops a key silently."
        )
    return out


def migrate_file(
    src: str | Path,
    dest: str | Path,
    *,
    validate: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Migrate a v1 YAML file to ``dest`` and return the v2 mapping.

    With ``validate`` (the default) the output is run through
    :class:`~ml_framework.config.schema.ExperimentConfig` **before** it is
    written, so a failed migration never leaves a broken config on disk.
    """
    source = Path(src)
    target = Path(dest)
    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (pass overwrite=True to replace it)")

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise MigrationError(f"{source} does not contain a YAML mapping")
    if not looks_like_v1(raw):
        log.warning("%s does not look like a v1 config; migrating anyway", source)

    migrated = migrate_mapping(raw)
    if validate:
        from .schema import ExperimentConfig

        ExperimentConfig.model_validate(migrated)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Migrated from "
        + source.name
        + " by `mlf migrate-config`.\n"
        + yaml.safe_dump(migrated, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    log.info("migrated %s → %s", source, target)
    return migrated


__all__ = [
    "MigrationError",
    "V1_TO_V2",
    "looks_like_v1",
    "migrate_file",
    "migrate_mapping",
]
