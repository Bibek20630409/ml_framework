"""config/schema.py — the v2 block schema.

A full rewrite, which §8.6 of the plan authorizes for exactly this file: the
schema break is the approved decision, so these assertions are about the *new*
contract. Every behaviour the v1 tests pinned is still pinned here, at its v2
field name — the required-by-kind checks, the val/test size sum, frozenness, and
`with_overrides` raising on an unknown key.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml_framework.config import ExperimentConfig
from ml_framework.core.plugins import IncompatibleCombinationError, MissingExtraError
from ml_framework.core.task import UnknownTaskError


def _tabular(**data) -> dict:
    return {
        "task": "binary",
        "data": {"kind": "tabular", "path": "d.csv", "target": "y", **data},
    }


# ── Required fields per data kind ─────────────────────────
@pytest.mark.unit
def test_tabular_requires_path_and_target():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"task": "binary", "data": {"kind": "tabular"}})


@pytest.mark.unit
def test_image_requires_a_train_path_and_a_test_dir():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"task": "multiclass", "model": {"name": "cnn"}, "data": {"kind": "image"}}
        )


@pytest.mark.unit
def test_invalid_task_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_tabular() | {"task": "clustering"})


@pytest.mark.unit
def test_a_task_in_the_vocabulary_without_a_taskspec_row_is_refused():
    """`seq2seq` is a valid Task literal but has no TaskSpec yet.

    The Literal is only a key; the table is what makes a task runnable, so
    accepting one with no row would defer the failure to mid-training.
    """
    with pytest.raises(UnknownTaskError, match="No TaskSpec"):
        ExperimentConfig.model_validate(_tabular() | {"task": "seq2seq"})


@pytest.mark.unit
def test_val_test_size_sum_validated():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_tabular(split={"val_size": 0.6, "test_size": 0.5}))


@pytest.mark.unit
def test_unknown_keys_are_rejected_in_every_fixed_block():
    for payload in (
        {"typo": 1},
        {"runtime": {"typo": 1}},
        {"fit": {"typo": 1}},
        {"fit": {"budget": {"typo": 1}}},
        {"tune": {"typo": 1}},
        {"data": {"kind": "tabular", "path": "d.csv", "target": "y", "typo": 1}},
    ):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(_tabular() | payload)


# ── Frozen + overrides ────────────────────────────────────
@pytest.mark.unit
def test_config_is_frozen(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        cfg.runtime.seed = 7  # frozen


@pytest.mark.unit
def test_with_overrides_returns_new_copy(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    new = cfg.with_overrides({"fit.params.lr": 0.05, "fit.budget.max_epochs": 9})
    assert new.fit.params["lr"] == 0.05
    assert new.fit.budget.max_epochs == 9
    assert cfg.fit.params.get("lr") != 0.05  # original untouched


@pytest.mark.unit
def test_with_overrides_unknown_key_raises(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    with pytest.raises(KeyError):
        cfg.with_overrides({"fit.nonexistent": 1})
    with pytest.raises(KeyError):
        cfg.with_overrides({"runtime.nonexistent": 1})


@pytest.mark.unit
def test_with_overrides_may_create_keys_only_under_the_plugin_owned_params(
    tabular_csv, make_config
):
    """Core cannot enumerate a plugin's knobs, so those dicts accept new keys.

    A typo there is still caught — one layer down, by the plugin's
    `extra="forbid"` params model.
    """
    cfg = make_config(tabular_csv, "multiclass")
    assert (
        cfg.with_overrides({"data.params.holdout_threshold": 10}).data.params["holdout_threshold"]
        == 10
    )
    assert cfg.with_overrides({"tune.overrides.x": 1}).tune.overrides["x"] == 1
    with pytest.raises(ValidationError):
        cfg.with_overrides({"model.params.not_a_knob": 1})


@pytest.mark.unit
def test_search_space_paths_are_applicable_as_overrides(tabular_csv, make_config):
    """The HPO contract: applying a trial is exactly `with_overrides(values)`."""
    from ml_framework.core.registry import MODELS

    cfg = make_config(tabular_csv, "multiclass")
    keys = list(MODELS.get_spec("mlp").search_space)
    assert keys  # would make the test vacuous
    applied = cfg.with_overrides(dict.fromkeys(keys, 0.25))
    assert applied.model.params["dropout"] == 0.25


# ── Plugin-validated model.params ─────────────────────────
@pytest.mark.unit
def test_model_params_are_validated_by_the_plugins_own_schema():
    """Rejecting the discriminated union cost IDE completion, not strictness."""
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _tabular() | {"model": {"name": "mlp", "params": {"drop": 1}}}
        )
    with pytest.raises(ValidationError, match="hidden_dims must be positive"):
        ExperimentConfig.model_validate(
            _tabular() | {"model": {"name": "mlp", "params": {"hidden_dims": [8, -1]}}}
        )


@pytest.mark.unit
def test_model_params_defaults_are_written_back_into_the_config():
    """config.json therefore records the *effective* params, not what was typed —
    and `backend.load()` can rebuild the network from the manifest alone."""
    cfg = ExperimentConfig.model_validate(_tabular() | {"model": {"name": "mlp", "params": {}}})
    assert cfg.model.params == {"hidden_dims": [128, 64, 32], "dropout": 0.3}


@pytest.mark.unit
def test_writing_defaults_back_does_not_recurse():
    cfg = ExperimentConfig.model_validate(
        _tabular() | {"model": {"name": "mlp", "params": {"dropout": 0.1}}}
    )
    assert cfg.model.params["dropout"] == 0.1
    # A second pass over the materialized dict must be a no-op, not an error.
    assert ExperimentConfig.model_validate(cfg.model_dump()).model.params == cfg.model.params


# ── validate_combination at load time ─────────────────────
@pytest.mark.unit
def test_an_impossible_combination_fails_at_load_rather_than_during_training():
    with pytest.raises(IncompatibleCombinationError, match="cannot consume data kind"):
        ExperimentConfig.model_validate(
            {
                "task": "multiclass",
                "model": {"name": "mlp"},
                "data": {"kind": "image", "path": "train", "params": {"test_dir": "test"}},
            }
        )


@pytest.mark.unit
def test_an_unknown_model_name_fails_at_load():
    with pytest.raises(KeyError, match="Unknown model"):
        ExperimentConfig.model_validate(_tabular() | {"model": {"name": "not_a_model"}})


@pytest.fixture
def unavailable_model():
    """Register a model whose extra can never be satisfied, and clean it up.

    The companion check in `test_plugins.py` gates on whether torchvision happens
    to be installed, so it self-skips on a full install. Doing the same here would
    mean the *config-load* refusal — the P2 addition — went untested in exactly
    the environment most people develop in. A synthetic requirement removes the
    dependence on the world instead of picking one.
    """
    from dataclasses import replace

    from ml_framework.core.registry import MODELS
    from ml_framework.core.types import Requirement

    original = MODELS.get_spec("cnn")
    MODELS.register(
        replace(
            original,
            requires=(Requirement("torchvision_not_installed", extra="image", min_version="0.15"),),
        ),
        override=True,
    )
    try:
        yield original.name
    finally:
        MODELS.register(original, override=True)


@pytest.mark.unit
def test_an_uninstalled_model_reports_the_pip_extra_at_load(unavailable_model):
    """The message that is the difference between a framework that feels finished
    and one that does not — emitted at config-load time, not 40 seconds in."""
    with pytest.raises(MissingExtraError, match=r"ml-framework\[image\]"):
        ExperimentConfig.model_validate(
            {
                "task": "multiclass",
                "model": {"name": unavailable_model},
                "data": {"kind": "image", "path": "train", "params": {"test_dir": "test"}},
            }
        )


@pytest.mark.unit
def test_an_impossible_combination_is_reported_before_the_missing_extra(unavailable_model):
    """Check order matters: telling someone to install 2 GB of torchvision for a
    combination that could never work would be the wrong instruction."""
    with pytest.raises(IncompatibleCombinationError, match="does not support task"):
        ExperimentConfig.model_validate(
            {
                "task": "regression",
                "model": {"name": unavailable_model},
                "data": {"kind": "image", "path": "train", "params": {"test_dir": "test"}},
            }
        )


# ── Split block ───────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize(
    ("split", "kind", "expected"),
    [
        ({}, "tabular", "random"),
        ({"time_col": "date"}, "tabular", "temporal"),
        ({"group_col": "patient"}, "tabular", "group"),
        ({}, "timeseries", "temporal"),
        ({"strategy": "random", "time_col": "date"}, "tabular", "random"),
    ],
)
def test_auto_split_strategy_resolution(split, kind, expected):
    """`auto` becomes temporal whenever time ordering matters — the config path
    that makes TemporalSplitter/GroupSplitter reachable at all."""
    from ml_framework.config import SplitConfig

    assert SplitConfig(**split).resolved_strategy(kind) == expected


@pytest.mark.unit
def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml(tmp_path / "nope.yaml")
