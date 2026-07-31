"""pipeline/tune.py — the search driver that replaced pipeline/hpo.py.

Three v1 defects have tests here, because each was silent rather than loud:
its space was hardcoded to the MLP's shape, its objective read a Lightning-only
`val/loss`, and it printed the winner instead of applying it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_framework.config.defaults import TUNE_BUDGETS, budget_for, parse_duration
from ml_framework.core.protocols import Categorical, Const, Float, Int
from ml_framework.pipeline.tune import (
    HPO_FILE,
    TuningError,
    describe_space,
    effective_space,
    resolve_tune_budget,
    tune,
)

optuna = pytest.importorskip("optuna", reason="the hpo extra is not installed")
pytest.importorskip("xgboost", reason="the gbdt extra is not installed")


def _tunable(csv, make_config, **overrides):
    """A GBDT config with a budget small enough for a test to finish."""
    base = {
        "tune.enabled": True,
        "tune.max_trials": 4,
        "tune.max_seconds": 60.0,
        "fit.params.n_estimators": 15,
        "fit.params.early_stopping_rounds": 0,
        **overrides,
    }
    return make_config(csv, "multiclass", model="xgboost", **base)


# ── Budgets ───────────────────────────────────────────────
@pytest.mark.unit
def test_budgets_are_backend_aware():
    """20 trials x 200 epochs on the Lightning path is hours. A uniform trial
    count is what would make `mlf train` feel broken."""
    assert budget_for("gbdt").max_trials > budget_for("lightning").max_trials
    assert budget_for("lightning").max_seconds > budget_for("gbdt").max_seconds
    # Only the epoch-loop backend caps epochs per trial; a one-shot fit has no
    # epochs, and `n_estimators` plus early stopping already bound it.
    assert budget_for("lightning").trial_max_epochs is not None
    assert budget_for("gbdt").trial_max_epochs is None


@pytest.mark.unit
def test_an_unknown_backend_gets_the_conservative_default():
    assert budget_for("no_such_backend") is TUNE_BUDGETS["default"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "seconds"),
    [("900", 900.0), ("30s", 30.0), ("10m", 600.0), ("2h", 7200.0), ("1.5m", 90.0), (45, 45.0)],
)
def test_durations_read_the_way_people_write_them(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "soon", "10d", "-5", "0"])
def test_a_bad_duration_is_refused_with_examples(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


@pytest.mark.unit
def test_explicit_config_values_beat_the_backend_default(tabular_csv, make_config):
    cfg = _tunable(tabular_csv, make_config, **{"tune.max_trials": 3})
    budget = resolve_tune_budget(cfg, "gbdt")
    assert budget.max_trials == 3  # the user's
    assert budget.max_seconds == 60.0  # also the user's


@pytest.mark.unit
def test_an_untouched_config_gets_the_backend_default(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass", model="xgboost", **{"tune.enabled": True})
    assert resolve_tune_budget(cfg, "gbdt").max_trials == budget_for("gbdt").max_trials


# ── Space assembly ────────────────────────────────────────
@pytest.mark.unit
def test_the_space_merges_model_then_backend(tabular_csv, make_config):
    """lr/batch_size are declared once on the backend rather than repeated in
    every plugin; tree shape comes from the plugin."""
    space = effective_space(_tunable(tabular_csv, make_config))
    assert "model.params.max_depth" in space  # plugin
    assert "fit.params.learning_rate" in space  # backend
    assert all("." in key for key in space)  # dotted config paths, always


@pytest.mark.unit
def test_tune_overrides_narrow_the_declared_space(tabular_csv, make_config):
    # The override key is the whole dotted path, so the block is set wholesale —
    # `with_overrides` cannot walk into a key that is itself a dotted string.
    cfg = _tunable(
        tabular_csv,
        make_config,
        **{"tune.overrides": {"model.params.max_depth": {"type": "int", "low": 3, "high": 4}}},
    )
    spec = effective_space(cfg)["model.params.max_depth"]
    assert isinstance(spec, Int) and (spec.low, spec.high) == (3, 4)


@pytest.mark.unit
def test_a_bare_override_value_pins_the_parameter(tabular_csv, make_config):
    cfg = _tunable(tabular_csv, make_config, **{"tune.overrides": {"fit.params.subsample": 0.9}})
    spec = effective_space(cfg)["fit.params.subsample"]
    assert isinstance(spec, Const) and spec.value == 0.9


@pytest.mark.unit
def test_every_paramspec_type_round_trips_from_yaml_shaped_data(tabular_csv, make_config):
    cfg = _tunable(
        tabular_csv,
        make_config,
        **{
            "tune.overrides": {
                "fit.params.learning_rate": {
                    "type": "float",
                    "low": 0.05,
                    "high": 0.2,
                    "log": True,
                },
                "model.params.tree_method": {
                    "type": "categorical",
                    "choices": ["hist", "exact"],
                },
            }
        },
    )
    space = effective_space(cfg)
    assert isinstance(space["fit.params.learning_rate"], Float)
    assert isinstance(space["model.params.tree_method"], Categorical)


@pytest.mark.unit
def test_a_malformed_override_is_refused_by_name(tabular_csv, make_config):
    cfg = _tunable(
        tabular_csv, make_config, **{"tune.overrides": {"model.params.max_depth": {"type": "wat"}}}
    )
    with pytest.raises(TuningError, match="max_depth.*unknown type"):
        effective_space(cfg)


@pytest.mark.unit
def test_the_recorded_space_shows_the_range_not_just_the_winner():
    """ "max_depth=6" is uninterpretable without knowing it came from [3, 12]."""
    described = describe_space({"model.params.max_depth": Int(3, 12)})
    assert "3" in described["model.params.max_depth"]
    assert "12" in described["model.params.max_depth"]


# ── Skipping, and why ─────────────────────────────────────
@pytest.mark.unit
def test_tuning_off_is_reported_not_silent(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass", model="xgboost")
    result = tune(cfg)
    assert not result.ran
    assert "enabled" in result.skipped
    assert result.config is cfg  # handed straight back, untouched


@pytest.mark.unit
def test_an_empty_space_auto_disables_with_a_reason(tabular_csv, make_config, monkeypatch):
    """A plugin may legitimately declare nothing worth tuning — the CNN does not
    tune its backbone. That is a skip with a reason, not a failure."""
    import sys

    from ml_framework.core.registry import MODELS

    # xgboost declares no `suggest` hook, so emptying the declarative space is
    # enough to reproduce the "nothing worth tuning" case.
    assert MODELS.get_spec("xgboost").suggest is None
    # Reached through sys.modules because `pipeline.tune` resolves to the
    # *function* — `pipeline/__init__` binds that name, shadowing the submodule.
    # Same collision `pipeline.train` has always had, and the same resolution.
    monkeypatch.setattr(
        sys.modules["ml_framework.pipeline.tune"], "effective_space", lambda _cfg: {}
    )

    result = tune(_tunable(tabular_csv, make_config))
    assert not result.ran and "empty" in result.skipped


# ── A real search ─────────────────────────────────────────
@pytest.mark.integration
def test_a_search_returns_a_config_with_the_winning_values(tabular_csv, make_config):
    """The v1 gap: it printed the winner. This returns it applied."""
    cfg = _tunable(tabular_csv, make_config)
    result = tune(cfg)

    assert result.ran
    assert result.n_trials == 4
    assert result.best_params
    assert result.metric == "acc"  # TaskSpec.primary_metric, not a hardcoded val/loss
    assert result.direction == "max"

    # Every winning value is present in the returned config, at its dotted path.
    for path, value in result.best_params.items():
        node: object = result.config
        for part in path.split("."):
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        assert node == value, path


@pytest.mark.integration
def test_the_objective_is_the_tasks_primary_metric(regression_csv, make_config):
    """Direction comes from the task table too: regression minimizes MAE."""
    cfg = make_config(
        regression_csv,
        "regression",
        model="xgboost",
        **{
            "tune.enabled": True,
            "tune.max_trials": 3,
            "fit.params.n_estimators": 15,
            "fit.params.early_stopping_rounds": 0,
        },
    )
    result = tune(cfg)
    assert result.metric == "mae"
    assert result.direction == "min"


@pytest.mark.integration
def test_an_explicit_tune_metric_wins_over_the_task_default(tabular_csv, make_config):
    cfg = _tunable(tabular_csv, make_config, **{"tune.metric": "f1"})
    assert tune(cfg).metric == "f1"


@pytest.mark.integration
def test_an_unreported_metric_says_what_was_reported(tabular_csv, make_config):
    """The v1 failure mode was reading a key no backend produced. Now it names
    the ones that exist instead."""
    cfg = _tunable(tabular_csv, make_config, **{"tune.metric": "not_a_metric"})
    with pytest.raises(TuningError, match="did not report 'not_a_metric'"):
        tune(cfg)


@pytest.mark.integration
def test_the_trial_budget_is_honoured(tabular_csv, make_config):
    cfg = _tunable(tabular_csv, make_config, **{"tune.max_trials": 2})
    assert tune(cfg).n_trials == 2


# ── Write-back, end to end ────────────────────────────────
@pytest.mark.integration
def test_train_applies_the_winner_and_records_it(tabular_csv, make_config):
    """The P4 exit gate: best params land in bundle/config.json."""
    from ml_framework.core.bundle import read_manifest
    from ml_framework.pipeline import train

    cfg = _tunable(tabular_csv, make_config)
    metrics = train(cfg)
    assert "test_acc" in metrics

    out = Path(cfg.runtime.output_dir)
    hpo = json.loads((out / HPO_FILE).read_text(encoding="utf-8"))
    assert hpo["ran"] is True
    assert hpo["metric"] == "acc"
    assert hpo["best_params"]
    assert hpo["space"]  # the ranges, not just the winner
    assert len(hpo["trials"]) == hpo["n_trials"]

    # config.json is the record of what actually trained.
    written = json.loads((out / "config.json").read_text(encoding="utf-8"))
    for path, value in hpo["best_params"].items():
        node = written
        for part in path.split("."):
            node = node[part]
        assert node == value, path

    # And the manifest carries it, so a *served* model can answer "how were these
    # numbers chosen?" without the training directory.
    assert read_manifest(out).hpo["best_params"] == hpo["best_params"]


@pytest.mark.integration
def test_a_skipped_search_still_writes_hpo_json_saying_so(tabular_csv, make_config):
    """Absence of the file would be indistinguishable from an old bundle."""
    from ml_framework.core.bundle import read_manifest
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass", model="xgboost")  # tuning off
    train(cfg)
    out = Path(cfg.runtime.output_dir)
    hpo = json.loads((out / HPO_FILE).read_text(encoding="utf-8"))
    assert hpo["ran"] is False and hpo["skipped"]
    assert read_manifest(out).hpo is None


@pytest.mark.integration
def test_emit_config_writes_a_config_that_reloads(tabular_csv, make_config, tmp_path):
    """bundle/config.json is one run's audit record; this is the file you keep."""
    from ml_framework.config import ExperimentConfig
    from ml_framework.pipeline import train

    dest = tmp_path / "tuned.yaml"
    cfg = _tunable(tabular_csv, make_config)
    train(cfg, emit_config=dest)

    reloaded = ExperimentConfig.from_yaml(dest)
    assert reloaded.model.name == "xgboost"
    hpo = json.loads((Path(cfg.runtime.output_dir) / HPO_FILE).read_text(encoding="utf-8"))
    for path, value in hpo["best_params"].items():
        node: object = reloaded
        for part in path.split("."):
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        assert node == value, path


@pytest.mark.integration
def test_refit_reuse_keeps_the_winning_trials_model(tabular_csv, make_config):
    """`best` retrains at full budget; `reuse` keeps the trial model, which was
    trained under the reduced one. Both are defensible, so both are spelled out."""
    from ml_framework.pipeline import train

    cfg = _tunable(tabular_csv, make_config, **{"tune.refit": "reuse"})
    metrics = train(cfg)
    assert "test_acc" in metrics
    assert (Path(cfg.runtime.output_dir) / "model" / "model.json").exists()


# ── The Lightning path ────────────────────────────────────
@pytest.mark.integration
def test_the_mlp_uses_its_conditional_suggest_hook(tabular_csv, make_config):
    """A conditional space — n_layers, then that many widths — cannot be written
    as a flat mapping, which is the whole reason `ModelSpec.suggest` exists. It
    runs after the declarative space and overwrites it, so it is an escape hatch
    rather than a second competing mechanism."""
    pytest.importorskip("pytorch_lightning")

    cfg = make_config(
        tabular_csv,
        "multiclass",
        **{
            "tune.enabled": True,
            "tune.max_trials": 2,
            "tune.max_seconds": 120.0,
            "fit.budget.max_epochs": 1,
        },
    )
    result = tune(cfg)
    assert result.ran
    # hidden_dims is conditional and only the suggest hook can produce it.
    assert "model.params.hidden_dims" in result.best_params
    assert isinstance(result.best_params["model.params.hidden_dims"], list)
    # The backend's own knobs are in the same trial, from the declarative half.
    assert "fit.params.lr" in result.best_params


@pytest.mark.integration
def test_a_lightning_trial_is_capped_below_the_configured_epochs(tabular_csv, make_config):
    """Without a per-trial cap one slow trial eats the whole wall budget and the
    search degenerates to a single sample."""
    from ml_framework.pipeline.tune import _trial_budget

    cfg = make_config(tabular_csv, "multiclass", **{"fit.budget.max_epochs": 500})
    capped = _trial_budget(cfg, budget_for("lightning"))
    assert capped.max_epochs == budget_for("lightning").trial_max_epochs

    # A one-shot backend has no epochs to cap, so the configured value survives.
    assert _trial_budget(cfg, budget_for("gbdt")).max_epochs == 500


@pytest.mark.integration
def test_a_configured_budget_below_the_cap_is_not_raised_to_it(tabular_csv, make_config):
    """The cap is a ceiling, never a floor."""
    from ml_framework.pipeline.tune import _trial_budget

    cfg = make_config(tabular_csv, "multiclass", **{"fit.budget.max_epochs": 3})
    assert _trial_budget(cfg, budget_for("lightning")).max_epochs == 3
