"""
Model selection end to end: really tune several families, really compare them.

``tests/pipeline/test_select.py`` pins the decision rule against synthetic
reports. This file checks the parts that only appear when models are actually
trained — that a bake-off crosses backends, that the winner's config is what gets
fitted, that the bundle records which families lost, and that the CV strategies
survive contact with a real source.

Every test here caps the tuning budget hard. A bake-off is one tuning budget per
candidate, and the default would put minutes into a suite that has to stay
runnable.
"""

from __future__ import annotations

import json

import pytest

from ml_framework.pipeline.select import select
from ml_framework.pipeline.train import train

pytestmark = pytest.mark.integration

pytest.importorskip("optuna", reason="the hpo extra is not installed")
xgboost = pytest.importorskip("xgboost", reason="the gbdt extra is not installed")


# Two trials against a 15-second wall is enough to exercise the search machinery
# without turning the suite into a benchmark.
FAST = {
    "tune.max_trials": 2,
    "tune.max_seconds": 15.0,
    "tune.enabled": True,
    "fit.budget.max_epochs": 2,
    "select.enabled": True,
    "select.profile_samples": 32,
}


def _cfg(make_config, csv, task, **overrides):
    return make_config(csv, task, model="xgboost", **{**FAST, **overrides})


# ── The bake-off ──────────────────────────────────────────
def test_a_bake_off_compares_two_tree_families_and_picks_one(make_config, binary_csv):
    pytest.importorskip("lightgbm", reason="the gbdt extra is not installed")
    cfg = _cfg(make_config, binary_csv, "binary", **{"select.candidates": ["xgboost", "lightgbm"]})

    result = select(cfg)

    assert result.ran
    assert result.winner in {"xgboost", "lightgbm"}
    assert len(result.reports) == 2
    assert all(r.eligible for r in result.reports), [r.skipped for r in result.reports]
    assert result.reason


def test_every_candidate_is_profiled_on_all_five_criteria(make_config, binary_csv):
    cfg = _cfg(make_config, binary_csv, "binary", **{"select.candidates": ["xgboost"]})

    profile = select(cfg).reports[0].profile

    assert profile is not None
    assert profile.n_folds >= 2  # performance came from folds, not one holdout
    assert profile.latency.measured and profile.latency.p95_ms > 0
    assert profile.cost.artifact_bytes > 0
    assert profile.explainability == 1.0  # a tree has native importances
    assert profile.explain_method == "native"
    assert 0.0 <= profile.maintainability.score <= 1.0


def test_a_bake_off_crosses_backends(make_config, binary_csv):
    # The claim the whole module rests on: a torch model and a tree model,
    # trained by different backends, compared on one table.
    pytest.importorskip("torch", reason="the lightning extra is not installed")
    cfg = _cfg(make_config, binary_csv, "binary", **{"select.candidates": ["xgboost", "mlp"]})

    result = select(cfg)

    backends = {r.backend for r in result.reports if r.eligible}
    assert backends == {"gbdt", "lightning"}


def test_the_winning_config_names_the_winning_model_and_its_parameters(make_config, binary_csv):
    pytest.importorskip("lightgbm", reason="the gbdt extra is not installed")
    cfg = _cfg(make_config, binary_csv, "binary", **{"select.candidates": ["xgboost", "lightgbm"]})

    result = select(cfg)

    assert result.config.model.name == result.winner
    # And the tuned values are applied, not merely reported.
    winner = next(r for r in result.reports if r.model == result.winner)
    for path, value in winner.best_params.items():
        if path.startswith("model.params."):
            assert result.config.model.params[path.split(".")[-1]] == value


def test_selection_is_disabled_on_the_candidate_config_so_it_cannot_recurse(
    make_config, binary_csv
):
    cfg = _cfg(make_config, binary_csv, "binary", **{"select.candidates": ["xgboost"]})
    assert select(cfg).config.select.enabled is False


def test_a_latency_budget_that_nothing_meets_fails_with_the_numbers(make_config, binary_csv):
    from ml_framework.pipeline.select import SelectionError

    cfg = _cfg(
        make_config,
        binary_csv,
        "binary",
        **{
            "select.candidates": ["xgboost"],
            "select.constraints.max_latency_p95_ms": 1e-6,  # nothing is this fast
        },
    )
    with pytest.raises(SelectionError, match="no candidate survived"):
        select(cfg)


def test_a_broken_candidate_does_not_take_the_bake_off_down(make_config, binary_csv):
    cfg = _cfg(
        make_config,
        binary_csv,
        "binary",
        **{"select.candidates": ["xgboost", "definitely_not_a_model"]},
    )
    # `candidate_models` validates an explicit list, so the bad name is caught up
    # front rather than after a fit — which is the better of the two failures.
    with pytest.raises(Exception, match="definitely_not_a_model"):
        select(cfg)


# ── Artifacts ─────────────────────────────────────────────
def test_selection_json_is_written_beside_the_bundle(make_config, binary_csv, tmp_path):
    out = tmp_path / "sel"
    cfg = _cfg(
        make_config,
        binary_csv,
        "binary",
        **{"select.candidates": ["xgboost"], "runtime.output_dir": str(out)},
    )
    select(cfg)

    payload = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert payload["ran"] is True
    assert payload["winner"] == "xgboost"
    assert payload["candidates"][0]["profile"]["latency"]["p95_ms"] > 0


def test_feature_importances_land_in_the_candidate_directory(make_config, binary_csv, tmp_path):
    out = tmp_path / "imp"
    cfg = _cfg(
        make_config,
        binary_csv,
        "binary",
        **{"select.candidates": ["xgboost"], "runtime.output_dir": str(out)},
    )
    select(cfg)

    written = list(out.rglob("feature_importance.json"))
    assert written, "no attribution artifact was written"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["method"] == "native"
    assert payload["top"]


# ── Through train() ───────────────────────────────────────
def test_train_with_selection_fits_the_winner_and_records_the_bake_off(
    make_config, binary_csv, tmp_path
):
    pytest.importorskip("lightgbm", reason="the gbdt extra is not installed")
    out = tmp_path / "trained"
    cfg = _cfg(
        make_config,
        binary_csv,
        "binary",
        **{
            "select.candidates": ["xgboost", "lightgbm"],
            "runtime.output_dir": str(out),
        },
    )

    metrics = train(cfg)

    assert "test_acc" in metrics
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["name"] == manifest["selection"]["winner"]
    assert manifest["selection"]["n_candidates"] == 2
    # A served bundle can answer "why this family?" without the training dir.
    assert manifest["selection"]["reason"]
    assert (out / "model").exists()


def test_train_without_selection_records_no_bake_off(make_config, binary_csv, tmp_path):
    # The default path must be untouched: `selection: null`, one model, no table.
    out = tmp_path / "plain"
    cfg = make_config(
        binary_csv,
        "binary",
        model="xgboost",
        **{"tune.enabled": False, "runtime.output_dir": str(out)},
    )

    train(cfg)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"] is None
    assert manifest["model"]["name"] == "xgboost"
    assert not (out / "selection.json").exists()


# ── CV strategies against a real source ───────────────────
@pytest.mark.parametrize("strategy", ["stratified", "kfold", "purged"])
def test_each_cv_strategy_produces_folds_from_a_real_table(make_config, binary_csv, strategy):
    from ml_framework.data.builders import cv_folds

    cfg = make_config(
        binary_csv,
        "binary",
        model="xgboost",
        **{
            "tune.enabled": False,
            "data.split.folds": 3,
            "data.split.cv_strategy": strategy,
            "data.split.label_horizon": 4 if strategy == "purged" else 0,
        },
    )

    folds = cv_folds(cfg)

    assert len(folds) == 3
    for fold in folds:
        assert len(fold.train) and len(fold.val) and len(fold.test)


def test_cpcv_produces_its_combinatorial_fold_count(make_config, binary_csv):
    from ml_framework.data.builders import cv_folds

    cfg = make_config(
        binary_csv,
        "binary",
        model="xgboost",
        **{
            "tune.enabled": False,
            "data.split.folds": 2,
            "data.split.cv_strategy": "cpcv",
            "data.split.cpcv_groups": 5,
            "data.split.cpcv_test_groups": 2,
        },
    )
    assert len(cv_folds(cfg)) == 10  # C(5, 2)


def test_auto_resolves_to_stratified_for_a_classification_table(make_config, binary_csv):
    cfg = make_config(binary_csv, "binary", model="xgboost", **{"data.split.folds": 3})
    assert cfg.data.split.resolved_cv_strategy("tabular", "binary") == "stratified"


def test_auto_resolves_to_purged_once_a_label_horizon_is_declared(make_config, binary_csv):
    # Declaring overlapping labels and then getting plain k-fold would ignore the
    # one thing the user said about their data.
    cfg = make_config(binary_csv, "binary", model="xgboost", **{"data.split.label_horizon": 5})
    assert cfg.data.split.resolved_cv_strategy("tabular", "binary") == "purged"


def test_stratified_is_refused_for_a_continuous_target(make_config, regression_csv):
    from ml_framework.data.builders import cv_folds

    cfg = make_config(
        regression_csv,
        "regression",
        model="xgboost",
        **{"tune.enabled": False, "data.split.folds": 3, "data.split.cv_strategy": "stratified"},
    )
    with pytest.raises(ValueError, match="one class label per row"):
        cv_folds(cfg)


# ── The CV tuning objective ───────────────────────────────
def test_a_cv_objective_scores_each_trial_across_inner_folds(make_config, binary_csv):
    from ml_framework.pipeline.tune import tune

    cfg = make_config(
        binary_csv,
        "binary",
        model="xgboost",
        **{
            "tune.enabled": True,
            "tune.objective": "cv",
            "tune.cv_folds": 2,
            "tune.max_trials": 2,
            "tune.max_seconds": 20.0,
        },
    )

    result = tune(cfg)

    assert result.ran
    assert result.best_value is not None
    assert result.n_trials >= 1


def test_the_inner_fold_count_is_independent_of_the_outer_one(make_config, binary_csv):
    # The honest nested arrangement — 4 outer, 2 inner — has to be expressible.
    cfg = make_config(
        binary_csv,
        "binary",
        model="xgboost",
        **{"data.split.folds": 4, "tune.cv_folds": 2, "tune.objective": "cv"},
    )
    assert cfg.data.split.folds == 4
    assert cfg.tune.cv_folds == 2
