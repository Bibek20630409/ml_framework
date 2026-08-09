"""
Cross-family model selection: gating, constraints and the decision rules.

Most of this file tests the *decision* against synthetic ``CandidateReport``s
rather than against trained models. That is deliberate: the rule that picks
between a 0.9012 model at 3 ms and a 0.9019 model at 40 ms is the part with the
interesting behaviour, and pinning it to real training runs would make the tests
slow, flaky and unable to express the cases that matter (an exact tie, an
unmeasurable latency, every candidate disqualified).

The end-to-end path — really tuning and profiling several families — is
exercised in ``tests/integration/test_selection_pipeline.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_framework.core.profile import (
    CostProfile,
    LatencyProfile,
    MaintainabilityProfile,
    ModelProfile,
)
from ml_framework.pipeline.select import (
    CRITERIA,
    SELECTION_FILE,
    CandidateReport,
    SelectionError,
    SelectionResult,
    apply_constraints,
    candidate_models,
    decide,
    evaluate_candidate,
    select,
    write_selection,
)

pytestmark = pytest.mark.unit


def report(
    model: str,
    *,
    score: float = 0.9,
    std: float = 0.0,
    folds: int = 5,
    latency: float | None = 10.0,
    mb: float = 1.0,
    explain: float = 1.0,
    method: str = "native",
    stability: float = 1.0,
    metric: str = "acc",
    skipped: str | None = None,
) -> CandidateReport:
    """A candidate with exactly the measurements a test cares about.

    ``latency=None`` means "could not be measured", which is a distinct state
    from "slow" and has its own behaviour in the tie-break.
    """
    if skipped:
        return CandidateReport(model=model, skipped=skipped)
    return CandidateReport(
        model=model,
        backend="test",
        best_params={"model.params.x": 1},
        profile=ModelProfile(
            model=model,
            backend="test",
            primary_metric=metric,
            score=score,
            score_std=std,
            n_folds=folds,
            latency=(
                LatencyProfile(p95_ms=latency, p50_ms=latency, n_calls=20)
                if latency is not None
                else LatencyProfile(error="not timed")
            ),
            cost=CostProfile(artifact_bytes=int(mb * 1024 * 1024)),
            explainability=explain,
            explain_method=method,
            maintainability=MaintainabilityProfile(fold_stability=stability, n_folds=folds),
        ),
    )


@pytest.fixture
def cfg(make_config, binary_csv):
    return make_config(binary_csv, "binary", model="mlp")


# ── The tolerance rule ────────────────────────────────────
def test_the_best_score_wins_when_nothing_is_close(cfg):
    winner, reason = decide([report("a", score=0.80), report("b", score=0.95)], cfg)
    assert winner.model == "b"
    assert "best acc (0.9500)" in reason


def test_a_statistically_tied_cheaper_model_wins(cfg):
    # The rule that makes this whole module worth having: 0.9019 at 40 ms and
    # 0.9012 at 3 ms are the same model as far as the data can tell.
    fast = report("fast", score=0.9012, std=0.02, latency=3.0)
    slow = report("slow", score=0.9019, std=0.02, latency=40.0)

    winner, reason = decide([slow, fast], cfg)

    assert winner.model == "fast"
    assert "within the" in reason
    assert "p95 latency (3.00 ms vs 40.00 ms)" in reason


def test_a_meaningfully_better_score_is_not_traded_for_speed(cfg):
    # The other half of the same rule. A 5-point gap on a tight CV is real, and
    # no amount of speed should buy it.
    fast = report("fast", score=0.85, std=0.001, latency=1.0)
    slow = report("slow", score=0.90, std=0.001, latency=40.0)

    winner, _ = decide([fast, slow], cfg)
    assert winner.model == "slow"


def test_the_default_tolerance_is_the_standard_error_of_the_cv_mean(cfg):
    # std 0.02 over 4 folds → std error 0.01. A 0.005 gap is inside it.
    best = report("best", score=0.90, std=0.02, folds=4, latency=40.0)
    near = report("near", score=0.895, std=0.02, folds=4, latency=1.0)
    assert decide([best, near], cfg)[0].model == "near"

    # The same gap against a *tight* CV is outside it: std error 0.0005.
    tight_best = report("best", score=0.90, std=0.001, folds=4, latency=40.0)
    tight_near = report("near", score=0.895, std=0.001, folds=4, latency=1.0)
    assert decide([tight_best, tight_near], cfg)[0].model == "best"


def test_an_explicit_tolerance_replaces_the_standard_error(cfg):
    cfg = cfg.with_overrides({"select.tolerance": 0.1})
    best = report("best", score=0.90, std=0.0001, latency=40.0)
    near = report("near", score=0.85, std=0.0001, latency=1.0)

    winner, reason = decide([best, near], cfg)
    assert winner.model == "near"
    assert "configured" in reason


def test_a_zero_tolerance_always_picks_the_top_score(cfg):
    cfg = cfg.with_overrides({"select.tolerance": 0.0})
    best = report("best", score=0.9001, latency=40.0)
    near = report("near", score=0.9000, latency=1.0)
    assert decide([best, near], cfg)[0].model == "best"


def test_a_minimized_metric_prefers_the_lower_score(make_config, regression_csv):
    # `mae` is minimize; the winner must be the smallest, not the largest.
    cfg = make_config(regression_csv, "regression", model="mlp")
    winner, _ = decide(
        [report("high", score=9.0, metric="mae"), report("low", score=2.0, metric="mae")], cfg
    )
    assert winner.model == "low"


def test_a_minimized_metric_still_breaks_ties_on_cost(make_config, regression_csv):
    cfg = make_config(regression_csv, "regression", model="mlp")
    slow = report("slow", score=2.00, std=0.5, metric="mae", latency=40.0)
    fast = report("fast", score=2.05, std=0.5, metric="mae", latency=1.0)
    assert decide([slow, fast], cfg)[0].model == "fast"


# ── The tie-break order ───────────────────────────────────
def test_latency_breaks_a_tie_before_size(cfg):
    # A bigger model that answers faster wins: the axis order is latency first,
    # because that is the one that turns into a user-visible failure. The top
    # scorer is the *slow* one, so the tie-break has to overturn it for the
    # ordering to be observable.
    big_fast = report("big_fast", score=0.900, std=0.05, latency=1.0, mb=50.0)
    small_slow = report("small_slow", score=0.901, std=0.05, latency=30.0, mb=0.1)

    winner, reason = decide([big_fast, small_slow], cfg)
    assert winner.model == "big_fast"
    assert "p95 latency (1.00 ms vs 30.00 ms)" in reason


def test_size_breaks_a_tie_when_latency_is_equal(cfg):
    big = report("big", score=0.901, std=0.05, latency=5.0, mb=50.0)
    small = report("small", score=0.900, std=0.05, latency=5.0, mb=0.5)

    winner, reason = decide([big, small], cfg)
    assert winner.model == "small"
    assert "artifact size (0.50 MB vs 50.00 MB)" in reason


def test_explainability_breaks_a_tie_when_speed_and_size_are_equal(cfg):
    opaque = report("opaque", score=0.901, std=0.05, explain=0.5, method="permutation")
    clear = report("clear", score=0.900, std=0.05, explain=1.0, method="native")

    winner, reason = decide([opaque, clear], cfg)
    assert winner.model == "clear"
    assert "explainability (1.00 (native) vs 0.50 (permutation))" in reason


def test_stability_is_the_last_tie_break(cfg):
    steady = report("steady", score=0.900, std=0.05, stability=0.99)
    jumpy = report("jumpy", score=0.901, std=0.05, stability=0.20)

    winner, reason = decide([jumpy, steady], cfg)
    assert winner.model == "steady"
    assert "fold stability (0.99 vs 0.20)" in reason


def test_the_top_scorer_winning_its_own_tie_says_so_without_a_tie_break(cfg):
    # When the best score is also the cheapest there is no axis to name, and the
    # reason must not invent one.
    _, reason = decide(
        [
            report("a", score=0.9, std=0.05, latency=1.0),
            report("b", score=0.9, std=0.05, latency=9.0),
        ],
        cfg,
    )
    assert "also cheapest of the 2" in reason
    assert "tie-break" not in reason


def test_an_unmeasured_latency_does_not_win_a_tie(cfg):
    # The failure mode this guards: treating "not timed" as 0 ms would make a
    # broken measurement the fastest candidate in the field.
    timed = report("timed", score=0.9, std=0.05, latency=25.0)
    untimed = report("untimed", score=0.9, std=0.05, latency=None)

    assert decide([untimed, timed], cfg)[0].model == "timed"


def test_ties_on_every_axis_resolve_by_name_for_determinism(cfg):
    a = report("aaa", score=0.9, std=0.05)
    b = report("bbb", score=0.9, std=0.05)
    assert decide([b, a], cfg)[0].model == "aaa"
    assert decide([a, b], cfg)[0].model == "aaa"


# ── Constraints ───────────────────────────────────────────
def test_a_latency_budget_disqualifies_and_says_by_how_much(cfg):
    cfg = cfg.with_overrides({"select.constraints.max_latency_p95_ms": 20.0})
    reports = apply_constraints([report("slow", latency=24.1)], cfg)

    assert not reports[0].eligible
    assert "24.10 ms exceeds the 20 ms budget" in reports[0].disqualified


def test_a_disqualified_candidate_cannot_win_however_good_its_score(cfg):
    cfg = cfg.with_overrides({"select.constraints.max_latency_p95_ms": 20.0})
    reports = apply_constraints(
        [report("brilliant_slow", score=0.99, latency=200.0), report("ok_fast", score=0.7)], cfg
    )
    assert decide(reports, cfg)[0].model == "ok_fast"


def test_a_size_budget_disqualifies(cfg):
    cfg = cfg.with_overrides({"select.constraints.max_model_mb": 10.0})
    reports = apply_constraints([report("fat", mb=64.0)], cfg)
    assert "64.00 MB exceeds the 10 MB budget" in reports[0].disqualified


def test_an_explainability_floor_disqualifies(cfg):
    cfg = cfg.with_overrides({"select.constraints.min_explainability": 0.8})
    reports = apply_constraints([report("opaque", explain=0.5, method="permutation")], cfg)
    assert "0.50 (permutation) is below the 0.80 floor" in reports[0].disqualified


def test_a_performance_floor_disqualifies(cfg):
    cfg = cfg.with_overrides({"select.constraints.min_performance": 0.8})
    reports = apply_constraints([report("weak", score=0.6)], cfg)
    assert "does not clear the 0.8 floor" in reports[0].disqualified


def test_a_performance_floor_on_a_minimized_metric_reads_the_other_way(make_config, regression_csv):
    cfg = make_config(regression_csv, "regression", model="mlp").with_overrides(
        {"select.constraints.min_performance": 5.0}
    )
    high = apply_constraints([report("high", score=9.0, metric="mae")], cfg)
    low = apply_constraints([report("low", score=2.0, metric="mae")], cfg)
    assert high[0].disqualified  # 9.0 is worse than a 5.0 ceiling on error
    assert not low[0].disqualified


def test_an_unmeasurable_latency_fails_a_latency_constraint(cfg):
    # Refusing is the safe direction: a model that cannot be timed has not shown
    # it meets the budget, and "unknown" must not read as "passed".
    cfg = cfg.with_overrides({"select.constraints.max_latency_p95_ms": 20.0})
    reports = apply_constraints([report("untimed", latency=None)], cfg)
    assert "could not be measured" in reports[0].disqualified


def test_no_constraints_disqualifies_nothing(cfg):
    reports = apply_constraints([report("a", latency=9999.0, mb=9999.0, explain=0.0)], cfg)
    assert reports[0].eligible


def test_every_candidate_disqualified_is_an_error_that_lists_them(cfg):
    cfg = cfg.with_overrides({"select.constraints.max_latency_p95_ms": 1.0})
    reports = apply_constraints([report("a", latency=50.0), report("b", latency=60.0)], cfg)

    with pytest.raises(SelectionError) as excinfo:
        decide(reports, cfg)
    message = str(excinfo.value)
    assert "no candidate survived" in message
    assert "a: rejected" in message and "b: rejected" in message


def test_a_skipped_candidate_is_reported_but_not_decided_on(cfg):
    reports = [report("ok"), report("broken", skipped="tuning failed: boom")]
    winner, _ = decide(reports, cfg)
    assert winner.model == "ok"


# ── The weighted rule ─────────────────────────────────────
def _weighted(cfg, **weights):
    return cfg.with_overrides(
        {"select.objective": "weighted", **{f"select.weights.{k}": v for k, v in weights.items()}}
    )


def test_weighting_performance_alone_picks_the_best_score(cfg):
    cfg = _weighted(cfg, performance=1.0)
    winner, reason = decide([report("a", score=0.8, latency=1.0), report("b", score=0.9)], cfg)
    assert winner.model == "b"
    assert "highest weighted score" in reason


def test_weighting_latency_alone_picks_the_fastest(cfg):
    cfg = _weighted(cfg, latency=1.0)
    winner, _ = decide(
        [report("a", score=0.99, latency=40.0), report("b", score=0.5, latency=1.0)], cfg
    )
    assert winner.model == "b"


def test_weights_shift_the_winner(cfg):
    accurate = report("accurate", score=0.95, latency=40.0)
    quick = report("quick", score=0.90, latency=1.0)

    assert (
        decide([accurate, quick], _weighted(cfg, performance=0.9, latency=0.1))[0].model
        == "accurate"
    )
    assert (
        decide([accurate, quick], _weighted(cfg, performance=0.1, latency=0.9))[0].model == "quick"
    )


def test_the_weighted_reason_shows_the_arithmetic(cfg):
    cfg = _weighted(cfg, performance=0.7, latency=0.3)
    _, reason = decide([report("a", score=0.8, latency=9.0), report("b", score=0.9)], cfg)
    assert "performance=" in reason and "latency=" in reason
    assert "x0.7" in reason


def test_a_criterion_identical_across_candidates_does_not_break_the_tie(cfg):
    # Every candidate has the same latency, so latency carries no information in
    # this run and must contribute equally rather than amplifying float noise.
    cfg = _weighted(cfg, performance=0.5, latency=0.5)
    winner, _ = decide(
        [report("a", score=0.8, latency=5.0), report("b", score=0.9, latency=5.0)], cfg
    )
    assert winner.model == "b"


def test_weights_need_not_sum_to_one(cfg):
    cfg = _weighted(cfg, performance=7.0, latency=3.0)
    assert decide([report("a", score=0.8, latency=1.0), report("b", score=0.9, latency=40.0)], cfg)


def test_an_unknown_criterion_is_refused_at_config_load(cfg):
    with pytest.raises(ValueError, match="unknown criteria"):
        cfg.with_overrides({"select.weights.speed": 1.0})


def test_the_weighted_objective_needs_at_least_one_weight(cfg):
    with pytest.raises(ValueError, match="select.weights is empty"):
        cfg.with_overrides({"select.objective": "weighted"})


def test_the_criteria_tuple_matches_the_schema(cfg):
    # The two lists are in different modules and would drift silently.
    assert set(CRITERIA) == {
        "performance",
        "latency",
        "cost",
        "explainability",
        "maintainability",
    }
    for name in CRITERIA:
        cfg.with_overrides({f"select.weights.{name}": 1.0})  # must not raise


# ── Candidate gating ──────────────────────────────────────
def test_candidates_default_to_every_compatible_installed_family(cfg):
    names = candidate_models(cfg)
    assert "mlp" in names  # torch is a dev dependency
    assert all(isinstance(n, str) for n in names)


def test_candidates_are_ranked_by_auto_priority(cfg):
    pytest.importorskip("xgboost", reason="the gbdt extra is not installed")
    names = candidate_models(cfg)
    # xgboost is auto_priority 30, mlp is 10.
    assert names.index("xgboost") < names.index("mlp")


def test_an_explicit_candidate_list_is_used_verbatim(cfg):
    cfg = cfg.with_overrides({"select.candidates": ["mlp"]})
    assert candidate_models(cfg) == ["mlp"]


def test_an_explicit_list_is_deduplicated(cfg):
    cfg = cfg.with_overrides({"select.candidates": ["mlp", "mlp"]})
    assert candidate_models(cfg) == ["mlp"]


def test_a_named_candidate_that_cannot_work_is_an_error_not_a_shorter_list(cfg):
    # Silently dropping it would leave the user believing their model competed.
    cfg = cfg.with_overrides({"select.candidates": ["cnn"]})
    with pytest.raises(Exception, match="cnn"):
        candidate_models(cfg)


def test_max_candidates_caps_the_pool(cfg):
    cfg = cfg.with_overrides({"select.max_candidates": 1})
    assert len(candidate_models(cfg)) == 1


def test_a_high_explainability_floor_gates_out_models_with_no_native_importances(cfg):
    pytest.importorskip("xgboost", reason="the gbdt extra is not installed")
    cfg = cfg.with_overrides({"select.constraints.min_explainability": 0.9})

    names = candidate_models(cfg)

    assert "xgboost" in names  # declares native_feature_importance
    assert "mlp" not in names  # cannot reach 0.9 by any route


def test_a_moderate_explainability_floor_keeps_permutation_capable_models(cfg):
    # 0.5 is the permutation tier, which any model with predict can reach, so
    # the a-priori gate must not fire.
    cfg = cfg.with_overrides({"select.constraints.min_explainability": 0.5})
    assert "mlp" in candidate_models(cfg)


def test_gating_everything_out_is_an_error_that_says_what_to_relax(
    cfg, monkeypatch: pytest.MonkeyPatch
):
    # Reached through `sys.modules` rather than `import ml_framework.pipeline.select`:
    # `pipeline/__init__.py` re-exports the *function* under that name, so the
    # dotted import resolves to the function and not to the module. The same is
    # already true of `train` and `tune`.
    import sys

    module = sys.modules["ml_framework.pipeline.select"]
    cfg = cfg.with_overrides({"select.constraints.min_explainability": 1.0})

    # Narrow the discovered pool to a model with no native importances, so the
    # a-priori gate removes the only candidate there was.
    original = module.models_for
    monkeypatch.setattr(
        module,
        "models_for",
        lambda *a, **k: [s for s in original(*a, **k) if s.name == "mlp"],
    )

    with pytest.raises(SelectionError, match="gated out before training"):
        candidate_models(cfg)


# ── Pass-through ──────────────────────────────────────────
def test_select_is_a_pass_through_when_it_is_disabled(cfg):
    # The property `train()` depends on: one code path whether a bake-off ran or
    # not. Disabled selection still returns a config and a TuneResult.
    result = select(cfg)

    assert not result.ran
    assert result.skipped == "select.enabled is false"
    assert result.winner == cfg.model.name
    assert result.config is cfg
    assert result.tuning is not None


def test_a_disabled_selection_still_returns_the_tuned_config(cfg, monkeypatch):
    # The regression this guards is silent and total: returning the *input*
    # config from the pass-through throws away the whole search one line after
    # running it, and every downstream artifact still looks correct because it
    # faithfully records the untuned config that was actually fitted.
    import sys

    from ml_framework.pipeline.tune import TuneResult

    module = sys.modules["ml_framework.pipeline.select"]
    tuned = cfg.with_overrides({"fit.params.lr": 0.00123})
    monkeypatch.setattr(
        module,
        "tune",
        lambda config, **kw: TuneResult(
            config=tuned, best_params={"fit.params.lr": 0.00123}, metric="acc"
        ),
    )

    result = select(cfg)

    assert result.config.fit.params["lr"] == 0.00123
    assert result.config is tuned


def test_a_disabled_selection_is_not_recorded_in_the_manifest(cfg):
    assert select(cfg).ran is False  # train() writes `selection: null` for this


# ── Reporting ─────────────────────────────────────────────
def test_the_table_lists_every_candidate_with_its_status(cfg):
    result = SelectionResult(
        config=cfg,
        tuning=select(cfg).tuning,
        winner="a",
        reports=[report("a"), report("b"), report("c", skipped="boom")],
    )
    table = result.table()

    assert "WINNER" in table
    assert "skipped: boom" in table
    for name in ("a", "b", "c"):
        assert name in table


def test_the_table_is_ascii_so_a_windows_console_can_print_it(cfg):
    # A crash in the *reporting* of a successful bake-off is an absurd way to
    # lose one, and cp1252 cannot render a box character.
    result = SelectionResult(config=cfg, tuning=select(cfg).tuning, reports=[report("a")])
    assert result.table().isascii()


def test_selection_json_records_every_candidate_not_just_the_winner(tmp_path: Path, cfg):
    # "We chose XGBoost" is not an answer to "why not the neural net?".
    result = SelectionResult(
        config=cfg,
        tuning=select(cfg).tuning,
        winner="a",
        reason="best score",
        reports=[report("a"), report("b", latency=99.0)],
    )
    path = write_selection(result, tmp_path)

    assert path.name == SELECTION_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["winner"] == "a"
    assert payload["n_candidates"] == 2
    assert {c["model"] for c in payload["candidates"]} == {"a", "b"}
    assert payload["candidates"][0]["profile"]["latency"]["p95_ms"] is not None


def test_a_report_for_a_broken_candidate_carries_the_reason(cfg):
    # A bake-off over five families must survive one of them failing.
    result = evaluate_candidate(cfg, "definitely_not_a_model")
    assert not result.eligible
    assert result.skipped
    assert "definitely_not_a_model" in result.skipped
