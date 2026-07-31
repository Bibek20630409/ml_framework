"""P6: forecasting — the task row, the temporal splitters, and the leakage guard.

The phase gates are here: `strategy: random` on `kind: timeseries` **raises**, and
a temporal split beats a shuffled one on a series with trend — which is the whole
argument for refusing the shuffle in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_framework.config import ExperimentConfig
from ml_framework.core.metrics import mase, smape
from ml_framework.data.splitters import RollingOriginSplitter, SplitError

# ── Fixtures ──────────────────────────────────────────────
POINTS = 300
SEASON = 7


@pytest.fixture
def series_csv(tmp_path: Path) -> Path:
    """Trend + weekly seasonality + noise: a series where order genuinely matters."""
    rng = np.random.default_rng(0)
    t = np.arange(POINTS)
    values = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0, 0.3, POINTS)
    frame = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=POINTS, freq="D"), "value": values}
    )
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    return path


def ts_config(csv: Path, tmp_path: Path, model: str = "ts.naive", **overrides) -> ExperimentConfig:
    raw = {
        "task": "forecasting",
        "runtime": {
            "output_dir": str(tmp_path / f"out_{model.replace('.', '_')}"),
            "seed": 0,
            "num_workers": 0,
        },
        "data": {
            "kind": "timeseries",
            "path": str(csv),
            "target": "value",
            "split": {"time_col": "date", "val_size": 0.15, "test_size": 0.15},
            "params": {"seasonality": SEASON, "window": 14},
        },
        "model": {"name": model},
        "fit": {"budget": {"max_epochs": 15}, "batch_size": 16},
        "tune": {"enabled": False},
        "logging": {"backend": "none"},
    }
    cfg = ExperimentConfig.model_validate(raw)
    return cfg.with_overrides(overrides) if overrides else cfg


# ── THE PHASE GATE: leakage ───────────────────────────────
@pytest.mark.unit
def test_a_shuffled_split_on_a_time_series_raises(series_csv, tmp_path):
    """**Phase gate.** The most damaging silent failure in this domain, refused.

    A random split puts future rows in training and past rows in test, and reports
    a *better* score for it. Nothing crashes, so nothing tells you — which is
    exactly why warning and proceeding would be the wrong choice.
    """
    with pytest.raises(ValidationError, match="shuffles the future into training"):
        ts_config(series_csv, tmp_path, **{"data.split.strategy": "random"})


@pytest.mark.unit
def test_the_escape_hatch_costs_typing_the_word_leakage(series_csv, tmp_path):
    """It exists — a cross-sectional model may legitimately carry a date column —
    but at a price roughly equal to the deliberation the decision deserves."""
    cfg = ts_config(
        series_csv,
        tmp_path,
        **{"data.split.strategy": "random", "data.split.allow_temporal_leakage": True},
    )
    assert cfg.data.split.strategy == "random"


@pytest.mark.unit
def test_tabular_data_may_still_be_shuffled(tabular_csv, make_config):
    """The guard is about time-ordered data, not about the word 'random'."""
    assert make_config(tabular_csv, "multiclass", **{"data.split.strategy": "random"})


@pytest.mark.unit
def test_auto_resolves_to_temporal_for_a_time_series(series_csv, tmp_path):
    cfg = ts_config(series_csv, tmp_path)
    assert cfg.data.split.strategy == "auto"
    assert cfg.data.split.resolved_strategy("timeseries") == "temporal"


# ── THE PHASE GATE: temporal beats shuffled ───────────────
@pytest.mark.integration
def test_a_temporal_split_scores_honestly_where_a_shuffled_one_flatters(series_csv, tmp_path):
    """**Phase gate.** Why the guard exists, demonstrated rather than asserted.

    Same data, same model, two splits. The shuffled split lets the model
    interpolate between points it has already seen on either side; the temporal
    split makes it extrapolate, which is the thing it will actually be asked to do.
    The shuffled score is *better* — and meaningless.
    """
    from ml_framework.data.splitters import RandomSplitter, TemporalSplitter

    frame = pd.read_csv(series_csv)
    values = frame["value"].to_numpy()
    n = len(values)

    def score(parts) -> float:
        # A one-nearest-neighbour forecast in *index* space: for each test point,
        # predict the value of the closest training point in time. Deliberately
        # simple, so what is measured is the split rather than a model.
        train_idx = np.sort(parts.train)
        nearest = np.searchsorted(train_idx, parts.test).clip(0, len(train_idx) - 1)
        return float(np.abs(values[parts.test] - values[train_idx[nearest]]).mean())

    shuffled = score(RandomSplitter(seed=0, task="regression").split(n, y=values))
    temporal = score(TemporalSplitter(val_size=0.15, test_size=0.15).split(n))

    # The shuffled split has training points interleaved among the test points, so
    # its "nearest neighbour" is often adjacent in time. That is the leak.
    assert shuffled < temporal, (shuffled, temporal)


# ── Rolling origin ────────────────────────────────────────
@pytest.mark.unit
def test_every_fold_trains_only_on_its_past():
    """The property that makes this cross-validation rather than leakage."""
    folds = RollingOriginSplitter(folds=3, horizon=5, gap=1).split(60)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train.max() < fold.val.min()
        assert fold.val.max() < fold.test.min()


@pytest.mark.unit
def test_the_gap_is_respected_between_segments():
    """With lag features the last train rows and the first test rows share source
    observations, so a zero gap leaks even though the split is chronological."""
    fold = RollingOriginSplitter(folds=1, horizon=4, gap=3).split(40)[0]
    assert fold.val.min() - fold.train.max() == 4  # 3 dropped + 1
    assert fold.test.min() - fold.val.max() == 4


@pytest.mark.unit
def test_the_origin_advances_each_fold():
    folds = RollingOriginSplitter(folds=3, horizon=5).split(60)
    starts = [f.test.min() for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == 3


@pytest.mark.unit
def test_expanding_grows_the_training_set_and_sliding_does_not():
    """An expanding window is what a production retrain does; a sliding one is for
    when old data is actively misleading."""
    expanding = RollingOriginSplitter(folds=3, horizon=5, expanding=True).split(60)
    sliding = RollingOriginSplitter(folds=3, horizon=5, expanding=False).split(60)

    assert [len(f.train) for f in expanding] == sorted(len(f.train) for f in expanding)
    assert len(expanding[-1].train) > len(expanding[0].train)
    assert len({len(f.train) for f in sliding}) == 1  # fixed width


@pytest.mark.unit
def test_a_series_too_short_for_the_horizon_is_refused():
    with pytest.raises(SplitError, match="too short|no rolling-origin folds"):
        RollingOriginSplitter(folds=5, horizon=20).split(30)


# ── Metrics ───────────────────────────────────────────────
@pytest.mark.unit
def test_smape_is_scale_free():
    """The reason MAE is a poor primary metric: it is not."""
    a, b = np.array([100.0, 200.0]), np.array([110.0, 220.0])
    assert smape(a, b) == pytest.approx(smape(a * 1000, b * 1000))


@pytest.mark.unit
def test_smape_scores_a_perfect_zero_forecast_rather_than_dividing_by_zero():
    """Both actual and predicted zero is *exactly right*; a NaN there would make a
    perfect forecast unscoreable."""
    assert smape(np.zeros(4), np.zeros(4)) == 0.0


@pytest.mark.unit
def test_mase_scales_by_the_series_step_size():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Average step change is 1.0, so an error of 0.5 per point scales to 0.5.
    assert mase(y, y + 0.5) == pytest.approx(0.5)


@pytest.mark.unit
def test_mase_is_undefined_for_a_constant_series():
    """The naive forecast is perfect, so the ratio has no meaning. NaN says so;
    0 or inf would both be read as a result."""
    assert np.isnan(mase(np.ones(5), np.ones(5) * 2))


@pytest.mark.unit
def test_forecasting_has_a_task_row_with_mase_as_its_objective():
    from ml_framework.core.task import get_task_spec

    spec = get_task_spec("forecasting")
    assert spec.primary_metric == "mase"
    assert spec.direction == "min"
    assert spec.output_kind == "series"
    assert not spec.is_classification


# ── The two payloads ──────────────────────────────────────
@pytest.mark.integration
def test_a_forecast_model_gets_the_series_itself(series_csv, tmp_path):
    """Prophet fits the series; handing it a feature matrix would be a lie about
    what it does."""
    from ml_framework.data import build_bundle

    bundle = build_bundle(ts_config(series_csv, tmp_path, model="ts.naive"))
    assert bundle.payload == "series"
    assert bundle.train.x is None  # no feature matrix
    assert bundle.train.y is not None and bundle.train.index is not None


@pytest.mark.integration
def test_a_lightning_model_gets_sliding_windows(series_csv, tmp_path):
    """Same source, same config, different payload — decided by what the model
    declares it accepts, not by a config flag."""
    from ml_framework.data import build_bundle

    bundle = build_bundle(ts_config(series_csv, tmp_path, model="ts.lstm"))
    assert bundle.payload == "arrays"
    assert bundle.train.x.shape[1] == 14  # data.params.window
    assert bundle.input_dim == 14


@pytest.mark.unit
def test_windows_never_span_the_split_boundary(series_csv, tmp_path):
    """Windows are built before the split, so a training window cannot contain a
    test observation."""
    from ml_framework.data.sources.timeseries import make_windows

    values = np.arange(20, dtype="float64")
    x, y, origin = make_windows(values, window=3, horizon=1)
    assert x.shape == (17, 3)
    assert np.array_equal(x[0], [0.0, 1.0, 2.0])
    assert y[0] == 3.0
    assert origin[0] == 3  # the target's position in the original series


# ── End to end ────────────────────────────────────────────
@pytest.mark.integration
@pytest.mark.parametrize("model", ["ts.naive", "ts.arima", "ts.prophet", "ts.lstm"])
def test_each_forecaster_trains_and_produces_a_loadable_bundle(model, series_csv, tmp_path):
    """Three backends' worth of models, one orchestrator, one bundle shape."""
    if model == "ts.arima":
        pytest.importorskip("statsmodels", reason="the timeseries extra is not installed")
    if model == "ts.prophet":
        pytest.importorskip("prophet", reason="the timeseries extra is not installed")
    if model == "ts.lstm":
        pytest.importorskip("pytorch_lightning", reason="the lightning extra is not installed")

    from ml_framework.core.bundle import read_manifest
    from ml_framework.core.inference import Inferencer
    from ml_framework.pipeline import train

    cfg = ts_config(series_csv, tmp_path, model=model)
    metrics = train(cfg)
    assert {"test_mase", "test_smape", "test_mae", "test_rmse"} <= set(metrics)

    out = Path(cfg.runtime.output_dir)
    manifest = read_manifest(out)
    assert manifest.task == "forecasting"
    assert manifest.data_kind == "timeseries"
    assert manifest.signature.output.kind == "series"

    inf = Inferencer.from_artifacts(out)
    assert not inf.produces_proba  # values over a horizon, never probabilities


@pytest.mark.integration
def test_the_naive_baseline_needs_no_optional_dependency():
    """MASE is defined against it, so it has to run wherever forecasting does."""
    from ml_framework.core.registry import MODELS

    assert MODELS.get_spec("ts.naive").requires == ()
    assert MODELS.is_available("ts.naive")


@pytest.mark.integration
def test_predictions_csv_leads_with_the_timestamp(series_csv, tmp_path):
    """A forecast without the point in time it belongs to is not interpretable."""
    from ml_framework.pipeline import train

    cfg = ts_config(series_csv, tmp_path, model="ts.naive")
    train(cfg)
    frame = pd.read_csv(Path(cfg.runtime.output_dir) / "predictions.csv")
    assert list(frame.columns[:3]) == ["index", "label", "prediction"]
    # Rendered as dates, not raw nanosecond epochs.
    assert str(frame["index"].iloc[0]).startswith("20")


@pytest.mark.integration
def test_the_report_renders_no_verdict(series_csv, tmp_path):
    """MASE < 1 is widely quoted as "better than naive" and over a multi-step
    horizon that is wrong. Printing a judgement the number does not support would
    be worse than printing none."""
    from ml_framework.pipeline import train

    cfg = ts_config(series_csv, tmp_path, model="ts.naive")
    train(cfg)
    report = (Path(cfg.runtime.output_dir) / "report.txt").read_text(encoding="utf-8")
    assert "MASE:" in report and "sMAPE:" in report
    assert "NOT a pass mark" in report
    assert "beats the" not in report


# ── Cross-validation routes to rolling origin ─────────────
@pytest.mark.integration
def test_time_series_cv_uses_rolling_origin_not_kfold(series_csv, tmp_path):
    """Shuffled folds here would be the same leakage the config validator refuses,
    wearing a different hat."""
    from ml_framework.data.builders import build_cv_bundles

    cfg = ts_config(series_csv, tmp_path, **{"data.split.folds": 3, "data.split.horizon": 10})
    bundles = list(build_cv_bundles(cfg))
    assert len(bundles) == 3
    for bundle in bundles:
        # Every training timestamp precedes every test timestamp.
        assert bundle.train.index.max() < bundle.test.index.min()


@pytest.mark.integration
def test_cross_validated_forecasting_reports_mean_and_spread(series_csv, tmp_path):
    from ml_framework.pipeline import train

    cfg = ts_config(series_csv, tmp_path, **{"data.split.folds": 3, "data.split.horizon": 10})
    metrics = train(cfg)
    assert "cv_mase_mean" in metrics and "cv_mase_std" in metrics

    cv = json.loads((Path(cfg.runtime.output_dir) / "cv.json").read_text(encoding="utf-8"))
    assert len(cv["per_fold"]) == 3


# ── Serving ───────────────────────────────────────────────
@pytest.mark.serving
def test_a_forecast_is_served_by_horizon_not_by_rows(series_csv, tmp_path):
    """`predict(X)` is a lying signature for forecasting, which is why the request
    schema varies by data kind."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ml_framework.pipeline import train
    from ml_framework.serving.api import create_app

    cfg = ts_config(series_csv, tmp_path, model="ts.naive")
    train(cfg)

    with TestClient(create_app(cfg.runtime.output_dir)) as client:
        assert client.get("/health").json()["data_kind"] == "timeseries"

        res = client.post("/predict", json={"horizon": 5})
        assert res.status_code == 200, res.text
        assert len(res.json()["forecast"]) == 5

        # Probabilities are refused from the manifest, as for regression.
        assert client.post("/predict_proba", json={"horizon": 5}).status_code == 400
        # Drift over a series has no named numeric features to compare.
        assert client.get("/drift").status_code == 501


@pytest.mark.serving
def test_a_forecast_request_rejects_a_row_payload(series_csv, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ml_framework.pipeline import train
    from ml_framework.serving.api import create_app

    cfg = ts_config(series_csv, tmp_path, model="ts.naive")
    train(cfg)
    with TestClient(create_app(cfg.runtime.output_dir)) as client:
        assert client.post("/predict", json={"instances": [[1.0]]}).status_code == 422
