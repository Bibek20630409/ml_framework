"""
data/sources/timeseries.py
──────────────────────────
A time-ordered table → :class:`~ml_framework.data.types.DataBundle`.

**One source, two payloads**, chosen by what the selected model can consume:

* ``series`` — the raw ordered values plus their timestamps, for the ``forecast``
  backend. Prophet and ARIMA fit the series itself; handing them a feature matrix
  would be a lie about what they do.
* ``arrays`` — sliding windows turned into supervised rows (``x`` = the previous
  ``window`` values, ``y`` = the next one), for anything on the ``lightning``
  backend. An LSTM needs (batch, time, features), which is a different object.

The choice is not a config flag: ``Capabilities.accepts`` already declares which
payloads a model can take, and ``registry.validate_combination`` already refuses
an impossible pairing at config-load time. This source simply reads the same
declaration. That is the machinery working as designed rather than a special case.

**Ordering is enforced, not assumed.** Rows are sorted by ``time_col`` before
anything else happens, and the splitters that reach this data are the temporal
ones. A shuffled split here is refused by the config validator — see
``SplitConfig.allow_temporal_leakage``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.types import Capabilities, FrameworkError
from ..preprocess.base import IdentityPreprocessor
from ..preprocess.tabular import TabularPreprocessor
from ..splitters import SplitError, TemporalSplitter
from ..types import DataBundle, FeatureSchema, Split
from .tabular import read_table

log = logging.getLogger(__name__)


class TimeSeriesSourceParams(PydanticModel):
    """``data.params`` for the time-series source."""

    model_config = {"frozen": True, "extra": "forbid"}

    # How many past observations a windowed model sees. Ignored by the forecast
    # backend, which consumes the series directly.
    window: int = Field(default=24, ge=1)
    # Steps ahead to predict. Also the width of each rolling-origin fold.
    horizon: int = Field(default=1, ge=1)
    # Period of the dominant seasonality: 7 for daily data with a weekly cycle, 12
    # for monthly with an annual one. Feeds the seasonal-naive baseline and MASE.
    seasonality: int = Field(default=1, ge=1)
    # Columns known for future periods (holidays, promotions). Carried through to
    # the models that accept them; ignored by those that do not.
    exog: list[str] = Field(default_factory=list)
    # Report ADF/KPSS at build time. Off by default: it costs a statsmodels import
    # and says nothing a model consumes.
    stationarity_test: bool = False
    freq: str | None = None


def check_stationarity(values: np.ndarray) -> dict[str, Any]:
    """ADF and KPSS, reported rather than acted on.

    The two tests answer *opposite* null hypotheses, which is why running both is
    worth the trouble: ADF's null is "has a unit root" (non-stationary), KPSS's is
    "is stationary". Agreement is informative; disagreement means the series is
    probably difference-stationary or trend-stationary and deserves a human.

    Nothing downstream branches on this. Differencing a series silently because a
    test said so would change what the model was asked to predict.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        log.info("statsmodels not installed — skipping the stationarity report")
        return {}

    out: dict[str, Any] = {}
    series = np.asarray(values, dtype="float64")
    try:
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        out["adf_statistic"] = float(adf_stat)
        out["adf_pvalue"] = float(adf_p)
        out["adf_says_stationary"] = bool(adf_p < 0.05)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
        log.debug("ADF failed: %s", exc)
    try:
        import warnings

        with warnings.catch_warnings():
            # KPSS warns loudly when the statistic falls outside its lookup table;
            # the p-value is then clipped, which is fine for a report.
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        out["kpss_statistic"] = float(kpss_stat)
        out["kpss_pvalue"] = float(kpss_p)
        out["kpss_says_stationary"] = bool(kpss_p >= 0.05)
    except Exception as exc:  # noqa: BLE001
        log.debug("KPSS failed: %s", exc)

    if out:
        log.info("stationarity: %s", out)
    return out


def make_windows(
    values: np.ndarray, window: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows as supervised rows: ``(x, y, origin_index)``.

    ``x[i]`` is ``values[i : i+window]`` and ``y[i]`` is ``values[i+window+horizon-1]``.
    ``origin_index[i]`` is the position of that target in the original series, which
    is what keeps predictions traceable to timestamps after the split.
    """
    n = len(values)
    last = n - window - horizon + 1
    if last <= 0:
        raise SplitError(
            f"a series of {n} points cannot form a window of {window} at horizon {horizon}"
        )
    x = np.stack([values[i : i + window] for i in range(last)]).astype("float32")
    y = np.asarray([values[i + window + horizon - 1] for i in range(last)], dtype="float32")
    origin = np.arange(window + horizon - 1, n, dtype="int64")[:last]
    return x, y, origin


def _payload_for(config) -> Literal["series", "arrays"]:
    """Which payload the selected model consumes.

    Reads ``Capabilities.accepts`` rather than adding a config flag: the plugin
    already declares this, and `validate_combination` already refuses a model that
    cannot take what this source produces.
    """
    from ...core.plugins import UnknownPluginError
    from ...core.registry import MODELS

    try:
        caps: Capabilities = MODELS.get_spec(config.model.name).capabilities
    except UnknownPluginError:
        return "series"
    return "series" if caps.can_accept("series") else "arrays"


def build_timeseries_bundle(config, *, indices: Any = None) -> DataBundle:
    """Materialize a time-series :class:`DataBundle` from a validated config."""
    params = TimeSeriesSourceParams.model_validate(dict(config.data.params))
    split_cfg = config.data.split

    frame = read_table(str(config.data.path))
    target = config.data.target
    if target not in frame.columns:
        raise KeyError(f"data.target '{target}' not in the table's columns")

    time_col = split_cfg.time_col
    if time_col is not None:
        if time_col not in frame.columns:
            raise KeyError(f"split.time_col '{time_col}' not in the table's columns")
        # Sorting here rather than trusting the file is the difference between "the
        # rows happened to be ordered" and "the rows are ordered".
        frame = frame.sort_values(time_col, kind="stable").reset_index(drop=True)
        stamps = pd.to_datetime(frame[time_col], errors="coerce")
        # Pinned to ns: every reader decodes this index with `pd.to_datetime(int)`,
        # which assumes ns, and pandas 3 parses to `datetime64[us]` by default.
        stamps = stamps.astype("datetime64[ns]")
        index = np.asarray(stamps.astype("int64") if stamps.notna().all() else frame[time_col])
    else:
        log.info("no split.time_col — treating row order as the chronology")
        index = np.arange(len(frame), dtype="int64")

    values = frame[target].to_numpy(dtype="float64")
    if len(values) < 3:
        raise FrameworkError(f"'{config.data.path}' has {len(values)} rows; too few to forecast")

    missing = [c for c in params.exog if c not in frame.columns]
    if missing:
        raise KeyError(f"data.params.exog names columns not in the table: {missing}")

    reference_stats = check_stationarity(values) if params.stationarity_test else {}
    payload = _payload_for(config)
    log.info(
        "timeseries: %d points, payload=%s, horizon=%d, seasonality=%d",
        len(values),
        payload,
        params.horizon,
        params.seasonality,
    )

    if payload == "series":
        return _series_bundle(config, params, values, index, frame, indices)
    return _windowed_bundle(config, params, values, index, indices, reference_stats)


def _split_indices(config, params: TimeSeriesSourceParams, n: int, indices: Any):
    """The temporal partition, unless one was injected (cross-validation)."""
    if indices is not None:
        return indices
    split_cfg = config.data.split
    return TemporalSplitter(
        val_size=split_cfg.val_size, test_size=split_cfg.test_size, gap=split_cfg.gap
    ).split(n)


def _series_bundle(
    config,
    params: TimeSeriesSourceParams,
    values: np.ndarray,
    index: np.ndarray,
    frame: pd.DataFrame,
    indices: Any,
) -> DataBundle:
    """Contiguous slices of the series itself, for the ``forecast`` backend.

    ``Split.y`` holds the values and ``Split.index`` the timestamps; ``Split.x``
    carries exogenous columns when there are any. There is no feature matrix
    because these models do not consume one — that is the honesty the ``series``
    payload buys.
    """
    parts = _split_indices(config, params, len(values), indices)
    exog = frame[params.exog] if params.exog else None

    def slice_of(rows: np.ndarray) -> Split:
        return Split(
            payload="series",
            x=None if exog is None else exog.iloc[rows].reset_index(drop=True),
            y=values[rows],
            index=index[rows],
        )

    schema = FeatureSchema(
        feature_names=tuple(params.exog),
        target_name=config.data.target,
        time_col=config.data.split.time_col,
        freq=params.freq,
    )
    return DataBundle(
        train=slice_of(parts.train),
        val=slice_of(parts.val),
        test=slice_of(parts.test),
        schema=schema,
        task=config.task,
        data_kind="timeseries",
        # A forecast model has no input width; the horizon is what it is asked for.
        input_dim=len(params.exog),
        output_dim=1,
        class_weights=None,
        preprocessor=IdentityPreprocessor(),
        reference_stats=None,
        meta={
            "horizon": params.horizon,
            "seasonality": params.seasonality,
            "freq": params.freq,
            # The history a forecaster continues from, so `predict_split` can align
            # its output with the split it was asked about.
            "series": values,
            "index": index,
        },
    )


def _windowed_bundle(
    config,
    params: TimeSeriesSourceParams,
    values: np.ndarray,
    index: np.ndarray,
    indices: Any,
    reference_stats: dict[str, Any],
) -> DataBundle:
    """Supervised rows from sliding windows, for the ``lightning`` backend.

    The split is applied to the *windows*, not the raw points, and the windows are
    built before the split — so a window never spans the boundary. Scaling is
    fitted on the training windows only, like everywhere else.
    """
    x, y, origin = make_windows(values, params.window, params.horizon)
    parts = _split_indices(config, params, len(x), indices)

    preprocessor = TabularPreprocessor(needs_scaling=True)
    x_train = preprocessor.fit_transform(Split(x=x[parts.train], y=y[parts.train]))
    schema = FeatureSchema(
        feature_names=tuple(f"lag_{params.window - i}" for i in range(params.window)),
        target_name=config.data.target,
        time_col=config.data.split.time_col,
        freq=params.freq,
    )
    return DataBundle(
        train=Split(
            payload="arrays", x=x_train, y=y[parts.train], index=index[origin[parts.train]]
        ),
        val=Split(
            payload="arrays",
            x=preprocessor.transform(x[parts.val]),
            y=y[parts.val],
            index=index[origin[parts.val]],
        ),
        test=Split(
            payload="arrays",
            x=preprocessor.transform(x[parts.test]),
            y=y[parts.test],
            index=index[origin[parts.test]],
        ),
        schema=schema,
        task=config.task,
        data_kind="timeseries",
        input_dim=params.window,
        output_dim=1,
        class_weights=None,
        preprocessor=preprocessor,
        reference_stats=reference_stats or None,
        meta={
            "horizon": params.horizon,
            "seasonality": params.seasonality,
            "window": params.window,
            "freq": params.freq,
        },
    )


__all__ = [
    "TimeSeriesSourceParams",
    "build_timeseries_bundle",
    "check_stationarity",
    "make_windows",
]
