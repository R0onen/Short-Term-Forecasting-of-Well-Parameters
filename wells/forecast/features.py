"""Origin-indexed hourly features for one well's target series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "pump_fillage"
HORIZONS = tuple(range(1, 25))
LAGS = (0, 1, 2, 3, 6, 12, 24)
LAG_COLUMNS = [f"lag_{k}" for k in LAGS]
CALENDAR_COLUMNS = ["target_hour_sin", "target_hour_cos"]
RIDGE_FEATURES = LAG_COLUMNS + CALENDAR_COLUMNS
CLIP_RANGE = (0.0, 100.0)
REGIME_WELL = "NK-7"
FRAME_COLUMNS = ["origin_ts", "target_ts", "horizon_h", "y_true", "seasonal_value"] + RIDGE_FEATURES


class ForecastError(RuntimeError):
    """A forecast stage failed; nothing may be published."""


def _hours(n: int) -> pd.Timedelta:
    return pd.Timedelta(hours=n)


def build_features(series: pd.Series, horizon_h: int) -> pd.DataFrame:
    """Return one row per origin hour `t` with the target at `t + horizon_h`.

    Predictors only read hours at or before the origin. Missing hours stay
    missing: there is no interpolation and no positional shift across holes.
    """
    if horizon_h not in HORIZONS:
        raise ValueError(f"horizon_h must be in 1..24, got {horizon_h}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    if series.index.has_duplicates:
        raise ValueError("series has duplicate timestamps")
    if series.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS)

    s = series.sort_index().astype(float)
    if (s.index != s.index.floor("h")).any():
        raise ValueError("series timestamps must be exact hours")
    known = s.dropna()
    if ((known < CLIP_RANGE[0]) | (known > CLIP_RANGE[1])).any():
        raise ValueError("usable target values must lie in [0, 100]")

    origin = pd.date_range(s.index[0], s.index[-1], freq="h")
    target = origin + _hours(horizon_h)

    def at(index: pd.DatetimeIndex) -> np.ndarray:
        return s.reindex(index).to_numpy()

    data = {
        "origin_ts": origin,
        "target_ts": target,
        "horizon_h": horizon_h,
        "y_true": at(target),
        "seasonal_value": at(target - _hours(24)),
    }
    for lag, column in zip(LAGS, LAG_COLUMNS):
        data[column] = at(origin - _hours(lag))
    angle = 2.0 * np.pi * target.hour.to_numpy() / 24.0
    data["target_hour_sin"] = np.sin(angle)
    data["target_hour_cos"] = np.cos(angle)
    return pd.DataFrame(data, columns=FRAME_COLUMNS)


def complete_inputs(frame: pd.DataFrame) -> pd.Series:
    """Mask of rows where every Ridge input is present."""
    return frame[RIDGE_FEATURES].notna().all(axis=1)


def eligible_fit_rows(frame: pd.DataFrame, *, well_id: str, cutoff, regime_start=None) -> pd.DataFrame:
    """Rows usable for fitting: complete inputs, known label, `target_ts < cutoff`.

    For NK-7 every lag and the target must lie on or after `regime_start`.
    """
    mask = complete_inputs(frame) & frame["y_true"].notna() & (frame["target_ts"] < pd.Timestamp(cutoff))
    if well_id == REGIME_WELL and regime_start is not None:
        mask &= frame["origin_ts"] - _hours(max(LAGS)) >= pd.Timestamp(regime_start)
    return frame[mask]
