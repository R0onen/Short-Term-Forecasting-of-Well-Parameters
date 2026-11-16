"""Train-only alpha selection, final refit, official backtest and latest release."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    CLIP_RANGE,
    HORIZONS,
    LAGS,
    TARGET,
    ForecastError,
    build_features,
    complete_inputs,
    eligible_fit_rows,
)
from .models import MODELS, fit_ridge, predict_all, predict_ridge

ALPHAS = (0.1, 1.0, 10.0)
PREFERRED_ALPHA = 1.0
PREDICTION_COLUMNS = ["well_id", "parameter", "origin_ts", "horizon_h", "model", "target_ts", "y_true", "y_pred"]
METRIC_COLUMNS = ["well_id", "parameter", "horizon_h", "model", "mae", "n_common", "n_actual", "coverage"]
PATH_COLUMNS = ["well_id", "parameter", "origin_ts", "horizon_h", "model", "target_ts", "y_pred"]

_HOUR = pd.Timedelta(hours=1)


def _require(rows: pd.DataFrame, well_id: str, horizon_h: int, stage: str) -> pd.DataFrame:
    if rows.empty:
        raise ForecastError(f"no usable rows: well={well_id} horizon={horizon_h} stage={stage} count=0")
    return rows


def validation_rows(frame: pd.DataFrame, *, validation_start, test_start) -> pd.DataFrame:
    start, end = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    mask = (
        complete_inputs(frame)
        & frame["y_true"].notna()
        & (frame["target_ts"] >= start)
        & (frame["target_ts"] < end)
        & (frame["origin_ts"] >= start - _HOUR)
    )
    return frame[mask]


def choose_alpha(macro_mae: dict[float, float]) -> float:
    """Lowest macro MAE; on a tie prefer 1.0, otherwise the smallest tied alpha."""
    best = min(macro_mae.values())
    tied = [alpha for alpha, mae in macro_mae.items() if np.isclose(mae, best, rtol=0.0, atol=1e-12)]
    return PREFERRED_ALPHA if PREFERRED_ALPHA in tied else min(tied)


def select_alpha(frames, *, validation_start, test_start, nk7_regime_start, alphas=ALPHAS, clip_range=CLIP_RANGE):
    """Pick one global alpha from train-only validation.

    `frames` maps `(well_id, horizon_h)` to a feature frame. Returns the alpha
    and rows keyed by `(alpha, well_id, horizon_h)` with `mae`, `n_validation`.
    """
    records = []
    for (well_id, horizon_h), frame in frames.items():
        fit = _require(
            eligible_fit_rows(frame, well_id=well_id, cutoff=validation_start, regime_start=nk7_regime_start),
            well_id, horizon_h, "internal_fit",
        )
        val = _require(
            validation_rows(frame, validation_start=validation_start, test_start=test_start),
            well_id, horizon_h, "validation",
        )
        for alpha in alphas:
            pred = predict_ridge(fit_ridge(fit, alpha), val, clip_range)
            mae = float((pred - val["y_true"]).abs().mean())
            if not np.isfinite(mae):
                raise ForecastError(
                    f"non-finite validation MAE: well={well_id} horizon={horizon_h} "
                    f"stage=validation alpha={alpha} count={len(val)}"
                )
            records.append(
                {"alpha": alpha, "well_id": well_id, "horizon_h": horizon_h, "mae": mae, "n_validation": len(val)}
            )
    table = pd.DataFrame(records, columns=["alpha", "well_id", "horizon_h", "mae", "n_validation"])
    if table.empty:
        raise ForecastError("no forecast groups to validate")
    macro = table.groupby("alpha")["mae"].mean().to_dict()
    return choose_alpha(macro), table


def fit_final(frames, alpha: float, *, test_start, nk7_regime_start):
    """Refit one pipeline per group on every eligible row with `target_ts < test_start`."""
    pipelines, counts = {}, []
    for (well_id, horizon_h), frame in frames.items():
        rows = _require(
            eligible_fit_rows(frame, well_id=well_id, cutoff=test_start, regime_start=nk7_regime_start),
            well_id, horizon_h, "final_fit",
        )
        pipelines[(well_id, horizon_h)] = fit_ridge(rows, alpha)
        counts.append({"well_id": well_id, "horizon_h": horizon_h, "n_fit": len(rows)})
    return pipelines, pd.DataFrame(counts, columns=["well_id", "horizon_h", "n_fit"])


def evaluate_backtest(frames, pipelines, *, test_start, test_end, parameter=TARGET, clip_range=CLIP_RANGE):
    """Score the three methods on their common keys in the official test period.

    `test_end` is inclusive. Every `(well, horizon, model)` gets a metric row,
    including groups without any scored key.
    """
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    predictions, metrics = [], []
    for (well_id, horizon_h), frame in frames.items():
        in_test = (frame["origin_ts"] >= start) & (frame["origin_ts"] <= end)
        n_actual = int(frame.loc[in_test, "lag_0"].notna().sum())

        rows = frame[
            (frame["origin_ts"] >= start - _HOUR) & (frame["target_ts"] >= start) & (frame["target_ts"] <= end)
        ]
        preds = predict_all(pipelines[(well_id, horizon_h)], rows, clip_range)
        common = rows["y_true"].notna()
        for pred in preds.values():
            common &= pred.notna()
        scored = rows[common]
        n_common = len(scored)
        for model in MODELS:
            y_pred = preds[model][common]
            if n_common:
                predictions.append(
                    pd.DataFrame(
                        {
                            "well_id": well_id,
                            "parameter": parameter,
                            "origin_ts": scored["origin_ts"],
                            "horizon_h": horizon_h,
                            "model": model,
                            "target_ts": scored["target_ts"],
                            "y_true": scored["y_true"],
                            "y_pred": y_pred,
                        }
                    )
                )
            metrics.append(
                {
                    "well_id": well_id,
                    "parameter": parameter,
                    "horizon_h": horizon_h,
                    "model": model,
                    "mae": float((y_pred - scored["y_true"]).abs().mean()) if n_common else None,
                    "n_common": n_common,
                    "n_actual": n_actual,
                    "coverage": n_common / n_actual if n_actual else None,
                }
            )
    prediction_table = (
        pd.concat(predictions, ignore_index=True)[PREDICTION_COLUMNS]
        if predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    return prediction_table, pd.DataFrame(metrics, columns=METRIC_COLUMNS).astype({"mae": object, "coverage": object})


def latest_origin(series: pd.Series, *, test_start):
    """Latest hour `t >= test_start - 1h` with every hour in `[t-24h, t]` usable, or None."""
    if series.empty:
        return None
    s = series.sort_index().astype(float)
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h"))
    window = max(LAGS) + 1
    full = s.notna().rolling(window).sum() == window
    full &= s.index >= pd.Timestamp(test_start) - _HOUR
    return full[full].index[-1] if full.any() else None


def build_forecast_path(series_by_well, pipelines, *, test_start, parameter=TARGET, clip_range=CLIP_RANGE) -> pd.DataFrame:
    """One 24-horizon trajectory per model from each well's latest eligible origin."""
    parts = []
    for well_id, series in series_by_well.items():
        origin = latest_origin(series, test_start=test_start)
        if origin is None:
            continue
        window = series.sort_index().loc[origin - pd.Timedelta(hours=max(LAGS)) : origin]
        for horizon_h in HORIZONS:
            row = build_features(window, horizon_h, clip_range).iloc[[-1]]
            for model, pred in predict_all(pipelines[(well_id, horizon_h)], row, clip_range).items():
                parts.append(
                    {
                        "well_id": well_id,
                        "parameter": parameter,
                        "origin_ts": origin,
                        "horizon_h": horizon_h,
                        "model": model,
                        "target_ts": row["target_ts"].iloc[0],
                        "y_pred": float(pred.iloc[0]),
                    }
                )
    return pd.DataFrame(parts, columns=PATH_COLUMNS)
