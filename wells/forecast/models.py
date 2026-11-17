"""Baselines and the direct lag-based Ridge model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CLIP_RANGE, LAG_COLUMNS, ORIGIN_SENSORS, RIDGE_FEATURES, complete_inputs

MODELS = ("naive", "seasonal_naive", "ridge")
BONUS_MODELS = ("gradient_boosting", "statsmodels_ar")
BOOSTING_FEATURES = [*RIDGE_FEATURES, *ORIGIN_SENSORS]
BOOSTING_PARAMS = {"max_iter": 80, "max_leaf_nodes": 15, "learning_rate": 0.08, "early_stopping": False}


def predict_naive(frame: pd.DataFrame) -> pd.Series:
    """Last value known at the origin; missing when that hour is missing."""
    return frame["lag_0"].rename("y_pred")


def predict_seasonal_naive(frame: pd.DataFrame) -> pd.Series:
    """Value 24 hours before the target hour; missing when that hour is missing."""
    return frame["seasonal_value"].rename("y_pred")


def fit_ridge(rows: pd.DataFrame, alpha: float) -> Pipeline:
    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    return pipeline.fit(rows[RIDGE_FEATURES], rows["y_true"])


def predict_ridge(pipeline: Pipeline, frame: pd.DataFrame, clip_range=CLIP_RANGE) -> pd.Series:
    """Clipped predictions aligned to `frame`; rows with incomplete inputs stay missing."""
    out = pd.Series(np.nan, index=frame.index, name="y_pred")
    mask = complete_inputs(frame)
    if mask.any():
        out[mask] = np.clip(pipeline.predict(frame.loc[mask, RIDGE_FEATURES]), *clip_range)
    return out


def _ar_design(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack((np.ones(len(rows)), rows[LAG_COLUMNS].to_numpy(dtype=float)))


def fit_bonus_models(rows: pd.DataFrame) -> dict:
    """Fit fixed-configuration boosting and direct autoregression on training rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from statsmodels.regression.linear_model import OLS

    return {
        "gradient_boosting": HistGradientBoostingRegressor(**BOOSTING_PARAMS).fit(
            rows[BOOSTING_FEATURES], rows["y_true"],
        ),
        "statsmodels_ar": OLS(rows["y_true"].to_numpy(), _ar_design(rows), missing="raise").fit(),
    }


def predict_all(pipeline: Pipeline, frame: pd.DataFrame, clip_range=CLIP_RANGE, *, bonus=None) -> dict[str, pd.Series]:
    predictions = {
        "naive": predict_naive(frame),
        "seasonal_naive": predict_seasonal_naive(frame),
        "ridge": predict_ridge(pipeline, frame, clip_range),
    }
    if bonus is not None:
        mask = complete_inputs(frame)
        rows = frame.loc[mask]
        for model, fitted in bonus.items():
            values = pd.Series(np.nan, index=frame.index, name="y_pred")
            if not rows.empty:
                design = rows[BOOSTING_FEATURES] if model == "gradient_boosting" else _ar_design(rows)
                values.loc[mask] = np.clip(fitted.predict(design), *clip_range)
            predictions[model] = values
    return predictions
