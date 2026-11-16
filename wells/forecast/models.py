"""Baselines and the direct lag-based Ridge model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CLIP_RANGE, RIDGE_FEATURES, complete_inputs

MODELS = ("naive", "seasonal_naive", "ridge")


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


def predict_all(pipeline: Pipeline, frame: pd.DataFrame, clip_range=CLIP_RANGE) -> dict[str, pd.Series]:
    return {
        "naive": predict_naive(frame),
        "seasonal_naive": predict_seasonal_naive(frame),
        "ridge": predict_ridge(pipeline, frame, clip_range),
    }
