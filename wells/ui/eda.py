"""Exploratory statistics. Functions take data frames and return data frames."""

from itertools import product

import numpy as np
import pandas as pd

SERIES_KEYS = ("well_id", "parameter", "unit")
DESCRIBE_COLUMNS = ["count", "mean", "median", "std", "min", "q25", "q75", "max"]
PROFILE_COLUMNS = ["well_id", "parameter", "unit", "hour", "median", "q25", "q75", "n"]
CORRELATION_COLUMNS = ["parameter_x", "parameter_y", "rho", "n_pairs"]
COUNT_COLUMNS = ["n_expected", "n_observed", "n_usable", "n_flagged"]
TYPE_COVERAGE_COLUMNS = ["well_type", "period_start", "n_series", *COUNT_COLUMNS, "coverage"]
WELL_COVERAGE_COLUMNS = ["well_id", "well_type", "period_start", "applicable", "n_series",
                         *COUNT_COLUMNS, "coverage"]


def _quantile(share):
    return lambda values: values.quantile(share)


def describe_observations(history, by=SERIES_KEYS) -> pd.DataFrame:
    """Count, mean, median, sample standard deviation, extremes and quartiles of ``display_value``.

    Missing and masked hours are excluded. Groups without values keep count 0 and missing statistics.
    """
    by = list(by)
    grouped = history.groupby(by, sort=True, dropna=False)["display_value"]
    stats = grouped.agg(
        count="count", mean="mean", median="median", std=lambda values: values.std(ddof=1),
        min="min", q25=_quantile(0.25), q75=_quantile(0.75), max="max",
    )
    return stats.reset_index().loc[:, [*by, *DESCRIBE_COLUMNS]]


def hourly_profile(history) -> pd.DataFrame:
    """Median, quartiles and sample count of ``display_value`` by source-clock hour of day."""
    values = history.dropna(subset=["display_value"])
    if values.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    values = values.assign(hour=values["timestamp"].dt.hour)
    grouped = values.groupby([*SERIES_KEYS, "hour"], sort=True)["display_value"]
    profile = grouped.agg(median="median", q25=_quantile(0.25), q75=_quantile(0.75), n="count")
    return profile.reset_index().loc[:, PROFILE_COLUMNS]


def pairwise_spearman(matrix, min_pairs=30) -> pd.DataFrame:
    """Spearman correlation of every parameter pair on the hours where both are present.

    Ranks are recomputed inside each pair. The coefficient is missing when fewer than
    ``min_pairs`` shared hours exist or when either paired series is constant.
    """
    rows = []
    for left, right in product(matrix.columns, repeat=2):
        pair = matrix[[left]].join(matrix[[right]].rename(columns={right: "__other__"})).dropna()
        x, y = pair[left], pair["__other__"]
        rho = np.nan
        if len(pair) >= min_pairs and x.nunique() > 1 and y.nunique() > 1:
            rho = 1.0 if left == right else float(x.rank().corr(y.rank()))
        rows.append((left, right, rho, len(pair)))
    return pd.DataFrame.from_records(rows, columns=CORRELATION_COLUMNS)


def _with_coverage(frame):
    frame["coverage"] = (frame["n_observed"] / frame["n_expected"]).where(frame["n_expected"] > 0)
    return frame


def coverage_by_type(aggregates) -> pd.DataFrame:
    """Sum expected, observed, usable and flagged hours over applicable pairs, then divide."""
    if aggregates.empty:
        return pd.DataFrame(columns=TYPE_COVERAGE_COLUMNS)
    grouped = aggregates.groupby(["well_type", "period_start"], sort=True)
    totals = grouped[COUNT_COLUMNS].sum()
    totals["n_series"] = grouped.size()
    return _with_coverage(totals.reset_index()).loc[:, TYPE_COVERAGE_COLUMNS]


def coverage_by_well(aggregates, wells) -> pd.DataFrame:
    """Per-well period coverage summed over the pairs present in ``aggregates``.

    ``wells`` has ``well_id`` and ``well_type`` for every row to display. A well without any
    applicable pair in ``aggregates`` gets ``applicable=False`` and missing counts, which is a
    different state from an applicable sensor with zero observations.
    """
    wells = wells[["well_id", "well_type"]].drop_duplicates()
    periods = pd.Index(sorted(aggregates["period_start"].unique()), name="period_start")
    if len(periods) == 0 or wells.empty:
        return pd.DataFrame(columns=WELL_COVERAGE_COLUMNS)
    grouped = aggregates.groupby(["well_id", "period_start"], sort=True)
    totals = grouped[COUNT_COLUMNS].sum()
    totals["n_series"] = grouped.size()
    grid = wells.merge(periods.to_frame(index=False), how="cross")
    merged = grid.merge(totals.reset_index(), on=["well_id", "period_start"], how="left")
    merged["applicable"] = merged["well_id"].isin(aggregates["well_id"].unique())
    merged = _with_coverage(merged).sort_values(["well_type", "well_id", "period_start"])
    return merged.loc[:, WELL_COVERAGE_COLUMNS].reset_index(drop=True)


def period_summary(history, periods) -> pd.DataFrame:
    """Describe ``display_value`` inside explicit [start, end) periods given as (label, start, end)."""
    rows = []
    for label, start, end in periods:
        inside = history.loc[history["timestamp"].ge(start) & history["timestamp"].lt(end)]
        stats = describe_observations(inside.assign(period=label), by=("period",))
        if stats.empty:
            stats = pd.DataFrame([{"period": label, "count": 0}], columns=["period", *DESCRIBE_COLUMNS])
        stats.insert(1, "start", start)
        stats.insert(2, "end", end)
        stats.insert(3, "n_hours", len(inside))
        rows.append(stats)
    columns = ["period", "start", "end", "n_hours", *DESCRIBE_COLUMNS]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
