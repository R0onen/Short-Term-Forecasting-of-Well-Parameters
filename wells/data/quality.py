"""Validate raw records, retain measurements for audit, and define the usable mask."""

import math
import re
from datetime import timedelta
from functools import cache

import pandas as pd

from wells.data.ingest import RAW_COLUMNS, parse_hour

KEY = ["well_id", "timestamp", "parameter"]
LONG_COLUMNS = [*KEY, "value", "unit", "n_obs", "flag", "split", "is_usable"]
ISSUE_COLUMNS = ["well_id", "parameter", "timestamp", "end_ts", "source_file", "source_row", "issue_code", "action"]
SOURCE_KEY = ["source_file", "source_row"]
MAY_START = "2026-05-01 00:00:00"
JUNE_START = "2026-06-01 00:00:00"
NK82_SHIFT = "2026-05-08 00:00:00"
NK7_PARAMETERS = ("SPM", "dynamometer_area", "pump_fillage")


def calendar_bounds(meta):
    bounds = tuple(parse_hour(meta[key]) for key in ("train_start", "train_end", "test_start", "test_end"))
    if any(bound is None for bound in bounds):
        raise ValueError("Calendar bounds must use hourly labels YYYY-MM-DD HH:00:00")
    train_start, train_end, test_start, test_end = bounds
    if not (train_start <= train_end < test_start <= test_end) or test_start - train_end != timedelta(hours=1):
        raise ValueError("Train and test bounds must be ordered and contiguous")
    return bounds


def _finite_value(text):
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _positive_count(text):
    if not isinstance(text, str) or not re.fullmatch(r"[+]?[0-9]+", text):
        return None
    digits = text.lstrip("+").lstrip("0")
    if not digits or len(digits) > 19:
        return None
    count = int(digits)
    return count if count <= 2**63 - 1 else None


def _annotation_rows(catalog, archive_start, archive_end):
    for row in catalog.itertuples(index=False):
        periods = []
        if row.well_type == "sucker_rod_pump" and row.parameter == "dynamometer_area":
            periods.append((MAY_START, JUNE_START, "area_scale_shift"))
            if row.well_id == "NK-82":
                periods.append((NK82_SHIFT, "2026-05-09 00:00:00", "area_scale_shift"))
        if row.well_id == "NK-7" and row.parameter in NK7_PARAMETERS:
            periods.append((JUNE_START, None, "nk7_regime_change"))
        for start, end, code in periods:
            if start <= archive_end and (end is None or end > archive_start):
                yield (row.well_id, row.parameter, start, end, None, None, code, "annotate")


def validate_measurements(raw, catalog, meta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unique typed long rows and source-linked reasons for each audit action."""
    train_start, train_end, test_start, test_end = calendar_bounds(meta)
    bounds = {"train": (train_start, train_end), "test": (test_start, test_end)}
    applicable = {
        (row.well_id, row.parameter): (row.well_type, row.unit)
        for row in catalog.itertuples(index=False)
    }
    wells = set(catalog["well_id"])
    parameters = set(catalog["parameter"])
    parse_cached_hour = cache(parse_hour)
    records = []
    issues = []
    raw_records = raw.loc[:, RAW_COLUMNS].itertuples(index=False, name=None)
    for file, number, well, well_type, label, parameter, value_text, unit, count_text, flag, split in raw_records:
        timestamp = parse_cached_hour(label)
        value = _finite_value(value_text)
        count = _positive_count(count_text)
        expected = applicable.get((well, parameter))
        reasons = []
        if timestamp is None:
            reasons.append("invalid_timestamp")
        elif split not in bounds or not bounds[split][0] <= timestamp <= bounds[split][1]:
            reasons.append("outside_split")
        if well not in wells:
            reasons.append("unknown_well")
        if expected is None:
            reasons.append("unknown_parameter" if parameter not in parameters else "inapplicable_parameter")
        else:
            if well_type != expected[0]:
                reasons.append("wrong_well_type")
            if unit != expected[1]:
                reasons.append("wrong_unit")
        if value is None:
            reasons.append("invalid_value")
        if count is None:
            reasons.append("invalid_n_obs")
        if reasons:
            context = (well if well in wells else None, parameter if parameter in parameters else None,
                       label if timestamp is not None else None, None, file, number)
            issues.extend((*context, code, "reject") for code in reasons)
            continue
        records.append((well, timestamp, parameter, value, unit, count, flag, split, 1, file, number))

    long = pd.DataFrame.from_records(records, columns=[*LONG_COLUMNS, *SOURCE_KEY])
    duplicates = long.loc[long.duplicated(KEY, keep=False)].sort_values(SOURCE_KEY)
    discard = []
    for _, group in duplicates.groupby(KEY, sort=False):
        if len(group[["value", "unit", "n_obs", "flag", "split"]].drop_duplicates()) == 1:
            removed = group.iloc[1:]
            code, action = "duplicate_exact", "deduplicate"
        else:
            removed = group
            code, action = "duplicate_conflict", "reject"
        discard.extend(removed.index)
        issues.extend(_issue(row, code, action) for row in removed.itertuples())
    long = long.drop(index=discard)

    flagged = long.loc[long["flag"].str.strip(" \t\n\r\v\f").ne("")]
    long.loc[flagged.index, "is_usable"] = 0
    for row in flagged.itertuples():
        codes = dict.fromkeys(code.strip() for code in row.flag.split(";") if code.strip())
        issues.extend(_issue(row, code, "mask") for code in codes or ["published_flag"])

    outside = long.loc[long["parameter"].eq("pump_fillage") & ~long["value"].between(0, 100)]
    long.loc[outside.index, "is_usable"] = 0
    issues.extend(_issue(row, "pump_fillage_out_of_range", "mask") for row in outside.itertuples())

    keys = ["well_id", "timestamp"]
    lower = long.loc[long["parameter"].eq("min_rod_weight"), [*keys, "value"]].reset_index(names="min_index")
    upper = long.loc[long["parameter"].eq("max_rod_weight"), [*keys, "value"]].reset_index(names="max_index")
    pairs = lower.merge(upper, on=keys, suffixes=("_min", "_max"), validate="one_to_one")
    reversed_pairs = pairs.loc[pairs["value_min"] > pairs["value_max"]]
    reversed_rows = reversed_pairs[["min_index", "max_index"]].to_numpy().ravel().tolist()
    long.loc[reversed_rows, "is_usable"] = 0
    issues.extend(_issue(row, "reversed_rod_load", "mask") for row in long.loc[reversed_rows].itertuples())

    issues.extend(_annotation_rows(catalog, meta["train_start"], meta["test_end"]))
    audit = pd.DataFrame.from_records(issues, columns=ISSUE_COLUMNS)
    audit["source_row"] = pd.array([issue[5] for issue in issues], dtype="Int64")
    return long.loc[:, LONG_COLUMNS].reset_index(drop=True), audit


def _issue(row, code, action):
    return (row.well_id, row.parameter, row.timestamp.isoformat(sep=" "), None,
            row.source_file, row.source_row, code, action)


def summarize_anomalies(long) -> pd.DataFrame:
    """Compare published values, including masked rows, in explicit half-open periods."""
    rows = []
    for (well, parameter), series in long.groupby(["well_id", "parameter"], sort=True):
        comparisons = []
        if parameter == "dynamometer_area":
            comparisons.append(("area_scale_shift", ("2026-04-01 00:00:00", MAY_START, JUNE_START)))
            if well == "NK-82":
                comparisons.append(("nk82_may8_detail", ("2026-05-07 00:00:00", NK82_SHIFT, "2026-05-09 00:00:00")))
        if well == "NK-7" and parameter in NK7_PARAMETERS:
            comparisons.append(("nk7_regime_change", ("2026-01-01 00:00:00", JUNE_START, "2026-10-01 00:00:00")))
        if not comparisons:
            continue
        timestamps = pd.to_datetime(series["timestamp"])
        for comparison, (start, boundary, end) in comparisons:
            for period, left, right in (("before", start, boundary), ("after", boundary, end)):
                values = series.loc[timestamps.ge(left) & timestamps.lt(right), "value"]
                rows.append((well, parameter, comparison, period, left, right, len(values),
                             values.median() if len(values) else None, int(values.lt(0).sum())))
    return pd.DataFrame.from_records(rows, columns=[
        "well_id", "parameter", "comparison", "period", "start_ts", "end_ts",
        "n_observed", "median_value", "n_negative",
    ])
