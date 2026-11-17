"""Forecast runner: read the built SQLite DB, evaluate, publish three result tables.

    python -m wells.forecast.run --db PATH [--diagnostic]
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import subprocess
import sys
from contextlib import closing, nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .evaluation import (
    ALPHAS,
    METRIC_COLUMNS,
    PATH_COLUMNS,
    PREDICTION_COLUMNS,
    PREFERRED_ALPHA,
    build_forecast_path,
    evaluate_backtest,
    fit_bonus,
    fit_final,
    select_alpha,
    validation_rows,
)
from .features import (
    CLIP_RANGE,
    HORIZONS,
    LAG_COLUMNS,
    ORIGIN_SENSORS,
    REGIME_WELL,
    RIDGE_FEATURES,
    TARGET,
    TARGET_RANGES,
    ForecastError,
    build_features,
    eligible_fit_rows,
)
from .models import BONUS_MODELS, BOOSTING_FEATURES, BOOSTING_PARAMS, MODELS

DEFAULT_DB = Path("data/processed/wells.sqlite")
DEFAULT_VALIDATION_START = "2026-07-15 00:00:00"
DEFAULT_NK7_REGIME_START = "2026-06-01 00:00:00"
PINNED_REVISION = "4004bb78ca8811a87c2892a89e457c59d22ace2b"
PINNED_WELLS = ["NK-68", "NK-7", "NK-77", "NK-82", "NK-86", "NK-91", "NK-93"]
DIAGNOSTIC_HORIZONS = (1, 24)
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
RESULT_TABLES = {
    "backtest_predictions": PREDICTION_COLUMNS,
    "metrics": METRIC_COLUMNS,
    "forecast_path": PATH_COLUMNS,
}
REQUIRED_TABLES = {
    "dataset_meta": ["key", "value"],
    "series_catalog": ["well_id", "parameter"],
    "measurements_wide": ["well_id", "timestamp", "split", TARGET],
    **RESULT_TABLES,
}
META_KEYS = ["revision", "train_end", "test_start", "test_end", "train_sha256", "test_sha256"]
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "data/sql/schema.sql"


def _connect(db_path: Path, mode: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise ForecastError(f"database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode={mode}", uri=True)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _load(conn: sqlite3.Connection, *, train_only: bool = False, target: str = TARGET):
    """Return metadata and one hourly target series per well that has the target sensor."""
    if target not in TARGET_RANGES:  # allowlist: the name is used as a column identifier
        raise ForecastError(f"unsupported target {target!r}; choose from {sorted(TARGET_RANGES)}")
    for table, columns in REQUIRED_TABLES.items():
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = [c for c in columns if c not in present]
        if missing:
            raise ForecastError(f"incompatible schema: {table} lacks {missing}")
    meta = dict(conn.execute("SELECT key, value FROM dataset_meta"))
    absent = [k for k in META_KEYS if k not in meta]
    if absent:
        raise ForecastError(f"dataset_meta lacks {absent}")
    wells = [r[0] for r in conn.execute(
        "SELECT well_id FROM series_catalog WHERE parameter = ? ORDER BY well_id", (target,)
    )]
    if not wells:
        raise ForecastError(f"catalog has no {target} series")
    if target == TARGET and meta["revision"] == PINNED_REVISION and wells != sorted(PINNED_WELLS):
        raise ForecastError(f"unexpected {TARGET} wells for the pinned revision: {wells}")
    where = "WHERE split = 'train'" if train_only else ""
    wide = pd.read_sql_query(
        f"SELECT well_id, timestamp, {target} AS y FROM measurements_wide {where} ORDER BY well_id, timestamp",
        conn,
        parse_dates=["timestamp"],
    )
    if wide.duplicated(["well_id", "timestamp"]).any():
        raise ForecastError("measurements_wide has duplicate well-hours")
    series = {
        well: wide.loc[wide["well_id"] == well].set_index("timestamp")["y"].astype(float) for well in wells
    }
    return meta, series


def _sensors_by_well(conn: sqlite3.Connection, wells) -> dict[str, pd.DataFrame]:
    columns = ", ".join(ORIGIN_SENSORS)
    wide = pd.read_sql_query(f"SELECT well_id, timestamp, {columns} FROM measurements_wide",
                             conn, parse_dates=["timestamp"]).set_index(["well_id", "timestamp"])
    return {well: wide.loc[well] for well in wells}


def _frames(series_by_well, horizons=HORIZONS, value_range=CLIP_RANGE, sensors_by_well=None):
    return {
        (well, h): build_features(s, h, value_range, origin_sensors=(
            None if sensors_by_well is None else sensors_by_well[well]
        ))
        for well, s in series_by_well.items()
        for h in horizons
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON- and SQLite-safe rows: timestamps as text, missing as None."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime(TS_FORMAT)
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


def _check_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame, path: pd.DataFrame, n_groups: int,
                   clip_range=CLIP_RANGE, models=MODELS) -> None:
    """Writer-side contract checks before anything is committed."""
    for name, table in (("backtest_predictions", predictions), ("forecast_path", path)):
        if table.empty:
            continue
        step = pd.to_timedelta(table["horizon_h"].astype(int), unit="h")
        if not (table["target_ts"] == table["origin_ts"] + step).all():
            raise ForecastError(f"{name}: target_ts != origin_ts + horizon_h")
        y = table["y_pred"].astype(float)
        low, high = clip_range
        in_range = np.isfinite(y) & (y >= low) & (True if high is None else y <= high)
        if not in_range.all():
            raise ForecastError(f"{name}: prediction outside the valid range or non-finite")
    if not predictions.empty:
        per_key = predictions.groupby(["well_id", "origin_ts", "horizon_h"])["model"].nunique()
        if not (per_key == len(models)).all():
            raise ForecastError("backtest_predictions: a scored key lacks a model")
    if len(metrics) != n_groups * len(models):
        raise ForecastError(f"metrics: expected {n_groups * len(models)} rows, got {len(metrics)}")
    if not path.empty:
        release = path.groupby("well_id").agg(n=("model", "size"), origins=("origin_ts", "nunique"))
        if not ((release["n"] == len(HORIZONS) * len(models)) & (release["origins"] == 1)).all():
            raise ForecastError("forecast_path: incomplete release group")


def _upgrade_result_tables(conn: sqlite3.Connection) -> None:
    """Replace legacy output table definitions using the current source schema."""
    definitions = dict(conn.execute("SELECT name, sql FROM sqlite_schema WHERE type = 'table'"))
    legacy = [name for name in RESULT_TABLES if not all(model in definitions[name] for model in BONUS_MODELS)]
    if legacy:
        with closing(sqlite3.connect(":memory:")) as reference:
            reference.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
            current = dict(reference.execute("SELECT name, sql FROM sqlite_schema WHERE type = 'table'"))
        for name in legacy:
            conn.execute(f"DROP TABLE {name}")
            conn.execute(current[name])


def _publish(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame], *, upgrade_schema=False) -> None:
    """Replace all three result sets in one transaction; roll back on any failure."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        if upgrade_schema:
            _upgrade_result_tables(conn)
        for name in RESULT_TABLES:
            conn.execute(f"DELETE FROM {name}")
        for name, columns in RESULT_TABLES.items():
            rows = [tuple(r[c] for c in columns) for r in _records(tables[name])]
            marks = ", ".join("?" * len(columns))
            conn.executemany(f"INSERT INTO {name} ({', '.join(columns)}) VALUES ({marks})", rows)


def _code_state() -> dict:
    try:
        root = Path(__file__).resolve().parents[2]
        git = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _config(meta, wells, horizons, validation_start, nk7_regime_start, target=TARGET) -> dict:
    return {
        "target": target,
        "validation_start": str(validation_start),
        "train_end": meta["train_end"],
        "test_start": meta["test_start"],
        "test_end": meta["test_end"],
        "nk7_regime_start": str(nk7_regime_start),
        "wells": list(wells),
        "horizons": list(horizons),
        "feature_order": RIDGE_FEATURES,
        "scaler": "StandardScaler",
        "ridge": {"fit_intercept": True},
        "clip_range": list(TARGET_RANGES[target]),
        "alpha_candidates": list(ALPHAS),
        "tie_rule": f"prefer {PREFERRED_ALPHA} if tied, otherwise the smallest tied alpha",
    }


def _validation_report(alpha, table: pd.DataFrame) -> dict:
    return {
        "selected_alpha": alpha,
        "rows": _records(table),
        "macro_mae": {str(a): float(m) for a, m in table.groupby("alpha")["mae"].mean().items()},
    }


def _evaluate_target(conn, target: str, validation_start, nk7_regime_start, *, bonus_models=False) -> dict:
    """Select alpha, refit, backtest and build the release for one target. Writes nothing."""
    clip_range = TARGET_RANGES.get(target, CLIP_RANGE)
    meta, series = _load(conn, target=target)
    test_start, test_end = meta["test_start"], meta["test_end"]
    sensors = _sensors_by_well(conn, series) if bonus_models else None
    frames = _frames(series, value_range=clip_range, sensors_by_well=sensors)

    alpha, validation = select_alpha(
        frames, validation_start=validation_start, test_start=test_start, nk7_regime_start=nk7_regime_start,
        clip_range=clip_range,
    )
    pipelines, fit_counts = fit_final(frames, alpha, test_start=test_start, nk7_regime_start=nk7_regime_start)
    bonus = fit_bonus(frames, test_start=test_start, nk7_regime_start=nk7_regime_start) if bonus_models else None
    predictions, metrics = evaluate_backtest(
        frames, pipelines, test_start=test_start, test_end=test_end, parameter=target, clip_range=clip_range,
        bonus_pipelines=bonus,
    )
    path = build_forecast_path(series, pipelines, test_start=test_start, parameter=target, clip_range=clip_range,
                               bonus_pipelines=bonus, sensors_by_well=sensors)
    models = MODELS + BONUS_MODELS if bonus_models else MODELS
    _check_outputs(predictions, metrics, path, len(frames), clip_range, models)

    config = _config(meta, series, HORIZONS, validation_start, nk7_regime_start, target)
    if bonus_models:
        config["bonus_models"] = {
            "gradient_boosting": {"features": BOOSTING_FEATURES, "parameters": BOOSTING_PARAMS},
            "statsmodels_ar": {"estimator": "statsmodels OLS", "intercept": True, "features": LAG_COLUMNS,
                               "version": version("statsmodels")},
        }

    origins = {}
    if not path.empty:
        origins = path.groupby("well_id")["origin_ts"].first().dt.strftime(TS_FORMAT).to_dict()
    return {
        "meta": meta,
        "tables": {"backtest_predictions": predictions, "metrics": metrics, "forecast_path": path},
        "report": {
            "config": config,
            "validation": _validation_report(alpha, validation),
            "final_fit_counts": _records(fit_counts),
            "release_origins": {well: origins.get(well) for well in series},
        },
    }


def run_forecast(db_path: Path, *, validation_start=None, nk7_regime_start=None, targets=None,
                 bonus_models=False) -> dict:
    """Run the full experiment, publish results and return the run report.

    `targets` defaults to the approved `pump_fillage`. Extra targets (for
    example `tubing_pressure` of the fountain wells) are an opt-in extension;
    the top-level report sections always describe the first target and
    `extra_targets` holds the same sections for the others.
    """
    validation_start = validation_start or DEFAULT_VALIDATION_START
    nk7_regime_start = nk7_regime_start or DEFAULT_NK7_REGIME_START
    targets = list(dict.fromkeys(targets or [TARGET]))
    if bonus_models and TARGET not in targets:
        raise ForecastError("--bonus-models requires pump_fillage in --targets")
    threads = threadpool_limits(limits=1) if bonus_models else nullcontext()
    with closing(_connect(db_path, "rw")) as conn, threads:
        results = {
            t: _evaluate_target(conn, t, validation_start, nk7_regime_start, bonus_models=bonus_models and t == TARGET)
            for t in targets
        }
        tables = {}
        for name, columns in RESULT_TABLES.items():
            parts = [result["tables"][name] for result in results.values() if not result["tables"][name].empty]
            tables[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        try:
            _publish(conn, tables, upgrade_schema=bonus_models)
        except sqlite3.Error as exc:
            raise ForecastError(f"publication failed, previous results kept: {exc}") from exc

    primary = results[targets[0]]
    report = {
        "dataset": {k: primary["meta"][k] for k in ("revision", "train_sha256", "test_sha256")},
        "code": _code_state(),
        "environment": _environment(),
        **primary["report"],
        "table_counts": {name: len(table) for name, table in tables.items()},
    }
    if len(targets) > 1:
        report["extra_targets"] = {t: results[t]["report"] for t in targets[1:]}
    if bonus_models:
        metrics = results[TARGET]["tables"]["metrics"]
        summary = metrics.assign(mae=pd.to_numeric(metrics["mae"])).groupby("model").agg(
            macro_mae=("mae", "mean"), scored_well_horizons=("mae", "count"), scored_keys=("n_common", "sum"),
        )
        report["bonus_comparison"] = _records(summary.reindex(MODELS + BONUS_MODELS).reset_index())
    return report


def run_diagnostic(db_path: Path, *, validation_start=None, nk7_regime_start=None, well_id=REGIME_WELL) -> dict:
    """Read-only, train-only check on one well at horizons 1 and 24. Writes nothing."""
    validation_start = validation_start or DEFAULT_VALIDATION_START
    nk7_regime_start = nk7_regime_start or DEFAULT_NK7_REGIME_START
    with closing(_connect(db_path, "ro")) as conn:
        meta, series = _load(conn, train_only=True)
    if well_id not in series:
        raise ForecastError(f"diagnostic well {well_id} has no {TARGET} series")
    s = series[well_id]
    test_start = meta["test_start"]
    frames = _frames({well_id: s}, DIAGNOSTIC_HORIZONS)

    def fail(horizon_h, stage, detail):
        raise ForecastError(f"diagnostic failed: well={well_id} horizon={horizon_h} stage={stage} {detail}")

    counts = []
    cutoff, regime = pd.Timestamp(validation_start), pd.Timestamp(nk7_regime_start)
    for (_, h), frame in frames.items():
        fit = eligible_fit_rows(frame, well_id=well_id, cutoff=validation_start, regime_start=nk7_regime_start)
        val = validation_rows(frame, validation_start=validation_start, test_start=test_start)
        if not (fit["target_ts"] < cutoff).all():
            fail(h, "internal_fit", "label at or after the validation cutoff")
        if well_id == REGIME_WELL and not (fit["origin_ts"] - pd.Timedelta(hours=24) >= regime).all():
            fail(h, "internal_fit", "lag before the NK-7 regime start")
        if not (val["origin_ts"] >= cutoff - pd.Timedelta(hours=1)).all():
            fail(h, "validation", "origin before the first validation release")
        source = s.reindex(frame["origin_ts"] + pd.Timedelta(hours=h - 24)).to_numpy()
        if not np.array_equal(frame["seasonal_value"].to_numpy(), source, equal_nan=True):
            fail(h, "seasonal_source", "seasonal value is not y[t+h-24]")
        if h == 24 and not np.array_equal(
            frame["seasonal_value"].to_numpy(), frame["lag_0"].to_numpy(), equal_nan=True
        ):
            fail(h, "seasonal_source", "seasonal and naive differ at 24 h")
        counts.append({"well_id": well_id, "horizon_h": h, "n_fit": len(fit), "n_validation": len(val)})

    alpha, validation = select_alpha(
        frames, validation_start=validation_start, test_start=test_start, nk7_regime_start=nk7_regime_start
    )
    report = _validation_report(alpha, validation)
    report["provisional_alpha"] = report.pop("selected_alpha")
    return {
        "mode": "diagnostic",
        "dataset": {k: meta[k] for k in ("revision", "train_sha256", "test_sha256")},
        "config": _config(meta, [well_id], DIAGNOSTIC_HORIZONS, validation_start, nk7_regime_start),
        "eligible_counts": counts,
        "validation": report,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate and publish short-term well forecasts.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--diagnostic", action="store_true", help="read-only train-only check, no DB writes")
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGET_RANGES), default=[TARGET],
                        help="targets to forecast; the default is the approved pump_fillage only")
    parser.add_argument("--bonus-models", action="store_true",
                        help="also compare gradient boosting and statsmodels autoregression for pump_fillage")
    args = parser.parse_args(argv)
    if args.diagnostic and args.bonus_models:
        parser.error("--bonus-models is for a full run")
    try:
        report = (run_diagnostic(args.db) if args.diagnostic
                  else run_forecast(args.db, targets=args.targets, bonus_models=args.bonus_models))
    except (ForecastError, ValueError, sqlite3.Error) as exc:
        print(f"forecast failed: {exc}", file=sys.stderr)
        return 1
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
