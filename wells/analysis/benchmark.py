"""Time and memory measurements that justify the chosen data and model approach.

    python -m wells.analysis.benchmark --raw-dir data/raw --db data/processed/wells.sqlite --out reports

Part 1 compares ways to load the published CSVs (default dtypes, optimized
dtypes, chunked reading, Parquet). Part 2 compares Pandas with SQL for the same
pivot and aggregation. Part 3 measures the three forecasting methods.
Writes `<out>/benchmarks.json` and `<out>/benchmarks.md`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import tracemalloc
from contextlib import closing
from pathlib import Path

import pandas as pd

from wells.forecast.evaluation import evaluate_backtest, fit_final
from wells.forecast.features import TARGET, build_features
from wells.forecast.models import predict_naive, predict_ridge, predict_seasonal_naive
from wells.forecast.run import DEFAULT_NK7_REGIME_START

DEFAULT_RAW = Path("data/raw")
DEFAULT_DB = Path("data/processed/wells.sqlite")
DEFAULT_OUT = Path("reports")
CSV_FILES = ("train.csv", "test.csv")
CHUNK_ROWS = 50_000
OPTIMIZED_DTYPES = {
    "well_id": "category",
    "well_type": "category",
    "parameter": "category",
    "unit": "category",
    "flag": "category",
    "value": "float64",
    "n_obs": "int16",
}
MB = 1024 * 1024
BENCH_ALPHA = 10.0  # alpha selected by the official run; timing does not depend on it


def measure(action) -> tuple[object, float, float]:
    """Run `action` and return (result, seconds, peak megabytes allocated by Python)."""
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = action()
        seconds = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, seconds, peak / MB


def frame_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(deep=True).sum()) / MB


def read_default(raw_dir: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(raw_dir / name) for name in CSV_FILES], ignore_index=True)


def read_optimized(raw_dir: Path) -> pd.DataFrame:
    parts = [
        pd.read_csv(raw_dir / name, dtype=OPTIMIZED_DTYPES, parse_dates=["timestamp"], keep_default_na=False,
                    na_values={"value": [""]})
        for name in CSV_FILES
    ]
    frame = pd.concat(parts, ignore_index=True)
    for column, dtype in OPTIMIZED_DTYPES.items():  # concat turns differing categories into object
        if dtype == "category":
            frame[column] = frame[column].astype("category")
    return frame


def count_in_chunks(raw_dir: Path) -> dict[tuple[str, str], int]:
    """Aggregate without holding the whole file: rows per (well, parameter)."""
    counts: dict[tuple[str, str], int] = {}
    for name in CSV_FILES:
        for chunk in pd.read_csv(raw_dir / name, usecols=["well_id", "parameter"], chunksize=CHUNK_ROWS):
            for key, n in chunk.groupby(["well_id", "parameter"]).size().items():
                counts[key] = counts.get(key, 0) + int(n)
    return counts


def benchmark_loading(raw_dir: Path, work_dir: Path) -> list[dict]:
    csv_mb = sum((raw_dir / name).stat().st_size for name in CSV_FILES) / MB
    rows = []

    default, seconds, peak = measure(lambda: read_default(raw_dir))
    rows.append({"approach": "CSV, default dtypes", "seconds": seconds, "peak_mb": peak,
                 "in_memory_mb": frame_mb(default), "file_mb": csv_mb, "rows": len(default)})

    optimized, seconds, peak = measure(lambda: read_optimized(raw_dir))
    rows.append({"approach": "CSV, category/int16 dtypes", "seconds": seconds, "peak_mb": peak,
                 "in_memory_mb": frame_mb(optimized), "file_mb": csv_mb, "rows": len(optimized)})

    counts, seconds, peak = measure(lambda: count_in_chunks(raw_dir))
    rows.append({"approach": f"CSV, chunksize={CHUNK_ROWS} aggregation", "seconds": seconds, "peak_mb": peak,
                 "in_memory_mb": None, "file_mb": csv_mb, "rows": sum(counts.values())})

    parquet = work_dir / "measurements.parquet"
    _, write_seconds, _ = measure(lambda: optimized.to_parquet(parquet, index=False))
    restored, seconds, peak = measure(lambda: pd.read_parquet(parquet))
    rows.append({"approach": "Parquet (pyarrow), optimized dtypes", "seconds": seconds, "peak_mb": peak,
                 "in_memory_mb": frame_mb(restored), "file_mb": parquet.stat().st_size / MB, "rows": len(restored),
                 "write_seconds": write_seconds})
    if len(restored) != len(default) or sum(counts.values()) != len(default):
        raise RuntimeError("loading approaches disagree on the row count")
    return rows


def benchmark_pandas_vs_sql(conn: sqlite3.Connection) -> list[dict]:
    """Same two results computed in Pandas and in SQLite."""
    long = pd.read_sql_query(
        "SELECT well_id, timestamp, parameter, value FROM measurements_long WHERE is_usable = 1", conn
    )
    rows = []

    pivot, seconds, peak = measure(
        lambda: long.pivot(index=["well_id", "timestamp"], columns="parameter", values="value")
    )
    rows.append({"task": "long -> wide pivot", "approach": "Pandas pivot", "seconds": seconds, "peak_mb": peak,
                 "result_rows": len(pivot)})
    wide_sql = """
        SELECT well_id, timestamp,
               MAX(CASE WHEN parameter = 'SPM' THEN value END) AS SPM,
               MAX(CASE WHEN parameter = 'casing_pressure' THEN value END) AS casing_pressure,
               MAX(CASE WHEN parameter = 'dynamometer_area' THEN value END) AS dynamometer_area,
               MAX(CASE WHEN parameter = 'line_pressure' THEN value END) AS line_pressure,
               MAX(CASE WHEN parameter = 'max_rod_weight' THEN value END) AS max_rod_weight,
               MAX(CASE WHEN parameter = 'min_rod_weight' THEN value END) AS min_rod_weight,
               MAX(CASE WHEN parameter = 'pump_fillage' THEN value END) AS pump_fillage,
               MAX(CASE WHEN parameter = 'tubing_pressure' THEN value END) AS tubing_pressure
        FROM measurements_long WHERE is_usable = 1
        GROUP BY well_id, timestamp
    """
    pivot_sql, seconds, peak = measure(lambda: pd.read_sql_query(wide_sql, conn))
    rows.append({"task": "long -> wide pivot", "approach": "SQL conditional aggregation", "seconds": seconds,
                 "peak_mb": peak, "result_rows": len(pivot_sql)})

    daily, seconds, peak = measure(
        lambda: long.assign(day=long["timestamp"].str[:10]).groupby(["well_id", "parameter", "day"])["value"].mean()
    )
    rows.append({"task": "daily mean per series", "approach": "Pandas groupby", "seconds": seconds,
                 "peak_mb": peak, "result_rows": len(daily)})
    daily_sql, seconds, peak = measure(lambda: pd.read_sql_query(
        "SELECT well_id, parameter, substr(timestamp, 1, 10) AS day, AVG(value) AS mean_value "
        "FROM measurements_long WHERE is_usable = 1 GROUP BY well_id, parameter, day", conn))
    rows.append({"task": "daily mean per series", "approach": "SQL GROUP BY", "seconds": seconds,
                 "peak_mb": peak, "result_rows": len(daily_sql)})
    if len(pivot) != len(pivot_sql) or len(daily) != len(daily_sql):
        raise RuntimeError("Pandas and SQL results disagree on the row count")
    return rows


def benchmark_models(conn: sqlite3.Connection) -> list[dict]:
    """Fit and prediction cost of each method over all wells and horizons, with its MAE."""
    meta = dict(conn.execute("SELECT key, value FROM dataset_meta"))
    n_metrics = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    if n_metrics == 0:
        raise RuntimeError("metrics table is empty: run python -m wells.forecast.run first")
    wide = pd.read_sql_query(
        f"SELECT well_id, timestamp, {TARGET} AS y FROM measurements_wide "
        f"WHERE well_id IN (SELECT well_id FROM series_catalog WHERE parameter = '{TARGET}') "
        "ORDER BY well_id, timestamp", conn, parse_dates=["timestamp"])
    series = {well: part.set_index("timestamp")["y"] for well, part in wide.groupby("well_id")}

    frames, feature_seconds, feature_peak = measure(
        lambda: {(well, h): build_features(s, h) for well, s in series.items() for h in range(1, 25)}
    )
    (pipelines, _), fit_seconds, fit_peak = measure(
        lambda: fit_final(frames, BENCH_ALPHA, test_start=meta["test_start"], nk7_regime_start=DEFAULT_NK7_REGIME_START)
    )
    test_start = pd.Timestamp(meta["test_start"])
    test_frames = {key: frame[frame["target_ts"] >= test_start] for key, frame in frames.items()}

    predictors = {
        "naive": lambda: [predict_naive(f) for f in test_frames.values()],
        "seasonal_naive": lambda: [predict_seasonal_naive(f) for f in test_frames.values()],
        "ridge": lambda: [predict_ridge(pipelines[key], f) for key, f in test_frames.items()],
    }
    mae = dict(conn.execute("SELECT model, AVG(mae) FROM metrics GROUP BY model"))
    rows = []
    for model, action in predictors.items():
        _, seconds, peak = measure(action)
        rows.append({
            "model": model,
            "fit_seconds": fit_seconds if model == "ridge" else 0.0,
            "predict_seconds": seconds,
            "peak_mb": max(peak, fit_peak) if model == "ridge" else peak,
            "mean_mae": mae[model],
            "groups": len(test_frames),
        })
    _, backtest_seconds, _ = measure(
        lambda: evaluate_backtest(frames, pipelines, test_start=meta["test_start"], test_end=meta["test_end"])
    )
    rows.append({"model": "shared: feature building", "fit_seconds": 0.0, "predict_seconds": feature_seconds,
                 "peak_mb": feature_peak, "mean_mae": None, "groups": len(frames)})
    rows.append({"model": "shared: full backtest scoring", "fit_seconds": 0.0, "predict_seconds": backtest_seconds,
                 "peak_mb": None, "mean_mae": None, "groups": len(frames)})
    return rows


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def to_markdown(report: dict) -> str:
    sections = {
        "Loading the published CSVs": ("loading", ["approach", "seconds", "peak_mb", "in_memory_mb", "file_mb", "rows"]),
        "Pandas versus SQL": ("pandas_vs_sql", ["task", "approach", "seconds", "peak_mb", "result_rows"]),
        "Forecasting methods": ("models", ["model", "fit_seconds", "predict_seconds", "peak_mb", "mean_mae", "groups"]),
    }
    lines = ["# Benchmarks", "", "Generated by `python -m wells.analysis.benchmark`. Times are wall-clock seconds "
             "of one run on the development machine; peak memory is Python allocations traced by `tracemalloc`.", ""]
    for title, (key, columns) in sections.items():
        lines += [f"## {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in report[key]:
            lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure loading, SQL and model approaches.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        for name in CSV_FILES:
            if not (args.raw_dir / name).is_file():
                raise FileNotFoundError(f"missing input: {args.raw_dir / name}")
        if not args.db.is_file():
            raise FileNotFoundError(f"database not found: {args.db}")
        args.out.mkdir(parents=True, exist_ok=True)
        report = {"loading": benchmark_loading(args.raw_dir, args.db.parent)}
        with closing(sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            report["pandas_vs_sql"] = benchmark_pandas_vs_sql(conn)
            report["models"] = benchmark_models(conn)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    (args.out / "benchmarks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "benchmarks.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
