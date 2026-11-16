"""Static figures for the written report and the presentation.

    python -m wells.analysis.figures --db data/processed/wells.sqlite --out reports/figures
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from wells.analysis.sql_report import QUERIES, open_readonly  # noqa: E402

DEFAULT_DB = Path("data/processed/wells.sqlite")
DEFAULT_OUT = Path("reports/figures")
MODEL_STYLE = {
    "naive": ("Naive", "#7f8c8d"),
    "seasonal_naive": ("Seasonal naive", "#e67e22"),
    "ridge": ("Ridge (lag regression)", "#2471a3"),
}
EXAMPLE = {"well_id": "NK-82", "horizon_h": 6, "start": "2026-09-01 00:00:00", "end": "2026-09-08 00:00:00"}


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _horizon_chart(conn, out_dir: Path, query: str, ylabel: str, title: str, name: str):
    data = pd.read_sql_query(QUERIES[query][2], conn)
    if data.empty:  # the optional target was not run
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, (label, color) in MODEL_STYLE.items():
        ax.plot(data["horizon_h"], data[f"mae_{model}"], marker="o", markersize=3, label=label, color=color)
    ax.set(xlabel="Forecast horizon, hours", ylabel=ylabel, title=title)
    ax.set_xticks(range(1, 25, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir, name)


def mae_by_horizon(conn, out_dir: Path):
    return _horizon_chart(conn, out_dir, "12_mae_by_horizon", "MAE, percentage points",
                          "pump_fillage: mean MAE over 7 wells, official test period", "mae_by_horizon.png")


def pressure_mae_by_horizon(conn, out_dir: Path):
    return _horizon_chart(conn, out_dir, "14_pressure_mae_by_horizon", "MAE, atm",
                          "tubing_pressure: mean MAE over 3 fountain wells, official test period",
                          "pressure_mae_by_horizon.png")


def mae_per_well(conn, out_dir: Path) -> Path:
    data = pd.read_sql_query(QUERIES["11_mae_per_well"][2], conn).sort_values("well_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.27
    for i, (model, (label, color)) in enumerate(MODEL_STYLE.items()):
        ax.bar([x + (i - 1) * width for x in range(len(data))], data[f"mae_{model}"], width, label=label, color=color)
    ax.set_xticks(range(len(data)), data["well_id"])
    ax.set(ylabel="MAE, percentage points", title="pump_fillage: MAE per well, averaged over horizons 1–24 h")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out_dir, "mae_per_well.png")


def forecast_example(conn, out_dir: Path) -> Path:
    data = pd.read_sql_query(
        "SELECT target_ts, model, y_true, y_pred FROM backtest_predictions "
        "WHERE well_id = :well_id AND horizon_h = :horizon_h AND target_ts >= :start AND target_ts < :end "
        "ORDER BY target_ts", conn, params=EXAMPLE, parse_dates=["target_ts"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    actual = data[data["model"] == "ridge"]
    ax.plot(actual["target_ts"], actual["y_true"], color="black", linewidth=1.6, label="Actual")
    for model in ("naive", "ridge"):
        label, color = MODEL_STYLE[model]
        part = data[data["model"] == model]
        ax.plot(part["target_ts"], part["y_pred"], color=color, linewidth=1.1, label=label)
    ax.set(ylabel="pump_fillage, %",
           title=f"{EXAMPLE['well_id']}: {EXAMPLE['horizon_h']}-hour-ahead forecasts against actual values")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, out_dir, "forecast_example.png")


def area_scale_shift(conn, out_dir: Path) -> Path:
    data = pd.read_sql_query(QUERIES["05_area_scale_shift"][2], conn)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for well, part in data[data["mean_area"] > 0].groupby("well_id"):
        ax.plot(part["month"], part["mean_area"], marker="o", markersize=3, label=well)
    ax.set_yscale("log")
    ax.set(xlabel="Month", ylabel="Mean dynamometer_area (log scale)",
           title="dynamometer_area: scale shift in May 2026")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=4, fontsize=8)
    return _save(fig, out_dir, "area_scale_shift.png")


def nk7_regime(conn, out_dir: Path) -> Path:
    data = pd.read_sql_query(
        "SELECT substr(timestamp, 1, 10) AS day, AVG(SPM) AS spm, AVG(pump_fillage) AS fillage "
        "FROM measurements_wide WHERE well_id = 'NK-7' GROUP BY day ORDER BY day", conn, parse_dates=["day"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for ax, column, label in zip(axes, ("spm", "fillage"), ("SPM, strokes/min", "pump_fillage, %")):
        ax.plot(data["day"], data[column], color="#2471a3", linewidth=1)
        ax.axvline(pd.Timestamp("2026-06-01"), color="#c0392b", linestyle="--", label="1 June 2026")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title("NK-7: daily means before and after the June 2026 regime change")
    axes[0].legend()
    return _save(fig, out_dir, "nk7_regime.png")


def daily_profile(conn, out_dir: Path) -> Path:
    data = pd.read_sql_query(QUERIES["08_daily_profile"][2], conn)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, part) in zip(axes, data.groupby(["well_type", "parameter"])):
        ax.plot(part["hour"], part["mean_value"], marker="o", markersize=3, color="#2471a3")
        ax.set(xlabel="Hour of day", ylabel=key[1], title=f"{key[0]}: {key[1]}")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    fig.suptitle("Hour-of-day profile by well type (usable hourly values)")
    return _save(fig, out_dir, "daily_profile.png")


FIGURES = [mae_by_horizon, pressure_mae_by_horizon, mae_per_well, forecast_example, area_scale_shift, nk7_regime, daily_profile]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render report figures from the SQLite database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        with closing(open_readonly(args.db)) as conn:
            for make in FIGURES:
                path = make(conn, args.out)
                print(f"wrote {path}" if path else f"skipped {make.__name__}: no data")
    except (FileNotFoundError, sqlite3.Error) as error:
        print(f"figures failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
