"""Run the project's analytical SQL queries and load every result into Pandas.

    python -m wells.analysis.sql_report --db data/processed/wells.sqlite --out reports

Writes one CSV per query to `<out>/sql/` and a summary `<out>/sql_queries.md`.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path("data/processed/wells.sqlite")
DEFAULT_OUT = Path("reports")
PREVIEW_ROWS = 12

# name -> (what it shows, SQL features used, SQL text)
QUERIES: dict[str, tuple[str, str, str]] = {
    "01_split_boundaries": (
        "Chronological train/test split published with the dataset.",
        "GROUP BY, aggregates",
        """
        SELECT split, COUNT(*) AS n_rows, MIN(timestamp) AS first_hour, MAX(timestamp) AS last_hour,
               COUNT(DISTINCT well_id) AS wells
        FROM measurements_raw
        GROUP BY split
        ORDER BY first_hour
        """,
    ),
    "02_coverage_by_type": (
        "Observed and usable hourly values per well type and sensor.",
        "LEFT JOIN, GROUP BY",
        """
        SELECT c.well_type, c.parameter, c.unit,
               COUNT(DISTINCT c.well_id) AS wells,
               COUNT(l.value) AS n_observed,
               COALESCE(SUM(l.is_usable), 0) AS n_usable,
               ROUND(AVG(CASE WHEN l.is_usable = 1 THEN l.value END), 3) AS mean_usable
        FROM series_catalog AS c
        LEFT JOIN measurements_long AS l
               ON l.well_id = c.well_id AND l.parameter = c.parameter
        GROUP BY c.well_type, c.parameter, c.unit
        ORDER BY c.well_type, c.parameter
        """,
    ),
    "03_flagged_series": (
        "Series that carry published quality flags, with the flagged share.",
        "GROUP BY, HAVING",
        """
        SELECT well_id, parameter, COUNT(*) AS n_observed,
               SUM(flag <> '') AS n_flagged,
               ROUND(100.0 * SUM(flag <> '') / COUNT(*), 2) AS flagged_pct
        FROM measurements_long
        GROUP BY well_id, parameter
        HAVING SUM(flag <> '') > 0
        ORDER BY flagged_pct DESC
        """,
    ),
    "04_longest_gaps": (
        "Ten longest well-wide outages found with LAG over observed hours.",
        "CTE, window function LAG",
        """
        WITH hours AS (
            SELECT DISTINCT well_id, timestamp FROM measurements_long
        ), ordered AS (
            SELECT well_id, timestamp,
                   LAG(timestamp) OVER (PARTITION BY well_id ORDER BY timestamp) AS previous_ts
            FROM hours
        )
        SELECT well_id, previous_ts, timestamp,
               CAST(ROUND((julianday(timestamp) - julianday(previous_ts)) * 24) AS INTEGER) - 1 AS missing_hours
        FROM ordered
        WHERE previous_ts IS NOT NULL
          AND CAST(ROUND((julianday(timestamp) - julianday(previous_ts)) * 24) AS INTEGER) > 1
        ORDER BY missing_hours DESC, well_id
        LIMIT 10
        """,
    ),
    "05_area_scale_shift": (
        "Monthly mean dynamometer_area and its ratio to the previous month (May 2026 scale shift).",
        "CTE, window function LAG, GROUP BY",
        """
        WITH monthly AS (
            SELECT well_id, substr(timestamp, 1, 7) AS month, COUNT(*) AS n_hours, AVG(value) AS mean_area
            FROM measurements_long
            WHERE parameter = 'dynamometer_area'
            GROUP BY well_id, month
        )
        SELECT well_id, month, n_hours, ROUND(mean_area, 2) AS mean_area,
               ROUND(mean_area / LAG(mean_area) OVER (PARTITION BY well_id ORDER BY month), 3) AS ratio_to_previous
        FROM monthly
        ORDER BY well_id, month
        """,
    ),
    "06_nk7_regime": (
        "NK-7 before and after the June 2026 regime change.",
        "CASE, GROUP BY",
        """
        SELECT parameter,
               CASE WHEN timestamp < '2026-06-01 00:00:00' THEN 'before_june' ELSE 'from_june' END AS regime,
               COUNT(*) AS n_hours, ROUND(AVG(value), 3) AS mean_value,
               ROUND(MIN(value), 3) AS min_value, ROUND(MAX(value), 3) AS max_value
        FROM measurements_long
        WHERE well_id = 'NK-7' AND is_usable = 1
        GROUP BY parameter, regime
        ORDER BY parameter, regime
        """,
    ),
    "07_low_fillage_hours": (
        "Hours where pump fillage is below half of the well's own average.",
        "JOIN with an aggregated subquery, GROUP BY",
        """
        SELECT w.well_id, ROUND(a.mean_fillage, 2) AS mean_fillage,
               COUNT(*) AS low_hours, ROUND(100.0 * COUNT(*) / a.n_hours, 2) AS low_pct
        FROM measurements_wide AS w
        JOIN (
            SELECT well_id, AVG(pump_fillage) AS mean_fillage, COUNT(pump_fillage) AS n_hours
            FROM measurements_wide
            GROUP BY well_id
            HAVING COUNT(pump_fillage) > 0
        ) AS a ON a.well_id = w.well_id
        WHERE w.pump_fillage < 0.5 * a.mean_fillage
        GROUP BY w.well_id
        ORDER BY low_pct DESC
        """,
    ),
    "08_daily_profile": (
        "Hour-of-day profile of the main sensor of each well type.",
        "JOIN, GROUP BY on a derived hour",
        """
        SELECT c.well_type, l.parameter, CAST(strftime('%H', l.timestamp) AS INTEGER) AS hour,
               COUNT(*) AS n_hours, ROUND(AVG(l.value), 3) AS mean_value
        FROM measurements_long AS l
        JOIN series_catalog AS c ON c.well_id = l.well_id AND c.parameter = l.parameter
        WHERE l.is_usable = 1 AND l.parameter IN ('pump_fillage', 'tubing_pressure')
        GROUP BY c.well_type, l.parameter, hour
        ORDER BY c.well_type, l.parameter, hour
        """,
    ),
    "09_quality_issues": (
        "Audit journal by issue code, with the split of the source record.",
        "LEFT JOIN, GROUP BY",
        """
        SELECT q.issue_code, q.action, COUNT(*) AS n_issues,
               COUNT(DISTINCT q.well_id) AS wells,
               MIN(q.timestamp) AS first_ts, MAX(q.timestamp) AS last_ts,
               COALESCE(SUM(r.split = 'test'), 0) AS n_in_test
        FROM quality_issues AS q
        LEFT JOIN measurements_raw AS r
               ON r.source_file = q.source_file AND r.source_row = q.source_row
        GROUP BY q.issue_code, q.action
        ORDER BY n_issues DESC
        """,
    ),
    "10_model_wins": (
        "How often each forecasting method has the lowest MAE over 7 wells x 24 horizons.",
        "CTE, window function RANK",
        """
        WITH ranked AS (
            SELECT well_id, horizon_h, model, mae,
                   RANK() OVER (PARTITION BY well_id, horizon_h ORDER BY mae) AS place
            FROM metrics
            WHERE mae IS NOT NULL AND parameter = 'pump_fillage'
        )
        SELECT model, COUNT(*) AS groups_won, ROUND(AVG(mae), 3) AS mean_mae_when_best
        FROM ranked
        WHERE place = 1
        GROUP BY model
        ORDER BY groups_won DESC
        """,
    ),
    "11_mae_per_well": (
        "MAE per well and method averaged over 24 horizons, with coverage.",
        "JOIN, conditional aggregation",
        """
        SELECT m.well_id, c.well_type,
               ROUND(AVG(CASE WHEN m.model = 'naive' THEN m.mae END), 3) AS mae_naive,
               ROUND(AVG(CASE WHEN m.model = 'seasonal_naive' THEN m.mae END), 3) AS mae_seasonal_naive,
               ROUND(AVG(CASE WHEN m.model = 'ridge' THEN m.mae END), 3) AS mae_ridge,
               ROUND(AVG(m.coverage), 3) AS mean_coverage
        FROM metrics AS m
        JOIN series_catalog AS c ON c.well_id = m.well_id AND c.parameter = m.parameter
        WHERE m.parameter = 'pump_fillage'
        GROUP BY m.well_id, c.well_type
        ORDER BY mae_ridge
        """,
    ),
    "12_mae_by_horizon": (
        "Mean MAE across wells for every horizon and method.",
        "GROUP BY, conditional aggregation",
        """
        SELECT horizon_h,
               ROUND(AVG(CASE WHEN model = 'naive' THEN mae END), 3) AS mae_naive,
               ROUND(AVG(CASE WHEN model = 'seasonal_naive' THEN mae END), 3) AS mae_seasonal_naive,
               ROUND(AVG(CASE WHEN model = 'ridge' THEN mae END), 3) AS mae_ridge,
               SUM(CASE WHEN model = 'ridge' THEN n_common END) AS n_common
        FROM metrics
        WHERE parameter = 'pump_fillage'
        GROUP BY horizon_h
        ORDER BY horizon_h
        """,
    ),
    "13_pressure_mae_per_well": (
        "Extension: tubing_pressure MAE per fountain well and method (empty unless the pressure target was run).",
        "JOIN, conditional aggregation",
        """
        SELECT m.well_id, c.unit,
               ROUND(AVG(CASE WHEN m.model = 'naive' THEN m.mae END), 3) AS mae_naive,
               ROUND(AVG(CASE WHEN m.model = 'seasonal_naive' THEN m.mae END), 3) AS mae_seasonal_naive,
               ROUND(AVG(CASE WHEN m.model = 'ridge' THEN m.mae END), 3) AS mae_ridge,
               ROUND(AVG(m.coverage), 3) AS mean_coverage
        FROM metrics AS m
        JOIN series_catalog AS c ON c.well_id = m.well_id AND c.parameter = m.parameter
        WHERE m.parameter = 'tubing_pressure'
        GROUP BY m.well_id, c.unit
        ORDER BY m.well_id
        """,
    ),
    "14_pressure_mae_by_horizon": (
        "Extension: mean tubing_pressure MAE across fountain wells for every horizon and method.",
        "GROUP BY, conditional aggregation",
        """
        SELECT horizon_h,
               ROUND(AVG(CASE WHEN model = 'naive' THEN mae END), 3) AS mae_naive,
               ROUND(AVG(CASE WHEN model = 'seasonal_naive' THEN mae END), 3) AS mae_seasonal_naive,
               ROUND(AVG(CASE WHEN model = 'ridge' THEN mae END), 3) AS mae_ridge,
               SUM(CASE WHEN model = 'ridge' THEN n_common END) AS n_common
        FROM metrics
        WHERE parameter = 'tubing_pressure'
        GROUP BY horizon_h
        ORDER BY horizon_h
        """,
    ),
}


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an existing database read-only; a wrong path never creates a file."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def run_queries(conn: sqlite3.Connection, names=None) -> dict[str, pd.DataFrame]:
    """Execute the selected queries and return their results as DataFrames."""
    results = {}
    for name in names or QUERIES:
        results[name] = pd.read_sql_query(QUERIES[name][2], conn)
    return results


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table."""
    if frame.empty:
        return "_no rows_"
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def write_report(results: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Save each result as CSV and one Markdown file with SQL text and previews."""
    sql_dir = out_dir / "sql"
    sql_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SQL queries",
        "",
        "Generated by `python -m wells.analysis.sql_report`. Every result is read into Pandas "
        "and saved under `reports/sql/`.",
        "",
    ]
    for name, frame in results.items():
        title, features, sql = QUERIES[name]
        frame.to_csv(sql_dir / f"{name}.csv", index=False)
        body = "\n".join(line[8:] if line.startswith(" " * 8) else line for line in sql.strip("\n").splitlines())
        lines += [f"## {name}", "", f"{title} SQL features: {features}. Rows returned: {len(frame)}.", ""]
        lines += ["```sql", body, "```", "", _markdown_table(frame.head(PREVIEW_ROWS)), ""]
        if len(frame) > PREVIEW_ROWS:
            lines += [f"_First {PREVIEW_ROWS} rows shown; full result in `reports/sql/{name}.csv`._", ""]
    target = out_dir / "sql_queries.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the analytical SQL query set.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        with closing(open_readonly(args.db)) as conn:
            results = run_queries(conn)
    except (FileNotFoundError, sqlite3.Error) as error:
        print(f"sql report failed: {error}", file=sys.stderr)
        return 1
    target = write_report(results, args.out)
    for name, frame in results.items():
        print(f"{name}: {len(frame)} rows")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
