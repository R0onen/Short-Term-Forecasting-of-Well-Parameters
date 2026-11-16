"""Read-only SQLite access for the dashboard. Every filter value is bound, never interpolated."""

import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[1] / "data" / "sql"
SENSORS = ("SPM", "casing_pressure", "dynamometer_area", "line_pressure",
           "max_rod_weight", "min_rod_weight", "pump_fillage", "tubing_pressure")
MODES = ("raw", "clean")
GRAINS = ("day", "month")
MODELS = ("naive", "seasonal_naive", "ridge")
RESULT_TABLES = ("backtest_predictions", "metrics", "forecast_path")
CSV_FIELDS = ("well_id", "well_type", "timestamp", "parameter", "value", "unit", "n_obs", "flag")
REQUIRED_COLUMNS = {
    "dataset_meta": ("key", "value"),
    "series_catalog": ("well_id", "parameter", "well_type", "unit"),
    "measurements_raw": ("source_file", "source_row", *CSV_FIELDS, "split"),
    "measurements_wide": ("well_id", "timestamp", "well_type", "split", *SENSORS),
    "measurements_long": ("well_id", "timestamp", "parameter", "value", "unit", "n_obs", "flag",
                          "split", "is_usable"),
    "quality_issues": ("issue_id", "well_id", "parameter", "timestamp", "end_ts", "source_file",
                       "source_row", "issue_code", "action"),
    "backtest_predictions": ("well_id", "parameter", "origin_ts", "horizon_h", "model", "target_ts",
                             "y_true", "y_pred"),
    "metrics": ("well_id", "parameter", "horizon_h", "model", "mae", "n_common", "n_actual", "coverage"),
    "forecast_path": ("well_id", "parameter", "origin_ts", "horizon_h", "model", "target_ts", "y_pred"),
}
META_KEYS = ("revision", "source_url", "train_start", "train_end", "test_start", "test_end",
             "train_sha256", "test_sha256")
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

HISTORY_COLUMNS = ["well_id", "parameter", "timestamp", "unit", "value", "display_value",
                   "n_obs", "flag", "is_usable"]
OBSERVATION_COLUMNS = ["well_id", "well_type", "parameter", "unit", "timestamp", "value",
                       "display_value", "flag", "is_usable"]
GAP_COLUMNS = ["well_id", "parameter", "previous_ts", "timestamp", "delta_hours", "missing_hours"]
AGGREGATE_COLUMNS = ["well_id", "parameter", "well_type", "unit", "period_start", "n_expected",
                     "n_observed", "n_usable", "n_flagged", "n_values", "mean_value", "min_value",
                     "max_value", "coverage"]
ISSUE_COLUMNS = [*REQUIRED_COLUMNS["quality_issues"], *(f"raw_{field}" for field in CSV_FIELDS)]
BACKTEST_COLUMNS = list(REQUIRED_COLUMNS["backtest_predictions"])
METRIC_COLUMNS = list(REQUIRED_COLUMNS["metrics"])
PATH_COLUMNS = list(REQUIRED_COLUMNS["forecast_path"])


class IncompatibleDatabaseError(sqlite3.DatabaseError):
    """The file opened, but it does not carry the shared schema."""


def to_label(moment) -> str:
    """Format a date or timestamp as the stored hour label."""
    return pd.Timestamp(moment).strftime(TS_FORMAT)


def open_database(db_path: Path) -> sqlite3.Connection:
    """Open an existing database read-only. A wrong path never creates a file."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table, required in REQUIRED_COLUMNS.items():
            present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            missing = [column for column in required if column not in present]
            if missing:
                detail = "table is missing" if not present else f"missing columns {', '.join(missing)}"
                raise IncompatibleDatabaseError(f"{table}: {detail}")
        keys = {row[0] for row in conn.execute("SELECT key FROM dataset_meta")}
        absent = [key for key in META_KEYS if key not in keys]
        if absent:
            raise IncompatibleDatabaseError(f"dataset_meta: missing keys {', '.join(absent)}")
    except IncompatibleDatabaseError:
        conn.close()
        raise
    except sqlite3.DatabaseError as error:
        conn.close()
        raise IncompatibleDatabaseError(f"{path.name} is not a readable wells database: {error}") from error
    return conn


def _frame(conn, sql, bindings, columns, times=()):
    frame = pd.read_sql_query(sql, conn, params=bindings)
    frame = frame.loc[:, columns]
    for column in times:
        frame[column] = pd.to_datetime(frame[column], format=TS_FORMAT)
    return frame


def _check(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(map(str, allowed))}, got {value!r}")


def _sql_file(name):
    return (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")


def read_meta(conn) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM dataset_meta ORDER BY key"))


def read_catalog(conn) -> pd.DataFrame:
    """Applicability over the whole archive. It never depends on the selected dates."""
    return _frame(conn, "SELECT well_id, parameter, well_type, unit FROM series_catalog "
                        "ORDER BY well_type, well_id, parameter", {},
                  ["well_id", "parameter", "well_type", "unit"])


def read_history(conn, *, well_id, parameter, start_ts, end_ts, mode) -> pd.DataFrame:
    """One row per calendar hour in [start_ts, end_ts) for an applicable series.

    ``value`` keeps the published number. ``display_value`` hides masked rows in clean mode.
    """
    _check(mode, MODES, "mode")
    frame = _frame(conn, """
        SELECT w.well_id, c.parameter, w.timestamp, c.unit, l.value,
               CASE WHEN :mode = 'raw' OR l.is_usable = 1 THEN l.value END AS display_value,
               l.n_obs, l.flag, l.is_usable
        FROM series_catalog AS c
        JOIN measurements_wide AS w ON w.well_id = c.well_id
        LEFT JOIN measurements_long AS l
            ON l.well_id = w.well_id AND l.timestamp = w.timestamp AND l.parameter = c.parameter
        WHERE c.well_id = :well_id AND c.parameter = :parameter
          AND w.timestamp >= :start_ts AND w.timestamp < :end_ts
        ORDER BY w.timestamp
        """, dict(well_id=well_id, parameter=parameter, start_ts=start_ts, end_ts=end_ts, mode=mode),
        HISTORY_COLUMNS, times=["timestamp"])
    for column in ("value", "display_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("n_obs", "is_usable"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def read_observations(conn, *, start_ts, end_ts, mode) -> pd.DataFrame:
    """All validated observations of every series in the interval, for cross-well statistics."""
    _check(mode, MODES, "mode")
    frame = _frame(conn, """
        SELECT l.well_id, c.well_type, l.parameter, c.unit, l.timestamp, l.value,
               CASE WHEN :mode = 'raw' OR l.is_usable = 1 THEN l.value END AS display_value,
               l.flag, l.is_usable
        FROM measurements_long AS l
        JOIN series_catalog AS c ON c.well_id = l.well_id AND c.parameter = l.parameter
        WHERE l.timestamp >= :start_ts AND l.timestamp < :end_ts
        ORDER BY l.well_id, l.parameter, l.timestamp
        """, dict(start_ts=start_ts, end_ts=end_ts, mode=mode), OBSERVATION_COLUMNS, times=["timestamp"])
    frame["display_value"] = pd.to_numeric(frame["display_value"], errors="coerce").astype("float64")
    return frame


def read_well_matrix(conn, *, well_id, start_ts, end_ts, mode) -> pd.DataFrame:
    """Hour-indexed frame with one column per applicable parameter of the well."""
    _check(mode, MODES, "mode")
    listed = [row[0] for row in conn.execute(
        "SELECT parameter FROM series_catalog WHERE well_id = ? ORDER BY parameter", (well_id,))]
    parameters = [name for name in listed if name in SENSORS]  # allowlisted identifiers only
    bindings = dict(well_id=well_id, start_ts=start_ts, end_ts=end_ts)
    window = "well_id = :well_id AND timestamp >= :start_ts AND timestamp < :end_ts"
    if mode == "clean":
        columns = "".join(f', "{name}"' for name in parameters)
        matrix = pd.read_sql_query(
            f"SELECT timestamp{columns} FROM measurements_wide WHERE {window} ORDER BY timestamp",
            conn, params=bindings)
    else:
        calendar = pd.read_sql_query(
            f"SELECT timestamp FROM measurements_wide WHERE {window} ORDER BY timestamp", conn, params=bindings)
        long = pd.read_sql_query(
            f"SELECT timestamp, parameter, value FROM measurements_long WHERE {window}", conn, params=bindings)
        pivot = long.pivot(index="timestamp", columns="parameter", values="value")
        matrix = pivot.reindex(index=calendar["timestamp"], columns=parameters).reset_index()
    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], format=TS_FORMAT)
    matrix = matrix.set_index("timestamp").rename_axis(columns=None)
    return matrix.astype("float64")


def read_gaps(conn, *, well_id, parameter, start_ts, end_ts) -> pd.DataFrame:
    """Interior gaps from Student 1's LAG query. ``parameter=None`` asks for well-hour gaps."""
    return _frame(conn, _sql_file("gaps"),
                  dict(well_id=well_id, parameter=parameter, start_ts=start_ts, end_ts=end_ts),
                  GAP_COLUMNS, times=["previous_ts", "timestamp"])


def read_aggregates(conn, *, well_id, parameter, start_ts, end_ts, grain, mode) -> pd.DataFrame:
    """Daily or monthly counts and statistics from Student 1's query. None matches all pairs."""
    _check(grain, GRAINS, "grain")
    _check(mode, MODES, "mode")
    frame = _frame(conn, _sql_file("aggregates"),
                   dict(well_id=well_id, parameter=parameter, start_ts=start_ts, end_ts=end_ts,
                        grain=grain, mode=mode), AGGREGATE_COLUMNS)
    frame["period_start"] = pd.to_datetime(frame["period_start"], format="%Y-%m-%d")
    for column in ("mean_value", "min_value", "max_value", "coverage"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def read_issues(conn, *, well_id, parameter, start_ts, end_ts) -> pd.DataFrame:
    """Audit rows that overlap [start_ts, end_ts), with the raw CSV fields where a source exists.

    An hourly issue covers one hour. An annotation covers [timestamp, end_ts) and is open-ended
    when end_ts is NULL. Rejected records without a normalized hour are always returned so the
    audit view can list them separately. They match on whatever source context is known.
    ``parameter=None`` returns every parameter of the well.
    """
    raw = ", ".join(f"r.{field} AS raw_{field}" for field in CSV_FIELDS)
    frame = _frame(conn, f"""
        SELECT q.issue_id, q.well_id, q.parameter, q.timestamp, q.end_ts, q.source_file, q.source_row,
               q.issue_code, q.action, {raw}
        FROM quality_issues AS q
        LEFT JOIN measurements_raw AS r ON r.source_file = q.source_file AND r.source_row = q.source_row
        WHERE (q.well_id = :well_id OR (q.well_id IS NULL AND q.timestamp IS NULL))
          AND (:parameter IS NULL OR q.parameter = :parameter OR q.parameter IS NULL)
          AND (q.timestamp IS NULL
               OR (q.action = 'annotate' AND q.timestamp < :end_ts
                   AND (q.end_ts IS NULL OR q.end_ts > :start_ts))
               OR (q.action <> 'annotate' AND q.timestamp >= :start_ts AND q.timestamp < :end_ts))
        ORDER BY q.timestamp IS NULL, q.timestamp, q.parameter, q.issue_id
        """, dict(well_id=well_id, parameter=parameter, start_ts=start_ts, end_ts=end_ts),
        ISSUE_COLUMNS, times=["timestamp", "end_ts"])
    frame["source_row"] = pd.to_numeric(frame["source_row"], errors="coerce").astype("Int64")
    return frame


def read_annotations(conn) -> pd.DataFrame:
    """Period annotations of known dataset events over the whole archive."""
    return _frame(conn, """
        SELECT well_id, parameter, timestamp, end_ts, issue_code
        FROM quality_issues
        WHERE action = 'annotate' AND well_id IS NOT NULL AND parameter IS NOT NULL
        ORDER BY issue_code, well_id, parameter, timestamp
        """, {}, ["well_id", "parameter", "timestamp", "end_ts", "issue_code"], times=["timestamp", "end_ts"])


def read_result_counts(conn) -> dict[str, int]:
    """Row counts of the three forecast tables. All zero means the forecast has not been run."""
    return {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in RESULT_TABLES}


def _models(models):
    chosen = tuple(dict.fromkeys(models))
    for model in chosen:
        _check(model, MODELS, "model")
    bindings = {f"model_{index}": model for index, model in enumerate(chosen)}
    # An empty IN list is valid SQLite and matches nothing.
    return ", ".join(f":{name}" for name in bindings), bindings


def read_backtest(conn, *, well_id, parameter, horizon_h, start_ts, end_ts, models) -> pd.DataFrame:
    """Scored test predictions of one horizon whose target hour falls in [start_ts, end_ts)."""
    placeholders, bindings = _models(models)
    return _frame(conn, f"""
        SELECT well_id, parameter, origin_ts, horizon_h, model, target_ts, y_true, y_pred
        FROM backtest_predictions
        WHERE well_id = :well_id AND parameter = :parameter AND horizon_h = :horizon_h
          AND target_ts >= :start_ts AND target_ts < :end_ts AND model IN ({placeholders})
        ORDER BY target_ts, model
        """, dict(well_id=well_id, parameter=parameter, horizon_h=int(horizon_h),
                  start_ts=start_ts, end_ts=end_ts, **bindings),
        BACKTEST_COLUMNS, times=["origin_ts", "target_ts"])


def read_metrics(conn, *, well_id, parameter, models) -> pd.DataFrame:
    """Official test metrics of every stored horizon. Never restricted by the date slice."""
    placeholders, bindings = _models(models)
    frame = _frame(conn, f"""
        SELECT well_id, parameter, horizon_h, model, mae, n_common, n_actual, coverage
        FROM metrics
        WHERE well_id = :well_id AND parameter = :parameter AND model IN ({placeholders})
        ORDER BY horizon_h, model
        """, dict(well_id=well_id, parameter=parameter, **bindings), METRIC_COLUMNS)
    for column in ("mae", "coverage"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def read_forecast_path(conn, *, well_id, parameter, models) -> pd.DataFrame:
    """The stored latest release, independent of the date slice."""
    placeholders, bindings = _models(models)
    return _frame(conn, f"""
        SELECT well_id, parameter, origin_ts, horizon_h, model, target_ts, y_pred
        FROM forecast_path
        WHERE well_id = :well_id AND parameter = :parameter AND model IN ({placeholders})
        ORDER BY target_ts, model
        """, dict(well_id=well_id, parameter=parameter, **bindings),
        PATH_COLUMNS, times=["origin_ts", "target_ts"])
