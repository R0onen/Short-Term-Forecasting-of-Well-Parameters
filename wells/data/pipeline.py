"""Build the complete hourly calendar and publish a verified SQLite database."""

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from wells.data.ingest import RAW_COLUMNS, download_inputs, read_inputs
from wells.data.quality import calendar_bounds, validate_measurements

SQL_DIR = Path(__file__).with_name("sql")
SENSORS = ("SPM", "casing_pressure", "dynamometer_area", "line_pressure",
           "max_rod_weight", "min_rod_weight", "pump_fillage", "tubing_pressure")
CATALOG_COLUMNS = ("well_id", "parameter", "well_type", "unit")
WIDE_COLUMNS = ("well_id", "timestamp", "well_type", "split", *SENSORS)
META_KEYS = ("revision", "source_url", "train_start", "train_end",
             "test_start", "test_end", "train_sha256", "test_sha256")
PUMP_WELLS = ("NK-7", "NK-68", "NK-77", "NK-82", "NK-86", "NK-91", "NK-93")
PUMP_UNITS = {
    "SPM": "strokes/min", "dynamometer_area": "unitless", "max_rod_weight": "kg",
    "min_rod_weight": "kg", "pump_fillage": "%",
}


def _official_catalog() -> pd.DataFrame:
    rows = [(well, parameter, "sucker_rod_pump", unit)
            for well in PUMP_WELLS for parameter, unit in PUMP_UNITS.items()]
    for well in ("NK-75", "NK-76", "NK-83"):
        rows.extend((well, sensor, "fountain", "atm") for sensor in ("tubing_pressure", "line_pressure"))
        if well != "NK-76":
            rows.append((well, "casing_pressure", "fountain", "atm"))
    return pd.DataFrame.from_records(rows, columns=CATALOG_COLUMNS)


def build_wide(long, catalog, meta) -> pd.DataFrame:
    """Pivot usable observations onto the complete catalog well-hour grid."""
    train_start, _, test_start, test_end = calendar_bounds(meta)
    wells = catalog[["well_id", "well_type"]].drop_duplicates().set_index("well_id")["well_type"]
    if not wells.index.is_unique or not wells.isin(("fountain", "sucker_rod_pump")).all():
        raise ValueError("The catalog must assign one valid type to each well")
    if catalog.duplicated(["well_id", "parameter"]).any() or not catalog["parameter"].isin(SENSORS).all():
        raise ValueError("The catalog must contain unique pairs of wells and supported parameters")
    hours = pd.date_range(train_start, test_end, freq="h")
    index = pd.MultiIndex.from_product([wells.index.sort_values(), hours], names=["well_id", "timestamp"])

    usable = long.loc[long["is_usable"].eq(1), ["well_id", "timestamp", "parameter", "value"]].copy()
    usable["timestamp"] = pd.to_datetime(usable["timestamp"])
    pivot = usable.pivot(index=["well_id", "timestamp"], columns="parameter", values="value")
    if not pivot.index.isin(index).all() or not pivot.columns.isin(SENSORS).all():
        raise ValueError("Usable observations fall outside the supported well-hour grid")
    wide = pivot.reindex(index=index, columns=SENSORS).rename_axis(columns=None).reset_index()
    wide["well_type"] = wide["well_id"].map(wells)
    wide["split"] = wide["timestamp"].ge(test_start).map({True: "test", False: "train"})
    return wide.loc[:, WIDE_COLUMNS]


def _sqlite_rows(frame):
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            elif hasattr(value, "item"):
                value = value.item()
            values.append(value)
        yield tuple(values)


def _write_database(path, tables):
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        with conn:
            for table, frame in tables.items():
                columns = ", ".join(frame.columns)
                placeholders = ", ".join("?" for _ in frame.columns)
                conn.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", _sqlite_rows(frame))


def _verify_database(path, tables):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite foreign key check failed")
        for table, frame in tables.items():
            if conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != len(frame):
                raise ValueError(f"{table}: row count changed during write")
            columns = ", ".join(frame.columns)
            stored = conn.execute(f"SELECT {columns} FROM {table} ORDER BY rowid")
            for number, (actual, expected) in enumerate(zip(stored, _sqlite_rows(frame), strict=True), start=1):
                if actual != expected:
                    raise ValueError(f"{table}: row {number} did not round-trip through SQLite")
        for table in ("backtest_predictions", "metrics", "forecast_path"):
            if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                raise ValueError(f"{table}: expected an empty result table on rebuild")


def assemble_database(raw, catalog, meta, db_path: Path) -> None:
    """Build on a sibling temporary file and replace the destination after verification."""
    metadata = pd.DataFrame.from_records([(key, meta[key]) for key in META_KEYS], columns=["key", "value"])
    long, issues = validate_measurements(raw, catalog, meta)
    wide = build_wide(long, catalog, meta)
    tables = {
        "dataset_meta": metadata,
        "series_catalog": catalog.loc[:, CATALOG_COLUMNS],
        "measurements_raw": raw.loc[:, RAW_COLUMNS],
        "measurements_wide": wide,
        "measurements_long": long,
        "quality_issues": issues,
    }

    destination = db_path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        _write_database(temporary, tables)
        _verify_database(temporary, tables)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_database(raw_dir: Path, db_path: Path) -> None:
    download_inputs(raw_dir)
    raw, meta = read_inputs(raw_dir)
    assemble_database(raw, _official_catalog(), meta, db_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/wells.sqlite"))
    args = parser.parse_args()
    try:
        build_database(args.raw_dir, args.db)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Data build failed: {error}", file=sys.stderr)
        return 1
    print(f"Built {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
