"""Build and benchmark a reproducible 500,000-row sample of Petrobras 3W pressure."""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from wells.analysis.benchmark import measure

REVISION = "c86cc7c70063646910be7292de8a54a6fb2a8399"
SOURCE_ROOT = f"https://raw.githubusercontent.com/petrobras/3W/{REVISION}"
TARGET_ROWS = 500_000
PRESSURE_CHANNEL = "P-TPT"
REAL_INSTANCE = re.compile(r"dataset/[0-9]/(WELL-[0-9]{5})_[0-9]{14}\.parquet")
SOURCES = {
    "dataset/0/WELL-00001_20170201010207.parquet": "1df6fd74bc3b943570a70ffa075f53ca5b50d606eadea600acca977fc633146e",
    "dataset/0/WELL-00019_20170307235203.parquet": "5e44eb169fcf5007214f5f522d0f13c09317ff67c2340ca4f8145c2223163a41",
    "dataset/7/WELL-00022_20181026005827.parquet": "ac1c04f5fd9d4e40d3b6c20544a30b21ea578acbe1b8369eba2875319914cbaf",
}

SCHEMA = """
CREATE TABLE pressure_sources (
    path TEXT PRIMARY KEY NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE pressure_observations (
    well_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    pressure_pa REAL NOT NULL,
    source_file TEXT NOT NULL REFERENCES pressure_sources(path),
    PRIMARY KEY (well_id, timestamp)
) WITHOUT ROWID;
"""


def _verify_hash(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch (expected {expected}, got {actual})")


def _verified_source(raw_dir: Path, relative: str, sha256: str) -> Path:
    path = raw_dir / relative
    if path.exists():
        _verify_hash(path, sha256)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    output = NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False)
    temporary = Path(output.name)
    try:
        with output, urlopen(f"{SOURCE_ROOT}/{relative}", timeout=60) as source:
            shutil.copyfileobj(source, output)
        _verify_hash(temporary, sha256)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _pressure_rows(path: Path, well: str, relative: str):
    frame = pd.read_parquet(path, columns=[PRESSURE_CHANNEL], engine="pyarrow")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{relative}: expected a timestamp index")
    pressure = pd.to_numeric(frame[PRESSURE_CHANNEL], errors="raise")
    valid = frame.index.notna() & np.isfinite(pressure.to_numpy(dtype=float, na_value=np.nan))
    for timestamp, value in pressure.loc[valid].sort_index(kind="stable").items():
        yield well, timestamp.isoformat(sep=" "), float(value), relative


def _populate(conn: sqlite3.Connection, raw_dir: Path, sources: dict[str, str], target_rows: int) -> None:
    count = 0
    for relative, sha256 in sources.items():
        match = REAL_INSTANCE.fullmatch(relative)
        if not match:
            raise ValueError(f"{relative}: expected a real well instance")
        path = _verified_source(raw_dir, relative, sha256)
        conn.execute("INSERT INTO pressure_sources VALUES (?, ?)", (relative, sha256))
        inserted = conn.executemany(
            "INSERT INTO pressure_observations VALUES (?, ?, ?, ?) "
            "ON CONFLICT(well_id, timestamp) DO NOTHING",
            _pressure_rows(path, match[1], relative),
        )
        count += inserted.rowcount
        if count >= target_rows:
            break
    if count < target_rows:
        raise ValueError(f"expected {target_rows:,} distinct pressure rows, found {count:,}")
    if count > target_rows:
        conn.execute("""
            DELETE FROM pressure_observations
            WHERE (well_id, timestamp) IN (
                SELECT well_id, timestamp FROM pressure_observations
                ORDER BY well_id DESC, timestamp DESC LIMIT ?
            )
        """, (count - target_rows,))


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def build_pressure_database(
    raw_dir: Path, db_path: Path, *, sources: dict[str, str] = SOURCES, target_rows: int = TARGET_ROWS,
) -> None:
    """Keep one finite P-TPT value per well and timestamp in an independent DB."""
    if target_rows < 1:
        raise ValueError("target_rows must be positive")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=db_path.parent, prefix=".pressure-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(SCHEMA)
            with conn:
                _populate(conn, raw_dir, sources, target_rows)
                if conn.execute("SELECT count(*) FROM pressure_observations").fetchone()[0] != target_rows:
                    raise ValueError("pressure database row count check failed")
                if conn.execute("PRAGMA integrity_check").fetchone() != ("ok",) or conn.execute(
                    "PRAGMA foreign_key_check"
                ).fetchone() is not None:
                    raise ValueError("pressure database integrity check failed")
        if db_path.exists():
            with closing(_open_readonly(db_path)) as previous:
                if previous.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='pressure_observations'"
                ).fetchone() is None:
                    raise ValueError(f"{db_path}: existing database belongs to a different pipeline")
        temporary.replace(db_path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(db_path: Path) -> dict:
    """Check SQL and Pandas aggregations and report volume and source provenance."""
    with closing(_open_readonly(db_path)) as conn:
        sql, sql_seconds, sql_peak = measure(lambda: pd.read_sql_query("""
            SELECT well_id, count(*) AS rows, avg(pressure_pa) AS mean_pa,
                   min(pressure_pa) AS min_pa, max(pressure_pa) AS max_pa
            FROM pressure_observations GROUP BY well_id ORDER BY well_id
        """, conn))

        def pandas_groupby():
            frame = pd.read_sql_query("SELECT well_id, pressure_pa FROM pressure_observations", conn)
            return frame.groupby("well_id", as_index=False).agg(
                rows=("pressure_pa", "size"), mean_pa=("pressure_pa", "mean"),
                min_pa=("pressure_pa", "min"), max_pa=("pressure_pa", "max"),
            )

        pandas, pandas_seconds, pandas_peak = measure(pandas_groupby)
        pd.testing.assert_frame_equal(sql, pandas, atol=1e-6, rtol=1e-9)
        paths = conn.execute("""
            SELECT s.path, s.sha256, count(o.source_file)
            FROM pressure_sources AS s LEFT JOIN pressure_observations AS o ON o.source_file = s.path
            GROUP BY s.path, s.sha256 ORDER BY s.path
        """).fetchall()
        start, end = conn.execute("SELECT min(timestamp), max(timestamp) FROM pressure_observations").fetchone()
    return {
        "dataset": "Petrobras 3W",
        "revision": REVISION,
        "license": "CC BY 4.0",
        "variable": PRESSURE_CHANNEL,
        "unit": "Pa",
        "rows": int(sql["rows"].sum()),
        "wells": len(sql),
        "start_timestamp": start,
        "end_timestamp": end,
        "sources": [{"path": path, "sha256": sha256, "rows": count} for path, sha256, count in paths],
        "by_well": sql.to_dict(orient="records"),
        "timings": [
            {"approach": "SQLite GROUP BY", "seconds": sql_seconds, "python_peak_mib": sql_peak},
            {"approach": "Read SQLite rows and group with Pandas", "seconds": pandas_seconds,
             "python_peak_mib": pandas_peak},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/petrobras-3w"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/petrobras-pressure.sqlite"))
    args = parser.parse_args()
    try:
        build_pressure_database(args.raw_dir, args.db)
        print(json.dumps(summarize(args.db), indent=2))
    except (OSError, ValueError, sqlite3.Error, AssertionError) as error:
        print(f"Pressure benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
