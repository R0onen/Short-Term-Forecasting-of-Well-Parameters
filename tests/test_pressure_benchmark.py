import hashlib
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from wells.analysis.pressure_benchmark import build_pressure_database, summarize


def cached_pressure(raw_dir, source, readings, *, start="2017-01-01 00:00:00"):
    path = raw_dir / source
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(start, periods=len(readings), freq="s", name="timestamp")
    pd.DataFrame({"P-TPT": readings}, index=timestamps).to_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pressure_build_filters_and_deduplicates_in_source_order(tmp_path):
    raw_dir = tmp_path / "cache"
    first = "dataset/0/WELL-00001_20170101000000.parquet"
    overlapping = "dataset/1/WELL-00001_20170101000003.parquet"
    other_well = "dataset/0/WELL-00002_20170101000000.parquet"
    sources = {
        first: cached_pressure(raw_dir, first, [100_000.0, None, float("inf"), 103_000.0]),
        overlapping: cached_pressure(
            raw_dir, overlapping, [303_000.0, 104_000.0, 105_000.0], start="2017-01-01 00:00:03"
        ),
        other_well: cached_pressure(raw_dir, other_well, [200_000.0, 201_000.0]),
    }
    destination = tmp_path / "pressure.sqlite"
    build_pressure_database(raw_dir, destination, sources=sources, target_rows=5)

    with closing(sqlite3.connect(destination)) as conn:
        records = conn.execute(
            "SELECT well_id, datetime(timestamp), pressure_pa, source_file FROM pressure_observations "
            "ORDER BY well_id, timestamp"
        ).fetchall()
    assert records == [
        ("WELL-00001", "2017-01-01 00:00:00", 100_000.0, first),
        ("WELL-00001", "2017-01-01 00:00:03", 103_000.0, first),
        ("WELL-00001", "2017-01-01 00:00:04", 104_000.0, overlapping),
        ("WELL-00001", "2017-01-01 00:00:05", 105_000.0, overlapping),
        ("WELL-00002", "2017-01-01 00:00:00", 200_000.0, other_well),
    ]
    assert summarize(destination)["rows"] == 5


def test_pressure_build_preserves_destination_when_unique_count_is_short(tmp_path):
    raw_dir = tmp_path / "cache"
    first = "dataset/0/WELL-00001_20170101000000.parquet"
    overlap = "dataset/1/WELL-00001_20170101000000.parquet"
    sources = {
        first: cached_pressure(raw_dir, first, [100_000.0, 110_000.0]),
        overlap: cached_pressure(raw_dir, overlap, [300_000.0, float("-inf")]),
    }
    destination = tmp_path / "pressure.sqlite"
    build_pressure_database(raw_dir, destination, sources=sources, target_rows=2)
    previous_db = destination.read_bytes()

    with pytest.raises(ValueError, match="expected 3 distinct pressure rows, found 2"):
        build_pressure_database(raw_dir, destination, sources=sources, target_rows=3)

    assert destination.read_bytes() == previous_db
    assert summarize(destination)["rows"] == 2


def test_pressure_build_rejects_cached_file_with_wrong_hash(tmp_path):
    raw_dir = tmp_path / "cache"
    source = "dataset/0/WELL-00001_20170101000000.parquet"
    expected_hash = cached_pressure(raw_dir, source, [100_000.0])
    cached_pressure(raw_dir, source, [200_000.0])
    destination = tmp_path / "pressure.sqlite"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_pressure_database(raw_dir, destination, sources={source: expected_hash}, target_rows=1)

    assert not destination.exists()


def test_pressure_build_does_not_replace_the_nk_database(tmp_path):
    raw_dir = tmp_path / "cache"
    source = "dataset/0/WELL-00001_20170101000000.parquet"
    sources = {source: cached_pressure(raw_dir, source, [100_000.0])}
    destination = tmp_path / "wells.sqlite"
    with closing(sqlite3.connect(destination)) as conn:
        conn.execute("CREATE TABLE dataset_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    previous_db = destination.read_bytes()

    with pytest.raises(ValueError, match="existing database belongs to a different pipeline"):
        build_pressure_database(raw_dir, destination, sources=sources, target_rows=1)

    assert destination.read_bytes() == previous_db
