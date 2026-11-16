import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from wells.data import pipeline
from wells.data.ingest import CSV_COLUMNS, RAW_COLUMNS, _read_csv, read_inputs
from wells.data.quality import summarize_anomalies, validate_measurements

FIXTURE = Path(__file__).with_name("fixtures") / "seed.sql"


@pytest.fixture
def seeded():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript((pipeline.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        for table in ("backtest_predictions", "metrics", "forecast_path"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
        conn.executescript(FIXTURE.read_text(encoding="utf-8"))
        yield conn


def fixture_inputs(conn):
    return (
        pd.read_sql_query("SELECT * FROM measurements_raw ORDER BY source_file, source_row", conn),
        pd.read_sql_query("SELECT * FROM series_catalog ORDER BY well_id, parameter", conn),
        dict(conn.execute("SELECT key, value FROM dataset_meta")),
    )


def sql_rows(conn, name, **bindings):
    return conn.execute((pipeline.SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"), bindings).fetchall()


def test_schema_and_seed(seeded):
    tables = {
        "dataset_meta": ("key",), "series_catalog": ("well_id", "parameter"),
        "measurements_raw": ("source_file", "source_row"), "measurements_wide": ("well_id", "timestamp"),
        "measurements_long": ("well_id", "timestamp", "parameter"), "quality_issues": ("issue_id",),
        "metrics": ("well_id", "parameter", "horizon_h", "model"),
        "backtest_predictions": ("well_id", "parameter", "origin_ts", "horizon_h", "model"),
        "forecast_path": ("well_id", "parameter", "origin_ts", "horizon_h", "model"),
    }
    for table, primary_key in tables.items():
        columns = seeded.execute(f"PRAGMA table_info({table})").fetchall()
        assert tuple(row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]) == primary_key
    wide_columns = seeded.execute("PRAGMA table_info(measurements_wide)").fetchall()
    assert tuple(row[1] for row in wide_columns) == pipeline.WIDE_COLUMNS
    assert all(row[2] == "REAL" and row[3] == 0 for row in wide_columns[4:])
    long_fks = seeded.execute("PRAGMA foreign_key_list(measurements_long)").fetchall()
    assert {(r[2], r[3], r[4]) for r in long_fks} == {
        ("series_catalog", "well_id", "well_id"), ("series_catalog", "parameter", "parameter"),
        ("measurements_wide", "well_id", "well_id"), ("measurements_wide", "timestamp", "timestamp"),
    }
    assert [r[2] for r in seeded.execute("PRAGMA index_info(idx_measurements_long_series)")] == [
        "well_id", "parameter", "timestamp",
    ]
    assert seeded.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert seeded.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        seeded.execute(
            "INSERT INTO measurements_long VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NK-7", "2026-01-11 00:00:00", "pump_fillage", 23.0, "%", 1, "", "test", 1),
        )


def test_csv_structure_and_raw_fields():
    header = ",".join(CSV_COLUMNS).encode("ascii") + b"\n"
    frame = _read_csv(header + b'NK-7,sucker_rod_pump,bad,pump_fillage,00.0,%,2,"one;two"\n', "train.csv", "train")
    assert tuple(frame.columns) == RAW_COLUMNS
    assert tuple(frame.iloc[0]) == (
        "train.csv", 1, "NK-7", "sucker_rod_pump", "bad", "pump_fillage", "00.0", "%", "2", "one;two", "train"
    )
    with pytest.raises(ValueError, match="header"):
        _read_csv(b"invalid\n", "train.csv", "train")
    with pytest.raises(ValueError, match="record 1"):
        _read_csv(header + b"a,b\n", "train.csv", "train")
    with pytest.raises(ValueError, match="malformed CSV"):
        _read_csv(header + b'"unclosed', "train.csv", "train")


def test_hash_failure_preserves_database(tmp_path):
    (tmp_path / "train.csv").write_bytes(b"wrong bytes")
    dest = tmp_path / "wells.sqlite"
    dest.write_bytes(b"existing database")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_inputs(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pipeline.build_database(tmp_path, dest)
    assert dest.read_bytes() == b"existing database"


def test_validation_and_pivot(seeded, tmp_path):
    _, catalog, meta = fixture_inputs(seeded)
    catalog = pd.concat([catalog, pd.DataFrame([
        ("NK-7", "min_rod_weight", "sucker_rod_pump", "kg"),
        ("NK-7", "max_rod_weight", "sucker_rod_pump", "kg"),
    ], columns=pipeline.CATALOG_COLUMNS)], ignore_index=True)

    def record(number, *, hour="2026-01-01 00:00:00", parameter="pump_fillage",
               value="40", unit="%", count="1", flag="", well="NK-7", well_type="sucker_rod_pump"):
        return ("train.csv", number, well, well_type, hour, parameter, value, unit, count, flag, "train")

    raw = pd.DataFrame([
        record(2, value="0.00"), record(1, value="0"),
        record(3, hour="2026-01-01 01:00:00"), record(4, hour="2026-01-01 01:00:00", value="41"),
        record(5, parameter="SPM", unit="strokes/min", value="3", flag="first;second;first"),
        record(6, parameter="min_rod_weight", unit="kg", value="50"),
        record(7, parameter="max_rod_weight", unit="kg", value="49"),
        record(8, hour="2026-01-01 02:00:00", value="101"),
        record(9, hour="2026-02-30 00:00:00", value="inf", count="1.5", unit="atm", well_type="fountain"),
        record(10, hour=meta["test_start"]),
        record(11, well="unknown"),
        record(12, parameter="line_pressure"),
        record(13, parameter="other"),
    ], columns=RAW_COLUMNS)
    long, issues = validate_measurements(raw, catalog, meta)
    reasons = issues.groupby("source_row")["issue_code"].agg(set).to_dict()
    assert reasons == {
        2: {"duplicate_exact"}, 3: {"duplicate_conflict"}, 4: {"duplicate_conflict"},
        5: {"first", "second"}, 6: {"reversed_rod_load"}, 7: {"reversed_rod_load"},
        8: {"pump_fillage_out_of_range"},
        9: {"invalid_timestamp", "invalid_value", "invalid_n_obs", "wrong_unit", "wrong_well_type"},
        10: {"outside_split"}, 11: {"unknown_well", "inapplicable_parameter"},
        12: {"inapplicable_parameter"}, 13: {"unknown_parameter"},
    }
    assert issues.loc[issues["source_row"].eq(9), "timestamp"].isna().all()
    assert long.loc[long["parameter"].eq("SPM"), "flag"].item() == "first;second;first"
    wide = pipeline.build_wide(long, catalog, meta)
    assert len(wide) == 480 and not wide.duplicated(["well_id", "timestamp"]).any()
    start_rows = wide[wide["timestamp"].eq(meta["train_start"])].set_index("well_id")
    assert start_rows.loc["NK-7", "pump_fillage"] == 0
    assert start_rows.loc["NK-7", ["SPM", "min_rod_weight", "max_rod_weight"]].isna().all()
    assert pd.isna(start_rows.loc["NK-76", "casing_pressure"])
    assert wide.loc[wide["timestamp"].eq(meta["test_start"]), "split"].tolist() == ["test", "test"]
    path = tmp_path / "invalid-records.sqlite"
    pipeline.assemble_database(raw, catalog, meta, path)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM measurements_raw").fetchone() == (len(raw),)
        assert conn.execute("""
            SELECT r.timestamp, r.value, r.n_obs, q.timestamp
            FROM quality_issues q JOIN measurements_raw r USING (source_file, source_row)
            WHERE q.issue_code='invalid_timestamp'
        """).fetchall() == [("2026-02-30 00:00:00", "inf", "1.5", None)]


def test_failed_verification_leaves_prior_database(tmp_path, seeded, monkeypatch):
    raw, catalog, meta = fixture_inputs(seeded)
    dest = tmp_path / "wells.sqlite"
    pipeline.assemble_database(raw, catalog, meta, dest)
    prior = dest.read_bytes()

    def fail(*args):
        raise ValueError("simulated verification failure")

    monkeypatch.setattr(pipeline, "_verify_database", fail)
    with pytest.raises(ValueError, match="simulated verification failure"):
        pipeline.assemble_database(raw, catalog, meta, dest)
    assert dest.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [dest]


def test_gap_filters_and_coverage(seeded):
    result = ("NK-76", "line_pressure", "2026-01-01 01:00:00", "2026-01-01 04:00:00", 3, 2)
    filters = dict(well_id="NK-76", parameter="line_pressure",
                   start_ts="2026-01-01 03:00:00", end_ts="2026-01-01 04:00:00")
    assert sql_rows(seeded, "gaps", **filters) == [result]
    assert sql_rows(seeded, "gaps", **(filters | {"parameter": None})) == [
        (result[0], None, *result[2:]),
    ]
    assert sql_rows(seeded, "gaps", **(filters | {"start_ts": filters["end_ts"], "end_ts": "2026-01-02"})) == []
    assert sql_rows(seeded, "gaps", **(filters | {"start_ts": "2025-12-31", "end_ts": "2026-01-01 02:00:00"})) == []
    spm = filters | {"well_id": "NK-7", "parameter": "SPM"}
    assert sql_rows(seeded, "gaps", **(spm | {"start_ts": "2026-01-01", "end_ts": "2026-01-02"})) == []
    whole_day = spm | {"start_ts": "2026-01-01", "end_ts": "2026-01-02", "grain": "day"}
    raw = sql_rows(seeded, "aggregates", **whole_day, mode="raw")[0]
    clean = sql_rows(seeded, "aggregates", **whole_day, mode="clean")[0]
    assert raw[4:] == ("2026-01-01", 24, 2, 1, 1, 2, 5.0, 0.0, 10.0, 2 / 24)
    assert clean[4:] == ("2026-01-01", 24, 2, 1, 1, 1, 0.0, 0.0, 0.0, 2 / 24)
    partial = filters | {"grain": "month", "mode": "clean", "end_ts": "2026-01-01 05:00:00"}
    assert sql_rows(seeded, "aggregates", **partial)[0][4:] == (
        "2026-01-01", 2, 1, 1, 0, 1, 16.0, 16.0, 16.0, 0.5,
    )
    test_slice = partial | {"start_ts": "2026-01-08", "end_ts": "2026-01-11"}
    assert sql_rows(seeded, "aggregates", **test_slice)[0][5:] == (72, 0, 0, 0, 0, None, None, None, 0.0)
    assert sql_rows(seeded, "aggregates", **(test_slice | {"parameter": "casing_pressure"})) == []


@pytest.mark.skipif(os.environ.get("WELLS_RUN_OFFICIAL") != "1", reason="set WELLS_RUN_OFFICIAL=1 for pinned CSV checks")
def test_official_data(tmp_path):
    raw_dir = Path(os.environ.get("WELLS_RAW_DIR", "data/raw"))
    db_path = tmp_path / "official.sqlite"
    pipeline.build_database(raw_dir, db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT split, count(*) FROM measurements_raw GROUP BY split").fetchall() == [
            ("test", 51_681), ("train", 206_706),
        ]
        for table, expected in (("series_catalog", 43), ("measurements_wide", 65_520), ("measurements_long", 258_387)):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (expected,)
        assert conn.execute("SELECT count(DISTINCT timestamp) FROM measurements_long").fetchone() == (6_552 - 83,)
        assert conn.execute("SELECT count(*) FROM measurements_long WHERE flag<>'' AND is_usable=0").fetchone() == (7_041,)
        cases = dict(conn.execute("SELECT issue_code, count(*) FROM quality_issues GROUP BY issue_code"))
        assert cases == {
            "weight_overflow": 4098, "area_negative": 2921, "area_extreme": 97, "spm_outlier": 2,
            "reversed_rod_load": 26, "area_scale_shift": 8, "nk7_regime_change": 3,
        }
        assert conn.execute("""
            SELECT count(*) FROM (
                SELECT well_id, timestamp FROM quality_issues WHERE issue_code='reversed_rod_load'
                GROUP BY well_id, timestamp HAVING count(*)=2
            )
        """).fetchone() == (13,)
        assert conn.execute("""
            SELECT timestamp, end_ts FROM quality_issues
            WHERE well_id='NK-82' AND issue_code='area_scale_shift'
            ORDER BY timestamp
        """).fetchall() == [
            ("2026-05-01 00:00:00", "2026-06-01 00:00:00"),
            ("2026-05-08 00:00:00", "2026-05-09 00:00:00"),
        ]
        gaps = sql_rows(conn, "gaps", well_id="NK-86", parameter=None,
                        start_ts="2026-01-01", end_ts="2026-10-01")
        assert max(gaps, key=lambda row: row[-1])[2:] == ("2026-04-29 14:00:00", "2026-05-12 16:00:00", 314, 313)
        selected_test = dict(start_ts="2026-08-08 23:00:00", end_ts="2026-10-01", grain="month", mode="clean")
        rows = sql_rows(conn, "aggregates", well_id="NK-75", parameter="line_pressure", **selected_test)
        assert sum(row[5] for row in rows) == 1273
        assert all(row[6:13] == (0, 0, 0, 0, None, None, None) for row in rows)
        assert sql_rows(conn, "aggregates", well_id="NK-76", parameter="casing_pressure", **selected_test) == []
        source = pd.read_sql_query(
            "SELECT well_id, parameter, timestamp, value FROM measurements_long "
            "WHERE parameter='dynamometer_area' OR well_id='NK-7'", conn,
        )
    summary = summarize_anomalies(source)
    may8 = summary[summary["comparison"].eq("nk82_may8_detail")].set_index("period")
    assert may8.loc["before", "median_value"] == pytest.approx(12428, abs=1)
    assert may8.loc["after", "median_value"] == pytest.approx(207035, abs=1)
    nk7 = summary[summary["comparison"].eq("nk7_regime_change")].set_index(["parameter", "period"])
    assert nk7.loc[("dynamometer_area", "after"), "n_negative"] == 1510
    assert nk7.loc[("dynamometer_area", "after"), "n_observed"] == 2695
    assert nk7.loc[("SPM", "before"), "median_value"] == pytest.approx(4.16, abs=0.01)
    assert nk7.loc[("SPM", "after"), "median_value"] == pytest.approx(1.40, abs=0.01)
    assert nk7.loc[("pump_fillage", "before"), "median_value"] == pytest.approx(31.20, abs=0.01)
    assert nk7.loc[("pump_fillage", "after"), "median_value"] == pytest.approx(44.64, abs=0.01)
