"""Checks for the analysis scripts on the canonical fixture."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from wells.analysis import figures, sql_report
from wells.analysis.benchmark import measure

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "wells.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript((ROOT / "wells/data/sql/schema.sql").read_text(encoding="utf-8"))
        conn.executescript((ROOT / "tests/fixtures/seed.sql").read_text(encoding="utf-8"))
    return path


def test_every_query_runs_and_is_saved(db, tmp_path):
    with closing(sql_report.open_readonly(db)) as conn:
        results = sql_report.run_queries(conn)
    assert len(results) == len(sql_report.QUERIES) >= 8
    # Jan 1 line pressure of NK-76 has one interior two-hour gap (01:00 -> 04:00)
    gaps = results["04_longest_gaps"]
    assert gaps.loc[gaps["well_id"] == "NK-76", "missing_hours"].tolist() == [2]
    split = results["01_split_boundaries"].set_index("split")
    assert split.loc["train", "last_hour"] < split.loc["test", "first_hour"]
    flagged = results["03_flagged_series"]
    assert flagged[["well_id", "parameter", "n_flagged"]].values.tolist() == [["NK-7", "SPM", 1]]

    target = sql_report.write_report(results, tmp_path / "out")
    assert target.read_text(encoding="utf-8").count("```sql") == len(results)
    assert len(list((tmp_path / "out" / "sql").glob("*.csv"))) == len(results)


def test_missing_database_is_reported(tmp_path, capsys):
    missing = tmp_path / "absent.sqlite"
    assert sql_report.main(["--db", str(missing), "--out", str(tmp_path)]) == 1
    assert not missing.exists()
    assert "database not found" in capsys.readouterr().err


def test_pressure_figure_is_removed_when_pressure_results_are_missing(db, tmp_path):
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executemany(
            "INSERT INTO metrics VALUES ('NK-76', 'tubing_pressure', 1, ?, 1.0, 1, 2, 0.5)",
            [(model,) for model in ("naive", "seasonal_naive", "ridge")],
        )
        pump_figure = figures.mae_by_horizon(conn, tmp_path)
        pressure_figure = figures.pressure_mae_by_horizon(conn, tmp_path)
        assert pressure_figure.is_file()

        conn.execute("DELETE FROM metrics WHERE parameter = 'tubing_pressure'")
        assert figures.pressure_mae_by_horizon(conn, tmp_path) is None
        assert not pressure_figure.exists()
        assert pump_figure.is_file()


def test_measure_returns_result_time_and_memory():
    result, seconds, peak_mb = measure(lambda: [0] * 100_000)
    assert len(result) == 100_000 and seconds >= 0 and peak_mb > 0
