import math
import os
import sqlite3
from contextlib import closing
from importlib import import_module
from pathlib import Path

import pytest

from test_data import FIXTURE, fixture_inputs
from wells.data import pipeline

DATA_TABLES = ("dataset_meta", "series_catalog", "measurements_raw", "measurements_wide",
               "measurements_long", "quality_issues")
RESULT_TABLES = ("backtest_predictions", "metrics", "forecast_path")
MODELS = ("naive", "seasonal_naive", "ridge")


def _seed(conn):
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((pipeline.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executescript(FIXTURE.read_text(encoding="utf-8"))
    return fixture_inputs(conn)


def _assert_same_tables(first, second, tables):
    for table in tables:
        columns = first.execute(f"PRAGMA table_info({table})").fetchall()
        keys = [r[1] for r in sorted(columns, key=lambda row: row[5]) if r[5]]
        sql = f"SELECT * FROM {table} ORDER BY {', '.join(keys)}"
        for left, right in zip(first.execute(sql), second.execute(sql), strict=True):
            for column, a, b in zip(columns, left, right, strict=True):
                if column[2] == "REAL" and a is not None and b is not None:
                    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-9), (table, column[1], a, b)
                else:
                    assert a == b, (table, column[1], a, b)


def _consumer(module):
    # Integration handoff. These imports resolve to the owners' modules as soon as they land.
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name in {module, module.rsplit(".", 1)[0]}:
            pytest.skip(f"Integration pending: {module} has not been delivered")
        raise


def _assert_same_report(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_report(left[key], right[key])
    elif isinstance(left, list):
        for a, b in zip(left, right, strict=True):
            _assert_same_report(a, b)
    elif isinstance(left, float):
        assert math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-9)
    else:
        assert left == right


def test_fixture_rebuild_matches_seed(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with closing(sqlite3.connect(":memory:")) as seed:
        raw, catalog, meta = _seed(seed)
        for _ in range(2):
            pipeline.assemble_database(raw, catalog, meta, path)
            with closing(sqlite3.connect(path)) as rebuilt:
                _assert_same_tables(seed, rebuilt, DATA_TABLES)
                for table in RESULT_TABLES:
                    assert rebuilt.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
                assert rebuilt.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fixture_data_to_forecast_to_ui(tmp_path):
    forecast = _consumer("wells.forecast.run")
    ui = _consumer("wells.ui.queries")
    with closing(sqlite3.connect(":memory:")) as seed:
        raw, catalog, meta = _seed(seed)
    path = tmp_path / "fixture.sqlite"
    pipeline.assemble_database(raw, catalog, meta, path)
    report = forecast.run_forecast(
        path, validation_start="2026-01-05 00:00:00", nk7_regime_start="2026-01-02 00:00:00",
    )

    with closing(ui.open_database(path)) as opened:
        assert ui.read_meta(opened) == meta
        selection = dict(well_id="NK-76", parameter="tubing_pressure",
                         start_ts="2026-01-01 00:00:00", end_ts="2026-01-01 03:00:00", mode="clean")
        history = ui.read_history(opened, **selection)
        assert history["display_value"].iloc[:2].tolist() == [0.0, 10.0]
        assert history["display_value"].isna().tolist() == [False, False, True]
        target = dict(well_id="NK-7", parameter="pump_fillage", models=MODELS)
        metrics = ui.read_metrics(opened, **target)
        release = ui.read_forecast_path(opened, **target)
        assert len(metrics) == len(release) == 72
        assert set(metrics["n_actual"]) == {72}
        assert report["release_origins"]["NK-7"] == meta["test_end"]
        assert set(release.loc[release["horizon_h"].eq(24) & release["model"].ne("ridge"), "y_pred"]) == {62.0}
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            opened.execute("DELETE FROM dataset_meta")


@pytest.mark.skipif(os.environ.get("WELLS_RUN_OFFICIAL") != "1", reason="set WELLS_RUN_OFFICIAL=1 for pinned CSV checks")
def test_official_repeat_with_consumers(tmp_path):
    forecast = _consumer("wells.forecast.run")
    ui = _consumer("wells.ui.queries")
    raw_dir = Path(os.environ.get("WELLS_RAW_DIR", "data/raw"))
    paths = [tmp_path / "first.sqlite", tmp_path / "second.sqlite"]
    reports = []
    for path in paths:
        pipeline.build_database(raw_dir, path)
        reports.append(forecast.run_forecast(path))
    _assert_same_report(*reports)
    with closing(ui.open_database(paths[0])) as first, closing(ui.open_database(paths[1])) as second:
        _assert_same_tables(first, second, (*DATA_TABLES, *RESULT_TABLES))
        meta = ui.read_meta(first)
        gaps = ui.read_gaps(first, well_id="NK-86", parameter=None,
                            start_ts=meta["train_start"], end_ts="2026-10-01 00:00:00")
        assert gaps["missing_hours"].max() == 313
        pressure = ui.read_aggregates(first, well_id="NK-75", parameter="line_pressure",
                                      start_ts=meta["test_start"], end_ts="2026-10-01 00:00:00",
                                      grain="month", mode="clean")
        assert pressure["n_expected"].sum() == 1273 and pressure["n_observed"].sum() == 0
        target = dict(well_id="NK-91", parameter="pump_fillage", models=MODELS)
        assert len(ui.read_metrics(first, **target)) == len(ui.read_forecast_path(first, **target)) == 72
        assert reports[0]["table_counts"]["metrics"] == 504
        assert reports[0]["release_origins"]["NK-91"] == "2026-09-29 09:00:00"
