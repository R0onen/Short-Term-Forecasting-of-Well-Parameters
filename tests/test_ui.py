import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wells.data import pipeline
from wells.ui import charts, eda, queries

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "seed.sql"
MODELS = queries.MODELS
DAY = dict(start_ts="2026-01-01 00:00:00", end_ts="2026-01-02 00:00:00")


@pytest.fixture(scope="module")
def seeded_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("ui") / "fixture.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript((pipeline.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.executescript(FIXTURE.read_text(encoding="utf-8"))
    return path


@pytest.fixture
def db_copy(seeded_path, tmp_path):
    """A disposable copy for tests that change result tables or the schema."""
    path = tmp_path / "copy.sqlite"
    shutil.copy(seeded_path, path)
    return path


@pytest.fixture
def conn(seeded_path):
    with closing(queries.open_database(seeded_path)) as opened:
        yield opened


def _edit(path, *statements):
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA foreign_keys = ON")
        with writer:
            for statement in statements:
                writer.execute(statement)


# ---------------------------------------------------------------- queries

def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "absent" / "wells.sqlite"
    with pytest.raises(FileNotFoundError):
        queries.open_database(path)
    assert not path.exists() and not path.parent.exists()


def test_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM dataset_meta")


def test_incompatible_schema_is_reported(db_copy):
    _edit(db_copy, "DROP TABLE forecast_path")
    with pytest.raises(queries.IncompatibleDatabaseError, match="forecast_path"):
        queries.open_database(db_copy)


def test_catalog_does_not_depend_on_observations(conn):
    catalog = queries.read_catalog(conn)
    assert list(catalog.columns) == ["well_id", "parameter", "well_type", "unit"]
    assert ("NK-76", "casing_pressure") not in set(zip(catalog["well_id"], catalog["parameter"]))
    assert set(catalog.loc[catalog["well_id"].eq("NK-76"), "parameter"]) == {"line_pressure", "tubing_pressure"}
    assert set(catalog["well_type"]) == {"fountain", "sucker_rod_pump"}
    assert queries.read_meta(conn)["test_end"] == "2026-01-10 23:00:00"


def test_history_keeps_zero_masks_flag_and_respects_half_open_end(conn):
    clean = queries.read_history(conn, well_id="NK-7", parameter="SPM", mode="clean",
                                 start_ts="2026-01-01 00:00:00", end_ts="2026-01-01 03:00:00")
    assert list(clean.columns) == queries.HISTORY_COLUMNS
    assert clean["timestamp"].dt.strftime("%H").tolist() == ["00", "01", "02"]
    assert clean["value"].iloc[:2].tolist() == [0.0, 10.0]
    assert clean["display_value"].iloc[0] == 0.0 and clean["display_value"].iloc[1:].isna().all()
    assert clean["flag"].iloc[1] == "spm_outlier" and clean["is_usable"].iloc[1] == 0
    raw = queries.read_history(conn, well_id="NK-7", parameter="SPM", mode="raw",
                               start_ts="2026-01-01 00:00:00", end_ts="2026-01-01 03:00:00")
    assert raw["display_value"].iloc[:2].tolist() == [0.0, 10.0]


def test_inclusive_end_date_becomes_next_midnight(conn):
    end_ts = queries.to_label(pd.Timestamp("2026-01-01") + pd.Timedelta(days=1))
    history = queries.read_history(conn, well_id="NK-7", parameter="pump_fillage", mode="clean",
                                   start_ts="2026-01-01 00:00:00", end_ts=end_ts)
    assert len(history) == 24 and history["timestamp"].max() == pd.Timestamp("2026-01-01 23:00:00")


def test_applicable_sensor_without_rows_stays_on_calendar(conn):
    window = dict(start_ts="2026-01-08 00:00:00", end_ts="2026-01-11 00:00:00")
    history = queries.read_history(conn, well_id="NK-76", parameter="line_pressure", mode="clean", **window)
    assert len(history) == 72 and history["value"].isna().all()
    totals = queries.read_aggregates(conn, well_id="NK-76", parameter="line_pressure", grain="month",
                                     mode="clean", **window)
    assert (totals["n_expected"].sum(), totals["n_observed"].sum()) == (72, 0)
    assert queries.read_history(conn, well_id="NK-76", parameter="casing_pressure", mode="raw", **window).empty


def test_bound_values_are_not_interpolated(conn):
    hostile = "NK-7' OR '1'='1"
    assert queries.read_history(conn, well_id=hostile, parameter="SPM", mode="raw", **DAY).empty
    assert queries.read_well_matrix(conn, well_id=hostile, mode="raw", **DAY).columns.empty
    with pytest.raises(ValueError):
        queries.read_aggregates(conn, well_id=None, parameter=None, grain="week", mode="raw", **DAY)
    with pytest.raises(ValueError):
        queries.read_metrics(conn, well_id="NK-7", parameter="pump_fillage", models=["drop table"])


def test_well_matrix_modes(conn):
    window = dict(start_ts="2026-01-01 00:00:00", end_ts="2026-01-01 06:00:00")
    raw = queries.read_well_matrix(conn, well_id="NK-7", mode="raw", **window)
    clean = queries.read_well_matrix(conn, well_id="NK-7", mode="clean", **window)
    assert list(raw.columns) == list(clean.columns) == ["SPM", "pump_fillage"]
    assert raw.index.equals(clean.index) and len(raw) == 6
    assert raw.loc["2026-01-01 01:00:00", "SPM"] == 10.0 and np.isnan(clean.loc["2026-01-01 01:00:00", "SPM"])


def test_gaps_report_full_interval_that_overlaps_selection(conn):
    selection = dict(well_id="NK-76", start_ts="2026-01-01 03:00:00", end_ts="2026-01-01 04:00:00")
    gaps = queries.read_gaps(conn, parameter="line_pressure", **selection)
    assert list(gaps.columns) == queries.GAP_COLUMNS
    assert gaps[["previous_ts", "timestamp"]].iloc[0].tolist() == [
        pd.Timestamp("2026-01-01 01:00:00"), pd.Timestamp("2026-01-01 04:00:00")]
    assert gaps["missing_hours"].tolist() == [2]
    assert queries.read_gaps(conn, parameter=None, **selection)["parameter"].isna().all()
    assert queries.read_gaps(conn, well_id="NK-76", parameter="line_pressure",
                             start_ts="2026-01-01 04:00:00", end_ts="2026-01-01 05:00:00").empty


def test_issues_join_raw_source(conn):
    issues = queries.read_issues(conn, well_id="NK-7", parameter="SPM", **DAY)
    assert list(issues.columns) == queries.ISSUE_COLUMNS
    row = issues.iloc[0]
    assert (row["issue_code"], row["action"], row["raw_value"], row["raw_flag"]) == (
        "spm_outlier", "mask", "10.0", "spm_outlier")
    assert queries.read_issues(conn, well_id="NK-7", parameter="SPM",
                               start_ts="2026-01-01 02:00:00", end_ts="2026-01-02 00:00:00").empty


def test_issue_annotations_overlap_and_unplaced_rejects(db_copy):
    _edit(db_copy,
          "INSERT INTO quality_issues (well_id, parameter, timestamp, end_ts, issue_code, action) VALUES "
          "('NK-7', 'SPM', '2026-01-02 00:00:00', NULL, 'nk7_regime_change', 'annotate')",
          "INSERT INTO quality_issues (well_id, parameter, timestamp, source_file, source_row, issue_code, action) "
          "VALUES (NULL, NULL, NULL, 'train.csv', 1, 'invalid_timestamp', 'reject')")
    with closing(queries.open_database(db_copy)) as opened:
        late = queries.read_issues(opened, well_id="NK-7", parameter="SPM",
                                   start_ts="2026-01-05 00:00:00", end_ts="2026-01-06 00:00:00")
        assert set(late["issue_code"]) == {"nk7_regime_change", "invalid_timestamp"}
        unplaced = late.loc[late["timestamp"].isna()].iloc[0]
        assert (unplaced["source_file"], unplaced["raw_well_id"]) == ("train.csv", "NK-7")
        early = queries.read_issues(opened, well_id="NK-7", parameter="SPM",
                                    start_ts="2026-01-01 00:00:00", end_ts="2026-01-01 12:00:00")
        assert "nk7_regime_change" not in set(early["issue_code"])
        assert len(queries.read_annotations(opened)) == 1


def test_issue_with_timestamp_and_unknown_well_remains_visible(db_copy, monkeypatch):
    _edit(db_copy,
          "UPDATE measurements_raw SET well_id = 'UNKNOWN-1' WHERE source_file = 'train.csv' AND source_row = 1",
          "INSERT INTO quality_issues (well_id, parameter, timestamp, source_file, source_row, issue_code, action) "
          "VALUES (NULL, 'SPM', '2026-01-01 01:00:00', 'train.csv', 1, 'unknown_well', 'reject')")
    with closing(queries.open_database(db_copy)) as opened:
        issues = queries.read_issues(opened, well_id="NK-7", parameter="SPM", **DAY)
        unknown = issues.loc[issues["issue_code"].eq("unknown_well")].iloc[0]
        assert pd.isna(unknown["well_id"]) and unknown["raw_well_id"] == "UNKNOWN-1"
        outside = queries.read_issues(opened, well_id="NK-7", parameter="SPM",
                                      start_ts="2026-01-02 00:00:00", end_ts="2026-01-03 00:00:00")
        assert outside.empty
    app = _app(db_copy, monkeypatch)
    assert "без нормализованного часа или известной скважины" in _texts(app, "markdown")


def test_forecast_reads_use_shared_keys(conn):
    target = dict(well_id="NK-7", parameter="pump_fillage")
    metrics = queries.read_metrics(conn, models=MODELS, **target)
    assert list(metrics.columns) == queries.METRIC_COLUMNS
    assert metrics.loc[metrics["horizon_h"].eq(24), "mae"].isna().all()
    release = queries.read_forecast_path(conn, models=["naive"], **target)
    assert len(release) == 24 and release["origin_ts"].nunique() == 1
    assert release["target_ts"].is_monotonic_increasing
    inside = queries.read_backtest(conn, horizon_h=1, models=MODELS, start_ts="2026-01-08 00:00:00",
                                   end_ts="2026-01-08 01:00:00", **target)
    assert sorted(inside["model"]) == sorted(MODELS) and set(inside["y_true"]) == {37.0}
    outside = queries.read_backtest(conn, horizon_h=1, models=MODELS, **DAY, **target)
    assert outside.empty
    # A date window outside test does not touch the official metrics.
    assert len(queries.read_metrics(conn, models=MODELS, **target)) == 6
    assert queries.read_forecast_path(conn, models=[], **target).empty


# ---------------------------------------------------------------- EDA

def _two_series():
    hours = pd.date_range("2026-03-01", periods=60, freq="h")
    x = pd.Series(np.arange(60, dtype=float), index=hours)
    y = pd.Series((np.arange(60) ** 2).astype(float), index=hours)
    x.iloc[[3, 10, 11]] = np.nan
    y.iloc[[5, 40]] = np.nan
    z = pd.Series(np.where(np.arange(60) % 2, 1.0, 2.0), index=hours)
    return pd.DataFrame({"x": x, "y": y, "z": z, "constant": 5.0})


def test_pairwise_spearman_uses_only_shared_hours():
    matrix = _two_series()
    result = eda.pairwise_spearman(matrix).set_index(["parameter_x", "parameter_y"])
    assert list(eda.CORRELATION_COLUMNS) == ["parameter_x", "parameter_y", "rho", "n_pairs"]
    assert result.loc[("x", "y"), "n_pairs"] == 55 and result.loc[("x", "y"), "rho"] == pytest.approx(1.0)
    expected = matrix[["x", "z"]].dropna()
    hand = np.corrcoef(expected["x"].rank(), expected["z"].rank())[0, 1]
    assert result.loc[("x", "z"), "rho"] == pytest.approx(hand)
    assert np.isnan(result.loc[("x", "constant"), "rho"]) and result.loc[("x", "constant"), "n_pairs"] == 57
    small = eda.pairwise_spearman(matrix.iloc[:20]).set_index(["parameter_x", "parameter_y"])
    assert np.isnan(small.loc[("x", "y"), "rho"]) and small.loc[("x", "y"), "n_pairs"] == 16


def test_describe_and_profile_match_hand_calculation():
    hours = pd.date_range("2026-03-01", periods=48, freq="h")
    values = np.r_[np.arange(24, dtype=float), np.arange(24, dtype=float) + 10]
    history = pd.DataFrame({"well_id": "W", "parameter": "p", "unit": "u", "timestamp": hours,
                            "display_value": values})
    history.loc[47, "display_value"] = np.nan
    stats = eda.describe_observations(history).iloc[0]
    present = values[:47]
    assert stats["count"] == 47 and stats["median"] == pytest.approx(np.median(present))
    assert stats["std"] == pytest.approx(np.std(present, ddof=1))
    assert stats["q25"] == pytest.approx(np.quantile(present, 0.25))
    profile = eda.hourly_profile(history).set_index("hour")
    assert profile.loc[5, ["median", "q25", "q75", "n"]].tolist() == [10.0, 7.5, 12.5, 2]
    assert profile.loc[23, "n"] == 1 and eda.hourly_profile(history.iloc[0:0]).empty


def test_type_coverage_weights_by_expected_hours():
    aggregates = pd.DataFrame({
        "well_id": ["A", "B", "C"], "parameter": ["p", "p", "q"],
        "well_type": ["fountain", "fountain", "sucker_rod_pump"],
        "period_start": pd.Timestamp("2026-01-01"),
        "n_expected": [24, 6, 10], "n_observed": [24, 0, 5], "n_usable": [20, 0, 5], "n_flagged": [4, 0, 0],
    })
    by_type = eda.coverage_by_type(aggregates).set_index("well_type")
    assert by_type.loc["fountain", "coverage"] == pytest.approx(24 / 30)  # not the mean of 1.0 and 0.0
    assert by_type.loc["fountain", "n_series"] == 2 and by_type.loc["fountain", "n_flagged"] == 4
    wells = pd.DataFrame({"well_id": ["A", "B", "C", "D"],
                          "well_type": ["fountain", "fountain", "sucker_rod_pump", "fountain"]})
    by_well = eda.coverage_by_well(aggregates, wells).set_index("well_id")
    assert by_well.loc["B", "coverage"] == 0 and bool(by_well.loc["B", "applicable"])
    assert not by_well.loc["D", "applicable"] and pd.isna(by_well.loc["D", "coverage"])


def test_eda_on_fixture_covers_both_types_and_modes(conn):
    window = dict(start_ts="2026-01-01 00:00:00", end_ts="2026-01-11 00:00:00")
    for mode, spm_count in (("raw", 2), ("clean", 1)):
        observations = queries.read_observations(conn, mode=mode, **window)
        stats = eda.describe_observations(observations, by=("well_type", "well_id", "parameter", "unit"))
        assert set(stats["well_type"]) == {"fountain", "sucker_rod_pump"}
        assert stats.set_index("parameter").loc["SPM", "count"] == spm_count
    coverage = eda.coverage_by_type(queries.read_aggregates(conn, well_id=None, parameter=None, grain="month",
                                                            mode="raw", **window)).set_index("well_type")
    assert coverage.loc["fountain", "n_expected"] == 2 * 240
    assert coverage.loc["fountain", "n_observed"] == 5


# ---------------------------------------------------------------- figures

def test_figures_render_with_hover_and_states(conn):
    history = queries.read_history(conn, well_id="NK-7", parameter="SPM", mode="clean", **DAY)
    aggregates = queries.read_aggregates(conn, well_id="NK-7", parameter="SPM", grain="day", mode="clean", **DAY)
    issues = queries.read_issues(conn, well_id="NK-7", parameter="SPM", **DAY)
    figure = charts.history_figure(history, aggregates, issues)
    names = [trace.name for trace in figure.data]
    assert "Флаг качества" in names and figure.data[0].connectgaps is False
    assert "n_obs" in figure.data[0].hovertemplate and "strokes/min" in figure.data[0].hovertemplate

    profile = charts.profile_figure(eda.hourly_profile(history))
    assert "наблюдений" in profile.data[-1].hovertemplate

    matrix = queries.read_well_matrix(conn, well_id="NK-7", mode="raw", **DAY)
    correlation = charts.correlation_figure(eda.pairwise_spearman(matrix))
    assert "совпавших часов" in correlation.data[0].hovertext[0][1]

    catalog = queries.read_catalog(conn)
    sensor = queries.read_aggregates(conn, well_id=None, parameter="tubing_pressure", grain="day", mode="raw", **DAY)
    coverage = charts.coverage_figure(eda.coverage_by_well(sensor, catalog[["well_id", "well_type"]].drop_duplicates()))
    assert any("неприменимо" in label for label in coverage.data[-1].y)


def test_forecast_figure_modes(conn):
    meta = queries.read_meta(conn)
    target = dict(well_id="NK-7", parameter="pump_fillage")
    metrics = queries.read_metrics(conn, models=MODELS, **target)
    window = dict(start_ts="2026-01-08 00:00:00", end_ts="2026-01-09 00:00:00")
    actuals = queries.read_history(conn, mode="clean", **target, **window)
    predictions = queries.read_backtest(conn, horizon_h=1, models=MODELS, **target, **window)
    bounds = dict(archive_end=meta["test_end"], test_start=meta["test_start"], test_end=meta["test_end"])
    backtest = charts.forecast_figure(actuals, predictions, metrics, mode="backtest", origin_ts=None, horizon_h=1, **bounds)
    assert {"Факт", "Naive", "Ridge"} <= {trace.name for trace in backtest.data}
    texts = [annotation.text for annotation in backtest.layout.annotations]
    assert any(text.startswith("Официальный test, 2026-01-08 00:00") for text in texts)

    release = queries.read_forecast_path(conn, models=MODELS, **target)
    figure = charts.forecast_figure(actuals, release, metrics, mode="release",
                                    origin_ts=release["origin_ts"].iloc[0], **bounds)
    assert any("час выпуска" in text for text in (a.text for a in figure.layout.annotations))
    empty = charts.forecast_figure(actuals.iloc[0:0], predictions.iloc[0:0], metrics.iloc[0:0], mode="backtest",
                                   origin_ts=None, **bounds)
    assert any("Нет опубликованной оценки" in a.text for a in empty.layout.annotations)


# ---------------------------------------------------------------- Streamlit states

def _app(path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setattr(sys, "argv", ["app.py", "--db", str(path)])
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def _texts(app, kind):
    return " ".join(str(element.value) for element in getattr(app, kind))


def test_app_missing_database(tmp_path, monkeypatch):
    path = tmp_path / "missing.sqlite"
    app = _app(path, monkeypatch)
    assert "не найден" in _texts(app, "error") and "wells.data.pipeline" in _texts(app, "code")
    assert not path.exists() and len(app.selectbox) == 0


def test_app_incompatible_database(db_copy, monkeypatch):
    _edit(db_copy, "DROP TABLE metrics")
    app = _app(db_copy, monkeypatch)
    assert "Несовместимая" in _texts(app, "error")


def test_app_filters_follow_catalog(seeded_path, monkeypatch):
    app = _app(seeded_path, monkeypatch)
    assert app.selectbox(key="well_type").value == "sucker_rod_pump"
    assert app.selectbox(key="parameter").value == "pump_fillage"
    assert app.session_state["dates"][1] == pd.Timestamp("2026-01-10").date()
    app.selectbox(key="well_type").set_value("fountain").run()
    assert app.selectbox(key="well_id").value == "NK-76"
    assert app.selectbox(key="parameter").options == ["line_pressure", "tubing_pressure"]
    assert "Прогнозы ещё не рассчитаны для NK-76 tubing_pressure" in _texts(app, "warning")
    assert "--targets pump_fillage tubing_pressure" in _texts(app, "code")
    assert len(app.get("plotly_chart")) >= 5


def test_app_partial_date_range_preserves_last_complete_selection(seeded_path, monkeypatch):
    app = _app(seeded_path, monkeypatch)
    days = [pd.Timestamp(f"2026-01-0{day}").date() for day in (3, 4, 5, 6)]
    app.date_input(key="dates").set_value((days[0], days[1])).run()
    app.date_input(key="dates").set_value((days[2],)).run()
    assert not app.exception
    assert app.date_input(key="dates").value == (days[2],)
    assert app.session_state["complete_dates"] == (days[0], days[1])
    assert "Пока используется предыдущий период" in _texts(app, "caption")
    app.date_input(key="dates").set_value((days[2], days[3])).run()
    assert app.session_state["complete_dates"] == (days[2], days[3])
    assert "Пока используется предыдущий период" not in _texts(app, "caption")


def test_app_pressure_forecasts_and_model_selection(db_copy, monkeypatch):
    _edit(db_copy,
          "INSERT INTO metrics VALUES ('NK-76', 'tubing_pressure', 1, 'naive', 0.25, 1, 1, 1.0)",
          "INSERT INTO backtest_predictions VALUES "
          "('NK-76', 'tubing_pressure', '2026-01-07 23:00:00', 1, 'naive', '2026-01-08 00:00:00', 7.0, 7.25)",
          "INSERT INTO forecast_path VALUES "
          "('NK-76', 'tubing_pressure', '2026-01-10 23:00:00', 1, 'naive', '2026-01-11 00:00:00', 7.25)")
    app = _app(db_copy, monkeypatch)
    app.selectbox(key="well_type").set_value("fountain").run()
    assert not app.exception and app.selectbox(key="parameter").value == "tubing_pressure"
    assert "MAE в atm" in _texts(app, "markdown")
    assert any("MAE, atm" in chart.proto.spec for chart in app.get("plotly_chart"))
    app.multiselect(key="models").set_value([]).run()
    assert "Выберите хотя бы одну модель" in _texts(app, "info")
    assert "нет опубликованных метрик" not in _texts(app, "warning")
    app.radio(key="forecast_mode").set_value("release").run()
    assert "Выберите хотя бы одну модель" in _texts(app, "info")
    assert "Нет полного окна" not in _texts(app, "warning")
    app.multiselect(key="models").set_value(["naive"]).run()
    assert "давность 0 ч" in _texts(app, "info") and "MAE в atm" in _texts(app, "markdown")


def test_app_shows_bonus_models_where_published(db_copy, monkeypatch):
    _edit(db_copy,
          "INSERT INTO metrics VALUES ('NK-7', 'pump_fillage', 1, 'gradient_boosting', 1, 1, 72, 1.0/72)",
          "INSERT INTO metrics VALUES ('NK-7', 'pump_fillage', 1, 'statsmodels_ar', 1, 1, 72, 1.0/72)",
          "INSERT INTO metrics VALUES ('NK-76', 'tubing_pressure', 1, 'naive', 1, 1, 1, 1)")
    app = _app(db_copy, monkeypatch)
    selected = app.multiselect(key="models").value
    assert set(selected) == set(queries.FORECAST_MODELS)
    assert any("Gradient boosting" in chart.proto.spec for chart in app.get("plotly_chart"))
    app.multiselect(key="models").set_value(["statsmodels_ar"]).run()
    assert not app.exception
    app.selectbox(key="well_type").set_value("fountain").run()
    assert not app.exception and app.multiselect(key="models").value == ["naive"]


def test_app_empty_slice_keeps_selection_and_official_mae(seeded_path, monkeypatch):
    app = _app(seeded_path, monkeypatch)
    app.selectbox(key="well_type").set_value("fountain").run()
    app.selectbox(key="parameter").set_value("line_pressure").run()
    assert app.session_state["dates"][0] == pd.Timestamp("2026-01-01").date()  # 14 days clamp to the archive
    app.date_input(key="dates").set_value((pd.Timestamp("2026-01-08").date(), pd.Timestamp("2026-01-10").date())).run()
    assert "0 из 72" in _texts(app, "info") and app.selectbox(key="parameter").value == "line_pressure"
    app.selectbox(key="well_type").set_value("sucker_rod_pump").run()
    app.date_input(key="dates").set_value((pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date())).run()
    assert "нет тестовых прогнозов" in _texts(app, "info")
    assert app.selectbox(key="well_id").value == "NK-7"
    app.slider(key="horizon").set_value(24).run()
    assert "нет оценки" in _texts(app, "warning").lower()


def test_app_release_states(db_copy, monkeypatch):
    app = _app(db_copy, monkeypatch)
    app.radio(key="forecast_mode").set_value("release").run()
    assert "давность 0 ч" in _texts(app, "info")
    _edit(db_copy, "UPDATE forecast_path SET origin_ts = '2026-01-10 21:00:00', "
                   "target_ts = datetime('2026-01-10 21:00:00', '+' || horizon_h || ' hours')")
    app.run()
    assert "давность 2 ч" in _texts(app, "warning") and "устаревший" in _texts(app, "warning")
    _edit(db_copy, "DELETE FROM forecast_path")
    app.run()
    assert "Нет полного окна" in _texts(app, "warning")
    _edit(db_copy, "DELETE FROM backtest_predictions", "DELETE FROM metrics")
    app.run()
    assert "Прогнозы ещё не рассчитаны" in _texts(app, "warning")


def test_app_event_shortcut(db_copy, monkeypatch):
    _edit(db_copy, "INSERT INTO quality_issues (well_id, parameter, timestamp, end_ts, issue_code, action) VALUES "
                   "('NK-7', 'SPM', '2026-01-05 00:00:00', NULL, 'nk7_regime_change', 'annotate')")
    app = _app(db_copy, monkeypatch)
    app.button(key="jump::nk7_regime_change").click().run()
    assert (app.selectbox(key="well_id").value, app.selectbox(key="parameter").value) == ("NK-7", "SPM")
    assert app.session_state["dates"] == (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-10").date())
