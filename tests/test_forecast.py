"""Forecast component tests on synthetic rows and the canonical fixture."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wells.forecast import run as run_module
from wells.forecast.evaluation import (
    build_forecast_path,
    choose_alpha,
    evaluate_backtest,
    fit_final,
    select_alpha,
)
from wells.forecast.features import ForecastError, build_features, eligible_fit_rows
from wells.forecast.models import fit_ridge, predict_naive, predict_ridge, predict_seasonal_naive
from wells.forecast.run import run_diagnostic, run_forecast

ROOT = Path(__file__).resolve().parents[1]
V = "2026-01-05 00:00:00"
REGIME = "2026-01-02 00:00:00"
TEST_START = "2026-01-08 00:00:00"
TEST_END = "2026-01-10 23:00:00"
DATES = {"validation_start": V, "nk7_regime_start": REGIME}


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def fixture_series() -> pd.Series:
    """Same NK-7 values as seed.sql: 0 at hour 0, then 30 + hour%24 + day."""
    index = pd.date_range("2026-01-01", TEST_END, freq="h")
    k = np.arange(len(index))
    return pd.Series(np.where(k == 0, 0.0, 30 + k % 24 + k // 24), index=index)


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "wells.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((ROOT / "wells/data/sql/schema.sql").read_text(encoding="utf-8"))
    conn.executescript((ROOT / "tests/fixtures/seed.sql").read_text(encoding="utf-8"))
    conn.close()
    return path


def query(path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def row_at(frame: pd.DataFrame, origin: str) -> pd.Series:
    return frame.set_index("origin_ts").loc[ts(origin)]


# ---- features and baselines -------------------------------------------------

@pytest.mark.parametrize("h, seasonal_source", [(1, "2026-01-02 13:00"), (6, "2026-01-02 18:00"), (24, "2026-01-03 12:00")])
def test_baseline_source_hours(h, seasonal_source):
    s = fixture_series()
    frame = build_features(s, h)
    row = row_at(frame, "2026-01-03 12:00")
    assert row["target_ts"] == ts("2026-01-03 12:00") + pd.Timedelta(hours=h)
    assert row["y_true"] == s[row["target_ts"]]
    assert predict_naive(frame)[frame["origin_ts"] == ts("2026-01-03 12:00")].iloc[0] == s[ts("2026-01-03 12:00")]
    assert row["seasonal_value"] == s[ts(seasonal_source)]
    assert row["lag_6"] == s[ts("2026-01-03 06:00")]
    assert row["lag_24"] == s[ts("2026-01-02 12:00")]


def test_baselines_agree_at_24h():
    frame = build_features(fixture_series(), 24)
    pd.testing.assert_series_equal(predict_naive(frame), predict_seasonal_naive(frame))


def test_holes_stay_missing_and_zero_survives():
    s = fixture_series().drop(ts("2026-01-02 05:00"))  # absent row, not just NaN
    frame = build_features(s, 1)
    assert len(frame) == len(s) + 1  # the hourly grid is restored
    assert np.isnan(row_at(frame, "2026-01-02 05:00")["lag_0"])
    assert np.isnan(row_at(frame, "2026-01-02 06:00")["lag_1"])
    assert row_at(frame, "2026-01-02 08:00")["lag_3"] != row_at(frame, "2026-01-02 08:00")["lag_2"]
    assert np.isnan(row_at(frame, "2026-01-02 04:00")["y_true"])
    assert row_at(frame, "2026-01-01 00:00")["lag_0"] == 0.0
    assert row_at(frame, "2026-01-01 01:00")["lag_1"] == 0.0
    # targets beyond the archive are allowed, with an unknown label
    assert frame["target_ts"].iloc[-1] == ts(TEST_END) + pd.Timedelta(hours=1)
    assert np.isnan(frame["y_true"].iloc[-1])
    # the hole makes every row that needs it incomplete for Ridge
    pred = predict_ridge(fit_ridge(eligible_fit_rows(frame, well_id="X", cutoff=TEST_START), 1.0), frame)
    assert np.isnan(pred[frame["origin_ts"] == ts("2026-01-02 11:00")].iloc[0])  # lag_6 hits the hole


def test_invalid_series_rejected():
    s = fixture_series()
    with pytest.raises(ValueError):
        build_features(s.where(s.index != s.index[5], 120.0), 1)
    with pytest.raises(ValueError):
        build_features(pd.concat([s, s.iloc[:1]]), 1)
    with pytest.raises(ValueError):
        build_features(s, 25)


def test_fit_cutoff_and_nk7_regime():
    frame = build_features(fixture_series(), 1)
    rows = eligible_fit_rows(frame, well_id="NK-7", cutoff=V, regime_start=REGIME)
    assert rows["target_ts"].max() == ts("2026-01-04 23:00")
    assert rows["origin_ts"].min() == ts("2026-01-03 00:00")  # lag_24 lands exactly on the regime start
    other = eligible_fit_rows(frame, well_id="NK-68", cutoff=V, regime_start=REGIME)
    assert other["origin_ts"].min() == ts("2026-01-02 00:00")  # only limited by lag_24 history


# ---- selection, leakage and scoring ----------------------------------------

def test_alpha_tie_rule():
    assert choose_alpha({0.1: 2.0, 1.0: 2.0, 10.0: 2.0}) == 1.0
    assert choose_alpha({0.1: 2.0, 1.0: 3.0, 10.0: 2.0}) == 0.1
    assert choose_alpha({0.1: 2.0, 1.0: 3.0, 10.0: 1.0}) == 10.0


def _train(series: pd.Series, horizons=(1, 24)):
    frames = {("NK-7", h): build_features(series, h) for h in horizons}
    alpha, table = select_alpha(frames, validation_start=V, test_start=TEST_START, nk7_regime_start=REGIME)
    pipelines, counts = fit_final(frames, alpha, test_start=TEST_START, nk7_regime_start=REGIME)
    return frames, alpha, table, pipelines, counts


def test_validation_keys_shared_and_bounded():
    _, alpha, table, _, counts = _train(fixture_series())
    assert alpha in (0.1, 1.0, 10.0)
    assert table.groupby(["well_id", "horizon_h"])["n_validation"].nunique().eq(1).all()
    # h=1: targets Jan 5 00:00 .. Jan 7 23:00; h=24: origins start at V-1h, so targets from Jan 5 23:00
    assert table.set_index(["alpha", "horizon_h"]).loc[(1.0, 1), "n_validation"] == 72
    assert table.set_index(["alpha", "horizon_h"]).loc[(1.0, 24), "n_validation"] == 49
    assert counts.set_index("horizon_h")["n_fit"].to_dict() == {1: 119, 24: 96}  # origins from Jan 3 00:00 (NK-7 regime + 24 h)


def test_test_period_values_do_not_affect_training():
    s = fixture_series()
    changed = s.copy()
    changed[changed.index >= ts(TEST_START)] = 5.0
    _, alpha_a, table_a, pipes_a, _ = _train(s)
    _, alpha_b, table_b, pipes_b, _ = _train(changed)
    assert alpha_a == alpha_b
    pd.testing.assert_frame_equal(table_a, table_b)
    for key in pipes_a:
        np.testing.assert_array_equal(pipes_a[key]["ridge"].coef_, pipes_b[key]["ridge"].coef_)
        np.testing.assert_array_equal(pipes_a[key]["scaler"].mean_, pipes_b[key]["scaler"].mean_)


def test_release_ignores_future_observations():
    s = fixture_series()
    _, _, _, pipelines, _ = _train(s, horizons=range(1, 25))
    cut = s[s.index <= ts("2026-01-09 12:00")]
    later = cut.copy()
    extra = pd.Series(99.0, index=pd.date_range("2026-01-09 13:00", periods=3, freq="h"))
    a = build_forecast_path({"NK-7": cut}, pipelines, test_start=TEST_START)
    assert len(a) == 72 and a["origin_ts"].nunique() == 1 and a["origin_ts"].iloc[0] == ts("2026-01-09 12:00")
    # a hole right after the origin keeps the origin; later data moves it only once 25 full hours exist
    b = build_forecast_path({"NK-7": pd.concat([later, extra.iloc[1:]])}, pipelines, test_start=TEST_START)
    pd.testing.assert_frame_equal(a, b)
    # no eligible origin -> no rows
    gappy = s.copy()
    gappy[(gappy.index >= ts("2026-01-07 00:00")) & (gappy.index.hour == 3)] = np.nan
    assert build_forecast_path({"NK-7": gappy}, pipelines, test_start=TEST_START).empty


class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_mae_and_common_keys():
    # flat history at 45, then actuals 40 and 50 in a two-hour "test" period
    index = pd.date_range("2026-02-01", periods=50, freq="h")
    s = pd.Series(45.0, index=index)
    s.iloc[-2:] = [40.0, 50.0]
    start, end = index[-2], index[-1]
    frames = {("W", 1): build_features(s, 1)}
    _, metrics = evaluate_backtest(frames, {("W", 1): _Constant(45.0)}, test_start=start, test_end=end)
    m = metrics.set_index("model")
    assert m.loc["ridge", "mae"] == 5.0  # |45-40|, |45-50|
    assert m.loc["naive", "mae"] == 7.5  # 45->40, 40->50
    assert (m["n_common"] == 2).all() and (m["n_actual"] == 2).all() and (m["coverage"] == 1.0).all()

    # drop the seasonal source of the second target: that key disappears for all three methods
    holed = s.copy()
    holed[end - pd.Timedelta(hours=24)] = np.nan
    frames = {("W", 1): build_features(holed, 1)}
    preds, metrics = evaluate_backtest(frames, {("W", 1): _Constant(45.0)}, test_start=start, test_end=end)
    assert (metrics["n_common"] == 1).all() and (metrics["coverage"] == 0.5).all()
    assert preds.groupby("origin_ts")["model"].nunique().tolist() == [3]

    # no usable test targets at all: NULL MAE and NULL coverage
    blank = s.copy()
    blank.iloc[-2:] = np.nan
    _, metrics = evaluate_backtest(
        {("W", 1): build_features(blank, 1)}, {("W", 1): _Constant(45.0)}, test_start=start, test_end=end
    )
    assert metrics["mae"].isna().all() and metrics["coverage"].isna().all() and (metrics["n_actual"] == 0).all()


# ---- runner on the canonical fixture ---------------------------------------

def test_run_publishes_fixture_results(db):
    long_before = query(db, "SELECT count(*), sum(value) FROM measurements_long")
    report = run_forecast(db, **DATES)
    assert report["table_counts"]["metrics"] == 72
    assert report["table_counts"]["forecast_path"] == 72
    assert report["release_origins"] == {"NK-7": TEST_END}
    assert report["config"]["wells"] == ["NK-7"]
    assert query(db, "SELECT count(*), sum(value) FROM measurements_long") == long_before

    # first origin, h=1: the hand-computed fixture example (fact 37, naive 59, seasonal 36)
    first = dict(query(db, "SELECT model, y_pred FROM backtest_predictions WHERE origin_ts='2026-01-07 23:00:00' AND horizon_h=1"))
    assert first["naive"] == 59 and first["seasonal_naive"] == 36 and 0 <= first["ridge"] <= 100
    assert query(db, "SELECT DISTINCT y_true FROM backtest_predictions WHERE origin_ts='2026-01-07 23:00:00' AND horizon_h=1") == [(37.0,)]

    counts = dict(query(db, "SELECT horizon_h, n_common FROM metrics WHERE model='ridge'"))
    assert counts[1] == 72 and counts[24] == 49  # first h=24 target is test_start + 23 h
    assert query(db, "SELECT DISTINCT n_actual FROM metrics") == [(72,)]
    assert query(db, "SELECT count(*) FROM backtest_predictions") == [(report["table_counts"]["backtest_predictions"],)]

    # stored MAE equals independent arithmetic over the stored predictions
    mismatched = query(db, """
        SELECT count(*) FROM metrics m JOIN (
            SELECT horizon_h, model, avg(abs(y_pred - y_true)) AS mae, count(*) AS n
            FROM backtest_predictions GROUP BY horizon_h, model
        ) p USING (horizon_h, model)
        WHERE abs(m.mae - p.mae) > 1e-9 OR m.n_common <> p.n
           OR abs(m.coverage - 1.0 * m.n_common / m.n_actual) > 1e-12""")
    assert mismatched == [(0,)]
    assert query(db, "SELECT count(*) FROM backtest_predictions WHERE target_ts <> datetime(origin_ts, '+' || horizon_h || ' hours')") == [(0,)]
    assert query(db, "PRAGMA foreign_key_check") == []

    # a rerun replaces rows without duplicates and reproduces the numbers
    snapshot = query(db, "SELECT * FROM metrics ORDER BY 1,2,3,4") + query(db, "SELECT * FROM forecast_path ORDER BY 1,2,3,4,5")
    again = run_forecast(db, **DATES)
    assert again["table_counts"] == report["table_counts"]
    assert again["validation"] == report["validation"]
    assert query(db, "SELECT * FROM metrics ORDER BY 1,2,3,4") + query(db, "SELECT * FROM forecast_path ORDER BY 1,2,3,4,5") == snapshot


def _results(db):
    return [query(db, f"SELECT * FROM {t} ORDER BY 1,2,3,4,5") for t in run_module.RESULT_TABLES]


@pytest.mark.parametrize("include_pressure", [False, True])
def test_run_publishes_backtest_when_no_latest_release_exists(db, include_pressure):
    assert query(db, "SELECT count(*) FROM forecast_path") == [(72,)]
    # Keep test facts and scoreable forecasts, but break every eligible 25-hour window.
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute("""
                UPDATE measurements_wide SET pump_fillage = NULL
                WHERE well_id = 'NK-7' AND timestamp IN (
                    '2026-01-07 23:00:00', '2026-01-09 00:00:00', '2026-01-10 01:00:00'
                )
            """)
    finally:
        conn.close()

    targets = ["pump_fillage"]
    if include_pressure:
        _with_pressure(db)
        targets.append("tubing_pressure")
    report = run_forecast(db, targets=targets, **DATES)

    assert report["release_origins"] == {"NK-7": None}
    assert report["table_counts"]["forecast_path"] == (72 if include_pressure else 0)
    assert report["table_counts"]["metrics"] == 72 * len(targets)
    assert report["table_counts"]["backtest_predictions"] > 0
    assert query(db, "SELECT count(*) FROM forecast_path WHERE parameter = 'pump_fillage'") == [(0,)]
    assert query(db, "SELECT DISTINCT n_actual FROM metrics WHERE parameter = 'pump_fillage'") == [(70,)]
    assert query(db, "SELECT count(*) FROM metrics WHERE parameter = 'pump_fillage' AND mae IS NOT NULL")[0][0] > 0
    if include_pressure:
        assert report["extra_targets"]["tubing_pressure"]["release_origins"] == {"NK-76": TEST_END}
        assert query(db, "SELECT count(*) FROM forecast_path WHERE parameter = 'tubing_pressure'") == [(72,)]


def test_failed_write_keeps_previous_results(db, monkeypatch):
    before = _results(db)
    real = run_module._records

    def broken(frame):
        rows = real(frame)
        if "y_true" not in frame.columns and "mae" not in frame.columns:  # forecast_path, written last
            rows[0]["model"] = "not-a-model"
        return rows

    monkeypatch.setattr(run_module, "_records", broken)
    with pytest.raises(ForecastError):
        run_forecast(db, **DATES)
    assert _results(db) == before


def test_empty_group_stops_before_publication(db):
    before = _results(db)
    with pytest.raises(ForecastError, match="stage=internal_fit"):
        run_forecast(db, validation_start="2026-01-02 00:00:00", nk7_regime_start=REGIME)
    with pytest.raises(ForecastError, match="stage=validation"):
        run_forecast(db, validation_start=TEST_START, nk7_regime_start=REGIME)
    assert _results(db) == before


def test_diagnostic_is_read_only(db):
    before = _results(db)
    report = run_diagnostic(db, **DATES)
    assert [c["horizon_h"] for c in report["eligible_counts"]] == [1, 24]
    assert all(c["n_fit"] > 0 and c["n_validation"] > 0 for c in report["eligible_counts"])
    assert all(np.isfinite(r["mae"]) for r in report["validation"]["rows"])
    assert _results(db) == before


def test_missing_database_is_not_created(tmp_path, capsys):
    missing = tmp_path / "absent.sqlite"
    assert run_module.main(["--db", str(missing)]) == 1
    assert not missing.exists()
    assert "database not found" in capsys.readouterr().err


# ---- extension: second target (tubing_pressure of fountain wells) ----------

def _with_pressure(db):
    """Give NK-76 a complete synthetic tubing_pressure series above 100 atm."""
    conn = sqlite3.connect(db)
    conn.execute("UPDATE measurements_wide SET tubing_pressure = 120 + CAST(strftime('%H', timestamp) AS INTEGER) "
                 "WHERE well_id = 'NK-76'")
    conn.commit()
    conn.close()


def test_unbounded_target_is_clipped_only_below():
    index = pd.date_range("2026-03-01", periods=80, freq="h")
    s = pd.Series(150.0 + np.arange(80) % 24, index=index)
    with pytest.raises(ValueError):
        build_features(s, 1)  # default range is the 0-100 % of pump_fillage
    frame = build_features(s, 1, (0.0, None))
    assert predict_ridge(_Constant(250.0), frame, (0.0, None)).max() == 250.0
    assert predict_ridge(_Constant(-5.0), frame, (0.0, None)).max() == 0.0
    assert predict_ridge(_Constant(250.0), frame).max() == 100.0


def test_run_with_pressure_target(db):
    _with_pressure(db)
    report = run_forecast(db, targets=["pump_fillage", "tubing_pressure"], **DATES)
    assert dict(query(db, "SELECT parameter, count(*) FROM metrics GROUP BY parameter")) == {
        "pump_fillage": 72, "tubing_pressure": 72}
    assert query(db, "SELECT DISTINCT well_id FROM forecast_path WHERE parameter = 'tubing_pressure'") == [("NK-76",)]
    assert query(db, "SELECT min(y_pred) > 100 FROM backtest_predictions WHERE parameter = 'tubing_pressure'") == [(1,)]
    assert report["config"]["target"] == "pump_fillage"
    extra = report["extra_targets"]["tubing_pressure"]
    assert extra["config"]["wells"] == ["NK-76"] and extra["config"]["clip_range"] == [0.0, None]
    assert query(db, "PRAGMA foreign_key_check") == []

    # the default run publishes the approved target only and drops the extension rows
    again = run_forecast(db, **DATES)
    assert "extra_targets" not in again
    assert query(db, "SELECT DISTINCT parameter FROM metrics") == [("pump_fillage",)]


def test_failing_extra_target_publishes_nothing(db):
    before = _results(db)  # NK-76 pressure in the fixture has two observations, so no fit rows
    with pytest.raises(ForecastError, match="well=NK-76"):
        run_forecast(db, targets=["pump_fillage", "tubing_pressure"], **DATES)
    with pytest.raises(ForecastError, match="unsupported target"):
        run_forecast(db, targets=["SPM"], **DATES)
    assert _results(db) == before
