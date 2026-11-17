import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from wells.analysis.anomaly_detection import evaluate


def test_anomaly_uses_clean_training_pairs_and_scores_flagged_test_rows():
    schema = Path(__file__).resolve().parents[1] / "wells/data/sql/schema.sql"
    readings = [
        (0, 100, ""),
        (1, 101, ""),
        (2, 102, ""),
        (3, 800, "weight_overflow"),
        (4, 103, ""),
        (5, 104, ""),
        (6, 500, "other; weight_overflow"),
        (7, 105, ""),
        (8, 106, ""),
        (10, 600, "weight_overflow"),
        (11, 600, "weight_overflow"),
        (12, 601, ""),
    ]
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO series_catalog VALUES ('NK-7', 'max_rod_weight', 'sucker_rod_pump', 'kg')")
        for hour, value, flag in readings:
            timestamp = (datetime(2026, 1, 1) + timedelta(hours=hour)).isoformat(sep=" ")
            split = "train" if hour < 6 else "test"
            conn.execute(
                "INSERT INTO measurements_wide (well_id, timestamp, well_type, split) "
                "VALUES ('NK-7', ?, 'sucker_rod_pump', ?)",
                (timestamp, split),
            )
            conn.execute(
                "INSERT INTO measurements_long VALUES ('NK-7', ?, 'max_rod_weight', ?, 'kg', 1, ?, ?, ?)",
                (timestamp, value, flag, split, int(not flag)),
            )
        report = evaluate(conn)

    well = report["per_well"][0]
    assert well["calibration_pairs"] == 3
    assert well["threshold_kg"] == 1.0
    summary = report["overall"]
    assert summary["observed_test_rows"] == 6
    assert summary["scored_test_rows"] == 5
    assert summary["observed_test_flags"] == 3
    assert summary["unscored_test_flags"] == 1
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["true_negatives"] == 2
    assert summary["precision"] == summary["recall"] == summary["f1"] == 0.5
