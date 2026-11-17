"""Evaluate causal one-hour rod-weight forecast alarms against published flags."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

from wells.analysis.sql_report import open_readonly

PARAMETER = "max_rod_weight"
LABEL = "weight_overflow"
QUANTILE = 0.99
DEFAULT_DB = Path("data/processed/wells.sqlite")

RESIDUALS = """
WITH history AS (
    SELECT well_id, timestamp, value, split, flag, is_usable,
           LAG(timestamp) OVER timeline AS prev_timestamp,
           LAG(value) OVER timeline AS prev_value,
           LAG(split) OVER timeline AS prev_split,
           LAG(is_usable) OVER timeline AS prev_usable
    FROM measurements_long
    WHERE parameter = ?
    WINDOW timeline AS (PARTITION BY well_id ORDER BY timestamp)
)
SELECT well_id, split, flag, is_usable, prev_split, prev_usable,
       CASE WHEN datetime(prev_timestamp, '+1 hour') = timestamp
            THEN ABS(value - prev_value) END AS residual
FROM history
ORDER BY well_id, timestamp
"""


def _classification_metrics(test: pd.DataFrame) -> dict:
    # A missing predecessor or threshold makes the target unscorable.
    scored = test.loc[test["residual"].notna() & test["threshold"].notna()]
    positive = scored["flagged"]
    alarm = scored["residual"].gt(scored["threshold"])
    tp = int((alarm & positive).sum())
    fp = int((alarm & ~positive).sum())
    fn = int((~alarm & positive).sum())
    tn = int((~alarm & ~positive).sum())
    flagged = int(test["flagged"].sum())
    return {
        "observed_test_rows": len(test),
        "scored_test_rows": len(scored),
        "observed_test_flags": flagged,
        "unscored_test_flags": flagged - tp - fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "alarm_rate": (tp + fp) / len(scored) if len(scored) else None,
    }


def evaluate(conn: sqlite3.Connection) -> dict:
    """Calibrate on clean training pairs and classify observed official test rows."""
    rows = pd.read_sql_query(RESIDUALS, conn, params=(PARAMETER,))
    calibration = rows.loc[
        rows["split"].eq("train")
        & rows["prev_split"].eq("train")
        & rows["is_usable"].eq(1)
        & rows["prev_usable"].eq(1)
        & rows["residual"].notna()
    ].groupby("well_id")["residual"]
    thresholds = calibration.quantile(QUANTILE)
    if thresholds.empty:
        raise ValueError(f"{PARAMETER}: no adjacent usable training observations for calibration")
    counts = calibration.size()

    test = rows.loc[rows["split"].eq("test")].copy()
    if test.empty:
        raise ValueError(f"{PARAMETER}: no observed test rows")
    test["threshold"] = test["well_id"].map(thresholds)
    # Raw target readings and raw lagged inputs enter the detector. Test labels
    # are used solely to score the alarms, including originally masked targets.
    test["flagged"] = test["flag"].str.contains(rf"(?:^|;)\s*{LABEL}\s*(?:;|$)", regex=True)
    metadata = dict(conn.execute(
        "SELECT key, value FROM dataset_meta WHERE key IN ('revision', 'train_end', 'test_start', 'test_end')"
    ))
    return {
        "dataset": metadata,
        "parameter": PARAMETER,
        "published_label": LABEL,
        "forecast": "previous observed value at exactly one hour before the target",
        "threshold": "per-well empirical quantile of absolute residuals from usable training pairs",
        "threshold_quantile": QUANTILE,
        "alarm_rule": "absolute residual strictly greater than threshold",
        "evaluation": "micro metrics on observed test rows with an exact prior-hour value and a threshold",
        "overall": _classification_metrics(test),
        "per_well": [
            {
                "well_id": well_id,
                "calibration_pairs": int(counts.get(well_id, 0)),
                "threshold_kg": float(thresholds[well_id]) if well_id in thresholds else None,
                **_classification_metrics(series),
            }
            for well_id, series in test.groupby("well_id", sort=True)
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, help="write a JSON report to this path")
    args = parser.parse_args(argv)
    try:
        with closing(open_readonly(args.db)) as conn:
            report = json.dumps(evaluate(conn), indent=2, allow_nan=False)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report + "\n", encoding="utf-8")
        else:
            print(report)
    except (OSError, ValueError, sqlite3.Error, pd.errors.DatabaseError) as error:
        print(f"anomaly detection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
