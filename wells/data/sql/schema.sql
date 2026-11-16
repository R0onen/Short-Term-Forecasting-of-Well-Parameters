-- Open each writer with PRAGMA foreign_keys = ON before its transaction.
CREATE TABLE dataset_meta (
    key TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE series_catalog (
    well_id TEXT NOT NULL,
    parameter TEXT NOT NULL,
    well_type TEXT NOT NULL CHECK (well_type IN ('fountain', 'sucker_rod_pump')),
    unit TEXT NOT NULL,
    PRIMARY KEY (well_id, parameter)
);

CREATE TABLE measurements_raw (
    source_file TEXT NOT NULL,
    source_row INTEGER NOT NULL CHECK (typeof(source_row) = 'integer' AND source_row > 0),
    well_id TEXT NOT NULL,
    well_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    parameter TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT NOT NULL,
    n_obs TEXT NOT NULL,
    flag TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    PRIMARY KEY (source_file, source_row)
);

CREATE TABLE measurements_wide (
    well_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    well_type TEXT NOT NULL CHECK (well_type IN ('fountain', 'sucker_rod_pump')),
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    SPM REAL,
    casing_pressure REAL,
    dynamometer_area REAL,
    line_pressure REAL,
    max_rod_weight REAL,
    min_rod_weight REAL,
    pump_fillage REAL,
    tubing_pressure REAL,
    PRIMARY KEY (well_id, timestamp)
);

CREATE TABLE measurements_long (
    well_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    parameter TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    n_obs INTEGER NOT NULL CHECK (typeof(n_obs) = 'integer' AND n_obs > 0),
    flag TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    is_usable INTEGER NOT NULL CHECK (is_usable IN (0, 1)),
    PRIMARY KEY (well_id, timestamp, parameter),
    FOREIGN KEY (well_id, parameter) REFERENCES series_catalog (well_id, parameter),
    FOREIGN KEY (well_id, timestamp) REFERENCES measurements_wide (well_id, timestamp)
);

CREATE INDEX idx_measurements_long_series
    ON measurements_long (well_id, parameter, timestamp);

CREATE TABLE quality_issues (
    issue_id INTEGER NOT NULL PRIMARY KEY,
    well_id TEXT,
    parameter TEXT,
    timestamp TEXT,
    end_ts TEXT,
    source_file TEXT,
    source_row INTEGER,
    issue_code TEXT NOT NULL,
    action TEXT NOT NULL,
    CHECK ((source_file IS NULL) = (source_row IS NULL)),
    CHECK (action <> 'annotate' OR (timestamp IS NOT NULL AND (end_ts IS NULL OR end_ts > timestamp))),
    FOREIGN KEY (source_file, source_row) REFERENCES measurements_raw (source_file, source_row)
);

CREATE TABLE backtest_predictions (
    well_id TEXT NOT NULL,
    parameter TEXT NOT NULL,
    origin_ts TEXT NOT NULL,
    horizon_h INTEGER NOT NULL CHECK (typeof(horizon_h) = 'integer' AND horizon_h BETWEEN 1 AND 24),
    model TEXT NOT NULL CHECK (model IN ('naive', 'seasonal_naive', 'ridge')),
    target_ts TEXT NOT NULL,
    y_true REAL NOT NULL,
    y_pred REAL NOT NULL,
    PRIMARY KEY (well_id, parameter, origin_ts, horizon_h, model),
    FOREIGN KEY (well_id, parameter) REFERENCES series_catalog (well_id, parameter),
    FOREIGN KEY (well_id, origin_ts) REFERENCES measurements_wide (well_id, timestamp),
    FOREIGN KEY (well_id, target_ts) REFERENCES measurements_wide (well_id, timestamp)
);

CREATE TABLE metrics (
    well_id TEXT NOT NULL,
    parameter TEXT NOT NULL,
    horizon_h INTEGER NOT NULL CHECK (typeof(horizon_h) = 'integer' AND horizon_h BETWEEN 1 AND 24),
    model TEXT NOT NULL CHECK (model IN ('naive', 'seasonal_naive', 'ridge')),
    mae REAL,
    n_common INTEGER NOT NULL CHECK (typeof(n_common) = 'integer' AND n_common >= 0),
    n_actual INTEGER NOT NULL CHECK (typeof(n_actual) = 'integer' AND n_actual >= n_common),
    coverage REAL CHECK (coverage BETWEEN 0.0 AND 1.0),
    PRIMARY KEY (well_id, parameter, horizon_h, model),
    CHECK ((mae IS NULL) = (n_common = 0)),
    CHECK (mae >= 0.0),
    CHECK ((coverage IS NULL) = (n_actual = 0)),
    FOREIGN KEY (well_id, parameter) REFERENCES series_catalog (well_id, parameter)
);

CREATE TABLE forecast_path (
    well_id TEXT NOT NULL,
    parameter TEXT NOT NULL,
    origin_ts TEXT NOT NULL,
    horizon_h INTEGER NOT NULL CHECK (typeof(horizon_h) = 'integer' AND horizon_h BETWEEN 1 AND 24),
    model TEXT NOT NULL CHECK (model IN ('naive', 'seasonal_naive', 'ridge')),
    target_ts TEXT NOT NULL,
    y_pred REAL NOT NULL,
    PRIMARY KEY (well_id, parameter, origin_ts, horizon_h, model),
    FOREIGN KEY (well_id, parameter) REFERENCES series_catalog (well_id, parameter),
    FOREIGN KEY (well_id, origin_ts) REFERENCES measurements_wide (well_id, timestamp)
);
