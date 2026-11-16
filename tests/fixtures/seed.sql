-- Apply to an empty database initialized with schema.sql and foreign keys enabled.
-- Forecast tests pass validation_start='2026-01-05 00:00:00' and
-- nk7_regime_start='2026-01-02 00:00:00' explicitly. The NK-7 target is
-- continuous for the internal fit, validation, test and latest release windows.
-- NK-76 casing pressure is inapplicable. Its line pressure is applicable but
-- has zero test observations. Jan 1 line pressure has an interior two-hour gap.

BEGIN;

INSERT INTO dataset_meta (key, value) VALUES
    ('revision', 'synthetic-fixture'),
    ('source_url', 'https://example.invalid/well-fixture'),
    ('train_start', '2026-01-01 00:00:00'),
    ('train_end', '2026-01-07 23:00:00'),
    ('test_start', '2026-01-08 00:00:00'),
    ('test_end', '2026-01-10 23:00:00'),
    ('train_sha256', '0000000000000000000000000000000000000000000000000000000000000000'),
    ('test_sha256', '1111111111111111111111111111111111111111111111111111111111111111');

INSERT INTO series_catalog (well_id, parameter, well_type, unit) VALUES
    ('NK-7', 'pump_fillage', 'sucker_rod_pump', '%'),
    ('NK-7', 'SPM', 'sucker_rod_pump', 'strokes/min'),
    ('NK-76', 'line_pressure', 'fountain', 'atm'),
    ('NK-76', 'tubing_pressure', 'fountain', 'atm');

WITH RECURSIVE hours(timestamp, hour_index) AS (
    SELECT value, 0 FROM dataset_meta WHERE key = 'train_start'
    UNION ALL
    SELECT datetime(timestamp, '+1 hour'), hour_index + 1
    FROM hours
    WHERE timestamp < (SELECT value FROM dataset_meta WHERE key = 'test_end')
), observations(well_id, timestamp, parameter, value, n_obs, flag) AS (
    SELECT 'NK-7', timestamp, 'pump_fillage',
           printf('%.2f', CASE WHEN hour_index = 0 THEN 0 ELSE 30 + hour_index % 24 + hour_index / 24 END),
           '4', ''
    FROM hours
    UNION ALL VALUES
    ('NK-7', '2026-01-01 00:00:00', 'SPM', '0.0', '4', ''),
    ('NK-7', '2026-01-01 01:00:00', 'SPM', '10.0', '4', 'spm_outlier'),
    ('NK-76', '2026-01-01 00:00:00', 'line_pressure', '12.0', '2', ''),
    ('NK-76', '2026-01-01 01:00:00', 'line_pressure', '14.0', '2', ''),
    ('NK-76', '2026-01-01 04:00:00', 'line_pressure', '16.0', '2', ''),
    ('NK-76', '2026-01-01 00:00:00', 'tubing_pressure', '0.00', '1', ''),
    ('NK-76', '2026-01-01 01:00:00', 'tubing_pressure', '10.0', '8', '')
), source AS (
    SELECT o.well_id, c.well_type, o.timestamp, o.parameter, o.value, c.unit, o.n_obs, o.flag,
           CASE WHEN o.timestamp < (SELECT value FROM dataset_meta WHERE key = 'test_start')
                THEN 'train' ELSE 'test' END AS split
    FROM observations AS o
    JOIN series_catalog AS c USING (well_id, parameter)
)
INSERT INTO measurements_raw (
    source_file, source_row, well_id, well_type, timestamp, parameter, value, unit, n_obs, flag, split
)
SELECT split || '.csv', row_number() OVER (PARTITION BY split ORDER BY timestamp, well_id, parameter),
       well_id, well_type, timestamp, parameter, value, unit, n_obs, flag, split
FROM source;

WITH RECURSIVE hours(timestamp) AS (
    SELECT value FROM dataset_meta WHERE key = 'train_start'
    UNION ALL
    SELECT datetime(timestamp, '+1 hour')
    FROM hours
    WHERE timestamp < (SELECT value FROM dataset_meta WHERE key = 'test_end')
), wells AS (
    SELECT DISTINCT well_id, well_type FROM series_catalog
)
INSERT INTO measurements_wide (
    well_id, timestamp, well_type, split, SPM, line_pressure, pump_fillage, tubing_pressure
)
SELECT w.well_id, h.timestamp, w.well_type,
       CASE WHEN h.timestamp < (SELECT value FROM dataset_meta WHERE key = 'test_start')
            THEN 'train' ELSE 'test' END,
       max(CASE WHEN r.parameter = 'SPM' THEN CAST(r.value AS REAL) END),
       max(CASE WHEN r.parameter = 'line_pressure' THEN CAST(r.value AS REAL) END),
       max(CASE WHEN r.parameter = 'pump_fillage' THEN CAST(r.value AS REAL) END),
       max(CASE WHEN r.parameter = 'tubing_pressure' THEN CAST(r.value AS REAL) END)
FROM wells AS w
CROSS JOIN hours AS h
LEFT JOIN measurements_raw AS r ON r.well_id = w.well_id AND r.timestamp = h.timestamp AND r.flag = ''
GROUP BY w.well_id, h.timestamp, w.well_type;

INSERT INTO measurements_long (well_id, timestamp, parameter, value, unit, n_obs, flag, split, is_usable)
SELECT well_id, timestamp, parameter, CAST(value AS REAL), unit, CAST(n_obs AS INTEGER), flag, split, flag = ''
FROM measurements_raw;

INSERT INTO quality_issues (well_id, parameter, timestamp, source_file, source_row, issue_code, action)
SELECT well_id, parameter, timestamp, source_file, source_row, flag, 'mask'
FROM measurements_raw
WHERE flag <> '';

-- Illustrative published results. The h=1 fact is 37. At the first origin,
-- the current hour's value is 59 and the seasonal source (Jan 7 00:00) is 36.
INSERT INTO backtest_predictions (well_id, parameter, origin_ts, horizon_h, model, target_ts, y_true, y_pred)
VALUES
    ('NK-7', 'pump_fillage', '2026-01-07 23:00:00', 1, 'naive', '2026-01-08 00:00:00', 37, 59),
    ('NK-7', 'pump_fillage', '2026-01-07 23:00:00', 1, 'seasonal_naive', '2026-01-08 00:00:00', 37, 36),
    ('NK-7', 'pump_fillage', '2026-01-07 23:00:00', 1, 'ridge', '2026-01-08 00:00:00', 37, 38);

INSERT INTO metrics (well_id, parameter, horizon_h, model, mae, n_common, n_actual, coverage) VALUES
    ('NK-7', 'pump_fillage', 1, 'naive', 22, 1, 72, 1.0 / 72),
    ('NK-7', 'pump_fillage', 1, 'seasonal_naive', 1, 1, 72, 1.0 / 72),
    ('NK-7', 'pump_fillage', 1, 'ridge', 1, 1, 72, 1.0 / 72),
    ('NK-7', 'pump_fillage', 24, 'naive', NULL, 0, 72, 0),
    ('NK-7', 'pump_fillage', 24, 'seasonal_naive', NULL, 0, 72, 0),
    ('NK-7', 'pump_fillage', 24, 'ridge', NULL, 0, 72, 0);

WITH RECURSIVE horizons(horizon_h) AS (
    VALUES (1)
    UNION ALL SELECT horizon_h + 1 FROM horizons WHERE horizon_h < 24
), models(model) AS (
    VALUES ('naive'), ('seasonal_naive'), ('ridge')
), release(origin_ts) AS (
    SELECT value FROM dataset_meta WHERE key = 'test_end'
)
INSERT INTO forecast_path (well_id, parameter, origin_ts, horizon_h, model, target_ts, y_pred)
SELECT 'NK-7', 'pump_fillage', origin_ts, horizon_h, model,
       datetime(origin_ts, '+' || horizon_h || ' hours'),
       CASE model WHEN 'naive' THEN 62 WHEN 'seasonal_naive' THEN 38 + horizon_h ELSE 40 + horizon_h * 0.5 END
FROM release CROSS JOIN horizons CROSS JOIN models;

COMMIT;
