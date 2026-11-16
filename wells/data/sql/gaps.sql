WITH observed AS (
    SELECT well_id, parameter, timestamp
    FROM measurements_long
    WHERE well_id = :well_id AND parameter = :parameter
    UNION ALL
    SELECT DISTINCT well_id, NULL, timestamp
    FROM measurements_long
    WHERE well_id = :well_id AND :parameter IS NULL
), ordered AS (
    SELECT well_id, parameter, timestamp,
           LAG(timestamp) OVER (PARTITION BY well_id, parameter ORDER BY timestamp) AS previous_ts
    FROM observed
), intervals AS (
    SELECT well_id, parameter, previous_ts, timestamp,
           (CAST(strftime('%s', timestamp) AS INTEGER) - CAST(strftime('%s', previous_ts) AS INTEGER)) / 3600
               AS delta_hours
    FROM ordered
)
SELECT well_id, parameter, previous_ts, timestamp, delta_hours, delta_hours - 1 AS missing_hours
FROM intervals
WHERE delta_hours > 1
  AND :start_ts < :end_ts
  AND datetime(previous_ts, '+1 hour') < :end_ts
  AND timestamp > :start_ts
ORDER BY well_id, parameter, timestamp;
