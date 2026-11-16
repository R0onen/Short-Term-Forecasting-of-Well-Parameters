WITH hours AS (
    SELECT c.well_id, c.parameter, c.well_type, c.unit,
           CASE :grain
               WHEN 'day' THEN date(w.timestamp)
               WHEN 'month' THEN date(w.timestamp, 'start of month')
           END AS period_start,
           l.value,
           l.is_usable,
           trim(l.flag, char(9, 10, 11, 12, 13, 32)) <> '' AS is_flagged,
           CASE WHEN :mode = 'raw' OR l.is_usable = 1 THEN l.value END AS analysis_value
    FROM series_catalog AS c
    JOIN measurements_wide AS w ON w.well_id = c.well_id
    LEFT JOIN measurements_long AS l
        ON l.well_id = w.well_id AND l.timestamp = w.timestamp AND l.parameter = c.parameter
    WHERE (:well_id IS NULL OR c.well_id = :well_id)
      AND (:parameter IS NULL OR c.parameter = :parameter)
      AND w.timestamp >= :start_ts AND w.timestamp < :end_ts
      AND :grain IN ('day', 'month') AND :mode IN ('raw', 'clean')
)
SELECT well_id, parameter, well_type, unit, period_start,
       count(*) AS n_expected,
       count(value) AS n_observed,
       count(CASE WHEN is_usable = 1 THEN 1 END) AS n_usable,
       count(CASE WHEN is_flagged THEN 1 END) AS n_flagged,
       count(analysis_value) AS n_values,
       avg(analysis_value) AS mean_value,
       min(analysis_value) AS min_value,
       max(analysis_value) AS max_value,
       1.0 * count(value) / count(*) AS coverage
FROM hours
GROUP BY well_id, parameter, well_type, unit, period_start
ORDER BY well_id, parameter, period_start;
