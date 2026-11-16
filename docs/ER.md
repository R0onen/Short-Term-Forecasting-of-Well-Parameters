# ER diagram

SQLite schema defined in [`wells/data/sql/schema.sql`](../wells/data/sql/schema.sql). GitHub renders the diagram below.

```mermaid
erDiagram
    series_catalog ||--o{ measurements_long : "applicable sensor"
    measurements_wide ||--o{ measurements_long : "well-hour"
    measurements_raw ||--o{ quality_issues : "source record"
    series_catalog ||--o{ backtest_predictions : "target series"
    series_catalog ||--o{ metrics : "target series"
    series_catalog ||--o{ forecast_path : "target series"
    measurements_wide ||--o{ backtest_predictions : "origin and target hour"
    measurements_wide ||--o{ forecast_path : "origin hour"

    dataset_meta {
        TEXT key PK
        TEXT value
    }
    series_catalog {
        TEXT well_id PK
        TEXT parameter PK
        TEXT well_type
        TEXT unit
    }
    measurements_raw {
        TEXT source_file PK
        INTEGER source_row PK
        TEXT well_id
        TEXT well_type
        TEXT timestamp
        TEXT parameter
        TEXT value
        TEXT unit
        TEXT n_obs
        TEXT flag
        TEXT split
    }
    measurements_wide {
        TEXT well_id PK
        TEXT timestamp PK
        TEXT well_type
        TEXT split
        REAL SPM
        REAL casing_pressure
        REAL dynamometer_area
        REAL line_pressure
        REAL max_rod_weight
        REAL min_rod_weight
        REAL pump_fillage
        REAL tubing_pressure
    }
    measurements_long {
        TEXT well_id PK, FK
        TEXT timestamp PK, FK
        TEXT parameter PK, FK
        REAL value
        TEXT unit
        INTEGER n_obs
        TEXT flag
        TEXT split
        INTEGER is_usable
    }
    quality_issues {
        INTEGER issue_id PK
        TEXT well_id
        TEXT parameter
        TEXT timestamp
        TEXT end_ts
        TEXT source_file FK
        INTEGER source_row FK
        TEXT issue_code
        TEXT action
    }
    backtest_predictions {
        TEXT well_id PK, FK
        TEXT parameter PK, FK
        TEXT origin_ts PK, FK
        INTEGER horizon_h PK
        TEXT model PK
        TEXT target_ts FK
        REAL y_true
        REAL y_pred
    }
    metrics {
        TEXT well_id PK, FK
        TEXT parameter PK, FK
        INTEGER horizon_h PK
        TEXT model PK
        REAL mae
        INTEGER n_common
        INTEGER n_actual
        REAL coverage
    }
    forecast_path {
        TEXT well_id PK, FK
        TEXT parameter PK, FK
        TEXT origin_ts PK, FK
        INTEGER horizon_h PK
        TEXT model PK
        TEXT target_ts
        REAL y_pred
    }
```

## Notes on the design

- `measurements_raw` keeps every published CSV row as text, so a rejected or masked value can always be traced to its source record.
- `measurements_long` is the normalized fact table: one row per well, hour and sensor, with unit, flag and usability mask.
- `measurements_wide` is the pivot: one row per well-hour on the full calendar (10 wells × 6,552 hours), with unusable, absent and inapplicable values as NULL. It is derived from `measurements_long` and kept for lag features and sensor comparison.
- `series_catalog` records which sensors apply to which well, so a missing series can be told apart from an inapplicable one.
- `dataset_meta` is a key–value table with the dataset revision, file hashes and the published train/test boundaries.
- The three forecast tables are written only by `wells.forecast.run`.
