# Implementation interfaces

This is the authoritative cross-module contract for Short-Term Forecasting of Well Parameters. [ARCHITECTURE.md](ARCHITECTURE.md) describes the design, [ML_STRATEGY.md](ML_STRATEGY.md) defines the experiment, and [TEAM_TASKS.md](TEAM_TASKS.md) assigns work. Student 1 coordinates contract revisions. A change to a shared schema, signature, SQL output, fixture or command requires review by Students 2 and 3 and matching updates to the implementation and tests in the same integration sequence.

## SHARED CONTRACT READY

**Status: pending implementation approval and A01 evidence.** Student 1 opens the A01 PR from the common planning baseline and coordinates sign-off. Students 2 and 3 each review and approve the contract and fixture before merge. Record the merged commit and the two approvals on the PR.

| A01 exit item | Evidence |
| --- | --- |
| SQLite schema and data types | `schema.sql` implements the tables, column names, affinities, nullability, primary and foreign keys below. Fresh DB initializes with foreign keys enabled and passes integrity checks. |
| Shared input/output and time conventions | All three owners confirm CSV/DB schemas, SQL bindings, Python entry points, timestamp and NULL semantics, CLI commands and paths below. |
| Dependency versions | Commit `requirements.txt` with tested exact versions of Pandas, NumPy, scikit-learn, Plotly, Streamlit and pytest. Record the agreed Python version (proposed 3.12) and actual SQLite runtime version in the PR. Verify clean install, imports and SQLite `LAG`. |
| Shared fixture | Commit `tests/fixtures/seed.sql`. Both consumers initialize it against the same DDL and verify representative input and output rows independently. |
| Ownership | Review the file/write boundaries below and confirm a single writer for each shared file. |

Before explicit implementation approval, work is limited to planning and review. After approval and before this milestone, Student 1 works on A01. Student 2 can review temporal formulas and supply hand-computed fixture cases. Student 3 can draft widget/state cases and review SQL contracts. A02–A08, M01–M08 and U01–U08 retain their task dependencies and start only after the milestone. After A01, fixture-based M01–M02 and U01–U05 can proceed alongside A02–A05, subject to their internal prerequisites. U01 may wire the named query adapters before A05, but execution of the real `gaps.sql` and `aggregates.sql` waits for A05. Before then, use expected tabular inputs or test doubles at the query boundary for EDA and UI tests. Student 1 supplies reference SQL output examples with the seed PR. Production code uses Student 1's SQL files. A04 enables the official-data ML diagnostic, A05 enables SQL reconciliation, M05 enables real U06, and A06, M07 and U07 enable A07.

## Inputs, paths and commands

Run from the repository root. `--raw-dir` defaults to `data/raw` and `--db` to `data/processed/wells.sqlite`. Pass the same database path to all stages when overriding it.

| Operation | Command |
| --- | --- |
| Build data from the pinned `train.csv` and `test.csv` | `python -m wells.data.pipeline --raw-dir PATH --db PATH` |
| Train-only diagnostic on the built DB | `python -m wells.forecast.run --db PATH --diagnostic` |
| Full official experiment and publication | `python -m wells.forecast.run --db PATH` |
| Read the populated database | `streamlit run app.py -- --db PATH` |
| Default fixture-based suite | `python -m pytest` |

The forecast command opens an existing DB and fails on missing input or incompatible schema. Only the data build acquires the CSVs. The input files are specified in [DATASET_ANALYSIS.md](DATASET_ANALYSIS.md#источник-и-разбиение), including revision and hashes. Preserve their published split. The CSV header is `well_id,well_type,timestamp,parameter,value,unit,n_obs,flag`. Source values are hourly means. Use finite REAL for validated `value`, a positive INTEGER for `n_obs`, and TEXT for the other CSV fields. Preserve raw field text. Blank `flag` means no published flag. Preserve nonblank flags and split combined codes on semicolons for auditing.

## Time, types and missing values

Store hour labels as SQLite TEXT in `YYYY-MM-DD HH:MM:SS`, with minutes and seconds zero for observations and forecast hour labels. Use source clock time with no assigned time zone. Python calculations use timezone-naive pandas timestamps or a DatetimeIndex. The hour labeled `t` covers `[t,t+1h)`. Under the availability assumption, its aggregate is known at `t+1h`. A release with `origin_ts=t` is made at `t+1h`. Its `target_ts=t+h` labels the target hour, whose aggregate becomes known at `t+h+1h`. Display the origin hour explicitly and label any displayed issue time as its closing time. Dataset metadata ends are inclusive. SQL UI intervals use `[start_ts,end_ts)`. For an inclusive selected end date, pass the next midnight as `end_ts`.

Use SQL NULL for unavailable numeric values and Pandas missing values internally. Keep observed zero. In long, a row with `is_usable=0` retains its published numeric value. In wide, absent, masked and inapplicable values are NULL. Use the catalog and long to distinguish those states. The catalog applies over the entire archive. An applicable sensor with no rows in a slice stays selectable. An inapplicable sensor is excluded from expected counts.

For the pinned revision, metadata labels are `train_start=2026-01-01 00:00:00`, `train_end=2026-08-08 22:00:00`, `test_start=2026-08-08 23:00:00` and `test_end=2026-09-30 23:00:00`. The first official scoring origin is `train_end`, after that hour closes. The first possible scored target at horizon `h` is `test_start + (h−1) hours`. Internal validation cutoffs and fit eligibility are specified in [ML_STRATEGY.md](ML_STRATEGY.md#хронологический-протокол).

## SQLite schema

Student 1 defines DDL in `wells/data/sql/schema.sql`. Columns below are NOT NULL unless suffixed with `?`. Keys inherit the listed affinities. All identifiers, split/model fields, dates, audit text and metadata values are TEXT except as explicitly typed. Enable `PRAGMA foreign_keys=ON` on each writer and fixture connection before opening a transaction. Use default restrictive FK behavior. Insert parent tables before children.

| Table | Primary key | Other columns and constraints | Foreign keys |
| --- | --- | --- | --- |
| `dataset_meta` | `key` | `value`. Keys `revision, source_url, train_start, train_end, test_start, test_end, train_sha256, test_sha256`. | None |
| `series_catalog` | `well_id, parameter` | `well_type, unit`. Types `fountain` or `sucker_rod_pump`. | None |
| `measurements_raw` | `source_file, source_row INTEGER` | Eight original CSV fields as TEXT, plus `split`. Basename and positive 1-based data record index, excluding header. | None. Keep rejected source content. |
| `measurements_wide` | `well_id, timestamp` | `well_type, split`, plus nullable REAL `SPM, casing_pressure, dynamometer_area, line_pressure, max_rod_weight, min_rod_weight, pump_fillage, tubing_pressure`. One row per well-hour. | None. The builder validates well IDs and type against the catalog. |
| `measurements_long` | `well_id, timestamp, parameter` | `value REAL, unit, n_obs INTEGER, flag, split, is_usable INTEGER`. Normalize absent flag to empty text. Positive `n_obs`. `is_usable` is 0 or 1. | `(well_id, parameter)` → `series_catalog`. `(well_id, timestamp)` → `measurements_wide`. |
| `quality_issues` | `issue_id INTEGER` | `well_id?, parameter?, timestamp?, end_ts?, source_file?, source_row? INTEGER, issue_code, action`. Source reference columns are both NULL or both set. | `(source_file, source_row)` → `measurements_raw`. Sensor context can describe rejected keys. |
| `backtest_predictions` | `well_id, parameter, origin_ts, horizon_h INTEGER, model` | `target_ts, y_true REAL, y_pred REAL`. One row per method per common scored key. | `(well_id, parameter)` → catalog. `(well_id, origin_ts)` and `(well_id, target_ts)` → wide well-hour keys. |
| `metrics` | `well_id, parameter, horizon_h INTEGER, model` | `mae? REAL, n_common INTEGER, n_actual INTEGER, coverage? REAL`. `0 <= n_common <= n_actual`. | `(well_id, parameter)` → catalog. |
| `forecast_path` | `well_id, parameter, origin_ts, horizon_h INTEGER, model` | `target_ts, y_pred REAL`. One common latest origin per series with 24 horizons per model, or zero rows. | `(well_id, parameter)` → catalog. `(well_id, origin_ts)` → wide well-hour key. The target may lie beyond the archive. |

`split` is `train` or `test`. The builder validates contiguous metadata bounds, split consistency and matching long units, and indexes long by `(well_id, parameter, timestamp)`. `model` is `naive`, `seasonal_naive` or `ridge`. `horizon_h` is in 1–24. The writer validates `target_ts = origin_ts + horizon_h hours`, prediction ranges, shared keys and complete release groups before committing. MAE is NULL when `n_common=0`. Coverage is NULL when `n_actual=0`, otherwise `n_common/n_actual`. MAE for `pump_fillage` is in percentage points. `n_actual` counts all usable official test targets for the well, even if they fall before the first eligible target for a longer horizon.

In the audit, an observation issue with a timestamp covers one hour. A period annotation has `action=annotate`, a required inclusive start and optional exclusive end. A NULL end on an annotation means ongoing within the archive. A NULL timestamp on a rejected record indicates it could not be placed on the calendar. Keep raw details accessible. For gaps, the well-level query uses distinct well-hours across sensors. Apply overlap to the missing interval `[previous_ts+1h,timestamp)` and return its full length. Aggregate coverage counts validated long rows as observed, including flagged rows. `n_flagged` counts each observed row with a nonblank published flag once. Raw mode uses validated pre-mask values. Clean mode uses usable values. An empty bucket has zero observations and NULL value statistics.

## Module interfaces

DataFrame and Series denote Pandas types. DB-facing DataFrames use the named columns, including when empty. Convert time labels at the SQLite boundary. Student 1 owns the SQL query results; Student 3 wraps the same SQL in Python.

### Data pipeline functions and SQL

- `download_inputs(raw_dir: Path) -> None` verifies both files. `read_inputs(raw_dir: Path) -> tuple[DataFrame, dict[str, str]]` returns raw fields plus provenance and `dataset_meta` keys. `validate_measurements(raw, catalog, meta) -> tuple[DataFrame, DataFrame]` returns the long and issue tables. `summarize_anomalies(long) -> DataFrame` computes period statistics used by the audit and acceptance review. These are internal data interfaces.
- `build_wide(long, catalog, meta) -> DataFrame` returns the ordered wide schema. `assemble_database(raw, catalog, meta, db_path: Path) -> None` runs validation, pivot, load and database checks on already-read inputs. `build_database(raw_dir: Path, db_path: Path) -> None` verifies the official inputs and invokes the assembly helper. `main() -> int` provides the CLI. Fixture tests pass raw rows and synthetic metadata to the assembly helper.
- Supply `gaps.sql` with bound `well_id, parameter, start_ts, end_ts`. Use NULL `parameter` for a well-hour gap query. Return `well_id, parameter, previous_ts, timestamp, delta_hours, missing_hours`. Use NULL `parameter` in well-hour results. Report interior gap lengths in full even when they only overlap the selection.
- Supply `aggregates.sql` with bound `well_id, parameter, start_ts, end_ts, grain, mode`. Allow NULL `well_id` or `parameter` to include all matching catalog pairs. Allow `grain=day/month` and `mode=raw/clean`. Return `well_id, parameter, well_type, unit, period_start, n_expected, n_observed, n_usable, n_flagged, n_values, mean_value, min_value, max_value, coverage`. `coverage = n_observed / n_expected`. `n_values` equals the count used for the statistics in the selected mode. The `n_usable` and `n_flagged` counts are independent of mode.

### Forecast module interfaces

- `build_features(series: Series, horizon_h: int) -> DataFrame` accepts one well's datetime-indexed hourly target. It returns `origin_ts, target_ts, horizon_h, y_true, seasonal_value` and the named predictors. Well identity is supplied by the caller. `eligible_fit_rows(frame, *, well_id, cutoff, regime_start) -> DataFrame` selects eligible fit rows. Pass fixture bounds as arguments in tests.
- `predict_naive(frame) -> Series` and `predict_seasonal_naive(frame) -> Series` align to the input row index. `fit_ridge(rows, alpha: float) -> Pipeline` selects only the Ridge inputs. `predict_ridge(pipeline, frame) -> Series` aligns results, leaving incomplete rows missing and clipping finite predictions.
- `select_alpha(frames, *, validation_start, test_start, nk7_regime_start) -> tuple[float, DataFrame]` returns the selected alpha and per-candidate, per-group validation rows keyed by `(alpha, well_id, horizon_h)` with `mae` and `n_validation`. `evaluate_backtest(frames, pipelines, *, test_start, test_end) -> tuple[DataFrame, DataFrame]` returns `backtest_predictions` and `metrics` in the shared schema. `test_end` is inclusive. `build_forecast_path(series_by_well, pipelines, *, test_start) -> DataFrame` returns release rows under that schema.
- `run_forecast(db_path: Path, *, validation_start=None, nk7_regime_start=None) -> dict` is the integration callable used by Student 1. It returns the successful run report after publishing results. Defaults apply the approved production dates. Tests can pass synthetic dates. The test boundary and archive end come from `dataset_meta`. `main() -> int` implements the CLI and diagnostic mode described below. Pandas, NumPy, scikit-learn, SQLite and pytest are the supported toolset.

### SQL and UI module interfaces

- `open_database(db_path: Path) -> sqlite3.Connection` opens a read-only connection. `read_meta(conn) -> dict[str, str]` and `read_catalog(conn) -> DataFrame` supply metadata and the four catalog columns. `read_history(conn, *, well_id, parameter, start_ts, end_ts, mode) -> DataFrame` returns hourly `well_id, parameter, timestamp, unit, value, display_value, n_obs, flag, is_usable` on the calendar. The mode is `raw` or `clean`. `read_well_matrix(conn, *, well_id, start_ts, end_ts, mode) -> DataFrame` returns a time-indexed frame with applicable parameter columns. Clean uses wide. Raw uses validated long pivoted onto the same calendar.
- `read_gaps(conn, *, well_id, parameter, start_ts, end_ts) -> DataFrame` executes `wells/data/sql/gaps.sql`. Use NULL parameter for well-hour gaps. Read `well_id, parameter, previous_ts, timestamp, delta_hours, missing_hours`. `read_aggregates(conn, *, well_id, parameter, start_ts, end_ts, grain, mode) -> DataFrame` executes `wells/data/sql/aggregates.sql`. Allow NULL well or parameter for all applicable matches. Grain is `day` or `month`. Read `well_id, parameter, well_type, unit, period_start, n_expected, n_observed, n_usable, n_flagged, n_values, mean_value, min_value, max_value, coverage`. Coverage uses `n_observed/n_expected`. `n_values` reflects the selected mode. `read_issues(conn, *, well_id, parameter, start_ts, end_ts) -> DataFrame` returns overlapping audit rows, including relevant well-wide annotations. Return all audit columns plus the eight raw CSV fields as `raw_<column>` where source references exist. For a rejected record without a normalized hour, expose it separately in the audit view. Use its valid source context for well and parameter matching when available. Unknown context stays visible in the unplaced-record list.
- `read_backtest(conn, *, well_id, parameter, horizon_h, start_ts, end_ts, models) -> DataFrame` uses `target_ts` for date filtering. `read_metrics(conn, *, well_id, parameter, models) -> DataFrame` returns all stored horizons for the official test. `read_forecast_path(conn, *, well_id, parameter, models) -> DataFrame` returns the stored latest release without applying the date slice. Sort forecasts by timestamp and model. Student 1 uses the read functions in the A07 integration test. `main() -> None` starts the app. `render_filters` returns a validated selection passed to the three section renderers. Support the shared `--db` argument after Streamlit's `--` separator.
- EDA functions accept data frames and return data frames. Use `describe_observations(history)` and `hourly_profile(history)` with `display_value`. Use `pairwise_spearman(matrix, min_pairs=30)` to return long-form `parameter_x, parameter_y, rho, n_pairs`. Use `coverage_by_type(aggregates)` to sum counts by `well_type, period_start` and derive coverage. Figure functions take prepared data frames and return Plotly figures. Use `history_figure(history, aggregates, issues)`, `profile_figure(profiles)`, `correlation_figure(correlations)`, `coverage_figure(coverage)`, and `forecast_figure(actuals, predictions, metrics, *, mode, origin_ts, archive_end, test_start, test_end)`. Pass labels and units from the prepared data or metadata. Query and fit work stays outside chart functions.

## Forecast diagnostic and run report

`--diagnostic` is a read-only run on NK-7 at horizons 1 and 24 after A04. Use only official train rows and the same inner-fit and validation predicates, candidates, feature order and clipping as the full run. Report eligible counts, successful per-candidate fitting and finite validation MAE. Check cutoff and seasonal-source assertions for both horizons. Return nonzero with well, horizon and stage if a required check fails. Diagnostic alpha is provisional. It must not select configuration for the full run. Diagnostic mode writes no DB rows and uses no test observations. Student 2 runs the analogous small check on a synthetic fixture before the real diagnostic. The full experiment still evaluates seven wells × 24 horizons × three methods.

For a successful full CLI run, emit one JSON object to stdout after committing results. Log progress and failures to stderr. `run_forecast` returns the same report as a dict. Capture stdout to an ignored local file such as `data/processed/forecast-report.json` for integration evidence. Use these top-level fields:

| Field | Required content |
| --- | --- |
| `dataset` | `revision`, `train_sha256`, `test_sha256`. |
| `code` | Git commit and dirty status, or explicit null values if unavailable. Run the defense build from a recorded clean revision. |
| `environment` | Actual Python, SQLite, NumPy, Pandas and scikit-learn versions. |
| `config` | Fit/validation/test cutoffs, NK-7 regime start, wells and horizons, feature order, scaler/Ridge settings, clipping range, candidate grid and tie rule. |
| `validation` | `selected_alpha`, validation rows with `alpha, well_id, horizon_h, mae, n_validation`, and macro MAE per candidate. |
| `final_fit_counts` | One row with `well_id, horizon_h, n_fit` per fitted group. |
| `table_counts` | Row counts keyed by `backtest_predictions`, `metrics` and `forecast_path`. |
| `release_origins` | Latest eligible hour label or null keyed by well. |

Use standard JSON values and serialize missing values as null. Keep the numerical report from the full experiment with the matching DB build. No trained model serialization is required.

## Fixture, publication and ownership

Student 1 maintains `tests/fixtures/seed.sql` with synthetic rows inserted into a disposable DB initialized by canonical DDL. Student 2 supplies forecast example rows and temporal cases. Student 3 supplies UI state cases. Include both well types, a zero, a masked flagged reading, a hole, an entirely absent applicable sensor in one slice, an inapplicable sensor, known forecast/metric rows, and enough continuous training, validation and test history for all fixture horizons in at least one target well. Include explicit synthetic cutoff values in the fixture test inputs. For A07 start from synthetic raw rows and rebuild outputs, then compare to independent expectations. Use a distinct fixture revision and fixture cutoffs. Never feed fixture data to the production CSV manifest checks. To simulate pre-forecast UI, leave all result tables empty. To simulate schema mismatch, remove a required table in a temporary test copy only.

| Owner | Writes |
| --- | --- |
| Student 1 | `wells/__init__.py`, `wells/data/`, `tests/fixtures/`, `tests/test_data.py`, `tests/test_integration.py`, `requirements.txt`, `.gitignore`, `README.md`. Owns all DDL and data-table rows. |
| Student 2 | `wells/forecast/` and `tests/test_forecast.py`. Writes rows to the three forecast tables through the runner. |
| Student 3 | `wells/ui/`, `app.py` and `tests/test_ui.py`. Opens SQLite read-only. |

All agents may read each module and test. Shared files have one writer. Changes to the contract, shared fixture, pins or DDL are coordinated by Student 1 and reviewed by both consumers. Keep downloaded CSVs, generated DBs, reports and environments outside Git. Use a separate local DB per agent or test. In the shared integration run, stop Streamlit and any forecast writer before rebuilding. Build and verify a temporary DB before replacing the destination. A successful data rebuild has empty forecast tables. The full forecast run computes and checks outputs before replacing all three result sets in a single transaction. On failure keep prior results on that DB. Run data build, forecast and UI in order and do not rebuild the DB concurrently with a forecast run.

## GitHub integration

Use one shared GitHub repository and one agreed planning commit on `main`. If the repository is still unborn, the students designate who creates and publishes that first commit. Share the baseline commit before creating feature branches. Student branches are `codex/student-1-data`, `codex/student-2-forecast` and `codex/student-3-dashboard`. Submit A01 first and base all database-dependent work on its approved commit. Use the dedicated branches for component PRs. Land data A02–A06, forecast M01–M06, then dashboard U01–U07, while reviewing independent work in parallel. Student 2 verifies M07 with Student 3 after U06. Student 1 coordinates the A07 integration PR once A06, M07 and U07 have passed. A08, M08 and U08 follow the verified A07 build.

Each PR names task IDs, the base/contract revision, owned files, fixture and official checks run, and dependencies still pending. Require another student's review for component PRs and both consumers for A01 or a shared change. The owning student resolves conflicts and defects in their module. For an interface blocker, report the task, producer and consumer revisions, command, expected contract, observed result and minimal reproduction. Resolve shared changes in one owner-led PR. A07 records the three reviewed revisions, runs fixture and official checks on the integrated revision, and obtains sign-off from all students before using that build for defense.
