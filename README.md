# Short-Term Forecasting of Well Parameters

Инженерная часть собирает SQLite из опубликованных почасовых записей десяти скважин. База хранит исходные строки, историю измерений, полный календарь, каталог применимости и журнал качества.

## Сборка данных

Из корня репозитория установите зависимости в окружение Python 3.12. Команды ниже используют PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m wells.data.pipeline --raw-dir data/raw --db data/processed/wells.sqlite
```

Сборка скачивает `train.csv` и `test.csv` ревизии `4004bb78ca8811a87c2892a89e457c59d22ace2b`. При каждом запуске она проверяет SHA-256, заголовок, число записей и границы опубликованных разделов. При несовпадении хеша сначала уберите повреждённый файл из каталога сырых данных, затем повторите команду. Обе границы каждого раздела включены. Почасовые метки хранятся в шкале источника без часового пояса. [Карточка NK v2](https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2), [train.csv](https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2/resolve/4004bb78ca8811a87c2892a89e457c59d22ace2b/train.csv), [test.csv](https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2/resolve/4004bb78ca8811a87c2892a89e457c59d22ace2b/test.csv), [манифест и контрольные суммы](wells/data/ingest.py).

Перед пересборкой остановите Streamlit и завершите запись прогнозов. Сборка проверяет временную SQLite перед заменой рабочего файла. После успешной замены три таблицы результатов пусты и прогнозы следует рассчитать заново.

### Подключение прогноза и интерфейса

После сборки данных выполните диагностику, опубликуйте прогноз по обеим целям и запустите дашборд на той же БД.

```powershell
python -m wells.forecast.run --db data/processed/wells.sqlite --diagnostic
python -m wells.forecast.run --db data/processed/wells.sqlite --targets pump_fillage tubing_pressure |
    Set-Content -Encoding utf8 data/processed/forecast-report.json
streamlit run app.py -- --db data/processed/wells.sqlite
```

Для автономной проверки чтения примените [схему SQLite](wells/data/sql/schema.sql), включите внешние ключи и загрузите [фикстуру](tests/fixtures/seed.sql) с примерами метрик и прогнозов. [Сквозные тесты](tests/test_integration.py) проверяют сборку, прогноз и чтение результатов.

## Проверки

```powershell
python -m pytest -q -rs
$env:WELLS_RUN_OFFICIAL = '1'
python -m pytest -q -rs
Remove-Item Env:WELLS_RUN_OFFICIAL
```

Первый запуск работает на синтетической фикстуре. Второй проверяет официальные CSV, сборку во временной БД и воспроизводимость сквозного сценария. Если входы лежат в другом каталоге, задайте `WELLS_RAW_DIR`.

## SQL для проверки и защиты

Выполните из корня проекта следующий пример, чтобы найти самый длинный разрыв NK-86 и вывести покрытие `line_pressure` для NK-75 по месяцам тестового раздела.

```powershell
@'
import sqlite3
from contextlib import closing
from pathlib import Path

sql = Path("wells/data/sql")
db = Path("data/processed/wells.sqlite").resolve()
with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
    gaps = conn.execute((sql / "gaps.sql").read_text(encoding="utf-8"), {
        "well_id": "NK-86", "parameter": None,
        "start_ts": "2026-01-01 00:00:00", "end_ts": "2026-10-01 00:00:00",
    }).fetchall()
    print(max(gaps, key=lambda row: row[-1]))
    coverage = conn.execute((sql / "aggregates.sql").read_text(encoding="utf-8"), {
        "well_id": "NK-75", "parameter": "line_pressure",
        "start_ts": "2026-08-08 23:00:00", "end_ts": "2026-10-01 00:00:00",
        "grain": "month", "mode": "clean",
    }).fetchall()
    print(coverage)
'@ | python -
```

Ожидаемые результаты — 313 пропущенных часов между наблюдениями NK-86 и 0 наблюдений из 1 273 ожидаемых часов для NK-75. Интервал запроса полуоткрытый `[start_ts, end_ts)`. При пересечении с разрывом `gaps.sql` возвращает его полные границы и длину. Знаменатель каждого суточного или месячного агрегата — ожидаемые часы внутри выбранного интервала, включая только применимые серии. Среднее одинаково взвешивает опубликованные часовые значения. Уровни измерений агрегируются по отдельным параметрам и единицам.

Для показа преобразования long → wide подставьте `NK-7` и `2026-01-03 09:00:00` в оба запроса. Опубликованный SPM за этот час равен примерно 5,69 с флагом `spm_outlier`. В wide SPM равен NULL, а `pump_fillage` остаётся примерно 22,29%.

```sql
SELECT well_id, timestamp, parameter, value, unit, flag, is_usable
FROM measurements_long
WHERE well_id = :well_id AND timestamp = :timestamp
ORDER BY parameter;

SELECT * FROM measurements_wide
WHERE well_id = :well_id AND timestamp = :timestamp;
```

При `is_usable = 0` в long сохраняется опубликованное значение, а в wide соответствующая ячейка становится NULL. Другие датчики часа сохраняют свою маску, и наблюдаемый ноль остаётся нулём. У пропущенного применимого часа нет строки в long. Неприменимость видна по отсутствию пары в каталоге. В режиме `raw` агрегаты используют численно корректные значения long до маски, в режиме `clean` — пригодные. В обоих режимах наблюдаемое покрытие включает также помеченные строки.

Для блока защиты сначала покажите официальный split и этот час до и после pivot. Затем покажите разрыв NK-86 и 13 пар, где минимальная нагрузка превышает максимальную. Обе стороны каждой пары замаскированы. В журнале выведите события следующим запросом.

```sql
SELECT well_id, parameter, timestamp, end_ts, issue_code
FROM quality_issues
WHERE action = 'annotate'
ORDER BY issue_code, well_id, parameter, timestamp;
```

Для пояснения событий `wells.data.quality.summarize_anomalies` принимает поля `well_id, parameter, timestamp, value` из long и возвращает медианы опубликованных значений по подписанным периодам. У NK-82 медиана площади 7 и 8 мая составляет примерно 12 428 и 207 035. У NK-7 медиана SPM меняется примерно с 4,16 за январь–май до 1,40 с июня, медиана площади — с 3 247 до −1 477, а fillage — с 31,20% до 44,64%. В SQLite эти значения сохраняются в опубликованных единицах. [Аудит качества в отчёте](REPORT.md#4-data-quality-audit).

## Прогноз и оценка

Модуль `wells.forecast` прогнозирует `pump_fillage` на 1–24 часа вперёд для семи насосных скважин и сравнивает три метода: naive, seasonal naive и Ridge-регрессию на лагах. Используется опубликованное хронологическое разбиение без перемешивания. Параметр `alpha` выбирается только на обучающей части (валидация с 2026-07-15), тестовые значения в обучении не участвуют.

Диагностика читает БД и проверяет границы обучения на NK-7. Команда с `--targets pump_fillage tubing_pressure` применяет те же три метода к `tubing_pressure` фонтанных скважин, публикует обе цели в таблицах `backtest_predictions`, `metrics` и `forecast_path` в одной транзакции и выводит JSON-отчёт.

Запуск без `--targets` публикует только `pump_fillage`, заменяя предыдущий набор прогнозов. Доля покрытия в метриках равна числу общих точек трёх моделей, делённому на число пригодных фактических наблюдений цели в официальном тесте.

## Дашборд

```powershell
streamlit run app.py -- --db data/processed/wells.sqlite
```

В боковой панели выбираются тип скважины, скважина, параметр и период. Три раздела показывают историю и EDA, качество данных, прогноз и оценку. В разделе прогноза выберите `pump_fillage` для насосной скважины или `tubing_pressure` для фонтанной. Для опубликованных прогнозов MAE показана в процентных пунктах или атм соответственно. Все пять графиков интерактивные (Plotly).

## SQL-запросы, замеры и графики отчёта

Команды полного отчёта используют БД после [прогона с обеими целями](#подключение-прогноза-и-интерфейса). Запуск на БД с одним `pump_fillage` даёт пустые таблицы MAE для давления и удаляет старый график `pressure_mae_by_horizon.png` из выходного каталога.

```powershell
python -m wells.analysis.sql_report --db data/processed/wells.sqlite --out reports
python -m wells.analysis.benchmark --raw-dir data/raw --db data/processed/wells.sqlite --out reports
python -m wells.analysis.figures --db data/processed/wells.sqlite --out reports/figures
```

- `sql_report` выполняет 14 аналитических запросов (JOIN, GROUP BY/HAVING, CTE, подзапросы, оконные функции `LAG` и `RANK`), загружает результаты в Pandas и сохраняет их в `reports/sql/*.csv` и [reports/sql_queries.md](reports/sql_queries.md).
- `benchmark` измеряет время и память: CSV с типами по умолчанию, оптимизированные типы, чтение по частям (`chunksize`), Parquet, Pandas против SQL и три метода прогноза. Результат — [reports/benchmarks.md](reports/benchmarks.md) и `reports/benchmarks.json`.
- `figures` строит графики для отчёта в `reports/figures/`.

Итоговые материалы доступны в репозитории. [Отчёт](REPORT.md), [ER-диаграмма](docs/ER.md), [презентация на русском](presentation/well-forecast-defense-ru.pptx), [презентация на английском](presentation/well-forecast-defense.pptx).

## Бенчмарк на 500 000 записей давления

```powershell
python -m wells.analysis.pressure_benchmark
```

Команда скачивает три файла Parquet из закреплённой ревизии Petrobras 3W, проверяет SHA-256 и собирает `data/processed/petrobras-pressure.sqlite`. В таблице `pressure_observations` ровно **500 000** конечных значений `P-TPT` (давление на подводной фонтанной арматуре, Па) с уникальными ключами «скважина–метка времени». При совпадении ключей сохраняется первая запись в порядке манифеста. Излишек отсекается по ключу в порядке «скважина, метка времени». Команда сравнивает количество, среднее, минимум и максимум по скважине через SQLite и Pandas и выводит JSON с замерами. [Результат проверки](reports/pressure_benchmark.json).

Файлы `WELL-*` содержат опубликованные ряды трёх реальных скважин Petrobras. Издатель привёл телеметрию к частоте 1 Гц линейной интерполяцией. Почасовой прогноз и его метрики рассчитываются на официальных 258 387 строках NK. Petrobras 3W используется для отдельной проверки производительности. [Источник Petrobras 3W и лицензия CC BY 4.0](https://github.com/petrobras/3W/blob/c86cc7c70063646910be7292de8a54a6fb2a8399/dataset/README.md), [методика подготовки рядов](https://arxiv.org/html/2507.01048v1#S2.SS5).

## Структура проекта

| Путь | Назначение |
| --- | --- |
| `wells/data/` | Загрузка CSV, проверка и аудит качества, pivot long → wide, сборка SQLite, SQL для пропусков и агрегатов |
| `wells/forecast/` | Признаки, baseline-методы, Ridge, оценка на тесте, публикация результатов |
| `wells/ui/`, `app.py` | Чтение SQLite, EDA, графики Plotly и приложение Streamlit |
| `wells/analysis/` | SQL-запросы, замеры времени и памяти на NK и Petrobras 3W, графики отчёта |
| `tests/` | Тесты данных, прогноза, интерфейса и сквозной сборки |
| `reports/` | Сохранённые результаты запросов, замеров и графики |
