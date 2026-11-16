"""Streamlit dashboard for NK oil-well history, data quality and short-term forecasts.

Run: streamlit run app.py -- --db data/processed/wells.sqlite
"""

import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from wells.ui import charts, eda, queries

DEFAULT_DB = Path("data/processed/wells.sqlite")
BUILD_COMMAND = "python -m wells.data.pipeline --raw-dir data/raw --db {db}"
FORECAST_COMMAND = 'python -m wells.forecast.run --db "{db}" --targets pump_fillage tubing_pressure'
FORECAST_TARGETS = {"sucker_rod_pump": "pump_fillage", "fountain": "tubing_pressure"}
TYPE_LABELS = {"fountain": "Фонтанная (fountain)", "sucker_rod_pump": "Насосная ШГН (sucker_rod_pump)"}
MODE_LABELS = {"raw": "Исходные (до маски)", "clean": "Очищенные (по маске качества)"}
GRAIN_LABELS = {"day": "День", "month": "Месяц"}
DEFAULT_DAYS = 14


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NK wells dashboard")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args


def _label(moment) -> str:
    return pd.Timestamp(moment).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- filters

def _keep_valid(key, options, default):
    """Restore a widget value only when it is still valid for the current parent choice."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]


def _date_period(first, last):
    complete = st.session_state.get("complete_dates")
    if complete is None or not first <= complete[0] <= complete[1] <= last:
        complete = (max(first, last - timedelta(days=DEFAULT_DAYS - 1)), last)

    draft = st.session_state.get("dates")
    if (not isinstance(draft, (tuple, list)) or len(draft) > 2
            or any(not first <= day <= last for day in draft)
            or (len(draft) == 2 and draft[0] > draft[1])):
        st.session_state["dates"] = complete

    picked = st.sidebar.date_input("Период (обе даты включительно)", key="dates", min_value=first, max_value=last)
    if len(picked) == 2:
        complete = picked
    else:
        st.sidebar.caption("Выберите обе даты. Пока используется предыдущий период.")
    st.session_state["complete_dates"] = complete
    return complete


def jump_to(*, well_type, well_id, parameter, start, end):
    """Callback for event shortcuts. It runs before widgets are created on the next rerun."""
    st.session_state.update(well_type=well_type, well_id=well_id, parameter=parameter,
                            dates=(pd.Timestamp(start).date(), pd.Timestamp(end).date()))


def render_filters(catalog, meta) -> dict:
    """Sidebar: type, well, parameter, dates. Child options always come from the catalog."""
    sidebar = st.sidebar
    sidebar.header("Фильтры")
    types = sorted(catalog["well_type"].unique())
    _keep_valid("well_type", types, "sucker_rod_pump")
    well_type = sidebar.selectbox("Тип скважины", types, key="well_type", format_func=lambda v: TYPE_LABELS.get(v, v))

    of_type = catalog.loc[catalog["well_type"].eq(well_type)]
    wells = sorted(of_type["well_id"].unique(), key=lambda name: (len(name), name))
    _keep_valid("well_id", wells, wells[0])
    well_id = sidebar.selectbox("Скважина", wells, key="well_id")

    of_well = of_type.loc[of_type["well_id"].eq(well_id)]
    parameters = sorted(of_well["parameter"])
    _keep_valid("parameter", parameters, FORECAST_TARGETS.get(well_type))
    parameter = sidebar.selectbox("Параметр", parameters, key="parameter")
    unit = of_well.set_index("parameter").loc[parameter, "unit"]

    first = pd.Timestamp(meta["train_start"]).date()
    last = pd.Timestamp(meta["test_end"]).date()
    start_date, end_date = _date_period(first, last)

    same = catalog.loc[catalog["parameter"].eq(parameter) & catalog["unit"].eq(unit)]
    others = sorted(set(same["well_id"]) - {well_id}, key=lambda name: (len(name), name))
    compare_key = f"compare::{parameter}::{unit}"
    if compare_key in st.session_state:
        st.session_state[compare_key] = [name for name in st.session_state[compare_key] if name in others]
    compare = sidebar.multiselect(f"Сравнить с другими скважинами ({parameter}, {unit})", others, key=compare_key)

    sidebar.caption(f"Время источника без часового пояса. Архив: {_label(meta['train_start'])} — "
                    f"{_label(meta['test_end'])}. Test с {_label(meta['test_start'])}.")
    return dict(
        well_type=well_type, well_id=well_id, parameter=parameter, unit=unit, compare=compare,
        start_date=start_date, end_date=end_date,
        start_ts=queries.to_label(start_date),
        end_ts=queries.to_label(pd.Timestamp(end_date) + pd.Timedelta(days=1)),  # inclusive end date
    )


def _well_colors(catalog, selection):
    """Slots follow each well's position within the wells measuring this parameter."""
    same = catalog.loc[catalog["parameter"].eq(selection["parameter"]) & catalog["unit"].eq(selection["unit"])]
    return charts.series_colors(sorted(same["well_id"].unique(), key=lambda name: (len(name), name)))


def _window(selection):
    return dict(start_ts=selection["start_ts"], end_ts=selection["end_ts"])


def _period_text(selection):
    return f"{selection['start_date']:%Y-%m-%d} — {selection['end_date']:%Y-%m-%d} включительно"


# ---------------------------------------------------------------- history and EDA

def render_history(conn, catalog, meta, selection) -> None:
    st.subheader(f"{selection['well_id']} · {selection['parameter']}, {selection['unit']}")
    left, middle, right = st.columns([2, 1, 2])
    mode = left.radio("Режим значений", queries.MODES, index=1, horizontal=True, key="mode",
                      format_func=MODE_LABELS.get)
    grain = middle.radio("Тренд", queries.GRAINS, horizontal=True, key="grain", format_func=GRAIN_LABELS.get)
    right.caption("Исходный режим показывает опубликованные значения, включая помеченные. "
                  "Очищенный скрывает строки с is_usable = 0. Пропущенные часы разрывают линию.")

    wells = [selection["well_id"], *selection["compare"]]
    window = _window(selection)
    history = pd.concat([queries.read_history(conn, well_id=well, parameter=selection["parameter"], mode=mode, **window)
                         for well in wells], ignore_index=True)
    aggregates = pd.concat([queries.read_aggregates(conn, well_id=well, parameter=selection["parameter"],
                                                    grain=grain, mode=mode, **window) for well in wells],
                           ignore_index=True)
    issues = queries.read_issues(conn, well_id=selection["well_id"], parameter=selection["parameter"], **window)
    colors = _well_colors(catalog, selection)

    observed = int(history.loc[history["well_id"].eq(selection["well_id"]), "value"].notna().sum())
    expected = int(history["well_id"].eq(selection["well_id"]).sum())
    if observed == 0:
        st.info(f"Для {selection['well_id']} {selection['parameter']} нет наблюдений за {_period_text(selection)}: "
                f"0 из {expected} ожидаемых часов. Датчик применим и остаётся в списке.")
    st.plotly_chart(charts.history_figure(history, aggregates, issues, colors=colors,
                                          trend_label=f"среднее за {GRAIN_LABELS[grain].lower()}"),
                    use_container_width=True, key="history_chart")

    st.markdown("#### Описательная статистика")
    st.caption(f"Все скважины с параметром {selection['parameter']} в единицах {selection['unit']}, "
               f"период {_period_text(selection)}, режим «{MODE_LABELS[mode]}». "
               "std — выборочное стандартное отклонение.")
    observations = queries.read_observations(conn, mode=mode, **window)
    same = observations.loc[observations["parameter"].eq(selection["parameter"])
                            & observations["unit"].eq(selection["unit"])]
    table = eda.describe_observations(same, by=("well_type", "well_id", "parameter", "unit"))
    st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Статистика всех параметров обоих типов скважин"):
        st.dataframe(eda.describe_observations(observations, by=("well_type", "well_id", "parameter", "unit")),
                     hide_index=True, width="stretch")

    st.markdown("#### Сравнение типов скважин: покрытие и качество")
    all_pairs = queries.read_aggregates(conn, well_id=None, parameter=None, grain="month", mode=mode, **window)
    by_type = eda.coverage_by_type(all_pairs)
    totals = by_type.groupby("well_type")[eda.COUNT_COLUMNS].sum().reset_index()
    totals["coverage"] = totals["n_observed"] / totals["n_expected"]
    totals["flagged_share"] = totals["n_flagged"] / totals["n_observed"]
    series = all_pairs.groupby("well_type")[["well_id", "parameter"]].apply(lambda f: len(f.drop_duplicates()))
    totals.insert(1, "n_series", totals["well_type"].map(series))
    st.caption("Знаменатель покрытия — сумма ожидаемых часов всех применимых пар скважина–параметр по каталогу. "
               "Помеченные строки считаются наблюдёнными. Доля флагов — от наблюдённых строк.")
    st.dataframe(totals, hide_index=True, width="stretch", column_config={
        "coverage": st.column_config.NumberColumn("coverage", format="percent"),
        "flagged_share": st.column_config.NumberColumn("flagged_share", format="percent")})

    st.markdown("#### Суточный профиль")
    profile = eda.hourly_profile(history)
    st.plotly_chart(charts.profile_figure(profile, colors=colors), use_container_width=True, key="profile_chart")

    st.markdown(f"#### Корреляции параметров {selection['well_id']}")
    matrix = queries.read_well_matrix(conn, well_id=selection["well_id"], mode=mode, **window)
    correlations = eda.pairwise_spearman(matrix)
    st.caption("Spearman по часам, где оба параметра есть. Коэффициент скрыт, если пар меньше 30 "
               "или один из рядов постоянен. Для площади в мае сравнивайте периоды до и после сдвига отдельно.")
    st.plotly_chart(charts.correlation_figure(correlations), use_container_width=True, key="correlation_chart")


# ---------------------------------------------------------------- data quality

def _event_shortcuts(catalog, meta, annotations):
    st.markdown("#### Известные события датасета")
    if annotations.empty:
        st.caption("В журнале нет аннотированных периодов.")
        return
    archive_end = pd.Timestamp(meta["test_end"])
    types = catalog.drop_duplicates("well_id").set_index("well_id")["well_type"]
    events = annotations.groupby("issue_code", sort=True)
    columns = st.columns(max(len(events), 1))
    for column, (code, rows) in zip(columns, events):
        first = rows.sort_values(["timestamp", "well_id"]).iloc[0]
        start = first.timestamp - pd.Timedelta(days=7)
        end = (first.end_ts if pd.notna(first.end_ts) else first.timestamp) + pd.Timedelta(days=7)
        start, end = max(start, pd.Timestamp(meta["train_start"])), min(end, archive_end)
        parameter = "dynamometer_area" if code == "area_scale_shift" else first.parameter
        if code == "nk7_regime_change" and "SPM" in set(rows["parameter"]):
            parameter = "SPM"
        label = charts.EVENT_LABELS.get(code, code)
        column.button(f"Показать: {label}", key=f"jump::{code}", on_click=jump_to,
                      kwargs=dict(well_type=types[first.well_id], well_id=first.well_id, parameter=parameter,
                                  start=start, end=end), width="stretch")
        ends = "до конца архива" if pd.isna(first.end_ts) else f"до {first.end_ts:%Y-%m-%d %H:%M} (не включительно)"
        column.caption(f"{rows['well_id'].nunique()} скв., {rows['parameter'].nunique()} парам. "
                       f"С {first.timestamp:%Y-%m-%d %H:%M} {ends}.")


def _before_after(conn, meta, selection, annotations):
    relevant = annotations.loc[annotations["well_id"].eq(selection["well_id"])
                               & annotations["parameter"].eq(selection["parameter"])]
    if relevant.empty:
        return
    archive_start, archive_end = pd.Timestamp(meta["train_start"]), pd.Timestamp(meta["test_end"]) + pd.Timedelta(hours=1)
    history = queries.read_history(conn, well_id=selection["well_id"], parameter=selection["parameter"], mode="raw",
                                   start_ts=queries.to_label(archive_start), end_ts=queries.to_label(archive_end))
    for event in relevant.itertuples():
        label = charts.EVENT_LABELS.get(event.issue_code, event.issue_code)
        end = event.end_ts if pd.notna(event.end_ts) else archive_end
        if event.issue_code == "area_scale_shift" and pd.notna(event.end_ts):
            periods = [("до события", event.timestamp - (event.end_ts - event.timestamp), event.timestamp),
                       ("в периоде события", event.timestamp, end), ("после события", end, end + (end - event.timestamp))]
        else:
            periods = [("до события", archive_start, event.timestamp), ("с начала события", event.timestamp, end)]
        periods = [(name, max(start, archive_start), min(stop, archive_end)) for name, start, stop in periods]
        st.markdown(f"**До и после: {label}, {selection['parameter']} {selection['well_id']}** "
                    f"(опубликованные значения, интервалы [начало, конец))")
        st.dataframe(eda.period_summary(history, periods), hide_index=True, width="stretch")


def render_quality(conn, catalog, meta, selection) -> None:
    st.subheader("Качество данных")
    window = _window(selection)
    annotations = queries.read_annotations(conn)
    _event_shortcuts(catalog, meta, annotations)
    _before_after(conn, meta, selection, annotations)

    st.markdown(f"#### Покрытие {selection['parameter']} по скважинам и дням")
    wells = catalog[["well_id", "well_type"]].drop_duplicates()
    wells = wells.loc[wells["well_type"].eq(selection["well_type"])] if selection["well_type"] else wells
    sensor = queries.read_aggregates(conn, well_id=None, parameter=selection["parameter"], grain="day", mode="raw", **window)
    sensor = sensor.loc[sensor["unit"].eq(selection["unit"])]
    st.plotly_chart(charts.coverage_figure(eda.coverage_by_well(sensor, wells)), use_container_width=True, key="coverage_sensor")

    scope = st.radio("Сводное покрытие всех применимых датчиков", ("Скважины обоих типов", "Типы скважин"),
                     horizontal=True, key="coverage_scope")
    everything = queries.read_aggregates(conn, well_id=None, parameter=None, grain="day", mode="raw", **window)
    combined = (eda.coverage_by_well(everything, catalog[["well_id", "well_type"]].drop_duplicates())
                if scope.startswith("Скважины") else eda.coverage_by_type(everything))
    st.caption("Ячейка — сумма наблюдённых часов по применимым датчикам, делённая на сумму ожидаемых. "
               "Неприменимые пары, например casing_pressure NK-76, в знаменатель не входят.")
    st.plotly_chart(charts.coverage_figure(combined), use_container_width=True, key="coverage_all")

    left, right = st.columns(2)
    series_gaps = queries.read_gaps(conn, well_id=selection["well_id"], parameter=selection["parameter"], **window)
    well_gaps = queries.read_gaps(conn, well_id=selection["well_id"], parameter=None, **window)
    left.markdown(f"**Разрывы ряда {selection['parameter']}**")
    left.caption("Интервалы, пересекающие выбранный период, с полной длиной. Края периода и полностью "
                 "пустой ряд видны по покрытию.")
    left.dataframe(series_gaps, hide_index=True, width="stretch")
    right.markdown(f"**Часы без единой строки по {selection['well_id']}**")
    right.caption("Разрывы на уровне скважины по всем датчикам.")
    right.dataframe(well_gaps, hide_index=True, width="stretch")

    counts = queries.read_aggregates(conn, well_id=selection["well_id"], parameter=selection["parameter"],
                                     grain="day", mode="raw", **window)
    expected, observed = int(counts["n_expected"].sum()), int(counts["n_observed"].sum())
    flagged, usable = int(counts["n_flagged"].sum()), int(counts["n_usable"].sum())
    metric = st.columns(4)
    metric[0].metric("Ожидаемых часов", f"{expected:,}".replace(",", " "))
    metric[1].metric("Наблюдено", f"{observed:,}".replace(",", " "),
                     f"{observed / expected:.1%} покрытие" if expected else None, delta_color="off")
    metric[2].metric("С флагом", f"{flagged:,}".replace(",", " "))
    metric[3].metric("Пригодно", f"{usable:,}".replace(",", " "))

    issues = queries.read_issues(conn, well_id=selection["well_id"], parameter=None, **window)
    located = issues["timestamp"].notna() & issues["well_id"].notna()
    placed = issues.loc[located]
    unplaced = issues.loc[~located]
    st.markdown(f"**Журнал качества {selection['well_id']}** — все параметры, пересечение с выбранным периодом")
    if not placed.empty:
        st.dataframe(placed.groupby(["parameter", "issue_code", "action"], dropna=False).size()
                     .rename("rows").reset_index(), hide_index=True, width="stretch")
    st.caption("Время записи — начало часа; проблема часа закрывается через час. raw_* — исходные поля CSV.")
    st.dataframe(placed, hide_index=True, width="stretch")
    if not unplaced.empty:
        st.markdown("**Отвергнутые записи без нормализованного часа или известной скважины**")
        st.dataframe(unplaced, hide_index=True, width="stretch")


# ---------------------------------------------------------------- forecast

def _metric_table(metrics):
    order = {model: index for index, model in enumerate(queries.MODELS)}
    table = metrics.sort_values(["horizon_h", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    table["model"] = table["model"].map(charts.MODEL_LABELS)
    table["mae"] = table["mae"].map(lambda value: "нет оценки" if pd.isna(value) else f"{value:.3f}")
    return table.loc[:, ["horizon_h", "model", "mae", "n_common", "n_actual", "coverage"]]


def render_forecast(conn, catalog, meta, selection) -> None:
    st.subheader("Прогноз и оценка")
    if selection["parameter"] != FORECAST_TARGETS.get(selection["well_type"]):
        st.info(f"Для {selection['well_id']} {selection['parameter']} доступны история и аудит. "
                "Прогноз поддерживает pump_fillage насосных и tubing_pressure фонтанных скважин.")
        return
    target = dict(well_id=selection["well_id"], parameter=selection["parameter"])
    published_metrics = queries.read_metrics(conn, models=queries.MODELS, **target)
    if published_metrics.empty:
        db = st.session_state.get("db_path", DEFAULT_DB)
        st.warning(f"Прогнозы ещё не рассчитаны для {target['well_id']} {target['parameter']}. "
                   "Выполните полный прогон на этой БД:")
        st.code(FORECAST_COMMAND.format(db=db), language="bash")
        return

    controls = st.columns([2, 3, 2])
    mode = controls[0].radio("Режим", ("backtest", "release"), horizontal=True, key="forecast_mode",
                             format_func={"backtest": "Backtest на test", "release": "Последний выпуск"}.get)
    models = controls[1].multiselect("Модели", queries.MODELS, default=list(queries.MODELS), key="models",
                                     format_func=charts.MODEL_LABELS.get)
    horizon = controls[2].slider("Горизонт, ч", 1, 24, 1, key="horizon", disabled=mode != "backtest")
    if not models:
        st.info("Выберите хотя бы одну модель.")
        return
    metrics = published_metrics.loc[published_metrics["model"].isin(models)]
    test_start, test_end = pd.Timestamp(meta["test_start"]), pd.Timestamp(meta["test_end"])
    archive_end = test_end

    if mode == "backtest":
        predictions = queries.read_backtest(conn, horizon_h=horizon, models=models, **target, **_window(selection))
        actuals = queries.read_history(conn, mode="clean", **target, **_window(selection))
        origin = None
        chosen = metrics.loc[metrics["horizon_h"].eq(horizon)]
        if metrics.empty:
            st.warning("Для этой скважины нет опубликованных метрик официального test.")
        elif not chosen.empty and chosen["n_common"].eq(0).all():
            st.warning(f"Горизонт {horizon} ч: нет оценки. Общих точек 0 из {int(chosen['n_actual'].max())} фактов test.")
        elif predictions.empty:
            st.info(f"В срезе {_period_text(selection)} нет тестовых прогнозов на горизонте {horizon} ч. "
                    f"Test: {_label(test_start)} — {_label(test_end)}. Панель MAE описывает весь test.")
    else:
        predictions = queries.read_forecast_path(conn, models=models, **target)
        origin = None if predictions.empty else predictions["origin_ts"].iloc[0]
        if origin is None:
            st.warning("Нет полного окна последнего выпуска для этой скважины. Backtest и MAE доступны.")
            actuals = queries.read_history(conn, mode="clean", **target, **_window(selection)).iloc[0:0]
        else:
            age = int((archive_end - origin) / pd.Timedelta(hours=1))
            state = "устаревший" if origin < test_end else "актуальный"
            message = (f"Час выпуска {_label(origin)} (прогноз сделан после его закрытия). Конец архива "
                       f"{_label(archive_end)}, давность {age} ч — выпуск {state}.")
            (st.warning if origin < test_end else st.info)(message)
            start = origin - pd.Timedelta(hours=48)
            actuals = queries.read_history(conn, mode="clean", **target, start_ts=queries.to_label(start),
                                           end_ts=queries.to_label(archive_end + pd.Timedelta(hours=1)))
            if origin < test_end:
                st.caption("Часть траектории лежит внутри архива; там поверх неё показан записанный факт.")

    st.plotly_chart(charts.forecast_figure(actuals, predictions, metrics, mode=mode, origin_ts=origin,
                                           archive_end=archive_end, test_start=test_start, test_end=test_end,
                                           horizon_h=horizon if mode == "backtest" else None, unit=selection["unit"]),
                    use_container_width=True, key="forecast_chart")
    mae_unit = "процентных пунктах" if selection["unit"] == "%" else selection["unit"]
    st.markdown(f"**Метрики официального test** ({_label(test_start)} — {_label(test_end)}), не зависят от фильтра дат. "
                f"MAE в {mae_unit}. n_common — общие точки всех трёх методов. "
                "Покрытие — доля этих точек среди пригодных наблюдений цели на test.")
    shown = metrics if mode == "release" else metrics.loc[metrics["horizon_h"].eq(horizon)]
    st.dataframe(_metric_table(shown), hide_index=True, width="stretch", column_config={
        "coverage": st.column_config.NumberColumn("coverage", format="percent")})


# ---------------------------------------------------------------- entry point

def main() -> None:
    st.set_page_config(page_title="NK wells: прогноз параметров", layout="wide")
    st.title("NK Oil Wells: история, качество и краткосрочный прогноз")
    db_path = parse_args().db
    st.session_state["db_path"] = db_path
    try:
        conn = queries.open_database(db_path)
    except FileNotFoundError:
        st.error(f"Файл базы не найден: {db_path}. Соберите базу командой:")
        st.code(BUILD_COMMAND.format(db=db_path), language="bash")
        return
    except sqlite3.DatabaseError as error:
        st.error(f"Несовместимая база {db_path}: {error}. Пересоберите её командой:")
        st.code(BUILD_COMMAND.format(db=db_path), language="bash")
        return
    with closing(conn):
        meta = queries.read_meta(conn)
        catalog = queries.read_catalog(conn)
        selection = render_filters(catalog, meta)
        st.caption(f"База: {db_path} · ревизия {meta['revision'][:12]} · период {_period_text(selection)}")
        history, quality, forecast = st.tabs(["История и EDA", "Качество данных", "Прогноз и оценка"])
        with history:
            render_history(conn, catalog, meta, selection)
        with quality:
            render_quality(conn, catalog, meta, selection)
        with forecast:
            render_forecast(conn, catalog, meta, selection)


if __name__ == "__main__":
    main()
