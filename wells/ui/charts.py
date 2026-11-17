"""Plotly figures built from prepared data frames. No queries or fitting happen here."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical slots in fixed order. A colour follows the entity, never its rank in a selection.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MODEL_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive (24 ч)",
    "ridge": "Ridge",
    "gradient_boosting": "Gradient boosting",
    "statsmodels_ar": "Statsmodels AR",
}
MODEL_COLORS = dict(zip(MODEL_LABELS, SERIES_COLORS))
ACTUAL_COLOR = "#6f6e69"
FLAG_COLOR = "#b3261e"
EVENT_COLOR = "#8a8880"
INAPPLICABLE_COLOR = "#c9c8c2"
SEQUENTIAL = [[0.0, "#e6f0fd"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#2a78d6"], [1.0, "#0d366b"]]
DIVERGING = [[0.0, "#c2410c"], [0.25, "#f1a27f"], [0.5, "#ecebe7"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
EVENT_LABELS = {
    "area_scale_shift": "сдвиг масштаба площади",
    "nk7_regime_change": "смена режима NK-7",
}
TIME_AXIS = "Время источника, без часового пояса"
STAMP = "%Y-%m-%d %H:%M"


def series_colors(names) -> dict:
    """Assign slots by position in a stable list of entities, such as the wells of one type."""
    return {name: SERIES_COLORS[index % len(SERIES_COLORS)] for index, name in enumerate(names)}


def _colors(names, colors):
    fallback = series_colors(names)
    return {name: (colors or {}).get(name, fallback[name]) for name in names}


def _rgba(color, alpha):
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _layout(fig, *, title, height=420, **extra):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        height=height, margin=dict(l=60, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        **extra,
    )
    return fig


def _empty_note(fig, text, **position):
    fig.add_annotation(text=text, showarrow=False, xref="x domain", yref="y domain", x=0.5, y=0.5,
                       font=dict(size=14, color=EVENT_COLOR), **position)


def _axis_label(frame):
    if frame.empty:
        return ""
    return ", ".join(dict.fromkeys(f"{row.parameter}, {row.unit}"
                                   for row in frame[["parameter", "unit"]].drop_duplicates().itertuples()))


def _vertical_marker(fig, moment, text, *, row=None, dash="dash", position="top left"):
    kwargs = {} if row is None else dict(row=row, col=1)
    fig.add_vline(x=moment, line=dict(color=EVENT_COLOR, width=1.5, dash=dash), **kwargs)
    fig.add_annotation(x=moment, y=1.0, yref="y domain", text=text, showarrow=False,
                       xanchor="right" if position.endswith("left") else "left", yanchor="top",
                       font=dict(size=11, color=EVENT_COLOR), bgcolor="rgba(255,255,255,0.6)", **kwargs)


def history_figure(history, aggregates, issues, *, colors=None, trend_label="среднее за период") -> go.Figure:
    """Hourly values with breaks at missing hours, flagged readings, the trend and event dates."""
    fig = go.Figure()
    wells = list(dict.fromkeys(history["well_id"]))
    palette = _colors(wells, colors)
    many = len(wells) > 1
    for well in wells:
        rows = history.loc[history["well_id"].eq(well)]
        unit = rows["unit"].iloc[0]
        flag = rows["flag"].fillna("").str.strip()
        fig.add_trace(go.Scatter(
            x=rows["timestamp"], y=rows["display_value"], mode="lines", connectgaps=False,
            name=well, legendgroup=well, line=dict(color=palette[well], width=2),
            customdata=pd.DataFrame({"n": rows["n_obs"].astype("object").where(rows["n_obs"].notna(), "—"),
                                     "flag": flag.where(flag.ne(""), "нет")}).to_numpy(),
            hovertemplate=(f"<b>{well}</b> %{{x|{STAMP}}}<br>%{{y:.4g}} {unit}"
                           "<br>n_obs: %{customdata[0]} · флаг: %{customdata[1]}<extra></extra>"),
        ))
        flagged = rows.loc[flag.ne("") & rows["value"].notna()]
        if not flagged.empty:
            masked = flagged["display_value"].isna().map({True: "скрыто маской", False: "показано"})
            fig.add_trace(go.Scatter(
                x=flagged["timestamp"], y=flagged["value"], mode="markers",
                name=f"{well}: флаг качества" if many else "Флаг качества", legendgroup=well,
                marker=dict(symbol="x-thin-open", size=9, color=FLAG_COLOR, line=dict(width=1.5, color=FLAG_COLOR)),
                customdata=pd.DataFrame({"flag": flagged["flag"], "state": masked}).to_numpy(),
                hovertemplate=(f"<b>{well}</b> %{{x|{STAMP}}}<br>опубликовано %{{y:.4g}} {unit}"
                               "<br>флаг: %{customdata[0]} · %{customdata[1]}<extra></extra>"),
            ))
        trend = aggregates.loc[aggregates["well_id"].eq(well) & aggregates["mean_value"].notna()]
        if not trend.empty:
            fig.add_trace(go.Scatter(
                x=trend["period_start"], y=trend["mean_value"], mode="lines+markers", line_shape="hv",
                name=f"{well}: {trend_label}" if many else f"Тренд: {trend_label}", legendgroup=well,
                line=dict(color=palette[well], width=1.5, dash="dot"), marker=dict(size=5), opacity=0.75,
                customdata=trend[["n_values", "n_expected"]].to_numpy(),
                hovertemplate=(f"<b>{well}</b> период с %{{x|%Y-%m-%d}}<br>среднее %{{y:.4g}} {unit}"
                               "<br>значений %{customdata[0]} из %{customdata[1]} ожидаемых часов<extra></extra>"),
            ))

    if history.empty or history["display_value"].notna().sum() == 0:
        _empty_note(fig, "Нет наблюдений в выбранном периоде")
    if not history.empty:
        start, end = history["timestamp"].min(), history["timestamp"].max() + pd.Timedelta(hours=1)
        fig.update_xaxes(range=[start, end])
        events = issues.loc[issues["action"].eq("annotate") & issues["timestamp"].notna()]
        events = events.drop_duplicates(["timestamp", "end_ts", "issue_code"])
        for event in events.itertuples():
            label = EVENT_LABELS.get(event.issue_code, event.issue_code)
            if start <= event.timestamp < end:
                _vertical_marker(fig, event.timestamp, f"{label}: начало {event.timestamp:%d.%m}", position="top right")
            if pd.notna(event.end_ts) and start < event.end_ts <= end:
                _vertical_marker(fig, event.end_ts, f"{label}: конец {event.end_ts:%d.%m}", dash="dot")
    fig.update_xaxes(title_text=TIME_AXIS)
    fig.update_yaxes(title_text=_axis_label(history))
    return _layout(fig, title="Почасовой ряд", hovermode="x unified", showlegend=True)


def profile_figure(profiles, *, colors=None) -> go.Figure:
    """Median and interquartile band by hour of day. Hover shows the sample count."""
    fig = go.Figure()
    wells = list(dict.fromkeys(profiles["well_id"]))
    palette = _colors(wells, colors)
    for well in wells:
        rows = profiles.loc[profiles["well_id"].eq(well)].sort_values("hour")
        unit, color = rows["unit"].iloc[0], palette[well]
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["q75"], mode="lines", line=dict(width=0),
                                 legendgroup=well, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["q25"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(color, 0.18),
                                 legendgroup=well, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=rows["hour"], y=rows["median"], mode="lines+markers", name=well, legendgroup=well,
            line=dict(color=color, width=2), marker=dict(size=6),
            customdata=rows[["q25", "q75", "n"]].to_numpy(),
            hovertemplate=(f"<b>{well}</b> %{{x}}:00<br>медиана %{{y:.4g}} {unit}"
                           "<br>квартили %{customdata[0]:.4g} – %{customdata[1]:.4g}"
                           "<br>наблюдений: %{customdata[2]}<extra></extra>"),
        ))
    if profiles.empty:
        _empty_note(fig, "Нет наблюдений для суточного профиля")
    fig.update_xaxes(title_text="Час суток по времени источника", dtick=2, range=[-0.5, 23.5])
    fig.update_yaxes(title_text=_axis_label(profiles))
    return _layout(fig, title="Суточный профиль: медиана и межквартильный диапазон",
                   hovermode="x unified", showlegend=len(wells) > 1)


def correlation_figure(correlations) -> go.Figure:
    """Spearman matrix. Suppressed cells stay empty and still report the paired count on hover."""
    fig = go.Figure()
    order = list(dict.fromkeys(correlations["parameter_x"]))
    if not order:
        _empty_note(fig, "Нет параметров для корреляции")
        return _layout(fig, title="Корреляция Spearman на совпавших часах")
    rho = correlations.pivot(index="parameter_y", columns="parameter_x", values="rho").reindex(index=order, columns=order)
    pairs = correlations.pivot(index="parameter_y", columns="parameter_x", values="n_pairs").reindex(index=order, columns=order)
    text = rho.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}")
    hover = [[
        f"{row} × {column}<br>" + (f"ρ = {rho.loc[row, column]:.3f}" if pd.notna(rho.loc[row, column])
                                   else "коэффициент не показан: мало пар или ряд постоянен")
        + f"<br>совпавших часов: {int(pairs.loc[row, column])}"
        for column in order] for row in order]
    fig.add_trace(go.Heatmap(
        z=rho.to_numpy(), x=order, y=order, zmin=-1, zmax=1, zmid=0, colorscale=DIVERGING,
        text=text.to_numpy(), texttemplate="%{text}", hovertext=hover, hoverinfo="text",
        xgap=2, ygap=2, colorbar=dict(title="ρ"),
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title="Корреляция Spearman на совпавших часах", height=120 + 60 * max(len(order), 3))


def coverage_figure(coverage) -> go.Figure:
    """Coverage heatmap by day with expected, observed and flagged counts on hover.

    Rows are wells when ``well_id`` is present, otherwise well types. Inapplicable rows are
    drawn in a separate neutral state, while applicable rows without data show a numeric zero.
    """
    fig = go.Figure()
    title = "Покрытие: наблюдаемые часы / ожидаемые часы"
    if coverage.empty:
        _empty_note(fig, "Нет календарных часов в выбранном периоде")
        return _layout(fig, title=title)
    key = "well_id" if "well_id" in coverage.columns else "well_type"
    frame = coverage.copy()
    if "applicable" not in frame.columns:
        frame["applicable"] = True
    applicable = frame.groupby(key, sort=False)["applicable"].first()
    labels = {name: name if ok else f"{name} · неприменимо" for name, ok in applicable.items()}
    frame["label"] = frame[key].map(labels)
    rows = list(labels.values())
    periods = sorted(frame["period_start"].unique())

    def grid(column):
        return frame.pivot(index="label", columns="period_start", values=column).reindex(index=rows, columns=periods)

    share, expected, observed, flagged = (grid(name) for name in ("coverage", "n_expected", "n_observed", "n_flagged"))
    series = grid("n_series") if "n_series" in frame.columns else None
    hover = []
    for label in rows:
        line = []
        for period in periods:
            day = f"{label}<br>{pd.Timestamp(period):%Y-%m-%d}<br>"
            if pd.isna(expected.loc[label, period]):
                line.append(day + "датчик неприменим: не входит в знаменатель")
                continue
            scope = "" if series is None else f" · серий в знаменателе: {int(series.loc[label, period])}"
            line.append(day + f"покрытие {share.loc[label, period]:.1%}"
                        f"<br>наблюдаемых {int(observed.loc[label, period])} из {int(expected.loc[label, period])}"
                        f" ожидаемых часов{scope}<br>с флагом: {int(flagged.loc[label, period])}")
        hover.append(line)
    blocked = expected.isna().astype(float).where(expected.isna())
    if blocked.notna().any().any():
        fig.add_trace(go.Heatmap(z=blocked.to_numpy(), x=periods, y=rows, showscale=False, ygap=2,
                                 colorscale=[[0, INAPPLICABLE_COLOR], [1, INAPPLICABLE_COLOR]], hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=share.to_numpy(), x=periods, y=rows, zmin=0, zmax=1, colorscale=SEQUENTIAL, ygap=2,
        hovertext=hover, hoverinfo="text", hoverongaps=True,
        colorbar=dict(title="Покрытие", tickformat=".0%"),
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(title_text="День по времени источника")
    return _layout(fig, title=title, height=140 + 34 * max(len(rows), 3))


def _hourly(rows, column):
    """Reindex onto consecutive hours so a line breaks where a model issued no prediction."""
    indexed = rows.set_index(column)
    return indexed.reindex(pd.date_range(indexed.index.min(), indexed.index.max(), freq="h"))


def forecast_figure(actuals, predictions, metrics, *, mode, origin_ts, archive_end, test_start, test_end,
                    horizon_h=None, unit=None) -> go.Figure:
    """Top panel: facts and predictions. Bottom panel: official test MAE by horizon.

    ``mode`` is ``backtest`` (one horizon inside the date slice) or ``release`` (the stored
    24-hour path from one origin). The MAE panel always describes the whole official test.
    """
    if mode not in ("backtest", "release"):
        raise ValueError(f"mode must be backtest or release, got {mode!r}")
    if unit is None:
        unit = actuals["unit"].iloc[0] if not actuals.empty else ""
    error_unit = "п.п." if unit == "%" else unit
    if mode == "backtest":
        top = "Факт и прогноз на тесте" + (f", горизонт {horizon_h} ч" if horizon_h else "")
    else:
        top = "Последний выпуск: траектория на 24 часа"
        if origin_ts is not None:
            top += f" от часа {pd.Timestamp(origin_ts):{STAMP}}"
    bottom = (f"Официальный test, {pd.Timestamp(test_start):{STAMP}} — {pd.Timestamp(test_end):{STAMP}}: "
              "MAE по горизонтам")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.2, subplot_titles=(top, bottom))

    if not actuals.empty:
        fig.add_trace(go.Scatter(
            x=actuals["timestamp"], y=actuals["display_value"], mode="lines", connectgaps=False,
            name="Факт", line=dict(color=ACTUAL_COLOR, width=2),
            hovertemplate=f"Факт %{{x|{STAMP}}}<br>%{{y:.4g}} {unit}<extra></extra>",
        ), row=1, col=1)
    models = [model for model in MODEL_COLORS if model in set(predictions["model"]) | set(metrics["model"])]
    for model in models:
        label, color = MODEL_LABELS[model], MODEL_COLORS[model]
        rows = predictions.loc[predictions["model"].eq(model)]
        if not rows.empty:
            rows = _hourly(rows, "target_ts")
            origin = rows["origin_ts"].dt.strftime(STAMP).fillna("")
            dense = rows["y_pred"].notna().sum() > 72
            fig.add_trace(go.Scatter(
                x=rows.index, y=rows["y_pred"], mode="lines" if dense else "lines+markers", connectgaps=False,
                name=label, legendgroup=model, line=dict(color=color, width=2), marker=dict(size=5),
                customdata=pd.DataFrame({"h": rows["horizon_h"], "origin": origin}).to_numpy(),
                hovertemplate=(f"<b>{label}</b> на %{{x|{STAMP}}}<br>%{{y:.4g}} {unit}"
                               "<br>горизонт %{customdata[0]:.0f} ч · час выпуска %{customdata[1]}<extra></extra>"),
            ), row=1, col=1)
        scores = metrics.loc[metrics["model"].eq(model)].sort_values("horizon_h")
        if not scores.empty:
            scores = scores.set_index("horizon_h").reindex(range(1, 25))
            fig.add_trace(go.Scatter(
                x=scores.index, y=scores["mae"], mode="lines+markers", connectgaps=False,
                name=label, legendgroup=model, showlegend=rows.empty,
                line=dict(color=color, width=2), marker=dict(size=6),
                customdata=scores[["n_common", "n_actual", "coverage"]].to_numpy(),
                hovertemplate=(f"<b>{label}</b>, горизонт %{{x}} ч<br>MAE %{{y:.3f}} {error_unit}"
                               "<br>общих точек %{customdata[0]:.0f} из %{customdata[1]:.0f} фактов"
                               " · покрытие %{customdata[2]:.1%}<extra></extra>"),
            ), row=2, col=1)

    if mode == "release" and origin_ts is not None:
        origin, end = pd.Timestamp(origin_ts), pd.Timestamp(archive_end)
        if origin == end:
            _vertical_marker(fig, origin, "час выпуска = конец архива", row=1)
        else:
            _vertical_marker(fig, origin, f"час выпуска {origin:%d.%m %H:%M}", row=1)
            _vertical_marker(fig, end, f"конец архива {end:%d.%m %H:%M}", row=1, dash="dot", position="top right")
    if actuals.empty and predictions.empty or (
            predictions.empty and actuals["display_value"].notna().sum() == 0):
        fig.add_annotation(text="Нет фактов и прогнозов в выбранном срезе", showarrow=False,
                           xref="x domain", yref="y domain", x=0.5, y=0.5,
                           font=dict(size=14, color=EVENT_COLOR), row=1, col=1)
    if metrics.empty or metrics["mae"].notna().sum() == 0:
        fig.add_annotation(text="Нет опубликованной оценки MAE", showarrow=False,
                           xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
                           font=dict(size=14, color=EVENT_COLOR))
    elif mode == "backtest" and horizon_h:
        fig.add_vline(x=horizon_h, line=dict(color=EVENT_COLOR, width=1, dash="dot"), row=2, col=1)
    fig.update_xaxes(title_text=TIME_AXIS, row=1, col=1)
    fig.update_yaxes(title_text=_axis_label(actuals) or unit, row=1, col=1)
    fig.update_xaxes(title_text="Горизонт, часов", dtick=1, range=[0.5, 24.5], row=2, col=1)
    fig.update_yaxes(title_text=f"MAE, {error_unit}" if error_unit else "MAE", rangemode="tozero", row=2, col=1)
    fig.update_annotations(selector=dict(text=top), x=0, xanchor="left", font=dict(size=15))
    fig.update_annotations(selector=dict(text=bottom), x=0, xanchor="left", font=dict(size=15))
    fig.update_layout(height=680, margin=dict(l=60, r=24, t=56, b=48), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0))
    return fig
