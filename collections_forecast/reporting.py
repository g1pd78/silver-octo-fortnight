from pathlib import Path
import json
import numpy as np
import pandas as pd
from .pipeline import json_write
from .config import Config


def diagnostic_panels(result, config):
    """Small screenshot-friendly panels: percentages only, no monetary amounts."""
    c = config.resolved()
    status = result.get("plan_status") or {}
    selection = result.get("selection") or {}
    score = result.get("metrics") or {}
    def number(value):
        return "—" if value is None or pd.isna(value) else f"{float(value):.3f}"
    def table(frame, columns):
        if frame is None or frame.empty:
            return "Нет результатов."
        columns = {key: value for key, value in columns.items() if key in frame}
        return frame[list(columns)].rename(columns=columns).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    metric_columns = {"month": "Месяц", "max_error_pp": "Макс.пп", "mae_pp": "MAE.пп",
                      "bias_pp": "Сдвиг.пп", "hit_rate": "Попадания", "complete": "Полный"}
    selected_id = (f"{selection['leaf']}|{selection.get('total') or 'bottom_up'}|{selection['weight']:g}"
                   if selection else None)
    lines = [f"1. ИТОГ {c.forecast_month} · screenshot-v2", f"Режим: {status.get('mode', 'неизвестен')}",
             f"Выбор: {selected_id or 'резерв'}"]
    if selection.get("weight") == 1 and selection.get("total"):
        lines.append(f"Общий итог: 100% {selection['total']}; leaf не влияет на сумму.")
    lines += [f"Оценено: {score.get('scored_days', 0)}; без оценки: {score.get('unscored_days', '—')}",
              f"Полный месяц оценён: {score.get('complete', False)}; цель на доступных датах: {score.get('available_days_goal_achieved', False)}",
              f"MAE: {number(score.get('mae_pp'))} п.п.; сдвиг: {number(score.get('bias_pp'))} п.п.; макс.: {number(score.get('max_error_pp'))} п.п.",
              f"Попадания: {number(None if score.get('hit_rate') is None else 100 * score['hit_rate'])}%; последняя дата: {score.get('last_scored_date') or '—'}; ошибка: {number(score.get('last_scored_error_pp'))} п.п."]
    comparison = result.get("comparison")
    if comparison is not None and not comparison.empty and "absolute_error_pp" in comparison:
        valid = comparison[comparison.absolute_error_pp.notna()].copy()
        errors = valid.signed_error_pp
        if len(errors) and errors.gt(0).all():
            lines.append("Прогноз выше факта на всех оценённых датах.")
        elif len(errors) and errors.lt(0).all():
            lines.append("Прогноз ниже факта на всех оценённых датах.")
        if len(valid):
            valid["period"] = np.where(pd.to_datetime(valid.date).dt.day.le(5), "1–5", "6–конец")
            groups = valid.groupby("period", sort=False).agg(days=("absolute_error_pp", "size"),
                mae_pp=("absolute_error_pp", "mean"), bias_pp=("signed_error_pp", "mean"))
            lines += ["Ошибка по части месяца:", table(groups.reset_index(),
                       {"period": "Дни", "days": "Оценено", "mae_pp": "MAE.пп", "bias_pp": "Сдвиг.пп"})]
    panels = ["\n".join(lines)]

    ranking = result.get("ranking")
    lines = ["2. ПОДБОР НА ПРОШЛЫХ МЕСЯЦАХ", "Критерий: сначала минимальный максимум ошибки, затем MAE."]
    if ranking is not None and not ranking.empty:
        ranked = ranking.copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        # At weight=1 all leaf choices have the same aggregate prediction.
        # Collapse these aliases for display only; never modify the selection.
        def identity(candidate):
            leaf, total, weight = str(candidate).rsplit("|", 2)
            return f"{total}|1" if float(weight) == 1 else str(candidate)
        ranked["identity"] = ranked.candidate.map(identity)
        shown = ranked.drop_duplicates("identity").head(3)
        baselines = ranked[ranked.candidate.isin([f"{leaf}|bottom_up|0" for leaf in
            ["plan_baseline", "plan_execution", "plan_execution_curve", "plan_execution_blend"]])]
        shown = pd.concat([shown, baselines]).drop_duplicates("identity")
        shown = shown.sort_values("rank")
        shown["candidate"] = shown.identity
        lines += ["Топ-3 разных общих прогнозов + простые варианты:", table(shown,
            {"rank": "№", "candidate": "Модель", "months": "Мес.", "max_error_pp": "Макс.пп", "mae_pp": "MAE.пп"})]
    else:
        lines.append("Подбор не выполнен.")
    dev = result.get("development_scores")
    if selected_id and dev is not None and not dev.empty:
        lines += ["Выбранная комбинация по месяцам подбора (не независимая проверка):",
                  table(dev[dev.candidate.eq(selected_id)], metric_columns)]
    panels.append("\n".join(lines))

    control = result.get("control")
    lines = ["3. КОНТРОЛЬ ВЫБРАННОЙ КОМБИНАЦИИ", "Попадания в таблице — доля от 0 до 1.", table(control, metric_columns)]
    if control is not None and not control.empty and "max_error_pp" in control:
        if control.max_error_pp.dropna().gt(c.tolerance_pp).any():
            lines.append(f"На контроле есть превышения {c.tolerance_pp:g} п.п.; точность не подтверждена.")
    for phase, part in status.get("validation_coverage", {}).items():
        lines.append(f"{phase}: исключены {', '.join(part.get('excluded_months', [])) or 'нет'}")
    lines += list(status.get("reasons", [])) + list(status.get("warnings", []))
    panels.append("\n".join(lines))
    model_status = result.get("model_status")
    if model_status is not None and not model_status.empty and {"model", "status"} <= set(model_status):
        missing = model_status[model_status.status.eq("unavailable")].drop_duplicates()
        if not missing.empty:
            panels.append("4. НЕДОСТУПНЫЕ МОДЕЛИ\n" + table(missing.head(8),
                {"month": "Месяц", "model": "Модель", "reason": "Причина"}))
    if result.get("diagnostic_load_warnings"):
        panels.append("НЕПОЛНЫЙ НАБОР ФАЙЛОВ\n" + "\n".join(result["diagnostic_load_warnings"]))
    return panels


def load_saved_run_diagnostics(output_dir):
    """Read saved results only. No database, training, selection or forecast writes."""
    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**{**manifest["config"], "output_dir": str(out)}).resolved()
    result = {"selection": manifest.get("selection"), "diagnostic_load_warnings": []}
    for key, filename in [("plan_status", "forecast_run_status.json"), ("metrics", "metrics.json")]:
        path = out / filename
        result[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if not path.exists():
            result["diagnostic_load_warnings"].append(f"Нет {filename}.")
    for key, filename in [("comparison", "comparison_total.csv"), ("ranking", "development_ranking.csv"),
                          ("development_scores", "development_scores.csv"), ("control", "control_scores.csv"),
                          ("model_status", "model_status.csv"), ("forecast", "forecast.csv")]:
        path = out / filename
        result[key] = pd.DataFrame()
        if path.exists():
            try:
                result[key] = pd.read_csv(path, encoding="utf-8-sig", **({"usecols": ["model"]} if key == "forecast" else {}))
            except pd.errors.EmptyDataError:
                pass
        elif key in {"comparison", "forecast"} or (result.get("plan_status") or {}).get("mode") == "selected_model":
            result["diagnostic_load_warnings"].append(f"Нет {filename}.")
    return result, config


def save_run_diagnostics(result, config):
    """One support log for the finished run; refresh after evaluating new actuals."""
    forecast = result.get("forecast")
    models = sorted(forecast.model.dropna().astype(str).unique().tolist()) if forecast is not None and "model" in forecast else []
    c = config.resolved()
    settings = {name: getattr(c, name) for name in [
        "forecast_month", "fact_mode", "plan_mode", "forecast_grain", "reference_months", "min_train_months",
        "development_start", "development_end", "control_start", "control_end", "enable_catboost",
        "enable_sarimax", "estimate_plan_tail", "allow_missing_plan_folds", "output_dir"]}
    def dump(value):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    summary = "\n\n".join(diagnostic_panels(result, c)) + "\n"
    lines = [summary, "ПОДРОБНОСТИ · screenshot-v2", "Настройки:", dump(settings),
             "Фактически использованные модели:", dump(models),
             "Выбранная комбинация:", dump(result.get("selection")),
             "Режим, причины, предупреждения и покрытие исторической проверки:", dump(result.get("plan_status")),
             "Метрики сравнения с фактом:", dump(result.get("metrics"))]
    comparison = result.get("comparison")
    if comparison is not None and not comparison.empty:
        eligible = comparison.absolute_error_pp.notna()
        columns = ["date", "plan_cumulative", "actual_efficiency_pct", "forecast_efficiency_pct",
                   "signed_error_pp", "absolute_error_pp"]
        worst = comparison.loc[eligible].nlargest(10, "absolute_error_pp")[columns].copy()
        worst["date"] = pd.to_datetime(worst.date).dt.strftime("%Y-%m-%d")
        lines += ["Десять худших оценённых дат:", worst.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
                  "Даты без оценки:", ", ".join(pd.to_datetime(comparison.loc[~eligible, "date"]).dt.strftime("%Y-%m-%d")) or "нет"]
    else:
        lines.append("Сравнения с фактом нет: передайте фактические снимки целевого месяца в df.")
    text = "\n".join(lines) + "\n"
    out = Path(c.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.txt").write_text(text, encoding="utf-8")
    (out / "diagnostics_summary.txt").write_text(summary, encoding="utf-8")
    return text


def save_forecast_chart(total, config, path):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Прогноз накопленного выполнения, %", "Накопленные суммы, руб."))
    fig.add_trace(go.Scatter(x=total.date, y=total.forecast_efficiency_pct, name="Прогноз выполнения"), row=1, col=1)
    for column, name in [("forecast_cumulative", "Прогноз сбора"), ("plan_cumulative", "Утверждённый план")]:
        fig.add_trace(go.Scatter(x=total.date, y=total[column], name=name), row=2, col=1)
    fig.update_layout(title=f"{config.forecast_month}: прогноз зафиксирован до начала месяца",
                      template="plotly_white", height=720, hovermode="x unified")
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def grouped_breakdown(segment, dimensions):
    def strict_sum(s):
        return s.sum(min_count=len(s))
    return segment.groupby([*dimensions, "date"], as_index=False).agg(
        plan_daily=("plan", strict_sum), forecast_daily=("forecast_daily", "sum"),
        fact_daily=("fact", strict_sum), forecast_cumulative=("forecast_cumulative", "sum"),
        fact_cumulative=("fact_cumulative", strict_sum),
        signed_error_cumulative=("signed_error_cumulative", strict_sum),
        reconciliation_cumulative=("reconciliation_cumulative", "sum"))


def save_report(segment, total, score, config):
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment.to_csv(out / "comparison_segments.csv", index=False, encoding="utf-8-sig")
    total.to_csv(out / "comparison_total.csv", index=False, encoding="utf-8-sig")
    json_write(out / "metrics.json", score)
    for dims in [["subproduct"], ["dpd_bucket"], ["pvr_bucket"], ["subproduct", "dpd_bucket"]]:
        if set(dims) <= set(segment):
            grouped_breakdown(segment, dims).to_csv(out / ("by_" + "_".join(dims) + ".csv"),
                                                  index=False, encoding="utf-8-sig")
    categories = total[["date", "absolute_error_pp"]].copy()
    from .data import Inputs
    from .features import FeatureBuilder
    calendar = FeatureBuilder(config, Inputs(config)).calendar(config.forecast_month)
    categories = categories.merge(calendar, on="date")
    category_rows = []
    for name, mask in [("first_5_days", categories.day.le(5)),
                       ("weekends", categories.is_weekend.eq(1)),
                       ("holidays", categories.is_holiday.eq(1)),
                       ("ordinary_workdays", categories.is_workday.eq(1) & categories.is_holiday.eq(0))]:
        errors = categories.loc[mask, "absolute_error_pp"].dropna()
        category_rows.append({"category": name, "scored_days": len(errors),
                              "max_error_pp": errors.max(), "mae_pp": errors.mean()})
    pd.DataFrame(category_rows).to_csv(out / "calendar_error_groups.csv", index=False, encoding="utf-8-sig")
    fig = comparison_figure(total, config)
    fig.write_html(out / "comparison.html", include_plotlyjs=True, full_html=True)
    save_static_chart(total, config, out / "comparison.png")
    if score["complete"]:
        outcome = "ЦЕЛЬ ДОСТИГНУТА" if score["goal_achieved"] else "ЦЕЛЬ НЕ ДОСТИГНУТА"
    else:
        outcome = ("ОЦЕНКА ПО ДОСТУПНЫМ ДАТАМ" if score["scored_days"] else "НЕТ ДАТ ДЛЯ ОЦЕНКИ")
    failed = total[total.absolute_error_pp.gt(config.tolerance_pp)]
    text = (f"# {config.forecast_month}: {outcome}\n\n"
            f"Допуск: {config.tolerance_pp} п.п. на каждую дату с положительным планом.\n\n"
            f"Максимальная ошибка: {score['max_error_pp']} п.п.\n\n"
            f"Средняя ошибка: {score['mae_pp']} п.п.\n\n"
            f"Доля попаданий: {score['hit_rate']}\n\n"
            f"Оценено дат: {score['scored_days']}; без оценки: {score['unscored_days']}. "
            f"Полный месяц оценён: {score['complete']}.\n\n"
            f"Последняя оценённая дата: {score['last_scored_date']}; "
            f"ошибка: {score['last_scored_error_pp']} п.п.\n\n"
            f"Даты превышения допуска: {', '.join(failed.date.dt.strftime('%Y-%m-%d')) or 'нет среди оценённых дат'}.\n\n"
            "Денежный вклад сегментов в отклонение и поправки согласования сохранены в comparison_segments.csv. "
            "Календарные группы и сегментные вклады описывают отклонения, но сами по себе не доказывают их причины.\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    return fig


def comparison_figure(total, config):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Накопленное выполнение плана, %", "Накопленный сбор и план, руб.",
                                        "Сбор за день, руб.", "Абсолютная ошибка эффективности, п.п."))
    x = total.date
    actual = total.actual_efficiency_pct
    fig.add_trace(go.Scatter(x=x, y=actual - config.tolerance_pp, line_width=0,
                            name="Нижняя граница", showlegend=False, connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=actual + config.tolerance_pp, line_width=0, fill="tonexty",
                            fillcolor="rgba(35,150,100,0.14)", name=f"Допуск ±{config.tolerance_pp} п.п.", connectgaps=False), row=1, col=1)
    for col, label, color in [("actual_efficiency_pct", "Факт, %", "#218c65"),
                              ("forecast_efficiency_pct", "Прогноз, %", "#356ad2")]:
        fig.add_trace(go.Scatter(x=x, y=total[col], name=label, line_color=color, connectgaps=False), row=1, col=1)
    for col, label in [("fact_cumulative", "Факт накопленный"), ("forecast_cumulative", "Прогноз накопленный"),
                       ("plan_cumulative", "План накопленный")]:
        fig.add_trace(go.Scatter(x=x, y=total[col], name=label, connectgaps=False), row=2, col=1)
    for col, label in [("fact_daily", "Факт за день"), ("forecast_daily", "Прогноз за день")]:
        fig.add_trace(go.Scatter(x=x, y=total[col], name=label, connectgaps=False), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=total.absolute_error_pp, name="Ошибка, п.п.",
                        marker_color=np.where(total.within_tolerance, "#218c65", "#cb4747")), row=4, col=1)
    fig.add_hline(y=config.tolerance_pp, line_dash="dash", line_color="#cb4747", row=4, col=1)
    fig.update_layout(title=f"Прогноз и факт · {config.forecast_month} · старт до начала месяца",
                      height=1150, template="plotly_white", hovermode="x unified")
    return fig


def save_static_chart(total, config, path):
    import os
    cache = Path(config.output_dir) / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    x = total.date
    axes[0].plot(x, total.actual_efficiency_pct, label="Факт", color="#218c65")
    axes[0].plot(x, total.forecast_efficiency_pct, label="Прогноз", color="#356ad2")
    axes[0].fill_between(x, total.actual_efficiency_pct - config.tolerance_pp,
                         total.actual_efficiency_pct + config.tolerance_pp, alpha=0.15, color="#218c65")
    axes[0].set_ylabel("Выполнение, %")
    for c, label in [("fact_cumulative", "Факт"), ("forecast_cumulative", "Прогноз"), ("plan_cumulative", "План")]:
        axes[1].plot(x, total[c], label=label)
    axes[1].set_ylabel("Накопленно, руб.")
    axes[2].plot(x, total.fact_daily, label="Факт")
    axes[2].plot(x, total.forecast_daily, label="Прогноз")
    axes[2].set_ylabel("За день, руб.")
    axes[3].bar(x, total.absolute_error_pp, color=np.where(total.within_tolerance, "#218c65", "#cb4747"))
    axes[3].axhline(config.tolerance_pp, ls="--", color="#cb4747", label="Допуск")
    axes[3].set_ylabel("Ошибка, п.п.")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.grid(alpha=0.15)
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle(f"{config.forecast_month}: прогноз из конца предыдущего месяца")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
