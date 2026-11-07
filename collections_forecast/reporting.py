from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline import json_write


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
        plan_daily=("plan", "sum"), forecast_daily=("forecast_daily", "sum"),
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
        outcome = "ОЦЕНКА НЕПОЛНАЯ: пропущенные факты или сегменты вне исходного плана"
    failed = total[total.absolute_error_pp.gt(config.tolerance_pp)]
    text = (f"# {config.forecast_month}: {outcome}\n\n"
            f"Допуск: {config.tolerance_pp} п.п. на каждую дату с положительным планом.\n\n"
            f"Максимальная ошибка: {score['max_error_pp']} п.п.\n\n"
            f"Средняя ошибка: {score['mae_pp']} п.п.\n\n"
            f"Доля попаданий: {score['hit_rate']}\n\n"
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
