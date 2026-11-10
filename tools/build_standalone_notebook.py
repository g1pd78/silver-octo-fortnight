"""Build the portable notebook from the tested package (no second implementation)."""
import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def inline(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    remove = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("collections_forecast")):
            remove.update(range(node.lineno - 1, node.end_lineno))
    result = "".join(line for index, line in enumerate(lines) if index not in remove)
    ast.parse(result)
    return result


def build():
    cells = [nbformat.v4.new_markdown_cell("""# Прогноз выполнения плана взысканий

Самостоятельный ноутбук для любого целевого месяца. Имя файла сохранено для совместимости.
Все определения находятся в ячейках; пакет проекта для запуска не нужен.

Прогноз фиксируется **до начала месяца**, использует известный план по датам и проверяется по разнице
`100 * abs(прогноз_накопленный - факт_накопленный) / план_накопленный` в процентных пунктах.
Цель — не больше 1 п.п. на каждой дате; достижение цели на рабочих данных пока не подтверждено.

Перед запуском выберите смысл входных сумм: `daily` — за день, `cumulative` — с начала месяца/входа когорты.
Режимы факта и плана независимы. Накопленные снимки должны быть полными, с устойчивыми ключами когорт.
Пропуски не заменяются нулями, отрицательные корректировки сохраняются.

Установите зависимости в используемое ядро, если нужно:
```python
%pip install "numpy>=1.26,<3" "pandas>=2.2,<4" "plotly>=5.20,<8" "matplotlib>=3.8,<4" "catboost>=1.2,<2" "statsmodels>=0.14,<0.16" "holidays>=0.60,<1"
```

**Простой вход**: pandas DataFrame `df` с `date, fact, plan` за историю и целевой месяц;
факт и план целевого месяца могут отсутствовать (`NaN`). При пропусках плана включается резервный денежный прогноз
по истории до конца месяца; проценты с неизвестным знаменателем не рассчитываются. Даты не подменяются.
Дополнительные сегментные колонки сохраняются.
**Детальный вход**: `run(history_df, plan_df, actual_df, config=config)` — история RR, планы всех месяцев,
отдельный необязательный факт целевого месяца. Ключи: `report_month, report_dt, dt_in, subproduct, dpd_bucket,
pvr_bucket, terbank, inflow`. История содержит `fact` или `w1,w2,cession,write_off` и `target`;
план содержит `target`. Для исходных RR-компонентов нужны также `acc_count,avg_dpd,avg_pvr`.

Если заданы `plan_available_column`/`raw_available_column`, используются только доступные на момент прогноза
версии. Без них предполагается, что пользователь передал исторически доступные планы и снимки;
поздние пересмотры из финальной выгрузки восстановить невозможно.

Первые месяцы используются для обучения, следующие — для выбора модели, последние три — для отдельной проверки.
При настройках по умолчанию нужно минимум 21 месяц полной истории. Границы можно задать явно.
Для технической проверки переключите `DEMO = True` ниже. Искусственный пример не подтверждает бизнес-точность.
""")]
    for name in ["config", "data", "features", "models", "fallback", "pipeline", "reporting", "demo"]:
        cells.append(nbformat.v4.new_markdown_cell(f"## {name}"))
        cells.append(nbformat.v4.new_code_cell(inline((ROOT / "collections_forecast" / f"{name}.py").read_text(encoding="utf-8"))))
    cells += [nbformat.v4.new_markdown_cell("## Настройки и запуск"), nbformat.v4.new_code_cell('''DEMO = False

if DEMO:
    config = replace(make_demo("outputs/notebook_demo_inputs"),
                     development_start="2026-04", development_end="2026-04",
                     control_start="2026-09", control_end="2026-09",
                     catboost_iterations=20)
    raw = demo_rows()
    raw["fact"] = raw.w1 - raw.w2 - raw.cession - raw.write_off
    df = raw.rename(columns={"report_dt": "date", "target": "plan"})
else:
    config = Config(
        forecast_month="2026-11",  # измените целевой месяц
        fact_mode="daily",         # либо cumulative после сверки с отчётом
        plan_mode="daily",         # выбирается отдельно от факта
        output_dir="outputs/forecast",
    )
    # df = ваш_коннектор(...)  # date, fact, plan; включая известный будущий план
    if "df" not in globals():
        raise ValueError("Передайте DataFrame df или включите DEMO = True")

result = run(df, config=config)
display(result["ranking"].head(10))
display(result["control"])
display(result["forecast"].head())
'''), nbformat.v4.new_code_cell('''if result["comparison"] is not None:
    display(result["metrics"])
    comparison_figure(result["comparison"], config).show()
else:
    print("Прогноз сохранён. Для оценки передайте факты месяца отдельным этапом.")
''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    # Stable IDs make regeneration deterministic and keep reviews readable.
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"forecast-{index:02d}"
    nbformat.validate(notebook)
    target = ROOT / "october_forecast_standalone.ipynb"
    nbformat.write(notebook, target)
    print(target)


if __name__ == "__main__":
    build()
