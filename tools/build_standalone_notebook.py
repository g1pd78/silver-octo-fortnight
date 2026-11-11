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

**Для вашего RR-запроса** `sum(w1-w2-cession-write_off)` и `sum(target)` — накопленные снимки на дату.
Поэтому рабочий конфиг ниже использует **`fact_mode="cumulative", plan_mode="cumulative"`**.
`daily` для таких снимков повторно накапливает деньги и сильно завышает график. Если вы передаёте
независимо рассчитанные дневные приращения, только тогда выбирайте `daily`.

**Рабочий путь загрузки:** оставьте старый `upload_model_effect.py`. После расчёта рядом с ноутбуком
создаётся `forecast_YYYY-MM.csv` — одна строка на дату, в формате его входа. Поместите этот файл рядом
с загрузчиком. В нём `dt_in` и `forecast_date` — первое число месяца, как в вашем рабочем коде;
истинный момент доступности данных остаётся в `origin` аналитического прогноза и `manifest.json`.
Для этого пути отдельный адаптер не требуется.

Перед запуском выберите смысл входных сумм: `daily` — за день, `cumulative` — с начала месяца/входа когорты.
Режимы факта и плана независимы. Накопленные снимки должны быть полными, с устойчивыми ключами когорт.
Пропуски не заменяются нулями, отрицательные корректировки сохраняются.

Для общего KPI в этом ноутбуке выбран `forecast_grain="total"`: на каждой дате сначала суммируется
полный снимок портфеля, затем накопленные суммы переводятся в дневные. Изменение DPD/PVR/продукта
не создаёт искусственные отсутствующие когорты. Вход должен содержать полный портфель на каждую дату;
заполненные суммы в отдельных строках сами по себе не доказывают полноту выгрузки.
При `forecast_grain="segments"` сохраняется строгая проверка устойчивых сегментных/когортных рядов.

Установите зависимости в используемое ядро, если нужно:
```python
%pip install "numpy>=1.26,<3" "pandas>=2.2,<4" "plotly>=5.20,<8" "matplotlib>=3.8,<4" "catboost>=1.2,<2" "statsmodels>=0.14,<0.16" "holidays>=0.60,<1"
```

**Простой вход**: pandas DataFrame `df` с `date, fact, plan` за историю и целевой месяц;
факт и план целевого месяца могут отсутствовать (`NaN`). При пропусках плана включается резервный денежный прогноз
по истории до конца месяца; проценты с неизвестным знаменателем не рассчитываются. Даты не подменяются.
Дополнительные сегментные колонки сохраняются.
Исходные имена `report_dt,target` и компоненты `w1,w2,cession,write_off` также поддерживаются.
При наличии `is_forecast` фактические и прогнозные строки разделяются до расчётов.
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

**Выгрузка для БД:** `forecast_upload.csv` — схема существующей таблицы `forecast`;
`fact` и `plan` — накопленные денежные суммы **на дату**, `is_forecast=1`, модель `collections.v2`.
`result["upload"]` содержит тот же набор. Одна строка на дату и сегмент модели, без размножения по `dt_in`.
Когорты уже суммированы, `dt_in` пустой. Итог на месяц — сумма `fact` по сегментам только на последнюю дату.
Повторная загрузка заменяет срез `(report_month,forecast_model,forecast_date)`, а не добавляет копии.
Файл `forecast_daily_totals.csv` / `result["daily_totals"]` — дневной общий прогноз для сверки,
одна строка на дату. Его не загружают вторым прогнозом той же модели.
Для агрегатного загрузчика дашборда используйте `forecast_upload_total.csv` / `result["upload_total"]`:
это схема таблицы `forecast` с накопленными суммами, ровно одна строка на дату.
Выбирайте один файл: сегментный или общий, не загружайте оба как один прогноз.
Если общий долг берётся из дашборда, используйте `aggregate_database_upload(upload, debt_by_day=dolg_df)`;
эта функция агрегирует сегменты до присоединения общего долга. Долг целевого месяца здесь служит только
для визуального сравнения RR и не поступает в обучение или оценку до начала месяца.
Нельзя присоединять прогноз к сырым строкам RR по дате или восстанавливать `dt_in` через JOIN: это размножит суммы.
Пустые `ranking/control` означают резервный режим: причины перечислены в `result["plan_status"]`.
При ошибке входа полный текст сохраняется в `input_error.txt` в каталоге результатов.
""")]
    for name in ["config", "data", "features", "models", "fallback", "export", "pipeline", "reporting", "demo"]:
        cells.append(nbformat.v4.new_markdown_cell(f"## {name}"))
        cells.append(nbformat.v4.new_code_cell(inline((ROOT / "collections_forecast" / f"{name}.py").read_text(encoding="utf-8"))))
    cells += [nbformat.v4.new_markdown_cell("## Настройки и запуск"), nbformat.v4.new_code_cell('''DEMO = False

if DEMO:
    config = replace(make_demo("outputs/notebook_demo_inputs"),
                     forecast_grain="total",
                     development_start="2026-04", development_end="2026-04",
                     control_start="2026-09", control_end="2026-09",
                     catboost_iterations=20)
    raw = demo_rows()
    raw["fact"] = raw.w1 - raw.w2 - raw.cession - raw.write_off
    df = raw.rename(columns={"report_dt": "date", "target": "plan"})
    df_layout = raw.copy()
    df_layout["plan"] = df_layout.groupby(["report_month", "dt_in", *config.segments]).target.cumsum()
else:
    config = Config(
        forecast_month="2026-11",  # измените целевой месяц
        forecast_grain="total",    # общий KPI; полный снимок портфеля на каждую дату
        fact_mode="cumulative",    # RR: w1 - w2 - cession - write_off — сбор НА ДАТУ
        plan_mode="cumulative",    # RR: target — план НА ДАТУ
        output_dir="outputs/forecast_rr_fixed",
    )
    # df = ваш_коннектор(...)  # date, fact, plan; включая известный будущий план
    if "df" not in globals():
        raise ValueError("Передайте DataFrame df или включите DEMO = True")

result = run(df, config=config)
display(result["ranking"].head(10))
display(result["control"])
display(result["forecast"].head())
display(result["dashboard_upload"])
print("Режим и полнота плана:", result["plan_status"])
dashboard_csv = Path(f"forecast_{config.forecast_month}.csv")
result["dashboard_upload"].to_csv(dashboard_csv, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
print("Файл для СТАРОГО загрузчика:", dashboard_csv.resolve())
print("Одна строка на дату. fact/plan — накопленные суммы. dt_in и forecast_date — первое число месяца.")
print("Поместите этот CSV рядом с upload_model_effect.py. Его чтение долга и загрузку в БД оставьте прежними.")
'''), nbformat.v4.new_markdown_cell('''## CSV с фильтрами продукта, DPD, PVR, ТБ и входов

Подготовьте `df_layout`: полный **известный на момент прогноза** план целевого месяца по строкам
`report_dt,dt_in,subproduct,dpd_bucket,terbank,pvr_bucket,inflow,plan,dolg`.
Поддерживаются также имена `date,target,w1`. `plan/target` в этой таблице **накопленный на дату**.
Если все эти поля есть в исходном `df` с накопленным планом, можно задать `df_layout = df.copy()`.
Общий прогноз распределяется по долям плана; при неизвестном/нулевом общем плане — по долям долга.
Это распределение, не отдельная модель качества каждого сегмента. Сумма прогноза по строкам
сохраняется на каждой дате. Долг каждой когорты берётся из `df_layout`, а не из общего итога.

**Этот файл загружается с сохранением всех полей**: не использовать блок старого загрузчика,
который стирает сегменты и заменяет `dt_in`/`dolg`. Вместо него после чтения файла:
```python
from dashboard_upload_adapter import read_filtered_dashboard_csv
df = read_filtered_dashboard_csv(CSV)
db.upload_forecast(df)
```
Существующий шаг обновления витрины сохраните. В дашборде выбирайте модель `collections.v2.allocated`.
Не объединяйте её строки с общим прогнозом `collections.v2` в одну сумму.
'''), nbformat.v4.new_code_cell('''if "df_layout" in globals():
    filtered = filtered_dashboard_upload(result["dashboard_upload"], df_layout, config)
    release = pd.Timestamp(filtered.forecast_date.iloc[0]).strftime("%Y%m%d")
    filtered_csv = Path(f"forecast_collections.v2.allocated_{release}.csv")
    filtered.to_csv(filtered_csv, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    result["filtered_upload"] = filtered
    print("CSV с фильтрами:", filtered_csv.resolve())
    print("Строк:", len(filtered), "| Уникальных дат:", filtered.report_dt.nunique())
    display(filtered.head())
else:
    print("Для CSV с фильтрами подготовьте df_layout по описанию выше. Общий прогноз уже сохранён.")
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
    # A companion file for the user's separate DB loader: no installed package
    # required, and the implementation is extracted from the same tested source.
    definitions = []
    for module, function in [("config", "days"), ("config", "month_start"), ("config", "origin"),
                             ("data", "parse_dates"), ("export", "aggregate_database_upload"),
                             ("export", "legacy_dashboard_upload"), ("export", "filtered_dashboard_upload"),
                             ("export", "read_filtered_dashboard_csv")]:
        source = (ROOT / "collections_forecast" / f"{module}.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
        definitions.append(ast.get_source_segment(source, node))
    adapter = ROOT / "dashboard_upload_adapter.py"
    adapter.write_text('"""Generated DB-loader adapter; rebuild with tools/build_standalone_notebook.py."""\n'
                       + "import numpy as np\nimport pandas as pd\n\n\n" + "\n\n\n".join(definitions) + "\n", encoding="utf-8")
    print(adapter)


if __name__ == "__main__":
    build()
