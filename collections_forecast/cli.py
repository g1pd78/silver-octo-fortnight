"""Command-line entry point; prediction is always frozen before the target month."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import pandas as pd
from .config import Config
from .pipeline import ForecastRunner, run


def read_frame(path, config):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(
        path, dtype={column: str for column in config.segments})


def parser():
    p = argparse.ArgumentParser(description="Прогноз следующего месяца с известным планом и ошибкой в п.п.")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--input", help="CSV/Parquet с date,fact,plan; факты целевого месяца необязательны")
    source.add_argument("--history", help="Исторические RR-строки; передайте также --plan")
    source.add_argument("--db", help="SQLite с rr1plus_data_raw (открывается только для чтения)")
    source.add_argument("--demo", action="store_true", help="Искусственные данные для проверки запуска")
    p.add_argument("--plan", help="Планы для исторических и целевого месяцев в формате RR")
    p.add_argument("--actual", help="Отдельный файл факта целевого месяца для оценки после прогноза")
    p.add_argument("--compare", action="store_true", help="После прогноза прочитать факт целевого месяца из SQLite")
    p.add_argument("--config", help="JSON с полями Config")
    p.add_argument("--month", help="Целевой месяц YYYY-MM")
    p.add_argument("--fact-mode", choices=["daily", "cumulative"])
    p.add_argument("--plan-mode", choices=["daily", "cumulative"])
    p.add_argument("--plan-available-column")
    p.add_argument("--raw-available-column")
    p.add_argument("--plan-is-forecast", type=int, choices=[0, 1])
    p.add_argument("--output", default="outputs/forecast")
    p.add_argument("--report-checks", help="CSV date,fact_daily,plan_daily для независимой сверки")
    p.add_argument("--fast", action="store_true", help="Только референсы и календарная регрессия")
    p.add_argument("--min-train-months", type=int)
    for name in ["development-start", "development-end", "control-start", "control-end"]:
        p.add_argument("--" + name)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.history and not args.plan:
            raise ValueError("--history requires --plan")
        if args.plan and not args.history:
            raise ValueError("--plan is used together with --history")
        if args.compare and (args.input or args.history):
            raise ValueError("--compare is for SQLite; use --actual with DataFrame files")
        options = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        if args.demo:
            from .demo import make_demo
            config = make_demo(Path(args.output) / "demo_inputs")
        else:
            if not any([args.input, args.history, args.db, args.config]):
                raise ValueError("Choose --input, --history/--plan, --db, --config or --demo")
            config = Config(**options)
        mapping = {"month": "forecast_month", "db": "db_path"}
        for name in ["month", "db", "fact_mode", "plan_mode", "plan_available_column", "raw_available_column",
                     "plan_is_forecast", "min_train_months", "development_start", "development_end", "control_start", "control_end"]:
            value = getattr(args, name)
            if value is not None:
                config = replace(config, **{mapping.get(name, name): value})
        config = replace(config, output_dir=args.output).resolved()
        if args.fast:
            config = replace(config, enable_catboost=False, enable_sarimax=False)
        config.validate()
        if args.input or args.history:
            frame = read_frame(args.input or args.history, config)
            plans = read_frame(args.plan, config) if args.plan else None
            actual = read_frame(args.actual, config) if args.actual else None
            result = run(frame, plans, actual, config=config, report_checks_path=args.report_checks)
            selection, control, score = result["selection"], result["control"], result["metrics"]
        else:
            runner = ForecastRunner(config)
            runner.audit()
            if args.report_checks:
                runner.verify_report(args.report_checks)
            if runner.needs_money_fallback():
                selection, control = None, pd.DataFrame()
            else:
                _, selection = runner.select()
                control = runner.control()
            runner.forecast()
            score = None
            if args.actual:
                _, _, score = runner.compare(actual_df=read_frame(args.actual, config))
            elif args.compare or args.demo:
                _, _, score = runner.compare()
        print(json.dumps({"forecast_month": config.forecast_month, "selection": selection,
                          "control_max_error_pp": float(control.max_error_pp.max()) if not control.empty else None,
                          "target_metrics": score, "output": str(Path(config.output_dir).resolve()),
                          "artificial_data": args.demo}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
