"""Deterministic artificial fixture. Never use it as evidence of business accuracy."""
from dataclasses import replace
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .config import Config, days, origin


def demo_rows():
    raw = []
    for month in pd.period_range("2026-01", "2026-10", freq="M"):
        for product, base in [("ПК", 10000.0), ("КК", 6500.0)]:
            dims = dict(subproduct=product, dpd_bucket="1. 1-30", pvr_bucket="1. [1-15]", terbank="МБ", inflow="0")
            for date in days(str(month)):
                plan = base * (0.45 if date.day_of_week >= 5 else 1.0) * (1.8 if date.day <= 3 else 1.0)
                fact = plan * (0.95 + 0.006 * np.sin(date.day / 4) + 0.002 * month.month)
                if date.day == 19 and product == "КК":
                    fact = -30.0  # A valid signed adjustment, not a missing value.
                raw.append(dict(report_month=str(month.start_time.date()), report_dt=str(date.date()),
                                dt_in=str(month.start_time.date()), **dims, w1=1_000_000.0,
                                w2=1_000_000.0-fact, cession=0.0, write_off=0.0, target=plan,
                                acc_count=100, avg_dpd=15.0, avg_pvr=10.0, is_forecast=0,
                                plan_known_at=str(origin(str(month)).date())))
    return pd.DataFrame(raw)


def make_demo(directory="outputs/demo_inputs"):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    database = out / "artificial.db"
    with sqlite3.connect(database) as conn:
        demo_rows().to_sql("rr1plus_data_raw", conn, index=False, if_exists="replace")
    conn.close()
    return Config(db_path=str(database), plan_available_column="plan_known_at", forecast_month="2026-10",
                  fact_mode="daily", plan_mode="daily",
                  output_dir=str(out.parent / "demo_results"), min_train_months=2,
                  development_start="2026-04", development_end="2026-06", control_start="2026-07", control_end="2026-09",
                  catboost_depths=(4,), catboost_iterations=60)
