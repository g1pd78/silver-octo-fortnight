from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd
from collections_forecast import Config, ForecastRunner
from collections_forecast.data import split_dated_frame


class RRDashboardContractTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2025-01-01", "2026-10-31")
        # Actual RR input semantics: snapshots ON the date, NOT daily flows.
        self.df = pd.DataFrame({"date": dates, "fact": 90. * dates.day, "plan": 100. * dates.day})
        self.c = Config(forecast_month="2026-10", forecast_grain="total",
                        fact_mode="cumulative", plan_mode="cumulative", min_train_months=2,
                        enable_catboost=False, enable_sarimax=False)
        self.spec = {"leaf": "reference_profile", "total": None, "weight": 0.}

    def runner(self, c):
        h, p, _ = split_dated_frame(self.df, c)
        return ForecastRunner(c, history_df=h, plan_df=p)

    def test_csv_through_existing_loader_preserves_kpi_rr_and_visible_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(self.c, output_dir=tmp)
            self.runner(c).forecast(self.spec, save=True)
            fc = pd.read_csv(Path(tmp) / "forecast_2026-10.csv", encoding="utf-8-sig")
            self.assertEqual(len(fc), 31)
            self.assertFalse(fc.report_dt.duplicated().any())
            self.assertTrue(fc.dt_in.eq("2026-10-01").all())
            self.assertTrue(fc.forecast_date.eq("2026-10-01").all())
            # The same mapping as the user's proven loader in the screenshots.
            report_month = pd.Timestamp("2026-10-01")
            dolg_map = pd.Series(10000., index=range(1, 32))
            day = pd.to_datetime(fc.report_dt).dt.day
            loaded = pd.DataFrame({
                "report_month": report_month, "report_dt": pd.to_datetime(fc.report_dt),
                "dt_in": report_month, "subproduct": "", "dpd_bucket": "", "terbank": "МБ",
                "pvr_bucket": "0. [undef]", "inflow": 0, "dolg": day.map(dolg_map),
                "plan": fc.plan.astype(float), "fact": fc.fact.astype(float),
                "is_forecast": 1, "forecast_date": report_month, "forecast_model": "test_rr",
            })
            with sqlite3.connect(":memory:") as conn:
                loaded.to_sql("forecast", conn, index=False)
                plotted = pd.read_sql_query("""
                    SELECT report_dt, COUNT(*) AS rows_count,
                           SUM(fact) AS fact, SUM(plan) AS plan, SUM(dolg) AS dolg,
                           100.0 * SUM(fact) / SUM(plan) AS kpi_pct,
                           100.0 * SUM(fact) / SUM(dolg) AS fact_rr_pct
                    FROM forecast
                    WHERE forecast_model='test_rr' AND forecast_date='2026-10-01 00:00:00'
                      AND dt_in='2026-10-01 00:00:00' AND subproduct='' AND dpd_bucket=''
                      AND terbank='МБ' AND pvr_bucket='0. [undef]' AND inflow=0
                    GROUP BY report_dt ORDER BY report_dt
                """, conn)
            self.assertEqual(len(plotted), 31)
            self.assertTrue(plotted.rows_count.eq(1).all())
            np.testing.assert_allclose(plotted.plan, 100. * np.arange(1, 32))
            np.testing.assert_allclose(plotted.fact, 90. * np.arange(1, 32))
            np.testing.assert_allclose(plotted.kpi_pct, 90.)
            self.assertAlmostEqual(plotted.fact_rr_pct.iloc[-1], 27.9)

    def test_daily_on_rr_snapshots_reproduces_sixteen_fold_end_amount(self):
        wrong = self.runner(replace(self.c, fact_mode="daily", plan_mode="daily")).forecast(self.spec, save=False)
        right = self.runner(self.c).forecast(self.spec, save=False)
        self.assertAlmostEqual(wrong.plan_cumulative.iloc[-1] / right.plan_cumulative.iloc[-1], 16.)
        self.assertAlmostEqual(wrong.forecast_cumulative.iloc[-1] / right.forecast_cumulative.iloc[-1], 16.)


if __name__ == "__main__":
    unittest.main()
