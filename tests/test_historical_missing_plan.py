from dataclasses import replace
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.models import ModelHistoryUnavailableError
from collections_forecast.export import legacy_dashboard_upload
from collections_forecast.data import split_dated_frame


class HistoricalMissingPlanTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", "2026-10-31")
        self.df = pd.DataFrame({"date": dates, "fact": dates.day * 90., "plan": dates.day * 100.})
        self.c = Config(forecast_month="2026-10", forecast_grain="total",
                        fact_mode="cumulative", plan_mode="cumulative", min_train_months=2,
                        development_start="2026-01", development_end="2026-01",
                        control_start="2026-09", control_end="2026-09",
                        enable_catboost=False, enable_sarimax=False).resolved()
        self.h, self.p, self.a = split_dated_frame(self.df, self.c)

    def check_forecast(self, result, month="2026-01"):
        self.assertEqual(result["plan_status"]["mode"], "money_fallback")
        self.assertEqual(result["plan_status"]["validation"], "not_run")
        self.assertIn(month, [i["month"] for i in result["plan_status"]["historical_plan_issues"]])
        self.assertTrue(result["ranking"].empty)
        self.assertTrue(result["control"].empty)
        self.assertIsNone(result["selection"])
        upload = result["dashboard_upload"]
        self.assertEqual(len(upload), 31)
        self.assertFalse(upload.report_dt.duplicated().any())
        np.testing.assert_allclose(upload.plan, np.arange(1, 32) * 100.)
        self.assertTrue(np.isfinite(upload.fact).all())
        self.assertTrue(np.isfinite(result["forecast"].forecast_efficiency_pct).all())

    def test_missing_development_plan_saves_csv_and_reasons(self):
        plans = self.p[self.p.report_month.ne(pd.Timestamp("2026-01-01"))]
        with tempfile.TemporaryDirectory() as directory:
            result = run(self.h, plans, config=replace(self.c, output_dir=directory))
            self.check_forecast(result)
            saved = pd.read_csv(Path(directory) / "forecast_2026-10.csv")
            self.assertEqual(len(saved), 31)
            np.testing.assert_allclose(saved.fact, result["dashboard_upload"].fact)
            log = pd.read_csv(Path(directory) / "fold_log.csv")
            self.assertEqual(log.iloc[0]["status"], "plan_unavailable")
            status = json.loads((Path(directory) / "forecast_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status, result["plan_status"])
            manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["selection"])
            self.assertIsNone(manifest["input_features_sha256"])

    def test_missing_control_plan_uses_same_fallback(self):
        plans = self.p[self.p.report_month.ne(pd.Timestamp("2026-09-01"))]
        self.check_forecast(run(self.h, plans, config=self.c, save=False), "2026-09")

    def test_null_plan_in_history_does_not_invalidate_known_cash(self):
        frame = self.df.copy()
        frame.loc[frame.date.lt("2026-10-01"), "plan"] = np.nan
        result = run(frame, config=self.c, save=False)
        self.check_forecast(result)
        self.assertTrue(result["metrics"]["complete"])

    def test_one_missing_plan_date_and_zero_month_do_not_stop_cash(self):
        for zero in [False, True]:
            with self.subTest(zero=zero):
                plans = self.p.copy()
                if zero:
                    plans.loc[plans.report_month.eq(pd.Timestamp("2026-01-01")), "target"] = 0.
                else:
                    plans = plans[plans.report_dt.ne(pd.Timestamp("2026-01-15"))]
                self.check_forecast(run(self.h, plans, config=self.c, save=False))

    def test_unavailable_plan_vintage_does_not_leak_later_plan(self):
        plans = self.p.copy()
        plans["available_at"] = plans.report_month - pd.Timedelta(days=1)
        plans.loc[plans.report_month.eq(pd.Timestamp("2026-01-01")), "available_at"] = pd.Timestamp("2026-02-01")
        c = replace(self.c, plan_available_column="available_at")
        self.check_forecast(run(self.h, plans, config=c, save=False))

    def test_target_actuals_never_change_fallback_prediction(self):
        frame = self.df.copy()
        frame.loc[frame.date.dt.strftime("%Y-%m").eq("2026-01"), "plan"] = np.nan
        before = run(frame, config=self.c, save=False)
        frame.loc[frame.date.ge("2026-10-01"), "fact"] *= 1000
        after = run(frame, config=self.c, save=False)
        assert_frame_equal(before["forecast"], after["forecast"])

    def test_corrupt_historical_plan_is_not_silently_bypassed(self):
        for corruption in [-100., np.inf, "bad"]:
            with self.subTest(corruption=corruption):
                plans = self.p.astype({"target": object})
                plans.loc[plans.report_dt.eq(pd.Timestamp("2026-01-01")), "target"] = corruption
                with self.assertRaises(ValueError):
                    run(self.h, plans, config=self.c, save=False)

    def test_too_few_training_months_still_export_cash(self):
        result = run(self.df, config=replace(self.c, min_train_months=50), save=False)
        self.assertEqual(result["plan_status"]["mode"], "money_fallback")
        self.assertIn("need 50", result["plan_status"]["model_history_issues"][0]["reason"])
        self.assertIsNone(result["selection"])
        self.assertTrue(result["ranking"].empty)
        self.assertEqual(len(result["dashboard_upload"]), 31)

    def test_no_plans_before_development_month_still_export_cash(self):
        plans = self.p[self.p.report_month.ge(pd.Timestamp("2026-01-01"))]
        result = run(self.h, plans, config=self.c, save=False)
        self.assertEqual(len(result["dashboard_upload"]), 31)
        self.assertIn("No complete training months", result["plan_status"]["model_history_issues"][0]["reason"])

    def test_unknown_raw_reference_plan_does_not_become_zero(self):
        history = self.h.copy()
        history["target"] = np.nan
        result = run(history, self.p, config=self.c, save=False)
        self.assertEqual(result["plan_status"]["mode"], "money_fallback")
        self.assertTrue(result["history_months"].raw_plan_sum.isna().all())
        self.assertTrue(result["dashboard_upload"].fact.gt(0).all())

    def test_bad_training_plan_is_not_ignored(self):
        plans = self.p.copy()
        plans.loc[plans.report_dt.eq(pd.Timestamp("2024-02-01")), "target"] = -100.
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            run(self.h, plans, config=self.c, save=False)

    def test_late_unavailable_history_discards_previous_selection(self):
        original = ForecastRunner._predict_candidates
        for failed_month, phase in [("2026-09", "control"), ("2026-10", "forecast")]:
            with self.subTest(phase=phase):
                def unavailable(runner, month):
                    if str(month) == failed_month:
                        raise ModelHistoryUnavailableError("reference history unavailable")
                    return original(runner, month)
                with patch.object(ForecastRunner, "_predict_candidates", unavailable):
                    result = run(self.df, config=self.c, save=False)
                self.assertIsNone(result["selection"])
                self.assertTrue(result["ranking"].empty)
                self.assertTrue(result["control"].empty)
                self.assertEqual(result["plan_status"]["model_history_issues"][0]["phase"], phase)
                self.assertEqual(len(result["dashboard_upload"]), 31)

    def test_month_lengths_and_whole_absent_historical_month(self):
        for target in ["2024-02", "2025-02", "2026-09", "2026-10", "2026-11"]:
            with self.subTest(target=target):
                c = replace(self.c, forecast_month=target, development_start=None, development_end=None,
                            control_start=None, control_end=None).resolved()
                dates = pd.date_range("2023-01-01", pd.Period(target).end_time.normalize())
                frame = pd.DataFrame({"date": dates, "fact": dates.day * 90., "plan": dates.day * 100.})
                frame = frame[frame.date.dt.strftime("%Y-%m").ne(c.development_start)]
                result = run(frame, config=c, save=False)
                upload = result["dashboard_upload"]
                self.assertEqual(len(upload), pd.Period(target).days_in_month)
                self.assertEqual(upload.report_dt.max(), pd.Period(target).end_time.normalize())
                np.testing.assert_allclose(upload.plan, upload.report_dt.dt.day * 100.)
                self.assertFalse(upload.report_dt.duplicated().any())

    def test_fallback_csv_adapter_and_sql_totals_do_not_multiply_money(self):
        plans = self.p[self.p.report_month.ne(pd.Timestamp("2026-01-01"))]
        with tempfile.TemporaryDirectory() as directory:
            result = run(self.h, plans, config=replace(self.c, output_dir=directory))
            csv = pd.read_csv(Path(directory) / "forecast_2026-10.csv", encoding="utf-8-sig")
            # Partial target debt is a display denominator only, just as in the loader.
            debt = pd.DataFrame({"report_day": range(1, 10), "dolg": 10000.})
            loaded = legacy_dashboard_upload(csv, debt_by_day=debt, report_month="2026-10")
            with sqlite3.connect(":memory:") as conn:
                loaded.to_sql("forecast", conn, index=False)
                rows = pd.read_sql_query("""SELECT report_dt, COUNT(*) AS n, SUM(fact) AS fact,
                    SUM(plan) AS plan, SUM(dolg) AS dolg,
                    100.0*SUM(fact)/SUM(plan) AS kpi,
                    100.0*SUM(fact)/SUM(dolg) AS rr
                    FROM forecast WHERE forecast_model='collections.v2'
                    AND forecast_date='2026-10-01 00:00:00'
                    GROUP BY report_dt ORDER BY report_dt""", conn)
            self.assertEqual(len(rows), 31)
            self.assertTrue(rows.n.eq(1).all())
            np.testing.assert_allclose(rows.fact, result["forecast"].forecast_cumulative)
            np.testing.assert_allclose(rows.plan, result["dashboard_upload"].plan)
            np.testing.assert_allclose(rows.dolg, 10000.)
            np.testing.assert_allclose(rows.kpi, result["forecast"].forecast_efficiency_pct)
            np.testing.assert_allclose(rows.rr, rows.fact / 100.)


if __name__ == "__main__":
    unittest.main()
