from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.data import split_dated_frame
from collections_forecast.export import upload_frame


class PlanAdjustmentTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", "2026-09-30")
        self.df = pd.DataFrame({"date": dates, "fact": dates.day * 90.,
            "plan": dates.day * 100. - np.where(dates.day >= 16, 400., 0.)})
        self.c = Config(forecast_month="2026-09", forecast_grain="total", fact_mode="cumulative", plan_mode="cumulative",
            min_train_months=2, development_start="2025-12", development_end="2025-12",
            control_start="2026-08", control_end="2026-08", enable_catboost=False, enable_sarimax=False).resolved()

    def runner(self, df=None):
        h, p, _ = split_dated_frame(self.df if df is None else df, self.c)
        return ForecastRunner(self.c, history_df=h, plan_df=p)

    def test_historical_and_target_plan_reconstruct_original_snapshots(self):
        r = self.runner()
        for month in ["2025-12", "2026-09"]:
            plan = r.inputs.plan(month).sort_values("date")
            self.assertEqual(plan.set_index("date").loc[month + "-16", "plan"], -300.)
            expected = self.df[self.df.date.dt.strftime("%Y-%m").eq(month)].plan
            np.testing.assert_allclose(plan.plan.cumsum(), expected)

    def test_calendar_does_not_predict_negative_cash_due_to_plan_revision(self):
        r = self.runner()
        _, leaves, _ = r._predict_candidates("2026-09")
        self.assertTrue((np.diff(np.r_[0., leaves["calendar_profile"]]) >= 0).all())

    def test_december_development_fold_and_september_csv_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(self.df, config=replace(self.c, output_dir=directory))
            self.assertFalse(result["ranking"].empty)
            self.assertFalse(result["control"].empty)
            self.assertTrue(result["metrics"]["complete"])
            csv = pd.read_csv(Path(directory)/"forecast_2026-09.csv")
            expected = self.df[self.df.date.dt.month.eq(9) & self.df.date.dt.year.eq(2026)].plan
            np.testing.assert_allclose(csv.plan, expected)
            self.assertEqual(csv.plan.iloc[15], 1200.)
            self.assertEqual(csv.plan.iloc[-1], 2600.)
            corrections = pd.read_csv(Path(directory)/"plan_adjustments.csv")
            self.assertTrue(corrections.plan_change.eq(-300.).all())
            self.assertTrue(corrections.date.eq("2025-12-16").any())
            self.assertTrue(corrections.date.eq("2026-09-16").any())

    def test_missing_future_plan_still_forecasts_with_adjusted_history(self):
        df = self.df.copy()
        df.loc[df.date.ge("2026-09-01"), "plan"] = np.nan
        result = run(df, config=self.c, save=False)
        self.assertEqual(len(result["dashboard_upload"]), 30)
        self.assertTrue(result["dashboard_upload"].plan.isna().all())

    def test_negative_source_snapshot_is_still_invalid(self):
        df = self.df.copy()
        df.loc[df.date.eq("2025-12-16"), "plan"] = -100.
        with self.assertRaisesRegex(ValueError, "Source plan amounts must be nonnegative"):
            self.runner(df)

    def test_export_accepts_adjustments_only_with_valid_cumulative_plan(self):
        r = self.runner()
        pred = r.forecast({"leaf": "reference_profile", "total": None, "weight": 0.}, save=False)
        upload_frame(pred, self.c)
        with self.assertRaisesRegex(ValueError, "daily source plans"):
            upload_frame(pred, replace(self.c, plan_mode="daily"))
        pred.loc[pred.date.eq("2026-09-01"), "plan"] = -100.
        with self.assertRaisesRegex(ValueError, "cumulative plan snapshots"):
            upload_frame(pred, self.c)


if __name__ == "__main__":
    unittest.main()
