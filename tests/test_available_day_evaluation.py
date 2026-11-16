from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner
from collections_forecast.data import split_dated_frame
from collections_forecast.pipeline import evaluate, metrics


class AvailableDayEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", "2026-10-31")
        df = pd.DataFrame({"date": dates, "fact": dates.day * 90., "plan": dates.day * 100.})
        cls.config = Config(forecast_month="2026-10", forecast_grain="total",
            fact_mode="cumulative", plan_mode="cumulative", min_train_months=2,
            development_start="2026-01", development_end="2026-01",
            control_start="2026-09", control_end="2026-09",
            enable_catboost=False, enable_sarimax=False).resolved()
        cls.history, cls.plan, cls.actual = split_dated_frame(df, cls.config)
        cls.runner = ForecastRunner(cls.config, history_df=cls.history, plan_df=cls.plan)
        cls.pred = cls.runner.forecast({"leaf": "reference_profile", "total": None, "weight": 0.}, save=False)
        cls.pred["forecast_daily"] += .5

    def test_six_missing_dates_anywhere_preserve_other_25_snapshot_scores(self):
        for missing in [range(1, 7), range(12, 18), range(26, 32), [2, 5, 10, 20, 25, 30]]:
            with self.subTest(missing=list(missing)):
                observed = self.actual[~self.actual.report_dt.dt.day.isin(missing)]
                before = self.pred.copy(deep=True)
                _, total, score = self.runner.compare(self.pred, actual_df=observed, save=False)
                assert_frame_equal(before, self.pred)
                known = ~total.date.dt.day.isin(missing)
                np.testing.assert_allclose(total.loc[known, "fact_cumulative"], total.loc[known, "date"].dt.day * 90.)
                np.testing.assert_allclose(total.loc[known, "absolute_error_pp"], .5, atol=1e-10)
                self.assertTrue(total.loc[~known, "absolute_error_pp"].isna().all())
                self.assertEqual(score["scored_days"], 25)
                self.assertEqual(score["unscored_days"], 6)
                self.assertEqual(score["hit_rate"], 1.)
                self.assertTrue(score["available_days_goal_achieved"])
                self.assertFalse(score["complete"])
                self.assertFalse(score["goal_achieved"])

    def test_gap_change_is_not_misreported_as_daily_collection(self):
        observed = self.actual[~self.actual.report_dt.dt.day.between(12, 17)]
        _, total, _ = self.runner.compare(self.pred, actual_df=observed, save=False)
        self.assertTrue(total.loc[total.date.dt.day.between(12, 18), "fact_daily"].isna().all())
        self.assertEqual(total.loc[total.date.dt.day.eq(19), "fact_daily"].iloc[0], 90.)
        self.assertEqual(total.loc[total.date.dt.day.eq(18), "fact_cumulative"].iloc[0], 1620.)

    def test_last_observed_date_is_not_labelled_month_end(self):
        observed = self.actual[self.actual.report_dt.dt.day.le(25)]
        _, _, score = self.runner.compare(self.pred, actual_df=observed, save=False)
        self.assertEqual(score["last_scored_date"], "2026-10-25")
        self.assertAlmostEqual(score["last_scored_error_pp"], .5)
        self.assertIsNone(score["end_error_pp"])

    def test_partial_segment_date_is_not_a_complete_portfolio_fact(self):
        c = replace(self.config, forecast_grain="segments")
        runner = ForecastRunner(c, history_df=self.history, plan_df=self.plan)
        pred = pd.concat([self.pred.assign(subproduct="A"), self.pred.assign(subproduct="B")])
        actual = pd.concat([self.actual.assign(subproduct="A"), self.actual.assign(subproduct="B")])
        actual = actual[~(actual.subproduct.eq("B") & actual.report_dt.dt.day.eq(5))]
        _, total, score = runner.compare(pred, actual_df=actual, save=False)
        self.assertTrue(total.loc[total.date.dt.day.eq(5), "fact_cumulative"].isna().all())
        self.assertEqual(score["scored_days"], 30)
        self.assertEqual(total.loc[total.date.dt.day.eq(6), "fact_cumulative"].iloc[0], 1080.)

    def test_daily_flows_cannot_recover_cumulative_amount_after_gap(self):
        actual = self.actual[["report_dt", *self.config.segments, "fact"]].rename(columns={"report_dt": "date"})
        actual["fact"] = 90.
        actual = actual[actual.date.dt.day.ne(12)]
        _, total = evaluate(self.pred, actual, replace(self.config, fact_mode="daily"))
        score = metrics(total, self.config)
        self.assertEqual(score["scored_days"], 11)
        self.assertTrue(total.loc[total.date.dt.day.ge(12), "fact_cumulative"].isna().all())
        self.assertEqual(score["hit_rate"], 1.)

    def test_saved_report_and_complete_month(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ForecastRunner(replace(self.config, output_dir=directory), history_df=self.history, plan_df=self.plan)
            observed = self.actual[self.actual.report_dt.dt.day.le(25)]
            runner.compare(self.pred, actual_df=observed, save=True)
            report = (Path(directory) / "report.md").read_text(encoding="utf-8")
            self.assertIn("ОЦЕНКА ПО ДОСТУПНЫМ ДАТАМ", report)
            self.assertIn("Оценено дат: 25; без оценки: 6", report)
        _, _, score = self.runner.compare(self.pred, actual_df=self.actual, save=False)
        self.assertTrue(score["complete"])
        self.assertTrue(score["goal_achieved"])
        self.assertEqual(score["scored_days"], 31)


if __name__ == "__main__":
    unittest.main()
