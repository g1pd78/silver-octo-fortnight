from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.data import split_dated_frame
from collections_forecast.fallback import money_forecast


class PartialPlanModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", "2026-10-31")
        daily = 100. + dates.day_of_week * 30. + np.where(dates.day == 1, dates.month * 120., 0.)
        amounts = pd.Series(daily).groupby(dates.to_period("M")).cumsum().to_numpy()
        cls.df = pd.DataFrame({"date": dates, "fact": 1.02 * amounts, "plan": amounts})
        cls.c = Config(forecast_month="2026-10", forecast_grain="total", fact_mode="cumulative", plan_mode="cumulative",
            development_start="2026-01", development_end="2026-01", control_start="2026-09", control_end="2026-09",
            min_train_months=2, enable_catboost=False, enable_sarimax=False).resolved()

    def partial(self):
        df = self.df.copy()
        df.loc[df.date.ge("2026-10-26"), "plan"] = np.nan
        return df

    def runner(self, df=None, config=None):
        c = config or self.c
        h, p, _ = split_dated_frame(self.partial() if df is None else df, c)
        return ForecastRunner(c, history_df=h, plan_df=p)

    def test_unknown_tail_keeps_model_selection_and_official_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(self.partial(), config=replace(self.c, output_dir=directory))
            self.assertEqual(result["plan_status"]["mode"], "selected_model")
            self.assertEqual(result["plan_status"]["model_plan_estimated_cells"], 6)
            self.assertTrue(result["plan_status"]["warnings"])
            self.assertFalse(result["ranking"].empty)
            self.assertFalse(result["control"].empty)
            forecast = result["forecast"]
            self.assertTrue(np.isfinite(forecast.forecast_daily).all())
            self.assertTrue(forecast.plan.iloc[25:].isna().all())
            self.assertTrue(forecast.plan_cumulative.iloc[25:].isna().all())
            self.assertTrue(forecast.model_plan.iloc[25:].notna().all())
            self.assertEqual(result["metrics"]["scored_days"], 25)
            self.assertLess(result["metrics"]["mae_pp"], 1e-8)
            csv = pd.read_csv(Path(directory) / "forecast_2026-10.csv")
            self.assertEqual(len(csv), 31)
            self.assertTrue(csv.plan.iloc[25:].isna().all())
            np.testing.assert_allclose(csv.plan.iloc[:25], self.df[self.df.date.ge("2026-10-01")].plan.iloc[:25])
            estimates = pd.read_csv(Path(directory) / "model_plan_estimates.csv")
            self.assertEqual(estimates.plan_estimated.sum(), 6)

    def test_tail_estimate_preserves_prefix_and_does_not_read_target_actuals(self):
        r = self.runner()
        filled = r.builder.model_plan(r.history, self.c.forecast_month)
        official = r.inputs.plan(self.c.forecast_month)
        np.testing.assert_allclose(filled.plan.iloc[:25], official.plan.iloc[:25])
        self.assertTrue(np.isfinite(filled.plan).all())
        changed = self.partial()
        changed.loc[changed.date.ge("2026-10-01"), "fact"] *= 1000
        other = self.runner(changed)
        assert_frame_equal(filled, other.builder.model_plan(other.history, self.c.forecast_month))
        spec = {"leaf": "plan_execution", "total": None, "weight": 0.}
        assert_frame_equal(r.forecast(spec, save=False), other.forecast(spec, save=False))

    def test_missing_historical_validation_plan_still_uses_known_target_plan(self):
        r = self.runner()
        plans = r.inputs._plan_rows.rename(columns={"plan": "target"})
        plans = plans[plans.report_month.ne(pd.Timestamp("2026-01-01"))]
        runner = ForecastRunner(self.c, history_df=r.history, plan_df=plans)
        self.assertTrue(runner.needs_money_fallback())
        forecast = runner.forecast(save=False)
        self.assertTrue(forecast.model.eq("fallback_plan_execution").all())
        np.testing.assert_allclose(forecast.forecast_cumulative.iloc[:25], forecast.plan_cumulative.iloc[:25] * 1.02)
        self.assertTrue(forecast.plan.iloc[25:].isna().all())

    def test_entire_missing_plan_and_interior_gap_are_not_silently_filled(self):
        for mask in [self.df.date.ge("2026-10-01"), self.df.date.eq("2026-10-12")]:
            df = self.df.copy()
            df.loc[mask, "plan"] = np.nan
            r = self.runner(df)
            self.assertTrue(r.needs_money_fallback())
            self.assertEqual(r.target_plan_status()["model_plan_estimated_cells"], 0)
            self.assertTrue(np.isfinite(r.forecast(save=False).forecast_daily).all())

    def test_completion_can_be_disabled_explicitly(self):
        r = self.runner(config=replace(self.c, estimate_plan_tail=False))
        self.assertTrue(r.needs_money_fallback())
        self.assertTrue(r.forecast(save=False).model.eq("fallback_history_curve").all())

    def test_plan_baseline_beats_unscaled_cash_curve_on_changed_plan_shape(self):
        r = self.runner()
        _, _, actual = split_dated_frame(self.partial(), self.c)
        old = money_forecast(r.history, r.inputs.plan(self.c.forecast_month), self.c)
        new = r.forecast({"leaf": "plan_execution", "total": None, "weight": 0.}, save=False)
        _, _, old_score = r.compare(old, actual_df=actual, save=False)
        _, _, new_score = r.compare(new, actual_df=actual, save=False)
        self.assertLess(new_score["mae_pp"], 1e-8)
        self.assertGreater(old_score["mae_pp"], new_score["mae_pp"] + 1.)

    def test_partial_plan_reaches_catboost_and_sarimax_candidates(self):
        c = replace(self.c, enable_catboost=True, enable_sarimax=True,
                    catboost_iterations=20, catboost_depths=(4,))
        r = self.runner(config=c)
        _, leaves, totals = r._predict_candidates(c.forecast_month)
        self.assertIn("plan_execution", leaves)
        self.assertIn("plan_baseline", leaves)
        import importlib.util
        if importlib.util.find_spec("catboost"):
            self.assertIn("catboost_daily_d4", leaves)
        self.assertTrue(any(row["model"] == "total_sarimax" for row in r.status))
        for values in [*leaves.values(), *totals.values()]:
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
