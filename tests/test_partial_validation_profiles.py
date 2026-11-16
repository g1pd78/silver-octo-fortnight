from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.data import split_dated_frame


class PartialValidationProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", "2026-10-31")
        plan = dates.day.to_numpy() * 100.
        ratio = 1.01 + .20 * np.exp(-(dates.day.to_numpy() - 1) / 2.5)
        cls.df = pd.DataFrame({"date": dates, "plan": plan, "fact": plan * ratio})
        cls.c = Config(forecast_month="2026-10", forecast_grain="total", fact_mode="cumulative", plan_mode="cumulative",
            development_start="2026-01", development_end="2026-02", control_start="2026-07", control_end="2026-09",
            min_train_months=2, enable_catboost=False, enable_sarimax=False).resolved()

    def missing_august(self):
        df = self.df.copy()
        df.loc[df.date.dt.strftime("%Y-%m").eq("2026-08"), "plan"] = np.nan
        df.loc[df.date.ge("2026-10-26"), "plan"] = np.nan
        return df

    def runner(self, df=None, config=None):
        c = config or self.c
        h, p, _ = split_dated_frame(self.missing_august() if df is None else df, c)
        return ForecastRunner(c, history_df=h, plan_df=p)

    def test_august_excluded_without_disabling_models_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(self.missing_august(), config=replace(self.c, output_dir=directory))
            status = result["plan_status"]
            self.assertEqual(status["mode"], "selected_model")
            self.assertEqual(status["validation"], "partial_model_selection_and_control")
            self.assertEqual(status["reasons"], [])
            self.assertEqual(status["validation_coverage"]["control"]["excluded_months"], ["2026-08"])
            self.assertEqual(result["control"].month.tolist(), ["2026-07", "2026-09"])
            self.assertFalse(result["ranking"].empty)
            self.assertTrue(result["ranking"].months.eq(2).all())
            self.assertEqual(len(result["dashboard_upload"]), 31)
            self.assertTrue(result["dashboard_upload"].plan.iloc[25:].isna().all())
            self.assertLess(result["metrics"]["max_error_pp"], 1e-6)
            self.assertGreater(result["forecast"].forecast_efficiency_pct.iloc[0], 120.)
            self.assertLess(result["forecast"].forecast_efficiency_pct.iloc[24], 102.)
            log = pd.read_csv(Path(directory) / "fold_log.csv")
            self.assertEqual(log.loc[log.month.eq("2026-08"), "status"].tolist(), ["plan_unavailable"])
            diagnostic = (Path(directory) / "diagnostics.txt").read_text(encoding="utf-8")
            self.assertIn("partial_model_selection_and_control", diagnostic)
            self.assertIn("2026-08", diagnostic)
            self.assertIn("Десять худших оценённых дат", diagnostic)
            self.assertIn("max_error_pp", diagnostic)
            self.assertIn(result["forecast"].model.iloc[0], diagnostic)

    def test_missing_development_fold_competes_on_same_remaining_months(self):
        frame = self.missing_august()
        frame.loc[frame.date.dt.strftime("%Y-%m").eq("2026-01"), "plan"] = np.nan
        r = self.runner(frame)
        ranking, selection = r.select()
        self.assertIsNotNone(selection)
        self.assertTrue(ranking.months.eq(1).all())
        self.assertEqual(r.development_scores.month.unique().tolist(), ["2026-02"])

    def test_explicit_strict_mode_and_all_missing_control_still_use_fallback(self):
        strict = self.runner(config=replace(self.c, allow_missing_plan_folds=False))
        self.assertTrue(strict.needs_money_fallback())
        no_control = self.missing_august()
        no_control.loc[no_control.date.between("2026-07-01", "2026-09-30"), "plan"] = np.nan
        self.assertTrue(self.runner(no_control).needs_money_fallback())

    def test_daily_profile_reproduces_start_spike_without_target_actuals(self):
        r = self.runner()
        flat = r.forecast({"leaf": "plan_execution", "total": None, "weight": 0.}, save=False)
        profile = r.forecast({"leaf": "plan_execution_curve", "total": None, "weight": 0.}, save=False)
        _, _, actual = split_dated_frame(self.missing_august(), self.c)
        _, _, flat_score = r.compare(flat, actual_df=actual, save=False)
        _, _, profile_score = r.compare(profile, actual_df=actual, save=False)
        self.assertGreater(flat_score["max_error_pp"], 19.)
        self.assertLess(profile_score["max_error_pp"], 1e-8)
        changed = self.missing_august()
        changed.loc[changed.date.ge("2026-10-01"), "fact"] *= 10000
        other = self.runner(changed)
        assert_frame_equal(profile, other.forecast({"leaf": "plan_execution_curve", "total": None, "weight": 0.}, save=False))

    def test_profile_day_one_is_not_stretched_for_february(self):
        c = replace(self.c, forecast_month="2026-03", development_start="2025-06", development_end="2025-06",
                    control_start="2026-01", control_end="2026-02", reference_months=1)
        r = self.runner(self.df[self.df.date.lt("2026-04-01")], c)
        pred = r.forecast({"leaf": "plan_execution_curve", "total": None, "weight": 0.}, save=False)
        self.assertAlmostEqual(pred.forecast_efficiency_pct.iloc[0], 121.)
        self.assertAlmostEqual(pred.forecast_efficiency_pct.iloc[1], 100 * (1.01 + .2 * np.exp(-1 / 2.5)))
        self.assertTrue(np.isfinite(pred.forecast_daily).all())


if __name__ == "__main__":
    unittest.main()
