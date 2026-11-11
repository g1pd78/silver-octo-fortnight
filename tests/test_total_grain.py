from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.config import origin
from collections_forecast.data import prepare_actual, raw_asof, daily_actual
from collections_forecast.demo import demo_rows


class TotalGrainTests(unittest.TestCase):
    def setUp(self):
        self.raw = demo_rows()
        for col in ["report_dt", "report_month", "dt_in", "plan_known_at"]:
            self.raw[col] = pd.to_datetime(self.raw[col])
        self.raw["fact"] = self.raw.w1 - self.raw.w2 - self.raw.cession - self.raw.write_off
        self.c = Config(forecast_month="2026-10", fact_mode="daily", plan_mode="daily",
                        forecast_grain="total", min_train_months=2,
                        development_start="2026-04", development_end="2026-04",
                        control_start="2026-09", control_end="2026-09",
                        enable_catboost=False, enable_sarimax=False).resolved()
        self.spec = {"leaf": "reference_profile", "total": None, "weight": 0.}

    def moving(self, raw):
        raw = raw.copy()
        raw["subproduct"] = raw.subproduct + "_" + raw.report_dt.dt.strftime("%Y%m%d")
        return raw

    def runner(self, raw, config=None):
        return ForecastRunner(config or self.c, history_df=raw[raw.report_dt.lt("2026-10-01")], plan_df=raw)

    def test_nonnull_source_plans_do_not_turn_into_synthetic_missing_cells(self):
        raw = self.moving(self.raw)
        self.assertTrue(raw.target.notna().all())
        old = self.runner(raw, replace(self.c, forecast_grain="segments"))
        self.assertGreater(old.target_plan_status()["missing_plan_cells"], 0)
        self.assertGreater(old.target_plan_status()["missing_segments"], 0)
        runner = self.runner(raw)
        status = runner.target_plan_status()
        self.assertEqual(status["mode"], "selected_model")
        self.assertEqual(status["missing_plan_cells"], 0)
        self.assertEqual(status["missing_segments"], 0)
        plans = runner.inputs.plan("2026-10")
        expected = raw[raw.report_dt.ge("2026-10-01")].groupby("report_dt").target.sum()
        np.testing.assert_allclose(plans.plan, expected)
        self.assertEqual(len(plans), 31)

    def test_moving_cumulative_segments_match_independent_total_series(self):
        raw = self.raw.copy()
        keys = ["report_month", "dt_in", *self.c.segments]
        raw[["fact", "target"]] = raw.groupby(keys)[["fact", "target"]].cumsum()
        config = replace(self.c, fact_mode="cumulative", plan_mode="cumulative")
        runner = self.runner(self.moving(raw), config)
        reference = self.runner(self.raw)
        forecast = runner.forecast(self.spec, save=False)
        expected = reference.forecast(self.spec, save=False)
        assert_frame_equal(forecast, expected, atol=1e-7, rtol=1e-10)
        _, _, score = runner.compare(actual_df=self.moving(raw[raw.report_dt.ge("2026-10-01")]), save=False)
        self.assertEqual(score["missing_actual_cells"], 0)
        self.assertIsNotNone(score["max_error_pp"])

    def test_real_missing_target_day_still_has_unknown_plan(self):
        raw = self.moving(self.raw)
        plan = raw[~raw.report_dt.eq("2026-10-12")]
        runner = ForecastRunner(self.c, history_df=raw[raw.report_dt.lt("2026-10-01")], plan_df=plan)
        self.assertTrue(runner.needs_money_fallback())
        self.assertIn("2026-10-12", runner.target_plan_status()["missing_plan_dates"])
        self.assertEqual(len(runner.forecast(save=False)), 31)

    def test_explicit_null_plan_cell_is_not_summed_away(self):
        raw = self.moving(self.raw)
        plan = raw.copy()
        plan.loc[plan[plan.report_dt.eq("2026-10-12")].index[0], "target"] = np.nan
        runner = ForecastRunner(self.c, history_df=raw[raw.report_dt.lt("2026-10-01")], plan_df=plan)
        self.assertTrue(pd.isna(runner.inputs.plan("2026-10").set_index("date").loc[pd.Timestamp("2026-10-12"), "plan"]))
        self.assertTrue(runner.needs_money_fallback())

    def test_actual_vintages_selected_before_total_aggregation(self):
        raw = self.moving(self.raw[self.raw.report_dt.lt("2026-10-01")])
        raw["known_at"] = raw.report_dt
        revised = raw.copy()
        revised["known_at"] = pd.Timestamp("2026-10-10")
        revised["fact"] *= 100
        c = replace(self.c, raw_available_column="known_at")
        prepared = prepare_actual(pd.concat([raw, revised]), c, before="2026-10")
        actual = daily_actual(raw_asof(prepared, c, origin("2026-10")), c)
        expected = raw.groupby("report_dt").fact.sum()
        np.testing.assert_allclose(actual.fact, expected)

    def test_plan_vintages_selected_before_total_aggregation(self):
        raw = self.moving(self.raw)
        revision = raw.copy()
        revision["plan_known_at"] = revision.report_dt + pd.Timedelta(days=1)
        revision["target"] *= 100
        runner = ForecastRunner(replace(self.c, plan_available_column="plan_known_at"),
            history_df=raw[raw.report_dt.lt("2026-10-01")], plan_df=pd.concat([raw, revision]))
        expected = raw[raw.report_dt.ge("2026-10-01")].groupby("report_dt").target.sum()
        np.testing.assert_allclose(runner.inputs.plan("2026-10").plan, expected)

    def test_run_selects_and_controls_with_moving_segments(self):
        df = self.moving(self.raw).rename(columns={"report_dt": "date", "target": "plan"})
        result = run(df, config=self.c, save=False)
        self.assertFalse(result["ranking"].empty)
        self.assertFalse(result["control"].empty)
        self.assertEqual(result["metrics"]["missing_actual_cells"], 0)
        self.assertIsNotNone(result["metrics"]["max_error_pp"])
        self.assertEqual(len(result["upload"]), 31)
        self.assertTrue(result["upload"].plan.notna().all())

    def test_single_dataframe_preserves_explicit_unknown_segment_plan(self):
        df = self.moving(self.raw).rename(columns={"report_dt": "date", "target": "plan"})
        df.loc[df[df.date.eq("2026-10-12")].index[0], "plan"] = np.nan
        result = run(df, config=self.c, save=False)
        self.assertEqual(result["plan_status"]["mode"], "money_fallback")
        self.assertTrue(result["upload"].loc[result["upload"].report_dt.ge("2026-10-12"), "plan"].isna().all())

    def test_duplicates_are_rejected_before_total_aggregation(self):
        raw = pd.concat([self.raw, self.raw.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate raw grain"):
            self.runner(raw)

    def test_partial_fact_snapshot_is_not_reported_as_complete_total(self):
        df = self.moving(self.raw).rename(columns={"report_dt": "date", "target": "plan"})
        df.loc[df[df.date.eq("2026-09-12")].index[0], "fact"] = np.nan
        with self.assertRaisesRegex(ValueError, "partially missing fact"):
            run(df, config=self.c, save=False)


if __name__ == "__main__":
    unittest.main()
