from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner
from collections_forecast.config import origin
from collections_forecast.data import Inputs, prepare_actual, raw_asof, daily_actual, split_dated_frame
from collections_forecast.demo import demo_rows
from collections_forecast.features import FeatureBuilder
from collections_forecast.models import profile_forecast
from collections_forecast.pipeline import evaluate, metrics


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.raw = demo_rows()
        self.raw["fact"] = self.raw.w1 - self.raw.w2 - self.raw.cession - self.raw.write_off
        self.config = Config(forecast_month="2026-10", fact_mode="daily", plan_mode="daily",
                             plan_available_column="plan_known_at", min_train_months=2,
                             development_start="2026-04", development_end="2026-04",
                             control_start="2026-09", control_end="2026-09",
                             enable_catboost=False, enable_sarimax=False).resolved()
        self.spec = {"leaf": "reference_profile", "total": None, "weight": 0.0}

    def runner(self, raw=None, config=None):
        raw = self.raw if raw is None else raw
        return ForecastRunner(config or self.config,
                              history_df=raw[raw.report_dt.lt("2026-10-01")], plan_df=raw)

    def cumulative(self, columns):
        raw = self.raw.copy()
        keys = ["report_month", "dt_in", *self.config.segments]
        raw[columns] = raw.groupby(keys)[columns].cumsum()
        return raw

    def test_explicit_semantics_required_and_arbitrary_target_month(self):
        with self.assertRaisesRegex(ValueError, "Explicitly set"):
            Config(forecast_month="2027-03").resolved().validate()
        config = Config(forecast_month="2027-03", fact_mode="daily", plan_mode="cumulative").resolved()
        config.validate()
        self.assertEqual(config.development_start, "2026-06")
        self.assertEqual(config.control_start, "2026-12")
        self.assertEqual(config.control_end, "2027-02")

    def test_cumulative_inputs_match_daily_predictions_and_evaluation(self):
        daily = self.runner()
        raw = self.cumulative(["fact", "target"])
        cumulative = self.runner(raw, replace(self.config, fact_mode="cumulative", plan_mode="cumulative"))
        a = daily.forecast(self.spec, save=False)
        b = cumulative.forecast(self.spec, save=False)
        assert_frame_equal(a, b, atol=1e-7, rtol=1e-10)
        _, ta, _ = daily.compare(actual_df=self.raw[self.raw.report_dt.ge("2026-10-01")], save=False)
        _, tb, _ = cumulative.compare(actual_df=raw[raw.report_dt.ge("2026-10-01")], save=False)
        assert_frame_equal(ta, tb, atol=1e-7, rtol=1e-10)

    def test_fact_and_plan_modes_are_independent(self):
        raw = self.cumulative(["fact"])
        r = self.runner(raw, replace(self.config, fact_mode="cumulative"))
        assert_frame_equal(self.runner().forecast(self.spec, save=False), r.forecast(self.spec, save=False),
                           atol=1e-7, rtol=1e-10)

    def test_cumulative_gaps_are_not_bridged_or_treated_as_zero(self):
        raw = self.cumulative(["fact"])
        raw = raw[raw.report_dt.ne("2026-09-15")]
        r = self.runner(raw, replace(self.config, fact_mode="cumulative"))
        with self.assertRaisesRegex(ValueError, "missing cohort snapshots"):
            r.forecast(self.spec, save=False)

    def test_cumulative_snapshots_are_not_differenced_twice(self):
        raw = self.cumulative(["fact", "target"])
        c = replace(self.config, fact_mode="cumulative", plan_mode="cumulative")
        prepared = prepare_actual(raw[raw.report_dt.lt("2026-10-01")], c, before="2026-10")
        first = raw_asof(prepared, c, origin("2026-10"))
        second = raw_asof(first, c, origin("2026-10"))
        assert_frame_equal(first, second)
        self.assertTrue(first.fact.lt(0).any(), "Signed corrections must remain signed")

    def test_cumulative_revision_selected_before_differencing(self):
        raw = self.cumulative(["fact"])
        raw["available_at"] = raw.report_dt
        revision = raw[raw.report_dt.eq("2026-09-29")].copy()
        revision["available_at"] = "2026-10-02"
        revision["fact"] += 1000000
        c = replace(self.config, fact_mode="cumulative", raw_available_column="available_at")
        base = self.runner(raw, c)
        changed = self.runner(pd.concat([raw, revision], ignore_index=True), c)
        # Plans are supplied separately to avoid inventing a duplicate plan vintage.
        changed.inputs = Inputs(c, plan_df=raw)
        changed.builder = FeatureBuilder(c, changed.inputs)
        assert_frame_equal(base.forecast(self.spec, save=False), changed.forecast(self.spec, save=False))

    def test_reference_endpoint_preserved_for_different_month_lengths(self):
        for target in ["2026-02", "2026-03"]:
            c = replace(self.config, forecast_month=target, plan_available_column=None)
            dates = pd.date_range("2026-01-01", pd.Period(target, "M").end_time.normalize())
            frame = pd.DataFrame({"date": dates, "plan": 10.0, "fact": 12.0})
            frame.loc[frame.date.ge(pd.Period(target, "M").start_time), "fact"] = np.nan
            history, plans, _ = split_dated_frame(frame, c)
            history = prepare_actual(history, c, before=target)
            builder = FeatureBuilder(c, Inputs(c, plan_df=plans))
            future = builder.build(history, target)
            prediction = profile_forecast(history, future, c)
            self.assertAlmostEqual(prediction[-1], future.plan.sum() * 1.2)

    def test_dataframe_plan_filters_actual_and_forecast_kinds(self):
        other = self.raw.copy()
        other["is_forecast"] = 1
        other["target"] *= 10
        plan = Inputs(self.config, plan_df=pd.concat([self.raw, other])).plan("2026-10")
        assert_frame_equal(plan, Inputs(self.config, plan_df=self.raw).plan("2026-10"))

    def test_control_facts_cannot_change_model_selection(self):
        original = self.runner()
        _, a = original.select()
        changed = self.raw.copy()
        changed.loc[changed.report_dt.ge("2026-07-01"), "fact"] *= 100
        _, b = self.runner(changed).select()
        self.assertEqual(a, b)

    def test_partial_forecast_cannot_claim_full_month_success(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        truth = prepare_actual(self.raw[self.raw.report_dt.ge("2026-10-01")], self.config, month="2026-10")
        actual = daily_actual(truth, self.config)
        with self.assertRaisesRegex(ValueError, "every date"):
            evaluate(f[f.date.dt.day.le(5)], actual, self.config)
        _, total = evaluate(f, actual, self.config)
        result = metrics(total.iloc[:5], self.config)
        self.assertFalse(result["complete"])
        self.assertFalse(result["goal_achieved"])

    def test_segment_and_total_errors_use_daily_values_consistently(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        truth = prepare_actual(self.raw[self.raw.report_dt.ge("2026-10-01")], self.config, month="2026-10")
        actual = daily_actual(truth, self.config)
        f["forecast_daily"] += 100
        segment, total = evaluate(f, actual, self.config)
        np.testing.assert_allclose(segment.groupby("date").signed_error_cumulative.sum(), total.signed_error_money)

    def test_saved_forecast_cannot_be_overwritten_with_a_different_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            r = self.runner(config=replace(self.config, output_dir=temp))
            f = r.forecast(self.spec, save=True)
            saved = (Path(temp) / "forecast.csv").read_bytes()
            r.save_forecast(f)  # Replaying exactly the same frozen prediction is safe.
            changed = f.copy()
            changed["forecast_daily"] += 10
            with self.assertRaisesRegex(FileExistsError, "frozen forecast"):
                r.save_forecast(changed)
            self.assertEqual((Path(temp) / "forecast.csv").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
