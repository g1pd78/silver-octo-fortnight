import json
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, ForecastRunner, run
from collections_forecast.config import origin
from collections_forecast.data import Inputs, daily_actual, load_raw, load_plan, parse_dates, readonly_connection
from collections_forecast.demo import demo_rows, make_demo
from collections_forecast.features import FeatureBuilder
from collections_forecast.models import candidates, reconcile
from collections_forecast.pipeline import evaluate, metrics


@contextmanager
def test_connection(path):
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = replace(make_demo(Path(cls.tmp.name) / "inputs"), enable_catboost=False, enable_sarimax=False,
                             development_start="2026-04", development_end="2026-04",
                             control_start="2026-09", control_end="2026-09")
        cls.spec = {"leaf": "reference_profile", "total": "total_reference", "weight": 0.5}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def runner(self):
        return ForecastRunner(self.config)

    @contextmanager
    def copied_database(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "copy.db"
            with test_connection(self.config.db_path) as source, test_connection(path) as target:
                source.backup(target)
            yield replace(self.config, db_path=str(path))

    def test_plan_from_database_without_external_inputs(self):
        self.assertFalse(hasattr(self.config, "plans_path"))
        inputs = Inputs(self.config)
        plan = inputs.plan("2026-10")
        with test_connection(self.config.db_path) as conn:
            expected = conn.execute("SELECT SUM(target) FROM rr1plus_data_raw WHERE report_month='2026-10-01'").fetchone()[0]
        self.assertAlmostEqual(plan.plan.sum(), expected)
        self.assertIsNone(self.config.calendar_path)
        calendar = FeatureBuilder(self.config, inputs).calendar("2026-01")
        self.assertEqual(calendar.loc[calendar.date.eq(pd.Timestamp('2026-01-09')), 'is_workday'].iloc[0], 0)

    def test_dataframe_forecast_and_comparison_use_no_sqlite(self):
        raw = demo_rows()
        keys = ["report_month", "report_dt", "dt_in", *self.config.segments]
        is_october = raw.report_dt.ge("2026-10-01")
        history = raw.loc[~is_october].rename(columns={"report_dt": "date"})
        plan = raw[[*keys, "target", "plan_known_at"]].rename(columns={"report_dt": "date"})
        actual = raw.loc[is_october, [*keys, "w1", "w2", "cession", "write_off"]].rename(columns={"report_dt": "date"})
        expected = self.runner().forecast(self.spec, save=False)
        config = replace(self.config, output_dir=str(Path(self.tmp.name) / "frame_results"))
        with patch("collections_forecast.data.readonly_connection", side_effect=AssertionError("Unexpected SQLite access")):
            r = ForecastRunner(config, history_df=history, plan_df=plan)
            forecast = r.forecast(self.spec, save=True)
            _, total, score = r.compare(actual_df=actual, save=True)
        assert_frame_equal(forecast, expected)
        self.assertEqual(len(total), 31)
        self.assertTrue(score["complete"])
        self.assertEqual(r.audit()[0]["input_mode"], "dataframes")

    def test_run_single_frame_keeps_dates_and_holds_out_october_actuals(self):
        raw = demo_rows()
        raw["fact"] = raw.w1 - raw.w2 - raw.cession - raw.write_off
        df = (raw.groupby("report_dt", as_index=False).agg(fact=("fact", "sum"), plan=("target", "sum"))
              .rename(columns={"report_dt": "date"}))
        df["date"] = pd.to_datetime(df.date)
        config = replace(self.config, plan_available_column=None)
        original = run(df, config=config, save=False)
        changed = df.copy()
        changed.loc[changed.date.ge("2026-10-01"), "fact"] *= 10
        rerun = run(changed, config=config, save=False)
        assert_frame_equal(original["forecast"], rerun["forecast"])
        expected_dates = pd.date_range("2026-10-01", "2026-10-31")
        self.assertTrue(pd.DatetimeIndex(original["forecast"].date).equals(expected_dates))
        self.assertTrue(pd.DatetimeIndex(original["comparison"].date).equals(expected_dates))
        self.assertNotEqual(original["metrics"]["max_error_pp"], rerun["metrics"]["max_error_pp"])

    def test_dataframe_history_rejects_october_actuals(self):
        raw = demo_rows()
        with self.assertRaisesRegex(ValueError, "df_history must contain only dates before"):
            ForecastRunner(self.config, history_df=raw, plan_df=raw)

    def test_plan_query_has_no_access_to_fact_or_portfolio_fields(self):
        original_connection = readonly_connection
        forbidden = {"w1", "w2", "cession", "write_off", "fact", "acc_count", "avg_dpd", "avg_pvr"}
        @contextmanager
        def guarded_connection(path):
            with original_connection(path) as conn:
                def authorize(action, table, column, database, trigger):
                    if action == sqlite3.SQLITE_READ and table == self.config.raw_table and column in forbidden:
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                conn.set_authorizer(authorize)
                yield conn
        with patch("collections_forecast.data.readonly_connection", guarded_connection):
            plan = load_plan(self.config, "2026-10")
        self.assertEqual(len(plan), 62)

    def test_sql_history_excludes_october(self):
        h = load_raw(self.config, before="2026-10")
        self.assertLess(h.date.max(), pd.Timestamp("2026-10-01"))
        with readonly_connection(self.config.db_path) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM rr1plus_data_raw")

    def test_sql_boundary_with_timezone(self):
        with test_connection(self.config.db_path) as conn:
            row = pd.read_sql_query("SELECT * FROM rr1plus_data_raw LIMIT 1", conn)
            row["report_month"] = "2026-10-01"
            row["report_dt"] = "2026-10-01T01:00:00+03:00"
            row["dt_in"] = "2026-10-01"
            row.to_sql("rr1plus_data_raw", conn, if_exists="append", index=False)
        try:
            h = load_raw(self.config, before="2026-10")
            self.assertLess(h.date.max(), pd.Timestamp("2026-10-01"))
        finally:
            with test_connection(self.config.db_path) as conn:
                conn.execute("DELETE FROM rr1plus_data_raw WHERE report_dt='2026-10-01T01:00:00+03:00'")

    def test_raw_late_revision_is_not_visible_at_origin(self):
        with test_connection(self.config.db_path) as conn:
            raw = pd.read_sql_query("SELECT * FROM rr1plus_data_raw", conn)
        raw["available_at"] = raw.report_dt
        changed = raw[raw.report_dt.eq("2026-09-30")].copy()
        changed["available_at"] = "2026-10-02"
        changed["w2"] = 0
        target = Path(self.tmp.name) / "vintaged.db"
        with test_connection(target) as conn:
            pd.concat([raw, changed]).to_sql("rr1plus_data_raw", conn, if_exists="replace", index=False)
        c = replace(self.config, db_path=str(target), raw_available_column="available_at")
        from collections_forecast.data import raw_asof
        h = load_raw(c, before="2026-10")
        eligible = raw_asof(h, c, origin("2026-10"))
        self.assertEqual(len(eligible[eligible.date.eq(pd.Timestamp("2026-09-30"))]), 2)
        self.assertTrue(eligible[eligible.date.eq(pd.Timestamp("2026-09-30"))].w2.gt(0).all())
        truth = ForecastRunner(c)._truth("2026-09")
        self.assertTrue(truth[truth.date.eq(pd.Timestamp("2026-09-30"))].fact.ge(1_000_000).all())

    def test_reference_report_reconciliation(self):
        r = self.runner()
        checks = r.history.groupby("date", as_index=False).agg(fact_daily=("fact", "sum"), plan_daily=("target", "sum")).tail(3)
        file = Path(self.tmp.name) / "checks.csv"
        checks.to_csv(file, index=False)
        self.assertEqual(len(r.verify_report(file)), 3)
        checks["fact_daily"] *= 2
        checks.to_csv(file, index=False)
        with self.assertRaisesRegex(ValueError, "disagree"):
            r.verify_report(file)

    def test_october_fact_change_and_removal_cannot_change_prediction(self):
        with self.copied_database() as c:
            a = ForecastRunner(c).forecast(self.spec, save=False)
            with test_connection(c.db_path) as conn:
                conn.execute("UPDATE rr1plus_data_raw SET w1=999999999, w2=0 WHERE report_dt >= '2026-10-01'")
            b = ForecastRunner(c).forecast(self.spec, save=False)
            assert_frame_equal(a, b)
            # Shared rows contain both plan and fact. Remove fact fields while keeping the plan fixed.
            with test_connection(c.db_path) as conn:
                conn.execute("UPDATE rr1plus_data_raw SET w1=NULL,w2=NULL,cession=NULL,write_off=NULL,acc_count=NULL WHERE report_dt >= '2026-10-01'")
            result = ForecastRunner(c).forecast(self.spec, save=False)
            assert_frame_equal(a, result)

    def test_actual_row_deletion_with_fixed_database_plan(self):
        with self.copied_database() as c:
            with test_connection(c.db_path) as conn:
                conn.execute("CREATE TABLE frozen_plan AS SELECT * FROM rr1plus_data_raw")
            c = replace(c, plan_table="frozen_plan")
            a = ForecastRunner(c).forecast(self.spec, save=False)
            with test_connection(c.db_path) as conn:
                conn.execute("DELETE FROM rr1plus_data_raw WHERE report_dt >= '2026-10-01'")
            result = ForecastRunner(c).forecast(self.spec, save=False)
            assert_frame_equal(a, result)

    def test_revision_after_origin_is_ignored(self):
        with self.copied_database() as c:
            before = Inputs(c).plan("2026-10")
            with test_connection(c.db_path) as conn:
                revision = pd.read_sql_query("SELECT * FROM rr1plus_data_raw WHERE report_month='2026-10-01'", conn)
                revision["plan_known_at"] = "2026-10-02"
                revision["target"] *= 9
                revision.to_sql(c.raw_table, conn, if_exists="append", index=False)
            assert_frame_equal(before, Inputs(c).plan("2026-10"))

    def test_plan_does_not_double_count_forecast_and_actual_rows(self):
        with self.copied_database() as c:
            before = Inputs(c).plan("2026-10")
            with test_connection(c.db_path) as conn:
                extra = pd.read_sql_query("SELECT * FROM rr1plus_data_raw WHERE report_month='2026-10-01'", conn)
                extra["is_forecast"] = 1
                extra["target"] *= 5
                extra.to_sql(c.raw_table, conn, if_exists="append", index=False)
            assert_frame_equal(before, Inputs(c).plan("2026-10"))
            alternate = Inputs(replace(c, plan_is_forecast=1)).plan("2026-10")
            np.testing.assert_allclose(alternate.plan, before.plan * 5)

    def test_moscow_timestamp_on_october_first_is_after_origin(self):
        values = parse_dates(pd.Series(["2026-10-01T01:00:00+03:00", "2026-09-30 23:00:00"]))
        self.assertGreater(values.iloc[0], origin("2026-10"))
        self.assertLess(values.iloc[1], origin("2026-10"))

    def test_training_month_features_ignore_its_observations(self):
        runner = self.runner()
        before = runner.builder.build(runner.history, "2026-08")
        changed = runner.history.copy()
        mask = changed.month.eq("2026-08")
        changed.loc[mask, ["fact", "w1", "acc_count", "avg_dpd"]] = 9e10
        after = runner.builder.build(changed, "2026-08")
        assert_frame_equal(before, after)

    def test_reconciliation_preserves_daily_and_cumulative_totals(self):
        r = self.runner()
        x, leaves, totals, _ = candidates(r.history, "2026-10", self.config, r.builder)
        leaf = leaves["reference_profile"]
        direct = totals["total_reference"] * 1.2
        f = reconcile(x, leaf, direct, 0.75, self.config)
        bottom = pd.Series(leaf).groupby(x.date).sum().to_numpy()
        expected = 0.25 * bottom + 0.75 * direct
        np.testing.assert_allclose(f.groupby("date").forecast_cumulative.sum(), expected)
        np.testing.assert_allclose(f.groupby("date").forecast_daily.sum().cumsum(), expected)

    def test_daily_metrics_and_strict_threshold(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        raw = load_raw(self.config, month="2026-10")
        actual = daily_actual(raw, self.config)
        exact = f.merge(actual[["date", *self.config.segments, "fact"]], on=["date", *self.config.segments])
        exact["forecast_daily"] = exact.fact + 0.009 * exact.plan
        _, total = evaluate(exact.drop(columns="fact"), actual, self.config)
        result = metrics(total, self.config)
        np.testing.assert_allclose(total.absolute_error_pp, 0.9)
        self.assertTrue(result["goal_achieved"])
        exact["forecast_daily"] = exact.fact + 0.011 * exact.plan
        _, total = evaluate(exact.drop(columns="fact"), actual, self.config)
        self.assertFalse(metrics(total, self.config)["goal_achieved"])

    def test_missing_actual_is_not_zero_or_success(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        actual = daily_actual(load_raw(self.config, month="2026-10"), self.config).iloc[1:]
        _, total = evaluate(f, actual, self.config)
        first_missing = total[total.missing_actual_cells.gt(0)].index[0]
        self.assertTrue(total.loc[first_missing:, "fact_cumulative"].isna().all())
        self.assertFalse(metrics(total, self.config)["goal_achieved"])

    def test_unplanned_segment_is_in_total(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        actual = daily_actual(load_raw(self.config, month="2026-10"), self.config)
        extra = actual.iloc[[0]].copy()
        extra["subproduct"], extra["fact"] = "NEW", 12345.0
        _, before = evaluate(f, actual, self.config)
        _, after = evaluate(f, pd.concat([actual, extra]), self.config)
        np.testing.assert_allclose(after.fact_cumulative - before.fact_cumulative, 12345.0)
        self.assertFalse(metrics(after, self.config)["complete"])

    def test_zero_plan_and_signed_adjustments(self):
        r = self.runner()
        f = r.forecast(self.spec, save=False)
        first_date = f.date.min()
        f.loc[f.date.eq(first_date), "plan"] = 0
        a = daily_actual(load_raw(self.config, month="2026-10"), self.config)
        self.assertTrue(a.fact.lt(0).any())
        _, total = evaluate(f, a, self.config)
        self.assertTrue(pd.isna(total.actual_efficiency_pct.iloc[0]))
        self.assertTrue(np.isfinite(total.forecast_daily).all())

    def test_new_segment_gets_features_without_future_portfolio(self):
        with self.copied_database() as c:
            with test_connection(c.db_path) as conn:
                extra = pd.read_sql_query("SELECT * FROM rr1plus_data_raw WHERE report_month='2026-10-01' AND subproduct='ПК'", conn)
                extra["subproduct"] = "NEW"
                extra.to_sql(c.raw_table, conn, if_exists="append", index=False)
            r = ForecastRunner(c)
            x = r.builder.build(r.history, "2026-10")
            new = x[x.subproduct.eq("NEW")]
            self.assertTrue(new.new_segment.eq(1).all())
            self.assertTrue(new.debt.eq(0).all())

    def test_missing_plan_day_and_duplicate_fail(self):
        with self.copied_database() as c:
            with test_connection(c.db_path) as conn:
                conn.execute("DELETE FROM rr1plus_data_raw WHERE report_dt='2026-10-01' AND subproduct='ПК'")
            with self.assertRaisesRegex(ValueError, "every segment"):
                Inputs(c).plan("2026-10")
        with self.copied_database() as c:
            with test_connection(c.db_path) as conn:
                row = pd.read_sql_query("SELECT * FROM rr1plus_data_raw WHERE report_dt='2026-10-01' LIMIT 1", conn)
                row.to_sql(c.raw_table, conn, if_exists="append", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                Inputs(c).plan("2026-10")

    def test_development_control_and_saved_artifacts(self):
        r = self.runner()
        rank, selection = r.select()
        frozen = dict(selection)
        control = r.control()
        self.assertEqual(r.selection, frozen)
        self.assertEqual(len(control), 1)
        self.assertTrue(rank.months.eq(1).all())
        out = Path(self.tmp.name) / "results"
        r.config = replace(r.config, output_dir=str(out))
        prediction = r.forecast(save=True)
        _, total, score = r.compare(save=True)
        self.assertTrue((out / "comparison.html").is_file())
        self.assertTrue((out / "comparison.png").is_file())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection"], frozen)
        self.assertEqual(len(total), 31)
        self.assertTrue(score["complete"])

    def test_all_model_families_smoke(self):
        c = replace(self.config, enable_catboost=True, enable_sarimax=True, catboost_iterations=8, catboost_depths=(3,))
        r = ForecastRunner(c)
        x, leaves, totals, status = candidates(r.history, "2026-10", c, r.builder)
        self.assertIn("catboost_daily_d3", leaves)
        self.assertIn("catboost_cumulative_d3", leaves)
        self.assertIn("total_catboost_d3", totals)
        self.assertTrue("total_sarimax" in totals or any(s["model"] == "total_sarimax" and s["status"] == "unavailable" for s in status))
        self.assertTrue(all(np.isfinite(v).all() for v in [*leaves.values(), *totals.values()]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
