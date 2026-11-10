from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from collections_forecast import Config, run
from collections_forecast.data import numeric, prepare_actual, split_dated_frame
from collections_forecast.export import upload_frame, database_upload_frame


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(forecast_month="2026-10", fact_mode="daily", plan_mode="daily", segments=("segment",))
        # 100 segment rows per day must become ONE portfolio row, without
        # repeating a portfolio total on all 100 source rows.
        self.pred = pd.MultiIndex.from_product(
            [pd.date_range("2026-10-01", "2026-10-31"), range(100)],
            names=["date", "segment"]).to_frame(index=False)
        self.pred["forecast_daily"] = 2.0
        self.pred["plan"] = 3.0
        self.pred["forecast_cumulative"] = self.pred.groupby("segment").forecast_daily.cumsum()

    def test_hundred_segments_export_once_and_sum_daily_not_cumulative(self):
        upload = upload_frame(self.pred.sample(frac=1, random_state=5), self.config)
        self.assertEqual(len(upload), 31)
        self.assertFalse(upload.date.duplicated().any())
        self.assertTrue(upload.forecast_daily.eq(200).all())
        self.assertEqual(upload.forecast_daily.sum(), 6200)
        self.assertEqual(upload.plan_daily.sum(), 9300)
        self.assertNotIn("forecast_cumulative", upload)
        self.assertNotIn("segment", upload)

    def test_repeated_rows_rejected_before_aggregation(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            upload_frame(pd.concat([self.pred, self.pred.iloc[:1]]), self.config)

    def test_missing_plan_is_not_partial_total_or_zero(self):
        self.pred.loc[0, "plan"] = np.nan
        upload = upload_frame(self.pred, self.config)
        self.assertTrue(pd.isna(upload.plan_daily.iloc[0]))
        self.assertEqual(upload.forecast_daily.iloc[0], 200)

    def test_incomplete_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "entire target month"):
            upload_frame(self.pred.iloc[1:], self.config)

    def test_rr_schema_preserves_segment_totals_and_has_no_cohort_expansion(self):
        config = replace(self.config, segments=("subproduct",))
        pred = self.pred.rename(columns={"segment": "subproduct"})
        pred["subproduct"] = pred.subproduct.astype(str)
        result = database_upload_frame(pred, config)
        self.assertEqual(len(result), 3100)
        self.assertTrue(result.dt_in.eq("").all())
        self.assertTrue(result.is_forecast.eq(1).all())
        self.assertTrue(result.forecast_model.eq("collections.v2").all())
        self.assertEqual(result.loc[result.report_dt.eq("2026-10-31"), "fact"].sum(), 6200)
        # Summing snapshots over dates is deliberately not a monthly amount.
        self.assertNotEqual(result.fact.sum(), 6200)
        self.assertFalse(result.duplicated(["report_dt", "subproduct", "dpd_bucket", "pvr_bucket", "terbank", "inflow"]).any())

    def test_rr_total_input_exports_one_row_per_date(self):
        config = replace(self.config, segments=("subproduct",))
        pred = self.pred[self.pred.segment.eq(0)].rename(columns={"segment": "subproduct"}).copy()
        pred["subproduct"] = "ВСЕ"
        pred.loc[pred.index[0], "plan"] = np.nan
        result = database_upload_frame(pred, config)
        self.assertEqual(len(result), 31)
        self.assertTrue(result.subproduct.eq("").all())
        self.assertTrue(result.plan.isna().all())
        self.assertEqual(result.fact.iloc[-1], 62)

    def test_rr_label_collision_is_rejected(self):
        config = replace(self.config, segments=("subproduct",))
        pred = self.pred[self.pred.segment.lt(2)].rename(columns={"segment": "subproduct"}).copy()
        pred["subproduct"] = pred.subproduct.map({0: "ВСЕ", 1: "__UNDEFINED__"})
        with self.assertRaisesRegex(ValueError, "collapse"):
            database_upload_frame(pred, config)


class InputDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(forecast_month="2026-10", fact_mode="daily", plan_mode="daily", segments=("segment",))

    def test_raw_aliases_and_source_flags_do_not_duplicate_plans(self):
        raw = pd.DataFrame({
            "report_dt": ["2026-09-01", "2026-09-01", "2026-10-01", "2026-10-01"],
            "is_forecast": [0, 1, 0, 1], "target": [10, 999, 20, 30],
            "w1": [100]*4, "w2": [90]*4, "cession": [0]*4, "write_off": [0]*4,
        })
        h, p, a = split_dated_frame(raw, replace(self.config, plan_is_forecast=1))
        self.assertEqual(len(h), 1)
        self.assertEqual(len(a), 1)
        self.assertEqual(p.target.tolist(), [10, 30])
        self.assertEqual(h.fact.iloc[0], 10)

    def test_bad_numeric_error_names_field_count_and_sample(self):
        with self.assertRaisesRegex(ValueError, "fact; 2/3 rows"):
            numeric(pd.DataFrame({"fact": [10, None, "abc"]}), ["fact"])

    def test_duplicate_error_exposes_actual_keys(self):
        frame = pd.DataFrame({"date": ["2026-09-01"]*2, "fact": [10]*2, "plan": [12]*2})
        h, _, _ = split_dated_frame(frame, self.config)
        with self.assertRaisesRegex(ValueError, "Duplicate raw grain.*2 rows.*segment"):
            prepare_actual(h, self.config, before="2026-10")

    def test_run_persists_diagnostic_without_silent_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(self.config, output_dir=tmp)
            with self.assertRaisesRegex(ValueError, "Missing"):
                run(pd.DataFrame({"date": ["2026-09-01"]}), config=c)
            self.assertIn("Missing", (Path(tmp) / "input_error.txt").read_text(encoding="utf-8"))
            self.assertFalse((Path(tmp) / "forecast_upload.csv").exists())


if __name__ == "__main__":
    unittest.main()
