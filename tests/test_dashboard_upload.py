from dataclasses import replace
import io
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, aggregate_database_upload
from collections_forecast.export import database_upload_frame


class DashboardUploadTests(unittest.TestCase):
    def setUp(self):
        c = Config(forecast_month="2026-10", segments=("subproduct",))
        pred = pd.MultiIndex.from_product([pd.date_range("2026-10-01", "2026-10-31"), range(100)],
                                         names=["date", "subproduct"]).to_frame(index=False)
        pred["forecast_daily"], pred["plan"], pred["debt"] = 2., 1., 10.
        self.rows = database_upload_frame(pred, c)
        self.debt = pd.DataFrame({"report_day": [1, 25], "dolg": [1000., 2000.]})

    def test_total_debt_is_attached_once_not_100_times(self):
        f = aggregate_database_upload(self.rows, debt_by_day=self.debt)
        self.assertEqual(len(f), 31)
        self.assertEqual(f.fact.iloc[-1], 6200)
        self.assertEqual(f.plan.iloc[-1], 3100)
        self.assertEqual(f.dolg.iloc[-1], 2000)
        self.assertAlmostEqual(f.fact.iloc[-1] / f.dolg.iloc[-1], 3.1)
        self.assertTrue((f.fact / f.plan).eq(2).all())
        self.assertTrue(f.forecast_date.eq("2026-09-30").all())

    def test_fill_uses_dates_not_segment_row_order(self):
        shuffled = self.rows.sample(frac=1, random_state=42)
        expected = aggregate_database_upload(self.rows, debt_by_day=self.debt)
        assert_frame_equal(aggregate_database_upload(shuffled, debt_by_day=self.debt), expected)
        self.assertTrue(expected.loc[expected.report_dt.lt("2026-10-25"), "dolg"].eq(1000).all())
        self.assertTrue(expected.loc[expected.report_dt.ge("2026-10-25"), "dolg"].eq(2000).all())

    def test_unknown_leading_debt_and_plan_stay_unknown(self):
        self.rows.loc[self.rows.report_dt.eq("2026-10-01"), "plan"] = np.nan
        f = aggregate_database_upload(self.rows, debt_by_day=self.debt.iloc[1:])
        self.assertTrue(f.loc[f.report_dt.lt("2026-10-25"), "dolg"].isna().all())
        self.assertTrue(pd.isna(f.plan.iloc[0]))

    def test_duplicate_forecast_and_duplicate_debt_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate forecast"):
            aggregate_database_upload(pd.concat([self.rows, self.rows.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "one total per date"):
            aggregate_database_upload(self.rows, debt_by_day=pd.concat([self.debt, self.debt]))

    def test_cannot_mix_models_vintages_or_relabel_month(self):
        for column, value in [("forecast_model", "another"), ("forecast_date", pd.Timestamp("2026-10-01"))]:
            f = self.rows.copy()
            f.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, "at a time"):
                aggregate_database_upload(f)
        with self.assertRaisesRegex(ValueError, "CSV month differs"):
            aggregate_database_upload(self.rows, report_month="2026-11-01")

    def test_changing_segment_counts_after_day25_rejected(self):
        f = self.rows[~(self.rows.report_dt.gt("2026-10-25") & self.rows.subproduct.eq("0"))]
        with self.assertRaisesRegex(ValueError, "entire month"):
            aggregate_database_upload(f)

    def test_csv_roundtrip_and_reaggregation_do_not_change_amounts(self):
        csv = pd.read_csv(io.StringIO(self.rows.to_csv(index=False)))
        total = aggregate_database_upload(csv, debt_by_day=self.debt)
        again = aggregate_database_upload(total, debt_by_day=self.debt)
        assert_frame_equal(total, again)

    def test_standalone_adapter_matches_package(self):
        from dashboard_upload_adapter import aggregate_database_upload as standalone
        assert_frame_equal(standalone(self.rows, debt_by_day=self.debt),
                           aggregate_database_upload(self.rows, debt_by_day=self.debt))


if __name__ == "__main__":
    unittest.main()
