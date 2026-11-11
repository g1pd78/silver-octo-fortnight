from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, filtered_dashboard_upload, read_filtered_dashboard_csv


class FilteredUploadTests(unittest.TestCase):
    def setUp(self):
        self.c = Config(forecast_month="2026-11", fact_mode="cumulative", plan_mode="cumulative", forecast_grain="total")
        rows = []
        for date in pd.date_range("2026-11-01", "2026-11-30"):
            rows.append(dict(date=date, dt_in=pd.Timestamp("2026-11-01"), subproduct="ПК", dpd_bucket="01. 1-30",
                terbank="СИБ", pvr_bucket="1. [1-15]", inflow=0, w1=10000., target=100.*date.day))
            if date.day >= 15:
                rows.append(dict(date=date, dt_in=pd.Timestamp("2026-11-15"), subproduct="ИК ГОС", dpd_bucket="00. 1-30",
                    terbank="ЦЧБ", pvr_bucket="0. [undef]", inflow=1, w1=2000., target=10.*(date.day-14)))
        self.layout = pd.DataFrame(rows)
        self.total = self.layout.groupby("date", as_index=False).target.sum().rename(columns={"date":"report_dt", "target":"plan"})
        self.total["fact"] = self.total.plan * .9
        self.total["forecast_date"] = pd.Timestamp("2026-11-01")

    def export(self, layout=None, total=None, config=None):
        return filtered_dashboard_upload(self.total if total is None else total,
            self.layout if layout is None else layout, config or self.c)

    def test_exact_columns_real_filters_cohorts_and_conserved_totals(self):
        f = self.export()
        self.assertEqual(f.columns.tolist(), ["report_month","report_dt","dt_in","subproduct","dpd_bucket","terbank",
            "pvr_bucket","inflow","dolg","plan","fact","is_forecast","forecast_date","forecast_model"])
        self.assertEqual(len(f), 46)
        self.assertEqual(set(f.subproduct), {"ПК", "ИК ГОС"})
        self.assertEqual(set(f.dpd_bucket), {"00. 1-30", "01. 1-30"})
        np.testing.assert_allclose(f.groupby("report_dt").fact.sum(), self.total.fact)
        np.testing.assert_allclose(f.groupby("report_dt").plan.sum(), self.total.plan)
        self.assertTrue(f.loc[f.inflow.eq(1), "dt_in"].eq("2026-11-15").all())
        self.assertTrue(f.loc[f.inflow.eq(1), "dolg"].eq(2000).all())
        self.assertFalse((f.report_dt < f.dt_in).any())

    def test_source_actual_fact_is_not_used_and_row_order_does_not_matter(self):
        source = self.layout.sample(frac=1, random_state=8).copy()
        source["fact"] = 1e18
        assert_frame_equal(self.export(source), self.export())

    def test_duplicated_or_incomplete_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.export(pd.concat([self.layout, self.layout.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "every forecast date"):
            self.export(self.layout[self.layout.date.ne("2026-11-12")])
        with self.assertRaisesRegex(ValueError, "real cohort fields"):
            self.export(self.layout.drop(columns="w1"))

    def test_different_plan_scope_or_daily_units_rejected(self):
        wrong = self.layout.copy()
        wrong["target"] /= wrong.date.dt.day
        with self.assertRaisesRegex(ValueError, "plan totals differ"):
            self.export(wrong)

    def test_unknown_plan_uses_debt_without_inventing_official_plan(self):
        source, total = self.layout.copy(), self.total.copy()
        source["target"] = np.nan
        total["plan"] = np.nan
        f = self.export(source, total)
        self.assertTrue(f.plan.isna().all())
        np.testing.assert_allclose(f.groupby("report_dt").fact.sum(), total.fact)
        end = f[f.report_dt.eq("2026-11-30")]
        np.testing.assert_allclose(end.fact / end.dolg, [total.fact.iloc[-1]/12000]*2)

    def test_future_plan_revision_excluded(self):
        source = self.layout.copy()
        source["known_at"] = pd.Timestamp("2026-10-30")
        future = source.copy()
        future["known_at"] = pd.Timestamp("2026-11-02")
        future["target"] *= 100
        c = replace(self.c, plan_available_column="known_at")
        assert_frame_equal(self.export(pd.concat([source, future]), config=c), self.export())

    def test_utf8_bom_iso_dates_and_filter_labels_survive_loader(self):
        f = self.export()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"forecast_collections.v2.allocated_20261101.csv"
            f.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            loaded = read_filtered_dashboard_csv(path)
            self.assertEqual(loaded.subproduct.tolist(), f.subproduct.tolist())
            self.assertEqual(loaded.dpd_bucket.tolist(), f.dpd_bucket.tolist())
            self.assertEqual(loaded.dt_in.tolist(), f.dt_in.tolist())
            np.testing.assert_allclose(loaded.fact, f.fact)
            from dashboard_upload_adapter import read_filtered_dashboard_csv as standalone
            assert_frame_equal(standalone(path), loaded)

    def test_excel_converted_bucket_rejected(self):
        bad = self.layout.copy()
        bad["dpd_bucket"] = "01.01.1930"
        with self.assertRaisesRegex(ValueError, "Excel-converted"):
            self.export(bad)

    def test_filter_subset_selects_its_own_money_and_debt(self):
        f = self.export()
        selected = f[(f.subproduct == "ИК ГОС") & (f.dpd_bucket == "00. 1-30") &
                     (f.pvr_bucket == "0. [undef]") & (f.terbank == "ЦЧБ") & (f.inflow == 1)]
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected.report_dt.min(), pd.Timestamp("2026-11-15"))
        self.assertAlmostEqual(selected.fact.iloc[-1], 144.)
        self.assertAlmostEqual(selected.plan.iloc[-1], 160.)
        self.assertAlmostEqual(selected.dolg.iloc[-1], 2000.)


if __name__ == "__main__":
    unittest.main()
