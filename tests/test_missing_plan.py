from dataclasses import replace
import tempfile
import sqlite3
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from collections_forecast import Config, run, ForecastRunner
from collections_forecast.demo import demo_rows, make_demo


class MissingPlanTests(unittest.TestCase):
    def setUp(self):
        raw = demo_rows()
        raw['fact'] = raw.w1 - raw.w2 - raw.cession - raw.write_off
        self.df = raw.groupby('report_dt', as_index=False).agg(fact=('fact','sum'), plan=('target','sum')).rename(columns={'report_dt':'date'})
        self.df.date = pd.to_datetime(self.df.date)
        self.c = Config(forecast_month='2026-10', fact_mode='daily', plan_mode='daily',
                        enable_catboost=False, enable_sarimax=False)

    def check_result(self, frame, config=None, save=False):
        result = run(frame, config=config or self.c, save=save)
        forecast = result['forecast']
        self.assertEqual(len(forecast), 31)
        self.assertEqual(forecast.date.max(), pd.Timestamp('2026-10-31'))
        self.assertTrue(np.isfinite(forecast.forecast_daily).all())
        self.assertTrue(forecast.model.str.startswith('fallback_').all())
        self.assertEqual(result['plan_status']['mode'], 'money_fallback')
        self.assertEqual(result['plan_status']['validation'], 'not_run')
        self.assertTrue(result['plan_status']['reasons'])
        self.assertTrue(result['ranking'].empty)
        self.assertTrue(result['control'].empty)
        return result

    def test_gap_forecasts_to_end_and_does_not_claim_success(self):
        self.df.loc[self.df.date.eq('2026-10-12'), 'plan'] = np.nan
        result = self.check_result(self.df)
        total = result['comparison']
        self.assertTrue(total.loc[total.date.ge('2026-10-12'), 'forecast_efficiency_pct'].isna().all())
        self.assertFalse(result['metrics']['goal_achieved'])
        self.assertEqual(result['metrics']['missing_plan_cells'], 1)

    def test_no_future_rows_still_forecasts_full_month_and_saves(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.check_result(self.df[self.df.date.lt('2026-10-01')], replace(self.c, output_dir=temp), save=True)
            self.assertTrue(result['forecast'].plan.isna().all())
            self.assertIsNone(result['comparison'])
            upload = pd.read_csv(temp + '/forecast_upload.csv')
            self.assertEqual(len(upload), 31)
            self.assertIn('fact', upload)
            self.assertNotIn('forecast_daily', upload)
            self.assertAlmostEqual(upload.fact.iloc[-1], result['forecast'].forecast_daily.sum())

    def test_all_nan_future_plan_and_modified_future_fact(self):
        self.df.loc[self.df.date.ge('2026-10-01'), 'plan'] = np.nan
        before = self.check_result(self.df)
        self.df.loc[self.df.date.ge('2026-10-01'), 'fact'] *= 1000
        after = self.check_result(self.df)
        assert_frame_equal(before['forecast'], after['forecast'])

    def test_cumulative_plan_gap_is_unknown_not_a_lumped_payment(self):
        self.df['plan'] = self.df.groupby(self.df.date.dt.to_period('M')).plan.cumsum()
        self.df.loc[self.df.date.eq('2026-10-12'), 'plan'] = np.nan
        result = self.check_result(self.df, replace(self.c, plan_mode='cumulative'))
        f = result['forecast'].set_index('date')
        self.assertTrue(pd.isna(f.loc['2026-10-12', 'plan']))
        self.assertTrue(pd.isna(f.loc['2026-10-13', 'plan']))

    def test_entire_segment_plan_missing(self):
        raw = demo_rows()
        plan = raw[~(raw.report_dt.ge('2026-10-01') & raw.subproduct.eq('ПК'))]
        r = ForecastRunner(self.c, history_df=raw[raw.report_dt.lt('2026-10-01')], plan_df=plan)
        f = r.forecast(save=False)
        self.assertEqual(len(f), 62)
        self.assertTrue(f.loc[f.subproduct.eq('ПК'), 'plan'].isna().all())

    def test_sqlite_null_future_plan_can_save_forecast_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            c = replace(make_demo(temp), output_dir=temp + '/results')
            with sqlite3.connect(c.db_path) as conn:
                conn.execute("UPDATE rr1plus_data_raw SET target=NULL WHERE report_month='2026-10-01'")
            conn.close()
            r = ForecastRunner(c)
            f = r.forecast(save=True)
            self.assertEqual(len(f), 62)
            _, _, score = r.compare(save=True)
            self.assertFalse(score['goal_achieved'])
            self.assertEqual(score['missing_plan_cells'], 62)


if __name__ == '__main__':
    unittest.main()
