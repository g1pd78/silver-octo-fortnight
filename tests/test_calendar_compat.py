from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import holidays
import pandas as pd
from collections_forecast import Config
from collections_forecast.features import FeatureBuilder, working_day


class LegacyRU(dict):
    country = 'RU'
    weekend = {5, 6}


class CalendarCompatibilityTests(unittest.TestCase):
    def builder(self, overrides=None):
        return FeatureBuilder(Config(forecast_month='2026-10'), SimpleNamespace(calendar=overrides))

    def test_october_without_working_day_method(self):
        with patch('holidays.country_holidays', return_value=LegacyRU()):
            calendar = self.builder().calendar('2026-10')
        self.assertEqual(len(calendar), 31)
        self.assertEqual(calendar.is_workday.sum(), 22)
        self.assertEqual(calendar.loc[calendar.date.eq('2026-10-03'), 'is_workday'].iloc[0], 0)

    def test_transfers_survive_missing_method_and_old_transfer_tables(self):
        old = LegacyRU()
        self.assertTrue(working_day(old, date(2024, 4, 27), 'RU'))
        self.assertTrue(working_day(old, date(2025, 11, 1), 'RU'))
        self.assertFalse(working_day(old, date(2025, 11, 3), 'RU'))
        self.assertFalse(working_day(old, date(2026, 1, 9), 'RU'))
        self.assertFalse(working_day(old, date(2026, 12, 31), 'RU'))

    def test_bundled_workdays_match_current_library_for_2024_to_2026(self):
        reference = holidays.country_holidays('RU', years=[2024, 2025, 2026])
        # This test checks the verified modern calendar; older dependency installs
        # are covered separately by fixtures that do not expose the method.
        if not callable(getattr(reference, 'is_working_day', None)):
            self.skipTest('Native working-day API unavailable')
        if tuple(int(p) for p in holidays.__version__.split('.')[:2]) < (0, 108):
            self.skipTest('Old native transfer tables are not the reference calendar')
        for timestamp in pd.date_range('2024-01-01', '2026-12-31'):
            value = timestamp.date()
            self.assertEqual(working_day(reference, value, 'RU'), reference.is_working_day(value), str(value))

    def test_explicit_calendar_override_wins(self):
        overrides = pd.DataFrame({'date': [pd.Timestamp('2026-10-03')], 'is_workday': [1]})
        with patch('holidays.country_holidays', return_value=LegacyRU()):
            calendar = self.builder(overrides).calendar('2026-10')
        self.assertEqual(calendar.loc[calendar.date.eq('2026-10-03'), 'is_workday'].iloc[0], 1)

    def test_generic_legacy_calendar_honors_holidays_and_transferred_weekends(self):
        old = LegacyRU({date(2030, 1, 1): 'Holiday'})
        old.weekend_workdays = {date(2030, 1, 5)}
        self.assertFalse(working_day(old, date(2030, 1, 1), 'OTHER'))
        self.assertTrue(working_day(old, date(2030, 1, 5), 'OTHER'))
        self.assertFalse(working_day(old, date(2030, 1, 6), 'OTHER'))


if __name__ == '__main__':
    unittest.main()
