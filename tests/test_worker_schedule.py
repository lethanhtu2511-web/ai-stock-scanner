import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from worker import daily_scan_due


VN = ZoneInfo("Asia/Ho_Chi_Minh")


class DailyScanScheduleTests(unittest.TestCase):
    def test_weekday_after_close_runs_when_not_done(self):
        now = datetime(2026, 11, 19, 15, 36, tzinfo=VN)
        self.assertTrue(daily_scan_due(now))
        self.assertFalse(daily_scan_due(now, "2026-11-19"))

    def test_before_daily_time_does_not_run(self):
        now = datetime(2026, 11, 19, 15, 34, tzinfo=VN)
        self.assertFalse(daily_scan_due(now))

    def test_weekend_does_not_run(self):
        now = datetime(2026, 11, 21, 16, 0, tzinfo=VN)
        self.assertFalse(daily_scan_due(now))


if __name__ == "__main__":
    unittest.main()
