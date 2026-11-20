import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from realtime_provider import _parse_source_time, _quote_freshness


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


class RealtimeValidationTests(unittest.TestCase):
    def test_parse_full_iso_timestamp(self):
        parsed = _parse_source_time("2026-11-19T10:15:00")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tzinfo, VN_TZ)
        self.assertEqual(parsed.hour, 10)

    def test_reject_date_only_and_time_only(self):
        self.assertIsNone(_parse_source_time("2026-11-19"))
        self.assertIsNone(_parse_source_time("10:15:00"))

    def test_fresh_quote_requires_source_timestamp_and_market_session(self):
        now = datetime(2026, 11, 19, 10, 15, tzinfo=VN_TZ)
        fresh = _quote_freshness("2026-11-19T10:14:00", now=now)
        self.assertTrue(fresh["is_fresh"])
        self.assertEqual(fresh["status"], "FRESH")

        stale = _quote_freshness("2026-11-19T10:00:00", now=now)
        self.assertFalse(stale["is_fresh"])
        self.assertEqual(stale["status"], "STALE")

        outside = _quote_freshness(
            "2026-11-19T15:59:00",
            now=datetime(2026, 11, 19, 16, 0, tzinfo=VN_TZ),
        )
        self.assertFalse(outside["is_fresh"])
        self.assertEqual(outside["status"], "OUTSIDE_SESSION")


if __name__ == "__main__":
    unittest.main()
