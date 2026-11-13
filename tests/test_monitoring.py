import unittest
from datetime import datetime, timezone, timedelta
from monitoring import build_monitoring_report


class MonitoringReportTests(unittest.TestCase):
    def test_first_observation_and_candle_age(self):
        now = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
        candle_ms = int((now - timedelta(minutes=5)).timestamp() * 1000)
        report = build_monitoring_report(
            {"last_closed_candle_timestamp": candle_ms, "closed_candle_count": 60},
            {},
            now,
        )
        self.assertEqual(report["cadence_status"], "FIRST_OBSERVATION")
        self.assertIsNone(report["estimated_missed_intervals"])
        self.assertIn("cannot be estimated", report["cadence_warning"])
        self.assertEqual(report["latest_closed_candle_age_seconds"], 300.0)
        self.assertFalse(report["state_restored"])

    def test_delayed_observation_gap(self):
        now = datetime(2026, 11, 9, 12, 45, tzinfo=timezone.utc)
        previous = {"last_observed_at_utc": "2026-11-09T12:00:00+00:00"}
        report = build_monitoring_report({}, previous, now)
        self.assertEqual(report["seconds_since_previous_observation"], 2700.0)
        self.assertEqual(report["cadence_status"], "DELAYED_OR_MISSED_INTERVAL")
        self.assertEqual(report["estimated_missed_intervals"], 2)
        self.assertIn("may have been delayed or missed", report["cadence_warning"])
        self.assertTrue(report["state_restored"])

    def test_near_schedule_gap(self):
        now = datetime(2026, 11, 9, 12, 14, tzinfo=timezone.utc)
        previous = {"last_observed_at_utc": "2026-11-09T12:00:00+00:00"}
        report = build_monitoring_report({}, previous, now)
        self.assertEqual(report["cadence_status"], "ON_OR_NEAR_SCHEDULE")
        self.assertEqual(report["estimated_missed_intervals"], 0)
        self.assertEqual(report["cadence_warning"], "No delayed observation gap detected.")


if __name__ == "__main__":
    unittest.main()
