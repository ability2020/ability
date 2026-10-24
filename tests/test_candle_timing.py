import unittest

import main


class CandleTimingTests(unittest.TestCase):
    def candle(self, timestamp, close="100"):
        return [str(timestamp), "100", "101", "99", close, "10", "1000"]

    def test_interval_duration_supports_requested_timeframes(self):
        self.assertEqual(main.interval_milliseconds("5m"), 300_000)
        self.assertEqual(main.interval_milliseconds("1H"), 3_600_000)
        self.assertEqual(main.interval_milliseconds("4H"), 14_400_000)

    def test_filter_sorts_and_excludes_forming_or_too_recent_candles(self):
        candles = [
            self.candle(300_000),   # forming at now=601999
            self.candle(0),
            self.candle(600_000),   # also not closed at now=601999
            self.candle("bad"),
            ["too-short"],
        ]
        result = main.filter_closed_candles(
            candles, interval="5m", now_ms=601_999, grace_ms=2_000
        )
        self.assertEqual([int(row[0]) for row in result], [0])

    def test_filter_accepts_candle_after_two_second_grace(self):
        result = main.filter_closed_candles(
            [self.candle(300_000)], interval="5m",
            now_ms=602_000, grace_ms=2_000
        )
        self.assertEqual(len(result), 1)

    def test_waits_until_two_seconds_after_nearby_close(self):
        sleeps = []
        main.wait_for_candle_close_grace(
            "5m", now_fn=lambda: 597.0, sleep_fn=sleeps.append
        )
        self.assertEqual(sleeps, [5.0])

    def test_waits_out_grace_when_called_just_after_close(self):
        sleeps = []
        main.wait_for_candle_close_grace(
            "5m", now_fn=lambda: 600.5, sleep_fn=sleeps.append
        )
        self.assertEqual(sleeps, [1.5])

    def test_does_not_sleep_mid_candle(self):
        sleeps = []
        main.wait_for_candle_close_grace(
            "5m", now_fn=lambda: 120.0, sleep_fn=sleeps.append
        )
        self.assertEqual(sleeps, [])

    def test_unknown_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            main.interval_milliseconds("2H")


if __name__ == "__main__":
    unittest.main()
