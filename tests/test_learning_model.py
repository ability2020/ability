import unittest
from unittest.mock import Mock, patch

import numpy as np

import learning_model


def make_candles(count=180):
    candles = []
    price = 100.0
    for i in range(count):
        # Deterministic changing path with both rises and falls.
        change = 0.002 if (i // 7) % 2 == 0 else -0.001
        open_price = price
        price = price * (1 + change)
        high = max(open_price, price) * 1.001
        low = min(open_price, price) * 0.999
        candles.append([str(1_700_000_000_000 + i * 300_000),
                        str(open_price), str(high), str(low), str(price),
                        str(100 + (i % 11)), "0"])
    return list(reversed(candles))  # mimic Bitget newest-first response


class LearningModelTests(unittest.TestCase):
    def test_features_are_chronological_and_label_next_candle(self):
        frame = learning_model.build_feature_frame(make_candles())
        self.assertTrue(frame["timestamp"].is_monotonic_increasing)
        self.assertTrue(np.isnan(frame.iloc[-1]["target"]))
        self.assertGreater(frame["return_1"].notna().sum(), 150)
        for feature in ("rsi_14", "macd_hist_pct", "atr_pct", "volume_zscore_20"):
            self.assertIn(feature, frame.columns)
        self.assertTrue(frame["rsi_14"].dropna().between(0, 100).all())

    def test_training_uses_chronological_holdout_and_emits_signal(self):
        report = learning_model.train_and_evaluate(make_candles())
        self.assertGreaterEqual(report["labelled_sample_count"], learning_model.MIN_SAMPLES)
        self.assertGreater(report["train_sample_count"], report["test_sample_count"])
        self.assertIn(report["model_signal"], ("BUY", "SELL", "WAIT"))
        self.assertIn("test_accuracy", report)
        self.assertIn("majority_baseline_accuracy", report)
        self.assertIn("majority_baseline_balanced_accuracy", report)
        self.assertIn("probability_brier_score", report)
        self.assertGreaterEqual(report["probability_brier_score"], 0.0)
        self.assertLessEqual(report["probability_brier_score"], 1.0)
        self.assertIn("buy_threshold_precision", report)
        self.assertIn("buy_threshold_recall", report)
        self.assertEqual(
            sum(report["confusion_matrix"].values()),
            report["test_sample_count"],
        )
        self.assertIn("does not control paper trades", report["safety_note"])

    def test_fetch_candles_paginates_backwards_and_deduplicates(self):
        def response(rows):
            mocked = Mock()
            mocked.raise_for_status.return_value = None
            mocked.json.return_value = {"code": "00000", "data": rows}
            return mocked

        # fetch_candles anchors its range to the current time; these rows are
        # relative to the first request's endTime, so they remain in-range.
        def page_for_call(call_index, request_args):
            end_time = int(request_args["params"]["endTime"])
            if call_index == 0:
                return response([
                    [str(end_time - 300_000), "100", "101", "99", "100.5", "10", "0"],
                    [str(end_time - 600_000), "100", "101", "99", "100.0", "10", "0"],
                ])
            if call_index == 1:
                return response([
                    [str(end_time - 300_000), "100", "101", "99", "100.0", "10", "0"],
                    [str(end_time - 600_000), "99", "100", "98", "99.5", "10", "0"],
                ])
            return response([])

        calls = []

        def fake_get(*args, **kwargs):
            calls.append(kwargs)
            return page_for_call(len(calls) - 1, kwargs)

        with patch.object(learning_model.requests, "get", side_effect=fake_get):
            candles = learning_model.fetch_candles(lookback_days=1)

        timestamps = [int(candle[0]) for candle in candles]
        self.assertEqual(timestamps, sorted(set(timestamps)))
        self.assertEqual(len(candles), 4)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0]["params"]["limit"], "1000")
        self.assertIn("startTime", calls[0]["params"])
        self.assertIn("endTime", calls[0]["params"])

    def test_fetch_candles_rejects_lookback_outside_regular_endpoint_window(self):
        with self.assertRaises(ValueError):
            learning_model.fetch_candles(lookback_days=90)

    def test_rejects_too_little_data(self):
        with self.assertRaises(ValueError):
            learning_model.train_and_evaluate(make_candles(40))


if __name__ == "__main__":
    unittest.main(verbosity=2)
