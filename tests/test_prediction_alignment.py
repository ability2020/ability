"""Regression tests for next-candle labels and out-of-sample return alignment."""

import unittest

import numpy as np

from entry_quality_diagnostics import build_predictions
from learning_model import FEATURE_COLUMNS, build_feature_frame
from walk_forward_backtest import _net_one_candle_return


def make_candles(count=155):
    candles = []
    close = 100.0
    for i in range(count):
        open_price = close
        # Deterministic movement with both positive and negative next-candle labels.
        change = (0.0015 if i % 3 else -0.0020) + 0.0004 * np.sin(i / 4.0)
        close = open_price * (1.0 + change)
        high = max(open_price, close) * 1.001
        low = min(open_price, close) * 0.999
        candles.append([
            str(1_700_000_000_000 + i * 300_000),
            str(open_price), str(high), str(low), str(close), str(100 + i % 9), "0",
        ])
    return candles


class PredictionAlignmentTests(unittest.TestCase):
    def test_target_at_t_describes_close_at_t_plus_1(self):
        frame = build_feature_frame(make_candles())
        for i in range(len(frame) - 1):
            expected = float(frame.iloc[i + 1]["close"]) > float(frame.iloc[i]["close"])
            self.assertEqual(bool(frame.iloc[i]["target"]), expected)
        self.assertTrue(np.isnan(frame.iloc[-1]["target"]))

    def test_walk_forward_prediction_return_uses_immediately_following_candle(self):
        frame = build_feature_frame(make_candles())
        usable = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        predictions = build_predictions(usable)
        self.assertGreater(len(predictions), 0)

        index_by_timestamp = {
            int(row["timestamp"]): i for i, (_, row) in enumerate(usable.iterrows())
        }
        for prediction in predictions:
            i = index_by_timestamp[prediction["timestamp"]]
            current = usable.iloc[i]
            following = usable.iloc[i + 1]
            expected_direction = float(following["close"]) > float(current["close"])
            expected_return = float(following["close"]) / float(current["close"]) - 1.0
            expected_net_return = _net_one_candle_return(
                float(current["close"]), float(following["close"])
            )
            self.assertEqual(prediction["actual_next_close_up"], expected_direction)
            self.assertAlmostEqual(prediction["next_close_return_fraction"], expected_return, places=12)
            self.assertAlmostEqual(prediction["net_return_fraction"], expected_net_return, places=12)
            self.assertEqual(prediction["timestamp"], int(current["timestamp"]))


if __name__ == "__main__":
    unittest.main()
