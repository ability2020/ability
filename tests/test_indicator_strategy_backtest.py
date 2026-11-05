import unittest
import numpy as np
import pandas as pd

import indicator_strategy_backtest as replay


class IndicatorReplayTests(unittest.TestCase):
    def make_frame(self, signals, opens=None, highs=None, lows=None, closes=None):
        n = len(signals)
        opens = opens or [100.0] * n
        highs = highs or [101.0] * n
        lows = lows or [99.0] * n
        closes = closes or [100.0] * n
        return pd.DataFrame({"timestamp": np.arange(1, n + 1), "open": opens,
                             "high": highs, "low": lows, "close": closes,
                             "signal": signals})

    def test_entry_uses_next_candle_open_not_signal_close(self):
        frame = self.make_frame(["BUY", "WAIT", "WAIT"], opens=[100, 105, 106])
        trades = replay.simulate(frame, "protective_only")
        self.assertEqual(trades[0]["entry_price"], 105.0)

    def test_sell_signal_exits_at_next_open(self):
        frame = self.make_frame(["BUY", "SELL", "WAIT", "WAIT"], opens=[100, 101, 102, 103])
        trades = replay.simulate(frame, "immediate_sell")
        self.assertEqual(trades[0]["exit_reason"], "SELL_SIGNAL")
        self.assertEqual(trades[0]["exit_price"], 102.0)

    def test_confirmed_sell_requires_two_consecutive_closed_sell_candles(self):
        frame = self.make_frame(["BUY", "SELL", "WAIT", "SELL", "SELL", "WAIT"])
        trades = replay.simulate(frame, "confirmed_sell_2")
        self.assertEqual(trades[0]["exit_reason"], "CONFIRMED_SELL_SIGNAL")
        self.assertEqual(trades[0]["exit_timestamp"], 6)

    def test_stop_loss_precedes_target_when_both_touched(self):
        frame = self.make_frame(["BUY", "WAIT", "WAIT"], opens=[100, 100, 100],
                                highs=[101, 104, 101], lows=[99, 97, 99])
        trades = replay.simulate(frame, "protective_only")
        self.assertEqual(trades[0]["exit_reason"], "STOP_LOSS")
        self.assertEqual(trades[0]["exit_price"], 98.0)

    def test_gap_through_stop_fills_at_open(self):
        frame = self.make_frame(["BUY", "WAIT", "WAIT"], opens=[100, 100, 95],
                                highs=[101, 101, 96], lows=[99, 99, 94])
        trades = replay.simulate(frame, "protective_only")
        self.assertEqual(trades[0]["exit_reason"], "STOP_LOSS_GAP")
        self.assertEqual(trades[0]["exit_price"], 95.0)

    def test_costs_are_included_and_reconcile(self):
        frame = self.make_frame(["BUY", "SELL", "WAIT"], opens=[100, 101, 102])
        trade = replay.simulate(frame, "immediate_sell")[0]
        self.assertAlmostEqual(trade["net_pnl_usdt"],
                               trade["gross_pnl_usdt"] - trade["estimated_costs_usdt"])
        self.assertGreater(trade["estimated_costs_usdt"], 0)


if __name__ == "__main__":
    unittest.main()
