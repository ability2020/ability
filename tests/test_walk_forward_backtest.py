import unittest

import numpy as np

import walk_forward_backtest


def make_candles(count=220):
    rng = np.random.default_rng(20261010)
    candles = []
    price = 100.0
    for i in range(count):
        open_price = price
        change = float(rng.normal(0.0001, 0.003))
        price = price * (1.0 + change)
        high = max(open_price, price) * 1.001
        low = min(open_price, price) * 0.999
        candles.append([
            str(1_700_000_000_000 + i * 300_000),
            str(open_price), str(high), str(low), str(price),
            str(100 + (i % 11)), "0",
        ])
    return list(reversed(candles))


class WalkForwardBacktestTests(unittest.TestCase):
    def test_walk_forward_is_chronological_and_reports_costs(self):
        report = walk_forward_backtest.walk_forward_evaluate(make_candles())
        self.assertGreaterEqual(report["walk_forward_prediction_count"], 20)
        self.assertGreater(report["first_test_timestamp"], 0)
        self.assertIn("fee_rate_each_side", report)
        self.assertIn("slippage_rate_each_side", report)
        self.assertIn("walk_forward_majority_baseline_accuracy", report)
        self.assertIn("directional_accuracy_beats_majority_baseline", report)
        self.assertIn("threshold_tuning_method", report)
        self.assertIn("positive", report["threshold_tuning_method"])
        self.assertGreaterEqual(report["selected_buy_threshold_min"], 0.5)
        self.assertLessEqual(report["selected_buy_threshold_max"], 1.0)
        self.assertIn("tuned_buy_only_fixed_stake_total_pnl_usdt", report)
        self.assertIn("tuned_threshold_beats_fixed_threshold_after_costs", report)
        self.assertIn("buy_only_net_total_return_percent_compounded", report)
        self.assertIn("always_buy_net_total_return_percent_compounded", report)
        self.assertLessEqual(
            report["buy_signal_count"] + report["sell_signal_count"] + report["wait_signal_count"],
            report["walk_forward_prediction_count"],
        )
        self.assertIn("no orders are submitted", " ".join(report["caveats"]))

    def test_rejects_too_little_data(self):
        with self.assertRaises(ValueError):
            walk_forward_backtest.walk_forward_evaluate(make_candles(80))


if __name__ == "__main__":
    unittest.main(verbosity=2)
