import unittest
from unittest.mock import patch

import model_selection


def make_candles(count=500):
    candles = []
    price = 100.0
    for i in range(count):
        change = 0.002 if (i // 7) % 2 == 0 else -0.001
        open_price = price
        price = price * (1 + change)
        high = max(open_price, price) * 1.001
        low = min(open_price, price) * 0.999
        candles.append([
            str(1_700_000_000_000 + i * 300_000), str(open_price),
            str(high), str(low), str(price), str(100 + (i % 11)), "0",
        ])
    return list(reversed(candles))


class ModelSelectionTests(unittest.TestCase):
    def test_uses_chronological_validation_then_untouched_test(self):
        report = model_selection.evaluate_model_selection(make_candles(500))
        self.assertTrue(report["available"])
        split = report["split"]
        self.assertEqual(split["chronological_order"], "train -> validation -> untouched test")
        self.assertFalse(split["test_used_for_selection"])
        self.assertEqual(split["training_count"] + split["validation_count"] + split["untouched_test_count"], report["usable_labelled_candle_count"])
        self.assertIn(report["selected_model"], ("baseline_features", "expanded_features"))
        self.assertIn(report["selected_buy_threshold"], model_selection.CANDIDATE_THRESHOLDS + (model_selection.DEFAULT_THRESHOLD,))
        self.assertIn("fixed_stake_net_pnl_usdt", report["holdout"])
        self.assertIn("model_diagnostics_at_fixed_threshold", report["holdout"])
        self.assertFalse(report["real_money_trading_eligible"])
        self.assertFalse(report["orders_sent"])

    def test_report_interval_uses_configured_interval(self):
        with patch.object(model_selection, "INTERVAL", "1H"):
            report = model_selection.evaluate_model_selection(make_candles(500))
        self.assertEqual(report["interval"], "1H")

    def test_reports_insufficient_data_without_live_eligibility(self):
        report = model_selection.evaluate_model_selection(make_candles(100))
        self.assertFalse(report["available"])
        self.assertFalse(report["real_money_trading_eligible"])
        self.assertFalse(report["orders_sent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
