import unittest

import feature_ablation


def make_candles(count=500):
    candles = []
    price = 100.0
    for i in range(count):
        change = 0.002 if (i // 7) % 2 == 0 else -0.001
        open_price = price
        price = price * (1 + change)
        high = max(open_price, price) * 1.001
        low = min(open_price, price) * 0.999
        candles.append([str(1_700_000_000_000 + i * 300_000), str(open_price), str(high), str(low), str(price), str(100 + (i % 11)), "0"])
    return list(reversed(candles))


class FeatureAblationTests(unittest.TestCase):
    def test_comparison_uses_disjoint_chronological_folds_and_never_live_eligible(self):
        report = feature_ablation.compare_feature_sets(make_candles(500), fold_count=3)
        self.assertTrue(report["available"])
        self.assertTrue(report["baseline_features"]["available"])
        self.assertTrue(report["expanded_features"]["available"])
        self.assertEqual(report["baseline_features"]["fold_count"], 3)
        self.assertEqual(report["expanded_features"]["fold_count"], 3)
        self.assertFalse(report["real_money_trading_eligible"])
        self.assertFalse(report["orders_sent"])
        for result in (report["baseline_features"], report["expanded_features"]):
            folds = result["folds"]
            self.assertTrue(all(fold["train_count"] > 0 and fold["test_count"] > 0 for fold in folds))
            self.assertTrue(all(0 <= fold["test_accuracy"] <= 1 for fold in folds))
            self.assertTrue(all(0 <= fold["brier_score"] <= 1 for fold in folds))

    def test_reports_insufficient_data_safely(self):
        report = feature_ablation.compare_feature_sets(make_candles(100))
        self.assertFalse(report["available"])
        self.assertFalse(report["real_money_trading_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
