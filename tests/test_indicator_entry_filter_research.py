import unittest

import pandas as pd

from indicator_entry_filter_research import (
    add_entry_features,
    apply_entry_filter,
    entry_filter_mask,
)


class IndicatorEntryFilterResearchTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "timestamp": [1, 2, 3, 4, 5],
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "sma_9": [101, 102, 103, 104, 105],
            "sma_21": [100, 101, 102, 103, 104],
            "rsi_14": [55, 60, 66, 52, 48],
            "signal": ["BUY", "BUY", "BUY", "SELL", "WAIT"],
        })

    def test_baseline_allows_all_rows(self):
        self.assertTrue(entry_filter_mask(self.frame, "baseline").all())

    def test_rsi_filter_blocks_overextended_and_low_rsi_buys(self):
        mask = entry_filter_mask(self.frame, "rsi_50_to_65")
        self.assertEqual(mask.tolist(), [True, True, False, True, False])

    def test_apply_filter_changes_only_blocked_buy_signals(self):
        filtered, blocked = apply_entry_filter(self.frame, "rsi_50_to_65")
        # Only the third row is a BUY outside the RSI range. The final row
        # has low RSI but is WAIT, so it must not count as a blocked BUY.
        self.assertEqual(blocked, 1)
        self.assertEqual(filtered["signal"].tolist(), ["BUY", "BUY", "WAIT", "SELL", "WAIT"])

    def test_trend_confirmation_requires_both_conditions(self):
        enriched = add_entry_features(self.frame)
        mask = entry_filter_mask(enriched, "trend_confirmation")
        self.assertFalse(bool(mask.iloc[0]))
        self.assertFalse(bool(mask.iloc[1]))
        self.assertFalse(bool(mask.iloc[2]))

    def test_unknown_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            entry_filter_mask(self.frame, "made_up_filter")


if __name__ == "__main__":
    unittest.main()
