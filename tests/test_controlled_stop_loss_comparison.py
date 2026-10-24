"""Focused safety tests for the controlled stop-loss research comparison."""
import unittest
import pandas as pd

import walk_forward_backtest


class ControlledStopLossComparisonTests(unittest.TestCase):
    def test_scenarios_are_predeclared_and_research_only(self):
        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 101.0, "low": 98.5, "close": 99.0},
            {"timestamp": 3, "open": 99.0, "high": 103.0, "low": 98.5, "close": 102.0},
            {"timestamp": 4, "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0},
        ])
        predictions = [
            {"timestamp": 1, "signal": "BUY", "probability_up": 0.61},
            {"timestamp": 2, "signal": "WAIT", "probability_up": 0.52},
            {"timestamp": 3, "signal": "WAIT", "probability_up": 0.51},
            {"timestamp": 4, "signal": "WAIT", "probability_up": 0.51},
        ]
        result = walk_forward_backtest.compare_stop_loss_scenarios(usable, predictions)
        self.assertEqual([x["stop_loss_percent"] for x in result["scenarios"]],
                         [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        self.assertTrue(result["same_predictions_and_cost_assumptions"])
        self.assertFalse(result["scenario_selection_performed"])
        self.assertTrue(all(not x["real_money_trading_eligible"] for x in result["scenarios"]))
        self.assertTrue(all(x["take_profit_percent"] == 3.0 for x in result["scenarios"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
