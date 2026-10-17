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
        self.assertIn("single_period_research_gate_passed", report)
        self.assertIn("single_period_research_gate_checks", report)
        self.assertFalse(report["real_money_trading_eligible"])
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

    def test_research_gate_requires_positive_after_costs_and_benchmark_outperformance(self):
        passing = walk_forward_backtest.evaluate_research_gate(
            860, 50, 0.002, -0.001, 1.0
        )
        self.assertTrue(passing["single_period_research_gate_passed"])
        self.assertFalse(passing["real_money_trading_eligible"])

        losing = walk_forward_backtest.evaluate_research_gate(
            860, 50, -0.003, -0.001, -1.0
        )
        self.assertFalse(losing["single_period_research_gate_passed"])
        self.assertFalse(losing["single_period_research_gate_checks"]["positive_average_net_buy_return"])

    def test_position_strategy_uses_stop_loss_and_includes_costs(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 100.5, "low": 97.0, "close": 98.0},
        ])
        predictions = [
            {"timestamp": 1, "signal": "BUY"},
            {"timestamp": 2, "signal": "WAIT"},
        ]
        result = walk_forward_backtest.simulate_position_strategy(
            usable, predictions, stop_loss_percent=2.0, take_profit_percent=3.0
        )
        self.assertEqual(result["position_strategy_trade_count"], 1)
        self.assertEqual(result["position_strategy_exit_reasons"]["STOP_LOSS"], 1)
        self.assertLess(result["position_strategy_closed_trade_pnl_usdt"], 0)
        self.assertFalse(result.get("real_money_trading_eligible", True))

    def test_position_strategy_uses_take_profit_ohlc_touch(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            # Target is 103; the candle touches it even though it closes below it.
            {"timestamp": 2, "open": 100.0, "high": 104.0, "low": 99.0, "close": 101.0},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 2, "signal": "WAIT"}],
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
        )
        self.assertEqual(result["position_strategy_exit_reasons"]["TAKE_PROFIT"], 1)
        trade = result["position_strategy_trades_sample"][0]
        self.assertAlmostEqual(trade["exit_price"], 103.0)
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertFalse(result["real_money_trading_eligible"])

    def test_position_strategy_checks_ohlc_when_prediction_is_missing(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            # No prediction for timestamp 2; the stop must still trigger.
            {"timestamp": 2, "open": 100.0, "high": 100.5, "low": 97.0, "close": 99.0},
            {"timestamp": 3, "open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 3, "signal": "WAIT"}],
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
        )
        self.assertEqual(result["position_strategy_exit_reasons"]["STOP_LOSS"], 1)
        self.assertEqual(result["position_strategy_exit_reasons"]["END_OF_TEST"], 0)
        self.assertAlmostEqual(result["position_strategy_trades_sample"][0]["exit_price"], 98.0)

    def test_position_strategy_stop_loss_wins_if_both_levels_touched(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 105.0, "low": 97.0, "close": 102.0},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 2, "signal": "WAIT"}],
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
        )
        self.assertEqual(result["position_strategy_exit_reasons"]["STOP_LOSS"], 1)
        self.assertEqual(result["position_strategy_exit_reasons"]["TAKE_PROFIT"], 0)
        self.assertAlmostEqual(result["position_strategy_trades_sample"][0]["exit_price"], 98.0)


    def test_position_strategy_reports_touch_counts_exit_prices_and_net_pnl(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            # Both levels touched: stop is the assumed exit, but both touches are counted.
            {"timestamp": 2, "open": 100.0, "high": 104.0, "low": 97.0, "close": 101.0},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}],
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
        )
        self.assertEqual(result["position_strategy_checked_open_position_candles"], 1)
        self.assertEqual(result["position_strategy_missing_prediction_candles_checked"], 1)
        self.assertEqual(result["position_strategy_stop_level_touch_candle_count"], 1)
        self.assertEqual(result["position_strategy_target_level_touch_candle_count"], 1)
        self.assertEqual(result["position_strategy_both_levels_touch_candle_count"], 1)
        self.assertAlmostEqual(result["position_strategy_exit_price_sample"][0]["exit_price"], 98.0)
        self.assertLess(result["position_strategy_exit_price_sample"][0]["net_pnl_usdt"], 0.0)

    def test_position_strategy_closes_remaining_position_at_end_of_test(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"timestamp": 3, "open": 100.5, "high": 101.0, "low": 100.0, "close": 100.25},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 2, "signal": "WAIT"}],
        )
        self.assertEqual(result["position_strategy_trade_count"], 1)
        self.assertEqual(result["position_strategy_exit_reasons"]["END_OF_TEST"], 1)
        trade = result["position_strategy_exit_price_sample"][0]
        self.assertEqual(trade["exit_timestamp"], 3)
        self.assertAlmostEqual(trade["exit_price"], 100.25)
        self.assertTrue(np.isfinite(trade["net_pnl_usdt"]))
        self.assertFalse(result["real_money_trading_eligible"])

    def test_position_strategy_does_not_enter_on_sell_or_wait(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ])
        predictions = [
            {"timestamp": 1, "signal": "WAIT"},
            {"timestamp": 2, "signal": "SELL"},
        ]
        result = walk_forward_backtest.simulate_position_strategy(usable, predictions)
        self.assertEqual(result["position_strategy_trade_count"], 0)
        self.assertEqual(result["position_strategy_closed_trade_pnl_usdt"], 0.0)

    def test_multiple_unseen_period_evaluation_is_disjoint_and_never_live_eligible(self):
        report = walk_forward_backtest.evaluate_multiple_unseen_periods(make_candles(600), period_count=3)
        self.assertTrue(report["multi_period_evaluation_available"])
        self.assertEqual(report["period_count_evaluated"], 3)
        self.assertTrue(report["periods_are_chronologically_ordered_and_non_overlapping"])
        self.assertFalse(report["real_money_trading_eligible"])
        self.assertIn("directional_accuracy_beats_majority_baseline_period_count", report)
        self.assertIn("buy_strategy_beats_one_candle_baseline_after_costs_period_count", report)
        self.assertIn("positive_buy_only_fixed_stake_pnl_period_count", report)
        self.assertIn("tuned_threshold_beats_fixed_threshold_after_costs_period_count", report)
        self.assertEqual(
            report["multi_period_research_gate_passed"],
            report["positive_research_gate_period_count"] == 3,
        )
        periods = report["periods"]
        for period in periods:
            self.assertIn("directional_accuracy_edge_over_majority_baseline_percentage_points", period)
            self.assertIn("buy_return_edge_over_all_candles_percentage_points", period)
            self.assertIn("buy_strategy_beats_one_candle_baseline_after_costs", period)
            self.assertIn("tuned_threshold_beats_fixed_threshold_after_costs", period)
        self.assertLess(periods[0]["last_candle_timestamp"], periods[1]["first_candle_timestamp"])
        self.assertLess(periods[1]["last_candle_timestamp"], periods[2]["first_candle_timestamp"])
        self.assertTrue(all(not period["real_money_trading_eligible"] for period in periods))

    def test_multiple_period_evaluation_reports_insufficient_data_safely(self):
        report = walk_forward_backtest.evaluate_multiple_unseen_periods(make_candles(220), period_count=3)
        self.assertFalse(report["multi_period_evaluation_available"])
        self.assertFalse(report["real_money_trading_eligible"])

    def test_rejects_too_little_data(self):
        with self.assertRaises(ValueError):
            walk_forward_backtest.walk_forward_evaluate(make_candles(80))


    def test_position_strategy_reports_gross_pnl_and_cost_breakdown(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 2, "signal": "SELL"}],
        )
        trade = result["position_strategy_trades_sample"][0]
        self.assertAlmostEqual(trade["gross_price_pnl_usdt"], 0.0, places=10)
        self.assertGreater(trade["estimated_total_costs_usdt"], 0.0)
        self.assertAlmostEqual(
            trade["pnl_usdt"],
            trade["gross_price_pnl_usdt"] - trade["estimated_total_costs_usdt"],
            places=10,
        )
        self.assertAlmostEqual(
            result["position_strategy_closed_trade_pnl_usdt"],
            result["position_strategy_gross_price_pnl_usdt"]
            - result["position_strategy_estimated_total_costs_usdt"],
            places=10,
        )
        self.assertFalse(result["real_money_trading_eligible"])

    def test_position_strategy_stop_loss_gap_fills_at_candle_open(self):
        import pandas as pd

        usable = pd.DataFrame([
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            # Stop trigger is 98, but the next candle opens at 95: assume the
            # stop executes at 95 rather than pretending a 98 fill was available.
            {"timestamp": 2, "open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5},
        ])
        result = walk_forward_backtest.simulate_position_strategy(
            usable,
            [{"timestamp": 1, "signal": "BUY"}, {"timestamp": 2, "signal": "WAIT"}],
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
        )
        trade = result["position_strategy_trades_sample"][0]
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertAlmostEqual(trade["exit_price"], 95.0)
        self.assertEqual(result["position_strategy_stop_gap_candle_count"], 1)
        self.assertFalse(result["real_money_trading_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
