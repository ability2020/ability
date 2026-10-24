import tempfile
import unittest
from pathlib import Path

import paper_trader


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = str(Path(self.temp_dir.name) / "paper_state.json")

    def test_buy_signal_opens_hypothetical_position(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0
        )
        self.assertEqual(event, "PAPER_BUY")
        self.assertAlmostEqual(state["position"]["entry_price"], 100.0)
        self.assertAlmostEqual(state["position"]["quantity"], 0.1)

    def test_wait_does_not_open_position(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "WAIT", 100.0
        )
        self.assertIsNone(state["position"])
        self.assertEqual(event, "HOLD_CASH")

    def test_buy_is_blocked_when_risk_decision_is_wait(self):
        self.assertEqual(paper_trader.risk_gated_paper_signal("BUY", "WAIT"), "WAIT")
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(),
            paper_trader.risk_gated_paper_signal("BUY", "WAIT"),
            100.0,
            10.0,
        )
        self.assertEqual(event, "HOLD_CASH")
        self.assertIsNone(state["position"])

    def test_buy_is_allowed_only_for_risk_candidate(self):
        self.assertEqual(
            paper_trader.risk_gated_paper_signal("BUY", "BUY_CANDIDATE"),
            "BUY",
        )

    def test_sell_can_close_hypothetical_position_without_real_btc_candidate(self):
        # Real-account balance checks must not prevent a paper exit from an
        # already-open simulated position.
        self.assertEqual(paper_trader.risk_gated_paper_signal("SELL", "WAIT"), "SELL")

    def test_stop_loss_closes_position_and_records_loss(self):
        state, _, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0, 2.0, 3.0
        )
        state, event, reason = paper_trader.process_cycle(
            state, "WAIT", 98.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_SELL")
        self.assertIn("STOP_LOSS", reason)
        self.assertIsNone(state["position"])
        self.assertAlmostEqual(state["realized_pnl_usdt"], -0.2)

    def test_take_profit_closes_position_and_records_profit(self):
        state, _, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0, 2.0, 3.0
        )
        state, event, reason = paper_trader.process_cycle(
            state, "WAIT", 103.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_SELL")
        self.assertIn("TAKE_PROFIT", reason)
        self.assertIsNone(state["position"])
        self.assertAlmostEqual(state["realized_pnl_usdt"], 0.3)

    def test_sell_signal_closes_only_an_existing_position(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "SELL", 100.0
        )
        self.assertEqual(event, "HOLD_CASH")
        self.assertIsNone(state["position"])
        state, _, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0
        )
        state, event, _ = paper_trader.process_cycle(state, "SELL", 101.0)
        self.assertEqual(event, "PAPER_SELL")
        self.assertIsNone(state["position"])
        self.assertAlmostEqual(state["realized_pnl_usdt"], 0.1)

    def test_existing_position_does_not_double_buy(self):
        state, _, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0
        )
        state, event, _ = paper_trader.process_cycle(state, "BUY", 99.0, 10.0)
        self.assertEqual(event, "HOLD_POSITION")
        self.assertAlmostEqual(state["position"]["entry_price"], 100.0)

    def test_complete_paper_cycle_records_one_closed_trade(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_BUY")
        self.assertIsNotNone(state["position"])

        state, event, _ = paper_trader.process_cycle(
            state, "WAIT", 102.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "HOLD_POSITION")
        self.assertIsNotNone(state["position"])

        state, event, reason = paper_trader.process_cycle(
            state, "SELL", 101.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_SELL")
        self.assertIn("SELL_SIGNAL", reason)
        self.assertIsNone(state["position"])
        self.assertEqual(len(state["closed_trades"]), 1)
        self.assertAlmostEqual(state["closed_trades"][0]["pnl_usdt"], 0.1)
        self.assertAlmostEqual(state["realized_pnl_usdt"], 0.1)

    def test_repeated_sell_without_position_never_opens_short(self):
        state = paper_trader.default_state()
        for _ in range(2):
            state, event, _ = paper_trader.process_cycle(
                state, "SELL", 100.0, 10.0, 2.0, 3.0
            )
            self.assertEqual(event, "HOLD_CASH")
            self.assertIsNone(state["position"])
        self.assertEqual(state["closed_trades"], [])
        self.assertEqual(state["realized_pnl_usdt"], 0.0)

    def test_state_persists_and_corrupt_state_fails_closed(self):
        state, _, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0
        )
        paper_trader.save_state(state, self.path)
        self.assertEqual(paper_trader.load_state(self.path), state)
        Path(self.path).write_text("{broken", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            paper_trader.load_state(self.path)

    def test_restored_position_triggers_stop_loss_after_reload(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_BUY")
        paper_trader.save_state(state, self.path)

        # A new workflow cycle reloads the persisted position from disk/cache.
        restored = paper_trader.load_state(self.path)
        self.assertIsNotNone(restored["position"])
        self.assertAlmostEqual(restored["position"]["entry_price"], 100.0)

        updated, event, reason = paper_trader.process_cycle(
            restored, "WAIT", 98.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_SELL")
        self.assertIn("STOP_LOSS", reason)
        self.assertIsNone(updated["position"])
        self.assertEqual(len(updated["closed_trades"]), 1)
        self.assertAlmostEqual(updated["realized_pnl_usdt"], -0.2)

    def test_restored_position_triggers_take_profit_after_reload(self):
        state, event, _ = paper_trader.process_cycle(
            paper_trader.default_state(), "BUY", 100.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_BUY")
        paper_trader.save_state(state, self.path)

        restored = paper_trader.load_state(self.path)
        updated, event, reason = paper_trader.process_cycle(
            restored, "WAIT", 103.0, 10.0, 2.0, 3.0
        )
        self.assertEqual(event, "PAPER_SELL")
        self.assertIn("TAKE_PROFIT", reason)
        self.assertIsNone(updated["position"])
        self.assertEqual(len(updated["closed_trades"]), 1)
        self.assertAlmostEqual(updated["realized_pnl_usdt"], 0.3)

    def test_invalid_market_data_is_rejected(self):
        for price in (0, -1, float("nan"), float("inf")):
            with self.subTest(price=price):
                with self.assertRaises(ValueError):
                    paper_trader.process_cycle(
                        paper_trader.default_state(), "BUY", price
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
