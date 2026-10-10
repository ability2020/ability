import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class PositionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = str(Path(self.temp_dir.name) / "bot_state.json")

    def test_missing_state_returns_safe_default(self):
        self.assertEqual(
            main.load_position_state(self.path),
            {"version": 1, "position": None, "last_order_ids": []},
        )

    def test_valid_state_round_trips(self):
        state = {
            "version": 1,
            "position": {"entry_price": 100.0, "quantity": 0.01},
            "last_order_ids": ["demo_order_1"],
        }
        main.save_position_state(state, self.path)
        self.assertEqual(main.load_position_state(self.path), state)

    def test_malformed_json_fails_closed(self):
        Path(self.path).write_text("{not-json", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            main.load_position_state(self.path)

    def test_unsupported_version_fails_closed(self):
        Path(self.path).write_text(json.dumps({"version": 2}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            main.load_position_state(self.path)

    def test_invalid_loaded_position_fails_closed(self):
        invalid_states = [
            {"version": 1, "position": {"entry_price": 0, "quantity": 1}},
            {"version": 1, "position": {"entry_price": 100, "quantity": -1}},
            {"version": 1, "position": {"entry_price": math.nan, "quantity": 1}},
            {"version": 1, "position": {"entry_price": 100}},
        ]
        for state in invalid_states:
            with self.subTest(state=state):
                Path(self.path).write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    main.load_position_state(self.path)

    def test_invalid_save_does_not_replace_existing_state(self):
        original = {
            "version": 1,
            "position": {"entry_price": 100.0, "quantity": 0.01},
            "last_order_ids": ["keep_me"],
        }
        main.save_position_state(original, self.path)
        bad = {
            "version": 1,
            "position": {"entry_price": 100.0, "quantity": math.inf},
            "last_order_ids": [],
        }
        with self.assertRaises(ValueError):
            main.save_position_state(bad, self.path)
        self.assertEqual(main.load_position_state(self.path), original)

    def test_order_ids_are_bounded_to_last_100(self):
        state = {
            "version": 1,
            "position": None,
            "last_order_ids": [f"id_{i}" for i in range(120)],
        }
        main.save_position_state(state, self.path)
        loaded = main.load_position_state(self.path)
        self.assertEqual(len(loaded["last_order_ids"]), 100)
        self.assertEqual(loaded["last_order_ids"][0], "id_20")

    def test_invalid_order_id_collection_fails_closed(self):
        Path(self.path).write_text(
            json.dumps({"version": 1, "position": None, "last_order_ids": "not-a-list"}),
            encoding="utf-8",
        )
        with self.assertRaises(RuntimeError):
            main.load_position_state(self.path)


class RiskSafetyTests(unittest.TestCase):
    def test_risk_engine_blocks_when_demo_mode_is_off(self):
        with patch.object(main, "DEMO_TRADING", False):
            decision, _ = main.evaluate_risk(
                "BUY", 100.0, {"indicator_score": 2}, 20.0, 0.0
            )
        self.assertEqual(decision, "WAIT")

    def test_risk_engine_blocks_invalid_balances_and_price(self):
        cases = [
            (float("nan"), 20.0, 0.0),
            (100.0, float("inf"), 0.0),
            (100.0, 20.0, -1.0),
            (0.0, 20.0, 0.0),
        ]
        for price, usdt, btc in cases:
            with self.subTest(price=price, usdt=usdt, btc=btc):
                decision, _ = main.evaluate_risk(
                    "BUY", price, {"indicator_score": 2}, usdt, btc
                )
                self.assertEqual(decision, "WAIT")

    def test_exit_checks_are_decision_only(self):
        self.assertEqual(main.evaluate_position_exit(100, 97)[0], "STOP_LOSS_CANDIDATE")
        self.assertEqual(main.evaluate_position_exit(100, 104)[0], "TAKE_PROFIT_CANDIDATE")
        self.assertEqual(main.evaluate_position_exit(100, 101)[0], "HOLD")
        self.assertEqual(main.evaluate_position_exit(100, float("nan"))[0], "WAIT")

    def test_order_id_is_deterministic_for_same_bucket(self):
        self.assertEqual(
            main.make_client_order_id("buy", time_bucket=123),
            main.make_client_order_id("buy", time_bucket=123),
        )
        self.assertNotEqual(
            main.make_client_order_id("buy", time_bucket=123),
            main.make_client_order_id("sell", time_bucket=123),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
