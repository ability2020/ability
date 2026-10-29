"""Validate predeclared entry filters on three independent historical windows.

Research-only: public Bitget candles, chronological predictions, no account access
or exchange orders. Results are exploratory and never enable live trading.
"""
import json
import os
from datetime import datetime, timezone

from learning_model import INTERVAL, SYMBOL, FEATURE_COLUMNS, MIN_SAMPLES, build_feature_frame, fetch_candles
from entry_quality_diagnostics import build_predictions
from walk_forward_backtest import build_candle_window_validation, simulate_position_strategy

LOOKBACK_DAYS = 30
OFFSETS_DAYS = (180, 240, 300)
MIN_TRADES = 10
STAKE = float(os.getenv("BACKTEST_STAKE_USDT", "10"))

def main():
    started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_windows = []
    candidate_names = (
        "baseline_unfiltered",
        "probability_at_least_0_62",
        "price_above_sma20",
        "rsi_45_to_65",
        "probability_0_62_and_above_sma20",
    )
    for offset in OFFSETS_DAYS:
        print(f"Entry-filter validation: {INTERVAL}, historical offset {offset} days.")
        candles = fetch_candles(lookback_days=LOOKBACK_DAYS, end_offset_days=offset)
        validation = build_candle_window_validation(
            candles, LOOKBACK_DAYS, offset, now_ms=started_ms, interval_name=INTERVAL
        )
        if not validation["offset_validation_passed"]:
            raise RuntimeError(f"Historical window validation failed at offset {offset}: {validation}")
        usable = build_feature_frame(candles).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        predictions = build_predictions(usable)
        filters = {
            "baseline_unfiltered": lambda row: True,
            "probability_at_least_0_62": lambda row: row["probability_up"] >= 0.62,
            "price_above_sma20": lambda row: row["sma_gap_20"] is not None and row["sma_gap_20"] > 0,
            "rsi_45_to_65": lambda row: row["rsi_14"] is not None and 45 <= row["rsi_14"] <= 65,
            "probability_0_62_and_above_sma20": lambda row: row["probability_up"] >= 0.62 and row["sma_gap_20"] is not None and row["sma_gap_20"] > 0,
        }
        comparisons = {}
        for name, predicate in filters.items():
            candidate = []
            blocked = 0
            for pred in predictions:
                row = dict(pred)
                if row["signal"] == "BUY" and not predicate(row):
                    row["signal"] = "WAIT"
                    blocked += 1
                candidate.append(row)
            sim = simulate_position_strategy(usable, candidate)
            trades = int(sim["position_strategy_trade_count"])
            pnl = float(sim["position_strategy_closed_trade_pnl_usdt"])
            comparisons[name] = {
                "closed_trade_count": trades,
                "net_pnl_usdt_after_estimated_costs": pnl,
                "gross_price_pnl_usdt": float(sim["position_strategy_gross_price_pnl_usdt"]),
                "estimated_total_costs_usdt": float(sim["position_strategy_estimated_total_costs_usdt"]),
                "maximum_drawdown_usdt": float(sim["position_strategy_maximum_drawdown_usdt"]),
                "buy_signals_blocked": blocked,
                "minimum_10_trades_met": trades >= MIN_TRADES,
                "positive_net_pnl": pnl > 0,
            }
        baseline_pnl = comparisons["baseline_unfiltered"]["net_pnl_usdt_after_estimated_costs"]
        for result in comparisons.values():
            result["pnl_difference_vs_baseline_usdt"] = result["net_pnl_usdt_after_estimated_costs"] - baseline_pnl
        all_windows.append({
            "end_offset_days": offset,
            "historical_window_validation": validation,
            "usable_candle_count": len(usable),
            "prediction_count": len(predictions),
            "prediction_sample_sufficient": len(predictions) >= 20,
            "candidate_comparisons": comparisons,
        })
        print(f"Offset {offset}: predictions={len(predictions)}; baseline trades={comparisons['baseline_unfiltered']['closed_trade_count']}; baseline net={baseline_pnl:.6f} USDT.")
    candidate_gates = {}
    for name in candidate_names[1:]:
        per_window = []
        for window in all_windows:
            result = window["candidate_comparisons"][name]
            baseline = window["candidate_comparisons"]["baseline_unfiltered"]
            per_window.append({
                "end_offset_days": window["end_offset_days"],
                "sample_sufficient": window["prediction_sample_sufficient"],
                "minimum_trades_met": result["minimum_10_trades_met"],
                "positive_net_pnl": result["positive_net_pnl"],
                "beats_baseline": result["net_pnl_usdt_after_estimated_costs"] > baseline["net_pnl_usdt_after_estimated_costs"],
                "net_pnl_usdt": result["net_pnl_usdt_after_estimated_costs"],
                "baseline_net_pnl_usdt": baseline["net_pnl_usdt_after_estimated_costs"],
            })
        passed = all(item["sample_sufficient"] and item["minimum_trades_met"] and item["positive_net_pnl"] and item["beats_baseline"] for item in per_window)
        candidate_gates[name] = {"windows": per_window, "passed_all_windows": passed}
    report = {
        "report_type": "multi-window-entry-filter-validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "lookback_days_per_window": LOOKBACK_DAYS,
        "end_offsets_days": list(OFFSETS_DAYS),
        "stake_usdt": STAKE,
        "fee_rate_each_side": float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
        "slippage_rate_each_side": float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005")),
        "minimum_predictions_for_interpretation": 20,
        "minimum_closed_trades_per_candidate_per_window": MIN_TRADES,
        "all_historical_windows_validated": all(w["historical_window_validation"]["offset_validation_passed"] for w in all_windows),
        "candidate_gate_rule": "A candidate passes only if all three windows have at least 20 predictions, at least 10 closed trades, positive net PnL after estimated costs, and net PnL strictly above the unfiltered baseline.",
        "candidate_gates": candidate_gates,
        "periods": all_windows,
        "real_money_trading_eligible": False,
        "orders_submitted": False,
        "limitations": "Predeclared filters are compared on three separate historical windows, but this remains exploratory research. Shared model design and limited sample sizes can still bias findings; a pass would require additional untouched validation before any strategy change.",
    }
    with open("multi_window_entry_filter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("Multi-window entry-filter report written.")
    for name, result in candidate_gates.items():
        print(name, "| passed all windows:", result["passed_all_windows"])
    print("Real-money trading eligible: False; no exchange orders submitted.")

if __name__ == "__main__":
    main()
