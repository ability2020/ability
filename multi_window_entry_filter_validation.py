"""Validate predeclared entry filters across independent historical windows.

Research-only: public Bitget Spot candles, chronological predictions, no orders.
A candidate is not recommended unless it is positive after estimated costs and
has at least 10 closed trades in every window; small samples remain inconclusive.
"""
import json
import os
from datetime import datetime, timezone

from learning_model import INTERVAL, SYMBOL, build_feature_frame, fetch_candles
from entry_quality_diagnostics import build_predictions
from walk_forward_backtest import build_candle_window_validation, simulate_position_strategy

LOOKBACK_DAYS = 30
OFFSETS_DAYS = (180, 240, 300)
MIN_TRADES = 10
MIN_PREDICTIONS = 20


def main():
    windows = []
    candidates = {
        "baseline_unfiltered": lambda r: True,
        "probability_at_least_0_62": lambda r: r["probability_up"] >= 0.62,
        "price_above_sma20": lambda r: r["sma_gap_20"] is not None and r["sma_gap_20"] > 0,
        "rsi_45_to_65": lambda r: r["rsi_14"] is not None and 45 <= r["rsi_14"] <= 65,
        "probability_0_62_and_above_sma20": lambda r: r["probability_up"] >= 0.62 and r["sma_gap_20"] is not None and r["sma_gap_20"] > 0,
    }
    started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for offset in OFFSETS_DAYS:
        print(f"Validating entry filters: interval={INTERVAL}, offset={offset} days")
        candles = fetch_candles(lookback_days=LOOKBACK_DAYS, end_offset_days=offset)
        validation = build_candle_window_validation(
            candles, LOOKBACK_DAYS, offset, now_ms=started_ms, interval_name=INTERVAL
        )
        if not validation["offset_validation_passed"]:
            raise RuntimeError(f"Historical window validation failed at offset {offset}: {validation}")

        frame = build_feature_frame(candles)
        usable = frame.dropna(subset=__import__("learning_model").FEATURE_COLUMNS).reset_index(drop=True)
        predictions = build_predictions(usable)
        comparisons = {}
        for name, predicate in candidates.items():
            altered = []
            blocked = 0
            for prediction in predictions:
                item = dict(prediction)
                if item["signal"] == "BUY" and not predicate(item):
                    item["signal"] = "WAIT"
                    blocked += 1
                altered.append(item)
            sim = simulate_position_strategy(usable, altered)
            trades = int(sim["position_strategy_trade_count"])
            pnl = float(sim["position_strategy_closed_trade_pnl_usdt"])
            comparisons[name] = {
                "closed_trade_count": trades,
                "net_pnl_usdt_after_estimated_costs": pnl,
                "buy_signals_blocked": blocked,
                "minimum_10_trades_met": trades >= MIN_TRADES,
                "positive_net_pnl": pnl > 0,
                "interpretable": len(predictions) >= MIN_PREDICTIONS and trades >= MIN_TRADES,
            }
        windows.append({
            "end_offset_days": offset,
            "historical_window_validation": validation,
            "prediction_count": len(predictions),
            "prediction_sample_sufficient": len(predictions) >= MIN_PREDICTIONS,
            "interpretation_status": "interpretable" if len(predictions) >= MIN_PREDICTIONS else "inconclusive_insufficient_predictions",
            "candidate_comparisons": comparisons,
        })

    summary = {}
    for name in candidates:
        per_window = [w["candidate_comparisons"][name] for w in windows]
        all_sufficient = all(w["prediction_sample_sufficient"] and x["minimum_10_trades_met"] for w, x in zip(windows, per_window))
        all_positive = all(x["net_pnl_usdt_after_estimated_costs"] > 0 for x in per_window)
        summary[name] = {
            "window_count": len(per_window),
            "positive_windows": sum(x["net_pnl_usdt_after_estimated_costs"] > 0 for x in per_window),
            "net_pnl_by_window_usdt": [x["net_pnl_usdt_after_estimated_costs"] for x in per_window],
            "trades_by_window": [x["closed_trade_count"] for x in per_window],
            "all_windows_have_sufficient_sample": all_sufficient,
            "positive_after_costs_in_every_window": all_positive,
            "robust_candidate_gate_passed": bool(all_sufficient and all_positive),
            "status": "candidate_for_further_untouched_testing" if all_sufficient and all_positive else "inconclusive_or_gate_failed",
        }

    report = {
        "report_type": "multi-window-entry-filter-validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "lookback_days_per_window": LOOKBACK_DAYS,
        "end_offsets_days": list(OFFSETS_DAYS),
        "candidate_filters_predeclared": True,
        "candidate_filters_tuned_on_these_windows": False,
        "minimum_predictions_per_window": MIN_PREDICTIONS,
        "minimum_closed_trades_per_candidate_per_window": MIN_TRADES,
        "windows": windows,
        "candidate_summary": summary,
        "real_money_trading_eligible": False,
        "orders_submitted": False,
        "methodology": "Same predeclared filters, chronological walk-forward prediction method, Spot position simulation, stake and estimated costs are applied independently to three historical windows. A candidate passes only if every window has at least 20 predictions, at least 10 closed trades, and positive net PnL after estimated costs. Passing is not proof of future profitability.",
    }
    with open("multi_window_entry_filter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("Multi-window entry filter report written.")
    for name, result in summary.items():
        print(name, "| positive windows:", result["positive_windows"], "/ 3", "| robust gate:", result["robust_candidate_gate_passed"])
    print("Real-money trading eligible: False; no exchange orders submitted.")


if __name__ == "__main__":
    main()
