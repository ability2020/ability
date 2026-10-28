"""Three independent historical-window validation for the bullish-regime entry filter.

Research-only: fetches public Bitget Spot candles, submits no orders, and never
changes paper-trading or live-trading eligibility.
"""
import json
import os
from datetime import datetime, timezone

from learning_model import INTERVAL, SYMBOL, fetch_candles
from walk_forward_backtest import (
    build_candle_window_validation,
    walk_forward_evaluate,
)

LOOKBACK_DAYS = 30
OFFSETS_DAYS = (180, 240, 300)
MIN_TRADES = 10


def main():
    period_results = []
    run_started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for offset in OFFSETS_DAYS:
        print(f"Validating {INTERVAL} historical window at {offset}-day offset.")
        candles = fetch_candles(
            lookback_days=LOOKBACK_DAYS,
            end_offset_days=offset,
        )
        validation = build_candle_window_validation(
            candles,
            LOOKBACK_DAYS,
            offset,
            now_ms=run_started_ms,
            interval_name=INTERVAL,
        )
        if not validation["offset_validation_passed"]:
            raise RuntimeError(
                f"Historical window validation failed for offset {offset}: {validation}"
            )

        report = walk_forward_evaluate(candles)
        comparison = report["bullish_regime_position_strategy_comparison"]
        baseline = comparison["baseline_unfiltered"]
        filtered = comparison["bullish_regime_filtered"]
        enough_trades = (
            baseline["trade_count"] >= MIN_TRADES
            and filtered["trade_count"] >= MIN_TRADES
        )
        baseline_positive = baseline["net_pnl_usdt_after_estimated_costs"] > 0
        filtered_positive = filtered["net_pnl_usdt_after_estimated_costs"] > 0
        period_results.append({
            "end_offset_days": offset,
            "historical_window_validation": validation,
            "prediction_count": report["walk_forward_prediction_count"],
            "directional_accuracy": report["directional_accuracy"],
            "majority_baseline_accuracy": report["walk_forward_majority_baseline_accuracy"],
            "baseline_unfiltered": baseline,
            "bullish_regime_filtered": filtered,
            "buy_signals_blocked_by_filter": comparison["buy_signals_blocked_by_filter"],
            "filtered_minus_baseline_net_pnl_usdt": comparison["net_pnl_difference_filtered_minus_baseline_usdt"],
            "minimum_trade_count_each_strategy": MIN_TRADES,
            "enough_trades_for_both_strategies": enough_trades,
            "baseline_positive_after_costs": baseline_positive,
            "filtered_positive_after_costs": filtered_positive,
            "period_gate_passed": bool(
                validation["offset_validation_passed"]
                and enough_trades
                and baseline_positive
                and filtered_positive
            ),
        })
        print(
            f"Offset {offset}: baseline trades={baseline['trade_count']}, "
            f"net={baseline['net_pnl_usdt_after_estimated_costs']:.6f}; "
            f"filtered trades={filtered['trade_count']}, "
            f"net={filtered['net_pnl_usdt_after_estimated_costs']:.6f}; "
            f"period gate={period_results[-1]['period_gate_passed']}"
        )

    passed = sum(item["period_gate_passed"] for item in period_results)
    report = {
        "report_type": "three-independent-window-bullish-regime-position-validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "lookback_days_per_window": LOOKBACK_DAYS,
        "end_offsets_days": list(OFFSETS_DAYS),
        "stake_usdt": float(os.getenv("BACKTEST_STAKE_USDT", "10")),
        "fee_rate_each_side": float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
        "slippage_rate_each_side": float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005")),
        "identical_strategy_rules_across_windows": True,
        "exit_rules": "Long-only Spot; 2% stop loss, 3% take profit, SELL-signal exit; stop-loss assumed first if both OHLC levels touched.",
        "filter_definition": "Allow BUY only when current-candle sma_gap_20 > 0; preserve SELL/WAIT signals.",
        "period_count_requested": len(OFFSETS_DAYS),
        "period_count_validated": sum(p["historical_window_validation"]["offset_validation_passed"] for p in period_results),
        "period_count_passing": passed,
        "all_three_periods_passed": bool(len(period_results) == len(OFFSETS_DAYS) and passed == len(OFFSETS_DAYS)),
        "real_money_trading_eligible": False,
        "orders_submitted": False,
        "method_and_limitations": (
            "Each 30-day window is fetched and boundary-validated independently at 180-, 240-, "
            "and 300-day offsets. Baseline and filtered strategies use identical chronological "
            "walk-forward predictions, OHLC exit rules, stake, fee and slippage assumptions within "
            "each window. A period passes only when both strategies have at least 10 closed trades "
            "and positive net PnL after estimated costs. Passing is a research screen, not proof of "
            "future profitability; results remain paper/research only."
        ),
        "periods": period_results,
    }
    with open("multi_window_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("Three-window validation report written.")
    print(f"Periods passing: {passed}/{len(OFFSETS_DAYS)}")
    print("Real-money trading eligible: False")
    print("No exchange orders were submitted.")


if __name__ == "__main__":
    main()
