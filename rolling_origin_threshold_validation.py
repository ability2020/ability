"""Repeated rolling-origin threshold validation; public candles only, no orders.

Each origin selects from predeclared probability thresholds on the immediately
older 30-day window, then evaluates the selected threshold on a separate,
later 30-day window. Windows are validated and never overlap within a pair.
This is a next-candle diagnostic, not a position-level execution simulation.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

from learning_model import (
    FEATURE_COLUMNS,
    INTERVAL,
    SYMBOL,
    build_feature_frame,
    fetch_candles,
)
from entry_quality_diagnostics import build_predictions
from walk_forward_backtest import build_candle_window_validation

LOOKBACK_DAYS = 30
TEST_END_OFFSETS_DAYS = (300, 270, 240, 210)
THRESHOLDS = (0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70)
FIXED_THRESHOLD = 0.62
MIN_ENTRIES = 10
MIN_PREDICTIONS = 100
STAKE = float(os.getenv("BACKTEST_STAKE_USDT", "10"))


def threshold_summary(predictions, threshold):
    selected = [row for row in predictions if float(row["probability_up"]) >= threshold]
    returns = [float(row["net_return_fraction"]) for row in selected]
    pnl = float(sum(returns) * STAKE)
    return {
        "threshold": float(threshold),
        "entry_count": len(selected),
        "net_pnl_usdt_after_estimated_costs": pnl,
        "average_net_return_percent_per_entry": 100.0 * float(np.mean(returns)) if returns else None,
        "minimum_entries_met": len(selected) >= MIN_ENTRIES,
        "positive_net_pnl": pnl > 0,
    }


def load_window(offset_days, started_ms):
    candles = fetch_candles(lookback_days=LOOKBACK_DAYS, end_offset_days=offset_days)
    validation = build_candle_window_validation(
        candles, LOOKBACK_DAYS, offset_days, now_ms=started_ms, interval_name=INTERVAL
    )
    if not validation["offset_validation_passed"]:
        raise RuntimeError(
            f"Historical window validation failed at offset {offset_days}: {validation}"
        )
    usable = build_feature_frame(candles).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return validation, usable, build_predictions(usable)


def main():
    started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    periods = []

    for test_offset in TEST_END_OFFSETS_DAYS:
        calibration_offset = test_offset + LOOKBACK_DAYS
        cal_validation, cal_candles, cal_predictions = load_window(calibration_offset, started_ms)
        test_validation, test_candles, test_predictions = load_window(test_offset, started_ms)

        candidates = [threshold_summary(cal_predictions, threshold) for threshold in THRESHOLDS]
        eligible = [row for row in candidates if row["minimum_entries_met"]]
        selected = max(
            eligible,
            key=lambda row: (
                row["net_pnl_usdt_after_estimated_costs"],
                row["entry_count"],
                -row["threshold"],
            ),
        ) if eligible else None

        fixed = threshold_summary(test_predictions, FIXED_THRESHOLD)
        selected_test = (
            threshold_summary(test_predictions, selected["threshold"])
            if selected is not None else None
        )
        enough_predictions = len(test_predictions) >= MIN_PREDICTIONS
        enough_entries = bool(selected_test and selected_test["entry_count"] >= MIN_ENTRIES)
        beats_fixed = bool(
            selected_test
            and selected_test["net_pnl_usdt_after_estimated_costs"]
            > fixed["net_pnl_usdt_after_estimated_costs"]
        )
        passed = bool(
            enough_predictions and enough_entries
            and selected_test["net_pnl_usdt_after_estimated_costs"] > 0
            and beats_fixed
        )
        periods.append({
            "test_end_offset_days": test_offset,
            "calibration_end_offset_days": calibration_offset,
            "calibration_window": {
                "historical_window_validation": cal_validation,
                "usable_candle_count": len(cal_candles),
                "prediction_count": len(cal_predictions),
                "threshold_candidates": candidates,
                "selected_threshold": selected["threshold"] if selected else None,
                "selection_rule": "Among predeclared thresholds with at least 10 entries, choose highest estimated-cost net PnL; tie-break by more entries, then lower threshold.",
            },
            "untouched_test_window": {
                "historical_window_validation": test_validation,
                "usable_candle_count": len(test_candles),
                "prediction_count": len(test_predictions),
                "fixed_threshold_0_62": fixed,
                "selected_threshold_result": selected_test,
                "minimum_prediction_count_met": enough_predictions,
                "minimum_entry_count_met": enough_entries,
                "selected_threshold_beats_fixed_0_62": beats_fixed,
                "period_gate_passed": passed,
            },
        })

    evaluable = [
        row for row in periods
        if row["untouched_test_window"]["minimum_prediction_count_met"]
        and row["untouched_test_window"]["minimum_entry_count_met"]
    ]
    passing = sum(row["untouched_test_window"]["period_gate_passed"] for row in evaluable)
    all_periods_evaluable = len(evaluable) == len(periods)
    all_evaluable_positive = bool(evaluable) and all(
        row["untouched_test_window"]["selected_threshold_result"]["net_pnl_usdt_after_estimated_costs"] > 0
        for row in evaluable
    )
    all_evaluable_beat_fixed = bool(evaluable) and all(
        row["untouched_test_window"]["selected_threshold_beats_fixed_0_62"]
        for row in evaluable
    )
    report = {
        "report_type": "repeated-rolling-origin-threshold-validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "lookback_days_per_window": LOOKBACK_DAYS,
        "test_end_offsets_days": list(TEST_END_OFFSETS_DAYS),
        "periods": periods,
        "aggregate": {
            "period_count": len(periods),
            "evaluable_period_count": len(evaluable),
            "passing_period_count": passing,
            "all_periods_evaluable": all_periods_evaluable,
            "all_evaluable_periods_positive_net_pnl": all_evaluable_positive,
            "selected_threshold_beats_fixed_0_62_in_all_evaluable_periods": all_evaluable_beat_fixed,
            "research_gate_passed": bool(
                all_periods_evaluable and passing == len(periods)
                and all_evaluable_positive and all_evaluable_beat_fixed
            ),
            "note": "Insufficient samples make the overall gate fail closed; an inconclusive timeframe is not treated as a pass.",
        },
        "cost_assumptions": {
            "fee_rate_each_side": float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
            "slippage_rate_each_side": float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005")),
            "stake_usdt_per_entry": STAKE,
            "note": "Estimated-cost next-candle returns, not position-level trading, actual fills, spread/depth, partial fills, or tax.",
        },
        "methodology": {
            "thresholds_predeclared": list(THRESHOLDS),
            "fixed_comparator_threshold": FIXED_THRESHOLD,
            "threshold_selection_uses_each_test_window": False,
            "calibration_and_test_windows_overlap_within_pair": False,
            "minimum_predictions_per_test_window": MIN_PREDICTIONS,
            "minimum_entries_per_selected_threshold": MIN_ENTRIES,
            "selection_is_repeated_for_each_origin": True,
            "orders_submitted": False,
            "real_money_trading_eligible": False,
        },
        "orders_submitted": False,
        "real_money_trading_eligible": False,
    }
    with open("rolling_origin_threshold_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("REPEATED ROLLING-ORIGIN VALIDATION — research only; no orders.")
    for row in periods:
        result = row["untouched_test_window"]
        selected = result["selected_threshold_result"]
        print(
            f"test offset {row['test_end_offset_days']}d | predictions {result['prediction_count']} "
            f"| selected threshold {selected['threshold'] if selected else None} "
            f"| net PnL {round(selected['net_pnl_usdt_after_estimated_costs'], 6) if selected else None} USDT "
            f"| period gate {result['period_gate_passed']}"
        )
    print("Research gate passed:", report["aggregate"]["research_gate_passed"])
    print("Report written: rolling_origin_threshold_report.json")
    print("No exchange orders submitted; real-money trading eligible: False.")


if __name__ == "__main__":
    main()
