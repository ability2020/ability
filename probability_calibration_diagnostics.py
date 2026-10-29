"""Probability calibration and threshold-selection diagnostic.

The threshold is selected using an earlier 30-day window only, then evaluated
once on a separate, later, non-overlapping 30-day historical window. This is
research-only: public candles, no account access, no paper-state mutation,
and no exchange orders. Results are not a deployment approval.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

from learning_model import (
    FEATURE_COLUMNS,
    INTERVAL,
    MIN_SAMPLES,
    SYMBOL,
    build_feature_frame,
    fetch_candles,
)
from entry_quality_diagnostics import build_predictions
from walk_forward_backtest import build_candle_window_validation

LOOKBACK_DAYS = 30
# Older calibration window ends 330 days ago; untouched test window ends
# 300 days ago. The windows are adjacent, not overlapping, in calendar time.
CALIBRATION_OFFSET_DAYS = 330
TEST_OFFSET_DAYS = 300
THRESHOLDS = (0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70)
MIN_ENTRIES = 10
STAKE = float(os.getenv("BACKTEST_STAKE_USDT", "10"))


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def calibration_summary(predictions):
    if not predictions:
        return {
            "prediction_count": 0,
            "brier_score": None,
            "log_loss": None,
            "mean_predicted_probability_percent": None,
            "observed_up_rate_percent": None,
            "calibration_bins": [],
            "sample_sufficient": False,
        }

    probabilities = np.asarray([float(row["probability_up"]) for row in predictions])
    actual = np.asarray([1.0 if row["actual_next_close_up"] else 0.0 for row in predictions])
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    bins = []
    edges = (0.0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0000001)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        bins.append({
            "probability_range": f"{low:.2f}_to_{high:.2f}",
            "count": count,
            "mean_predicted_probability_percent": 100.0 * float(probabilities[mask].mean()) if count else None,
            "observed_up_rate_percent": 100.0 * float(actual[mask].mean()) if count else None,
        })
    return {
        "prediction_count": len(predictions),
        "brier_score": float(np.mean((probabilities - actual) ** 2)),
        "log_loss": float(-np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))),
        "mean_predicted_probability_percent": 100.0 * float(probabilities.mean()),
        "observed_up_rate_percent": 100.0 * float(actual.mean()),
        "calibration_bins": bins,
        "sample_sufficient": len(predictions) >= 100,
        "interpretation_note": "Lower Brier score and log loss are better; bin-level rates are noisy and are descriptive, not proof of calibrated probabilities.",
    }


def threshold_summary(predictions, threshold):
    selected = [row for row in predictions if float(row["probability_up"]) >= threshold]
    returns = [float(row["net_return_fraction"]) for row in selected]
    pnl = float(sum(returns) * STAKE)
    return {
        "threshold": float(threshold),
        "entry_count": len(selected),
        "net_pnl_usdt_after_estimated_costs": pnl,
        "average_net_return_percent_per_entry": 100.0 * float(np.mean(returns)) if returns else None,
        "profitable_entry_rate_percent": 100.0 * sum(value > 0 for value in returns) / len(returns) if returns else None,
        "minimum_entries_met": len(selected) >= MIN_ENTRIES,
        "positive_net_pnl": pnl > 0,
        "entries": [
            {
                "timestamp": int(row["timestamp"]),
                "probability_up": float(row["probability_up"]),
                "actual_next_close_up": bool(row["actual_next_close_up"]),
                "net_return_fraction_after_estimated_costs": float(row["net_return_fraction"]),
            }
            for row in selected
        ],
    }


def load_window(offset_days, started_ms):
    candles = fetch_candles(lookback_days=LOOKBACK_DAYS, end_offset_days=offset_days)
    validation = build_candle_window_validation(
        candles, LOOKBACK_DAYS, offset_days, now_ms=started_ms, interval_name=INTERVAL
    )
    if not validation["offset_validation_passed"]:
        raise RuntimeError(f"Historical window validation failed at offset {offset_days}: {validation}")
    usable = build_feature_frame(candles).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    predictions = build_predictions(usable)
    return validation, usable, predictions


def main():
    started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"Probability calibration diagnostic: {SYMBOL} {INTERVAL}; research-only.")
    calibration_validation, calibration_candles, calibration_predictions = load_window(
        CALIBRATION_OFFSET_DAYS, started_ms
    )
    test_validation, test_candles, test_predictions = load_window(TEST_OFFSET_DAYS, started_ms)

    calibration_candidates = [threshold_summary(calibration_predictions, threshold) for threshold in THRESHOLDS]
    eligible = [item for item in calibration_candidates if item["minimum_entries_met"]]
    # Select using calibration-window PnL only. Test-window outcomes are not
    # read by this selection step.
    selected = max(
        eligible,
        key=lambda item: (item["net_pnl_usdt_after_estimated_costs"], item["entry_count"], -item["threshold"]),
    ) if eligible else None

    test_results = {
        "fixed_threshold_0_62": threshold_summary(test_predictions, 0.62),
    }
    if selected is not None:
        test_results["threshold_selected_on_calibration_window"] = threshold_summary(
            test_predictions, selected["threshold"]
        )

    selected_test = test_results.get("threshold_selected_on_calibration_window")
    fixed_test = test_results["fixed_threshold_0_62"]
    test_beats_fixed = (
        selected_test is not None
        and selected_test["entry_count"] >= MIN_ENTRIES
        and selected_test["net_pnl_usdt_after_estimated_costs"]
        > fixed_test["net_pnl_usdt_after_estimated_costs"]
    )
    report = {
        "report_type": "probability-calibration-and-untouched-threshold-test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "lookback_days_per_window": LOOKBACK_DAYS,
        "calibration_window": {
            "end_offset_days": CALIBRATION_OFFSET_DAYS,
            "historical_window_validation": calibration_validation,
            "usable_candle_count": len(calibration_candles),
            "prediction_count": len(calibration_predictions),
            "probability_calibration": calibration_summary(calibration_predictions),
            "threshold_candidates": calibration_candidates,
            "selected_threshold": selected["threshold"] if selected else None,
            "selection_rule": "Among predeclared thresholds with at least 10 entries, choose the highest after-estimated-cost net PnL in the calibration window; tie-break by more entries, then lower threshold.",
        },
        "untouched_test_window": {
            "end_offset_days": TEST_OFFSET_DAYS,
            "historical_window_validation": test_validation,
            "usable_candle_count": len(test_candles),
            "prediction_count": len(test_predictions),
            "probability_calibration": calibration_summary(test_predictions),
            "threshold_results": test_results,
            "selected_threshold_has_minimum_entries": bool(selected_test and selected_test["entry_count"] >= MIN_ENTRIES),
            "selected_threshold_beats_fixed_0_62_on_test": bool(test_beats_fixed),
        },
        "cost_assumptions": {
            "fee_rate_each_side": float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
            "slippage_rate_each_side": float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005")),
            "stake_usdt_per_entry": STAKE,
            "note": "Uses the existing estimated-cost next-candle return diagnostic, not simulated position-level trading or actual fills.",
        },
        "interpretation": {
            "threshold_selection_uses_test_window": False,
            "windows_overlap": False,
            "minimum_entries_for_threshold_interpretation": MIN_ENTRIES,
            "minimum_predictions_for_calibration_interpretation": 100,
            "research_gate_passed": bool(
                selected_test
                and selected_test["entry_count"] >= MIN_ENTRIES
                and selected_test["net_pnl_usdt_after_estimated_costs"] > 0
                and test_beats_fixed
                and calibration_summary(test_predictions)["sample_sufficient"]
            ),
            "note": "A single untouched window is still limited evidence. The next step would be a repeated rolling-origin test, and no threshold should be activated in the bot from this report alone.",
        },
        "orders_submitted": False,
        "real_money_trading_eligible": False,
    }
    with open("probability_calibration_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("Calibration predictions:", len(calibration_predictions))
    print("Untouched-test predictions:", len(test_predictions))
    print("Selected calibration threshold:", report["calibration_window"]["selected_threshold"])
    for name, result in test_results.items():
        print(name, "| entries:", result["entry_count"], "| net PnL:", round(result["net_pnl_usdt_after_estimated_costs"], 6), "USDT")
    print("Research gate passed:", report["interpretation"]["research_gate_passed"])
    print("No exchange orders submitted; real-money trading eligible: False.")


if __name__ == "__main__":
    main()
