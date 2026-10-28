"""Entry-quality diagnostics for the experimental Spot model.

Uses public historical candles and chronological walk-forward predictions only.
No account access, paper-state mutation, or exchange orders. Candidate filters are
predeclared diagnostics, not automatic strategy changes or deployment approvals.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learning_model import FEATURE_COLUMNS, INTERVAL, MIN_SAMPLES, SYMBOL, build_feature_frame, fetch_candles
from walk_forward_backtest import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    _net_one_candle_return,
    build_candle_window_validation,
    simulate_position_strategy,
)

STEP = int(os.getenv("WALK_FORWARD_STEP", "5"))
MIN_TRADES = 10
STAKE = float(os.getenv("BACKTEST_STAKE_USDT", "10"))


def safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def summarize_rows(rows):
    returns = [float(row["net_return_fraction"]) for row in rows]
    return {
        "buy_entry_count": len(rows),
        "profitable_next_candle_count": sum(value > 0 for value in returns),
        "loss_making_next_candle_count": sum(value <= 0 for value in returns),
        "win_rate_percent": 100.0 * sum(value > 0 for value in returns) / len(returns) if returns else None,
        "average_net_return_percent": 100.0 * float(np.mean(returns)) if returns else None,
        "fixed_stake_net_pnl_usdt": float(sum(returns) * STAKE),
        "minimum_trades_met": len(rows) >= MIN_TRADES,
    }


def build_predictions(usable):
    predictions = []
    for i in range(MIN_SAMPLES, len(usable) - 1, STEP):
        train = usable.iloc[:i].dropna(subset=["target"])
        current = usable.iloc[[i]]
        if len(train) < MIN_SAMPLES or train["target"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
        row = current.iloc[0]
        probability = float(model.predict_proba(current[FEATURE_COLUMNS])[:, 1][0])
        signal = "BUY" if probability >= BUY_THRESHOLD else ("SELL" if probability <= SELL_THRESHOLD else "WAIT")
        next_row = usable.iloc[i + 1]
        predictions.append({
            "timestamp": int(row["timestamp"]),
            "close": float(row["close"]),
            "probability_up": probability,
            "signal": signal,
            "actual_next_close_up": bool(float(next_row["close"]) > float(row["close"])),
            "next_close_return_fraction": float(next_row["close"]) / float(row["close"]) - 1.0,
            "net_return_fraction": _net_one_candle_return(float(row["close"]), float(next_row["close"])),
            "sma_gap_20": safe_float(row.get("sma_gap_20")),
            "rsi_14": safe_float(row.get("rsi_14")),
            "volatility_10": safe_float(row.get("volatility_10")),
            "return_1": safe_float(row.get("return_1")),
            "macd_hist_pct": safe_float(row.get("macd_hist_pct")),
            "range_pct": safe_float(row.get("range_pct")),
        })
    # Sparse higher-timeframe windows can legitimately produce fewer predictions.
    # Keep the workflow green and report the sample as inconclusive instead of failing.
    return predictions


def bucket_stats(buys, key_fn, labels):
    output = {}
    for label in labels:
        output[label] = summarize_rows([row for row in buys if key_fn(row) == label])
    return output


def main():
    lookback = int(os.getenv("CANDLE_LOOKBACK_DAYS", "30"))
    offset = int(os.getenv("CANDLE_END_OFFSET_DAYS", "300"))
    candles = fetch_candles(lookback_days=lookback, end_offset_days=offset)
    window_validation = build_candle_window_validation(candles, lookback, offset, interval_name=INTERVAL)
    if not window_validation["offset_validation_passed"]:
        raise RuntimeError(f"Historical window validation failed; refusing to interpret entry diagnostics: {window_validation}")

    frame = build_feature_frame(candles)
    usable = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    predictions = build_predictions(usable)
    buys = [row for row in predictions if row["signal"] == "BUY"]

    # Feature buckets are descriptive diagnostics of BUY entries only.
    probability_labels = ("0.58_to_0.62", "0.62_to_0.70", "0.70_and_above")
    def probability_bucket(row):
        p = row["probability_up"]
        return "0.58_to_0.62" if p < 0.62 else ("0.62_to_0.70" if p < 0.70 else "0.70_and_above")

    rsi_labels = ("below_40", "40_to_below_50", "50_to_60", "above_60")
    def rsi_bucket(row):
        value = row["rsi_14"]
        if value is None: return "missing"
        return "below_40" if value < 40 else ("40_to_below_50" if value < 50 else ("50_to_60" if value <= 60 else "above_60"))

    def trend_bucket(row):
        if row["sma_gap_20"] is None: return "missing"
        return "above_sma20" if row["sma_gap_20"] > 0 else "at_or_below_sma20"

    volatility_values = [row["volatility_10"] for row in buys if row["volatility_10"] is not None]
    volatility_median = float(np.median(volatility_values)) if volatility_values else None
    def volatility_bucket(row):
        if row["volatility_10"] is None or volatility_median is None: return "missing"
        return "higher_than_or_equal_median" if row["volatility_10"] >= volatility_median else "below_median"

    feature_analysis = {
        "probability_strength": bucket_stats(buys, probability_bucket, probability_labels),
        "trend_regime": bucket_stats(buys, trend_bucket, ("above_sma20", "at_or_below_sma20")),
        "rsi_regime": bucket_stats(buys, rsi_bucket, rsi_labels + ("missing",)),
        "volatility_regime": {
            "buy_entry_volatility_median": volatility_median,
            "buckets": bucket_stats(buys, volatility_bucket, ("below_median", "higher_than_or_equal_median", "missing")),
        },
    }

    candidate_definitions = {
        "baseline_unfiltered": lambda row: True,
        "probability_at_least_0_62": lambda row: row["probability_up"] >= 0.62,
        "price_above_sma20": lambda row: row["sma_gap_20"] is not None and row["sma_gap_20"] > 0,
        "rsi_45_to_65": lambda row: row["rsi_14"] is not None and 45 <= row["rsi_14"] <= 65,
        "probability_0_62_and_above_sma20": lambda row: row["probability_up"] >= 0.62 and row["sma_gap_20"] is not None and row["sma_gap_20"] > 0,
    }
    strategy_comparisons = {}
    for name, predicate in candidate_definitions.items():
        candidate_predictions = []
        blocked = 0
        for prediction in predictions:
            item = dict(prediction)
            if item["signal"] == "BUY" and not predicate(item):
                item["signal"] = "WAIT"
                blocked += 1
            candidate_predictions.append(item)
        sim = simulate_position_strategy(usable, candidate_predictions)
        count = int(sim["position_strategy_trade_count"])
        pnl = float(sim["position_strategy_closed_trade_pnl_usdt"])
        strategy_comparisons[name] = {
            "closed_trade_count": count,
            "net_pnl_usdt_after_estimated_costs": pnl,
            "gross_price_pnl_usdt": float(sim["position_strategy_gross_price_pnl_usdt"]),
            "estimated_total_costs_usdt": float(sim["position_strategy_estimated_total_costs_usdt"]),
            "maximum_drawdown_usdt": float(sim["position_strategy_maximum_drawdown_usdt"]),
            "stop_loss_trade_count": int(sim["position_strategy_stop_loss_diagnostics"]["stop_loss_trade_count"]),
            "buy_signals_blocked": blocked,
            "minimum_10_trades_met": count >= MIN_TRADES,
            "positive_net_pnl": pnl > 0,
            "interpretable_under_minimum_trade_rule": count >= MIN_TRADES,
        }

    baseline_pnl = strategy_comparisons["baseline_unfiltered"]["net_pnl_usdt_after_estimated_costs"]
    for result in strategy_comparisons.values():
        result["pnl_difference_vs_baseline_usdt"] = result["net_pnl_usdt_after_estimated_costs"] - baseline_pnl

    report = {
        "report_type": "experimental-entry-quality-diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "historical_window_validation": window_validation,
        "prediction_count": len(predictions),
        "minimum_predictions_for_interpretation": 20,
        "prediction_sample_sufficient": len(predictions) >= 20,
        "interpretation_status": "interpretable" if len(predictions) >= 20 else "inconclusive_insufficient_predictions",
        "buy_signal_count": len(buys),
        "buy_entry_next_candle_quality": summarize_rows(buys),
        "feature_bucket_analysis_of_buy_entries": feature_analysis,
        "position_strategy_candidate_comparisons": strategy_comparisons,
        "strategy_rules": {
            "stop_loss_percent": 2.0,
            "take_profit_percent": 3.0,
            "sell_signal_exit": True,
            "fee_rate_each_side": float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
            "slippage_rate_each_side": float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005")),
            "stake_usdt": STAKE,
            "minimum_closed_trades_for_interpretation": MIN_TRADES,
        },
        "methodology": {
            "prediction_method": "Chronological walk-forward LogisticRegression; each prediction uses only earlier labeled rows and scores the immediately following candle.",
            "filter_candidates_predeclared": True,
            "candidate_filters_tuned_on_results": False,
            "candidate_comparison_sample_is_independent": False,
            "minimum_predictions_for_interpretation": 20,
            "note": "When fewer than 20 chronological predictions are available, the script still writes a report but marks it inconclusive; do not interpret small-sample results. These candidate comparisons share one historical window and are exploratory. A candidate with fewer than 10 closed trades is inconclusive; even a positive result must be confirmed on new untouched windows before any strategy change.",
        },
        "orders_submitted": False,
        "real_money_trading_eligible": False,
    }
    with open("entry_quality_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("ENTRY QUALITY DIAGNOSTICS — public candles only; no orders.")
    print("Historical window validation passed:", window_validation["offset_validation_passed"])
    print("BUY entries:", len(buys), "| next-candle net PnL:", round(report["buy_entry_next_candle_quality"]["fixed_stake_net_pnl_usdt"], 6))
    for name, result in strategy_comparisons.items():
        print(name, "| trades:", result["closed_trade_count"], "| net PnL:", round(result["net_pnl_usdt_after_estimated_costs"], 6), "| >=10 trades:", result["minimum_10_trades_met"])
    print("Report written: entry_quality_report.json")
    print("Real-money trading eligible: False; no exchange orders submitted.")


if __name__ == "__main__":
    main()
