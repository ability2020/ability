"""Chronological walk-forward evaluation of the experimental model.

Research only: public candles, no account access, and no exchange orders.
Returns are reported as fixed-stake PnL and average per-trade return. Compounded
returns are illustrative only because stake size and idle cash are not simulated.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learning_model import (
    FEATURE_COLUMNS,
    INTERVAL,
    MIN_SAMPLES,
    SYMBOL,
    build_feature_frame,
    fetch_candles,
)

BUY_THRESHOLD = float(os.getenv("ML_BUY_THRESHOLD", "0.58"))
SELL_THRESHOLD = float(os.getenv("ML_SELL_THRESHOLD", "0.42"))
STAKE_USDT = float(os.getenv("BACKTEST_STAKE_USDT", "10"))
FEE_RATE = float(os.getenv("BACKTEST_FEE_RATE", "0.001"))  # 0.10% per side
SLIPPAGE_RATE = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))  # 0.05% per side


def _validate_settings():
    if not (0.5 < BUY_THRESHOLD <= 1.0):
        raise ValueError("ML_BUY_THRESHOLD must be greater than 0.5 and at most 1.")
    if not (0.0 <= SELL_THRESHOLD < 0.5):
        raise ValueError("ML_SELL_THRESHOLD must be at least 0 and less than 0.5.")
    if STAKE_USDT <= 0 or not np.isfinite(STAKE_USDT):
        raise ValueError("BACKTEST_STAKE_USDT must be a positive finite number.")
    if not (0 <= FEE_RATE < 0.1) or not np.isfinite(FEE_RATE):
        raise ValueError("BACKTEST_FEE_RATE must be finite and in [0, 0.1).")
    if not (0 <= SLIPPAGE_RATE < 0.1) or not np.isfinite(SLIPPAGE_RATE):
        raise ValueError("BACKTEST_SLIPPAGE_RATE must be finite and in [0, 0.1).")


def _net_one_candle_return(entry_close, next_close):
    """Return net fractional PnL for a fixed stake, including both-side costs."""
    entry_execution = entry_close * (1.0 + SLIPPAGE_RATE)
    exit_execution = next_close * (1.0 - SLIPPAGE_RATE)
    quantity = STAKE_USDT / entry_execution
    entry_fee = STAKE_USDT * FEE_RATE
    exit_proceeds_after_fee = quantity * exit_execution * (1.0 - FEE_RATE)
    return (exit_proceeds_after_fee - STAKE_USDT - entry_fee) / STAKE_USDT


def walk_forward_evaluate(candles):
    _validate_settings()
    frame = build_feature_frame(candles)
    usable = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(usable) < MIN_SAMPLES + 2:
        raise ValueError(
            f"Need at least {MIN_SAMPLES + 2} candles with features for walk-forward "
            f"evaluation; got {len(usable)}."
        )

    predictions = []
    trade_returns = []
    baseline_returns = []
    directional_actual = []
    directional_predictions = []
    buy_count = 0
    sell_count = 0
    wait_count = 0

    # At candle i close, labels for rows before i are known because they depend
    # only on closes through candle i. Candle i itself is never in training.
    for i in range(MIN_SAMPLES, len(usable) - 1):
        train = usable.iloc[:i].dropna(subset=["target"])
        current = usable.iloc[[i]]
        if len(train) < MIN_SAMPLES or train["target"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
        probability_up = float(model.predict_proba(current[FEATURE_COLUMNS])[:, 1][0])
        actual_up = int(float(current.iloc[0]["close"]) < float(usable.iloc[i + 1]["close"]))
        predicted_up = int(probability_up >= 0.5)
        directional_actual.append(actual_up)
        directional_predictions.append(predicted_up)

        if probability_up >= BUY_THRESHOLD:
            signal = "BUY"
            buy_count += 1
            trade_returns.append(_net_one_candle_return(
                float(usable.iloc[i]["close"]), float(usable.iloc[i + 1]["close"])
            ))
        elif probability_up <= SELL_THRESHOLD:
            signal = "SELL"
            sell_count += 1
        else:
            signal = "WAIT"
            wait_count += 1

        # Simple always-buy-for-one-candle benchmark, same dates and cost model.
        baseline_returns.append(_net_one_candle_return(
            float(usable.iloc[i]["close"]), float(usable.iloc[i + 1]["close"])
        ))
        predictions.append({
            "timestamp": int(current.iloc[0]["timestamp"]),
            "close": float(current.iloc[0]["close"]),
            "probability_up": probability_up,
            "signal": signal,
            "actual_next_close_up": bool(actual_up),
        })

    if len(predictions) < 20:
        raise ValueError("Walk-forward test produced fewer than 20 predictions.")

    # Fixed-stake PnL makes the assumed exposure explicit. Compounded returns
    # below are illustrative and are not a portfolio-level simulation.
    fixed_stake_pnl_usdt = float(sum(trade_returns) * STAKE_USDT)
    average_buy_return = float(np.mean(trade_returns)) if trade_returns else None
    average_market_return = float(np.mean(baseline_returns))
    net_total_return = float(np.prod([1.0 + r for r in trade_returns]) - 1.0) if trade_returns else 0.0
    baseline_total_return = float(np.prod([1.0 + r for r in baseline_returns]) - 1.0)
    return {
        "report_type": "experimental-walk-forward-model-evaluation",
        "model_version": "logistic-direction-v1",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candle_count_with_features": int(len(usable)),
        "walk_forward_prediction_count": int(len(predictions)),
        "first_test_timestamp": predictions[0]["timestamp"],
        "last_test_timestamp": predictions[-1]["timestamp"],
        "directional_accuracy": float(accuracy_score(directional_actual, directional_predictions)),
        "directional_balanced_accuracy": float(balanced_accuracy_score(directional_actual, directional_predictions)),
        "buy_signal_count": buy_count,
        "sell_signal_count": sell_count,
        "wait_signal_count": wait_count,
        "trade_test_method": "Each BUY prediction is a hypothetical one-candle Spot long, entered at that candle close and exited at the next close. Each trade uses a fixed stake; actual portfolio allocation, idle cash, and order execution are not simulated.",
        "stake_per_hypothetical_trade_usdt": STAKE_USDT,
        "fee_rate_each_side": FEE_RATE,
        "slippage_rate_each_side": SLIPPAGE_RATE,
        "estimated_round_trip_cost_percent_approx": float((2 * FEE_RATE + 2 * SLIPPAGE_RATE) * 100),
        "buy_only_trade_count": int(len(trade_returns)),
        "buy_only_fixed_stake_total_pnl_usdt": fixed_stake_pnl_usdt,
        "buy_only_return_on_total_deployed_stake_percent": average_buy_return * 100 if average_buy_return is not None else None,
        "buy_only_average_net_return_percent_per_trade": average_buy_return * 100 if average_buy_return is not None else None,
        "all_test_candles_average_net_return_percent": average_market_return * 100,
        "buy_signals_average_return_beats_all_candles_after_costs": bool(average_buy_return > average_market_return) if average_buy_return is not None else False,
        "buy_only_compounded_return_percent_illustrative_not_portfolio_simulation": net_total_return * 100,
        # Backward-compatible names used by existing offline safety tests and reports.
        # These compounded figures remain illustrative, not portfolio-level results.
        "buy_only_net_total_return_percent_compounded": net_total_return * 100,
        "always_buy_same_period_count": int(len(baseline_returns)),
        "always_buy_compounded_return_percent_illustrative": baseline_total_return * 100,
        "always_buy_net_total_return_percent_compounded": baseline_total_return * 100,
        "model_buy_strategy_beats_always_buy_after_costs": bool(average_buy_return > average_market_return) if average_buy_return is not None else False,
        "caveats": [
            "Research-only estimate; not proof of future profitability.",
            "Fixed-stake PnL assumes the stated stake is deployed for every BUY signal; it does not model a real account balance or position sizing.",
            "Compounded returns are illustrative only and are not a portfolio-level backtest; they should not be treated as expected account returns.",
            "Order-book depth, spread variation, partial fills, idle cash, and taxes are not modelled.",
            "Fee and slippage assumptions may differ from actual Bitget account costs.",
            "SELL predictions are reported but this Spot test does not short-sell.",
            "No exchange account is accessed and no orders are submitted.",
        ],
        "sample_predictions": predictions[-10:],
    }


def main():
    result = walk_forward_evaluate(fetch_candles())
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open("walk_forward_report.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("WALK-FORWARD EVALUATION — public data only; no orders.")
    print("Predictions:", result["walk_forward_prediction_count"])
    print("Directional accuracy:", round(result["directional_accuracy"], 4))
    print("Balanced accuracy:", round(result["directional_balanced_accuracy"], 4))
    print("BUY / SELL / WAIT:", result["buy_signal_count"], "/", result["sell_signal_count"], "/", result["wait_signal_count"])
    print("BUY-only fixed-stake total PnL (USDT):", round(result["buy_only_fixed_stake_total_pnl_usdt"], 4))
    print("BUY-only average net return per trade (%):", result["buy_only_average_net_return_percent_per_trade"])
    print("All-candle average net return (%):", round(result["all_test_candles_average_net_return_percent"], 4))
    print("Compounded figures are illustrative only, not portfolio returns.")
    print("Report written: walk_forward_report.json")
    print("This research report does not control paper trades or place orders.")


if __name__ == "__main__":
    main()
