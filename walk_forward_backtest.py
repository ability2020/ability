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

# Runtime controls: sparse, chronological evaluation preserves the next-candle
# target while avoiding an expensive model fit on every overlapping candle.
WALK_FORWARD_STEP = int(os.getenv("WALK_FORWARD_STEP", "5"))
THRESHOLD_REFRESH_PREDICTIONS = int(os.getenv("THRESHOLD_REFRESH_PREDICTIONS", "25"))


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
    if WALK_FORWARD_STEP < 1:
        raise ValueError("WALK_FORWARD_STEP must be a positive integer.")
    if THRESHOLD_REFRESH_PREDICTIONS < 1:
        raise ValueError("THRESHOLD_REFRESH_PREDICTIONS must be a positive integer.")


def _net_one_candle_return(entry_close, next_close):
    """Return net fractional PnL for a fixed stake, including both-side costs."""
    entry_execution = entry_close * (1.0 + SLIPPAGE_RATE)
    exit_execution = next_close * (1.0 - SLIPPAGE_RATE)
    quantity = STAKE_USDT / entry_execution
    entry_fee = STAKE_USDT * FEE_RATE
    exit_proceeds_after_fee = quantity * exit_execution * (1.0 - FEE_RATE)
    return (exit_proceeds_after_fee - STAKE_USDT - entry_fee) / STAKE_USDT


def choose_cost_aware_threshold(candidate_returns, baseline_average_return, fallback_threshold, minimum_trades=5):
    """Choose only a validation threshold with positive net returns that beat the same-window baseline."""
    best_threshold = float(fallback_threshold)
    best_average_return = None
    for threshold, returns in candidate_returns.items():
        if len(returns) < minimum_trades:
            continue
        average_return = float(np.mean(returns))
        if average_return <= 0.0 or average_return <= float(baseline_average_return):
            continue
        if best_average_return is None or average_return > best_average_return:
            best_average_return = average_return
            best_threshold = float(threshold)
    return best_threshold


def summarize_fixed_stake_drawdown(net_returns, stake_usdt=STAKE_USDT):
    """Summarize peak-to-trough drawdown of sequential fixed-stake trade PnL."""
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    maximum_drawdown = 0.0
    for net_return in net_returns:
        cumulative_pnl += float(net_return) * float(stake_usdt)
        peak_pnl = max(peak_pnl, cumulative_pnl)
        maximum_drawdown = max(maximum_drawdown, peak_pnl - cumulative_pnl)
    return {
        "maximum_drawdown_usdt": float(maximum_drawdown),
        "maximum_drawdown_percent_of_one_stake": (
            float(maximum_drawdown / float(stake_usdt) * 100.0) if float(stake_usdt) > 0 else None
        ),
        "drawdown_method": "Sequential fixed-stake trade PnL peak-to-trough; excludes idle cash, overlapping positions, and portfolio sizing.",
    }


def _select_buy_threshold(train, usable):
    """Select a threshold using only an inner chronological validation slice of training data."""
    candidates = (0.52, 0.55, 0.58, 0.62, 0.65, 0.70)
    split = int(len(train) * 0.8)
    fit = train.iloc[:split]
    validation = train.iloc[split:]
    # Too little validation data to compare thresholds meaningfully: keep the
    # configured threshold rather than tuning on a tiny sample.
    if len(fit) < MIN_SAMPLES or len(validation) < 20 or fit["target"].nunique() < 2:
        return BUY_THRESHOLD

    validation_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    validation_model.fit(fit[FEATURE_COLUMNS], fit["target"].astype(int))
    probabilities = validation_model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    validation_returns = []
    for position in range(len(validation)):
        row_index = int(validation.index[position])
        if row_index + 1 >= len(usable):
            continue
        validation_returns.append(_net_one_candle_return(
            float(usable.iloc[row_index]["close"]),
            float(usable.iloc[row_index + 1]["close"]),
        ))
    if not validation_returns:
        return BUY_THRESHOLD
    baseline_average_return = float(np.mean(validation_returns))

    candidate_returns = {}
    minimum_trades = 5
    for threshold in candidates:
        selected_positions = np.flatnonzero(probabilities >= threshold)
        returns = []
        for position in selected_positions:
            row_index = int(validation.index[position])
            if row_index + 1 >= len(usable):
                continue
            returns.append(_net_one_candle_return(
                float(usable.iloc[row_index]["close"]),
                float(usable.iloc[row_index + 1]["close"]),
            ))
        candidate_returns[threshold] = returns

    return choose_cost_aware_threshold(
        candidate_returns, baseline_average_return, BUY_THRESHOLD, minimum_trades
    )


def evaluate_research_gate(prediction_count, buy_count, average_buy_return, average_market_return, fixed_stake_pnl):
    """Report-only gate; never authorizes exchange orders or live trading."""
    checks = {
        "enough_walk_forward_predictions": int(prediction_count) >= 500,
        "enough_buy_signals": int(buy_count) >= 30,
        "positive_average_net_buy_return": average_buy_return is not None and float(average_buy_return) > 0.0,
        "buy_returns_beat_all_candles_after_costs": (
            average_buy_return is not None
            and average_market_return is not None
            and float(average_buy_return) > float(average_market_return)
        ),
        "positive_fixed_stake_pnl": float(fixed_stake_pnl) > 0.0,
    }
    return {
        "single_period_research_gate_passed": bool(all(checks.values())),
        "single_period_research_gate_checks": checks,
        "real_money_trading_eligible": False,
        "gate_note": (
            "This report-only gate is not connected to order placement. "
            "Real-money use remains disabled and would require positive after-cost "
            "results across multiple distinct unseen periods, realistic execution "
            "assumptions, and independent review."
        ),
    }




def _position_trade_with_cost_breakdown(position, exit_price, exit_execution, exit_reason, exit_timestamp):
    """Build one simulated trade and reconcile price PnL with estimated costs."""
    quantity = position["quantity"]
    entry_price = position["entry_price"]
    entry_execution = position["entry_execution"]
    entry_fee = position["entry_fee"]
    exit_fee = quantity * exit_execution * FEE_RATE
    entry_slippage = quantity * (entry_execution - entry_price)
    exit_slippage = quantity * (exit_price - exit_execution)
    gross_price_pnl = quantity * (exit_price - entry_price)
    net_pnl = gross_price_pnl - entry_slippage - exit_slippage - entry_fee - exit_fee
    return {
        "entry_timestamp": position["entry_timestamp"],
        "exit_timestamp": int(exit_timestamp),
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_price_pnl_usdt": float(gross_price_pnl),
        "estimated_entry_fee_usdt": float(entry_fee),
        "estimated_exit_fee_usdt": float(exit_fee),
        "estimated_entry_slippage_usdt": float(entry_slippage),
        "estimated_exit_slippage_usdt": float(exit_slippage),
        "estimated_total_costs_usdt": float(entry_fee + exit_fee + entry_slippage + exit_slippage),
        "pnl_usdt": float(net_pnl),
        "net_return_percent": float(net_pnl / STAKE_USDT * 100.0),
        "entry_probability_up": position.get("entry_probability_up"),
        "holding_candle_count": int(position.get("holding_candle_count", 0)),
        "max_adverse_excursion_percent": (
            float((position["minimum_holding_low"] / entry_price - 1.0) * 100.0)
            if position.get("minimum_holding_low") is not None else None
        ),
        "max_favorable_excursion_percent": (
            float((position["maximum_holding_high"] / entry_price - 1.0) * 100.0)
            if position.get("maximum_holding_high") is not None else None
        ),
        "entry_candle_range_percent": position.get("entry_candle_range_percent"),
        "average_holding_candle_range_percent": (
            float(position["holding_candle_range_sum_percent"] / position["holding_candle_count"])
            if position.get("holding_candle_count", 0) else None
        ),
        "stop_distance_in_average_holding_ranges": (
            float((position["stop_loss_percent"] / (
                position["holding_candle_range_sum_percent"] / position["holding_candle_count"]
            )))
            if position.get("holding_candle_count", 0)
            and position.get("holding_candle_range_sum_percent", 0.0) > 0
            and exit_reason == "STOP_LOSS" else None
        ),
        "stop_gap_exit": bool(
            exit_reason == "STOP_LOSS"
            and position.get("stop_price") is not None
            and float(exit_price) < float(position["stop_price"])
        ),
        "stop_gap_beyond_level_percent": (
            float((float(exit_price) / float(position["stop_price"]) - 1.0) * 100.0)
            if exit_reason == "STOP_LOSS" and position.get("stop_price") is not None else None
        ),
    }

def summarize_exit_diagnostics(trades):
    """Research-only aggregation by exit reason and cost assumptions."""
    by_reason = {}
    for reason in ("STOP_LOSS", "TAKE_PROFIT", "SELL_SIGNAL", "END_OF_TEST"):
        selected = [t for t in trades if t["exit_reason"] == reason]
        by_reason[reason] = {
            "trade_count": len(selected),
            "gross_price_pnl_usdt": float(sum(t["gross_price_pnl_usdt"] for t in selected)),
            "estimated_total_costs_usdt": float(sum(t["estimated_total_costs_usdt"] for t in selected)),
            "net_pnl_usdt": float(sum(t["pnl_usdt"] for t in selected)),
            "profitable_trade_count": sum(t["pnl_usdt"] > 0 for t in selected),
        }
    scenarios = []
    for fee in (0.0005, 0.001, 0.0015):
        for slip in (0.0002, 0.0005, 0.001):
            gross_total = costs_total = net_total = 0.0
            wins = 0
            for trade in trades:
                entry = float(trade["entry_price"])
                exit_price = float(trade["exit_price"])
                entry_exec = entry * (1 + slip)
                exit_exec = exit_price * (1 - slip)
                qty = STAKE_USDT / entry_exec
                gross = qty * (exit_price - entry)
                costs = STAKE_USDT * fee + qty * exit_exec * fee
                costs += qty * (entry_exec - entry) + qty * (exit_price - exit_exec)
                net = gross - costs
                gross_total += gross
                costs_total += costs
                net_total += net
                wins += net > 0
            scenarios.append({
                "fee_rate_each_side": fee, "slippage_rate_each_side": slip,
                "trade_count": len(trades), "gross_price_pnl_usdt": float(gross_total),
                "estimated_total_costs_usdt": float(costs_total), "net_pnl_usdt": float(net_total),
                "profitable_trade_count": int(wins), "real_money_trading_eligible": False,
            })
    entry_exec = 1.0 * (1 + SLIPPAGE_RATE)
    quantity = STAKE_USDT / entry_exec
    break_even_exit = (STAKE_USDT + STAKE_USDT * FEE_RATE) / (
        quantity * (1 - SLIPPAGE_RATE) * (1 - FEE_RATE)
    )
    return {
        "exit_pnl_by_reason": by_reason,
        "cost_sensitivity": {
            "base_fee_rate_each_side": FEE_RATE,
            "base_slippage_rate_each_side": SLIPPAGE_RATE,
            "break_even_gross_price_move_percent": (break_even_exit - 1.0) * 100.0,
            "scenarios": scenarios,
            "real_money_trading_eligible": False,
        },
    }


def simulate_position_strategy(usable, predictions, stop_loss_percent=2.0, take_profit_percent=3.0):
    """Simulate one long Spot position using every subsequent candle's OHLC.

    Entries use the signal candle close; exits are checked on every later candle,
    even if a prediction is absent for that candle. If stop-loss and take-profit
    are both touched, stop-loss is assumed first. No exchange orders are sent.
    """
    try:
        stop_loss_percent = float(stop_loss_percent)
        take_profit_percent = float(take_profit_percent)
    except (TypeError, ValueError) as error:
        raise ValueError("Stop loss and take profit must be numeric.") from error
    if not (
        np.isfinite(stop_loss_percent)
        and np.isfinite(take_profit_percent)
        and 0 < stop_loss_percent < 100
        and take_profit_percent > 0
    ):
        raise ValueError("Stop loss must be finite and in (0, 100); take profit must be positive and finite.")

    index_by_timestamp = {
        int(row["timestamp"]): idx for idx, (_, row) in enumerate(usable.iterrows())
    }
    signal_by_index = {
        index_by_timestamp[int(pred["timestamp"])]: pred["signal"]
        for pred in predictions
        if int(pred["timestamp"]) in index_by_timestamp
        and pred.get("signal") in ("BUY", "SELL", "WAIT")
    }
    probability_by_index = {
        index_by_timestamp[int(pred["timestamp"])]: float(pred["probability_up"])
        for pred in predictions
        if int(pred["timestamp"]) in index_by_timestamp
        and pred.get("probability_up") is not None
    }
    position = None
    trades = []
    checked_position_candles = 0
    missing_prediction_position_candles = 0
    stop_level_touch_candles = 0
    target_level_touch_candles = 0
    both_levels_touch_candles = 0
    stop_gap_candles = 0

    # Walk every candle, not only candles with a prediction. A missing signal
    # must never skip an OHLC stop/target check for an open position.
    for i in range(len(usable)):
        row = usable.iloc[i]
        signal = signal_by_index.get(i)
        exited_this_candle = False

        if position is not None and i > position["entry_index"]:
            entry = position["entry_price"]
            stop_price = entry * (1.0 - stop_loss_percent / 100.0)
            target_price = entry * (1.0 + take_profit_percent / 100.0)
            open_price = float(row["open"])
            low = float(row["low"])
            high = float(row["high"])
            checked_position_candles += 1
            position["stop_price"] = stop_price
            position["holding_candle_count"] += 1
            position["minimum_holding_low"] = min(position["minimum_holding_low"], low)
            position["maximum_holding_high"] = max(position["maximum_holding_high"], high)
            candle_range_percent = ((high - low) / entry * 100.0) if entry > 0 else 0.0
            position["holding_candle_range_sum_percent"] += candle_range_percent
            if i not in signal_by_index:
                missing_prediction_position_candles += 1
            stop_touched = low <= stop_price
            target_touched = high >= target_price
            if stop_touched:
                stop_level_touch_candles += 1
                if open_price <= stop_price:
                    stop_gap_candles += 1
            if target_touched:
                target_level_touch_candles += 1
            if stop_touched and target_touched:
                both_levels_touch_candles += 1
            exit_price = None
            exit_reason = None
            if stop_touched:
                # If the candle opens through the stop, assume execution at the
                # open (worse than the stop trigger), not at an unavailable price.
                exit_price = min(stop_price, open_price)
                exit_reason = "STOP_LOSS"
            elif target_touched:
                exit_price, exit_reason = target_price, "TAKE_PROFIT"
            elif signal == "SELL":
                exit_price, exit_reason = float(row["close"]), "SELL_SIGNAL"

            if exit_price is not None:
                exit_execution = exit_price * (1.0 - SLIPPAGE_RATE)
                trades.append(_position_trade_with_cost_breakdown(
                    position, exit_price, exit_execution, exit_reason, row["timestamp"]
                ))
                position = None
                exited_this_candle = True

        # Avoid an ambiguous stop/target exit followed by a new entry on the
        # same candle. New entries are considered at candle close only.
        if position is None and signal == "BUY" and not exited_this_candle:
            entry_reference = float(row["close"])
            entry_execution = entry_reference * (1.0 + SLIPPAGE_RATE)
            position = {
                "entry_index": i,
                "entry_timestamp": int(row["timestamp"]),
                "entry_price": entry_reference,
                "entry_execution": entry_execution,
                "quantity": STAKE_USDT / entry_execution,
                "entry_fee": STAKE_USDT * FEE_RATE,
                "entry_probability_up": probability_by_index.get(i),
                "holding_candle_count": 0,
                # Entry occurs at candle close, so do not count that candle's
                # earlier high/low as post-entry excursion.
                "minimum_holding_low": entry_reference,
                "maximum_holding_high": entry_reference,
                "entry_candle_range_percent": (
                    (float(row["high"]) - float(row["low"])) / entry_reference * 100.0
                    if entry_reference > 0 else None
                ),
                "holding_candle_range_sum_percent": 0.0,
                "stop_loss_percent": stop_loss_percent,
                "stop_price": entry_reference * (1.0 - stop_loss_percent / 100.0),
            }

    if position is not None:
        # Mark any remaining position at the final available candle, including
        # candles for which no prediction was produced.
        row = usable.iloc[-1]
        exit_price = float(row["close"])
        exit_execution = exit_price * (1.0 - SLIPPAGE_RATE)
        trades.append(_position_trade_with_cost_breakdown(
            position, exit_price, exit_execution, "END_OF_TEST", row["timestamp"]
        ))

    total_pnl = float(sum(trade["pnl_usdt"] for trade in trades))
    stop_trades = [trade for trade in trades if trade["exit_reason"] == "STOP_LOSS"]

    def _mean_or_none(values):
        return float(np.mean(values)) if values else None

    probability_bands = (
        ("below_0_55", lambda p: p is not None and p < 0.55),
        ("0_55_to_0_58", lambda p: p is not None and 0.55 <= p < 0.58),
        ("0_58_to_0_62", lambda p: p is not None and 0.58 <= p < 0.62),
        ("0_62_and_above", lambda p: p is not None and p >= 0.62),
        ("missing_probability", lambda p: p is None),
    )
    stop_loss_diagnostics = {
        "stop_loss_trade_count": len(stop_trades),
        "stop_gap_exit_count": sum(bool(t["stop_gap_exit"]) for t in stop_trades),
        "non_gap_stop_exit_count": sum(not bool(t["stop_gap_exit"]) for t in stop_trades),
        "mean_holding_candles_before_stop": _mean_or_none([t["holding_candle_count"] for t in stop_trades]),
        "mean_max_adverse_excursion_percent": _mean_or_none([t["max_adverse_excursion_percent"] for t in stop_trades if t["max_adverse_excursion_percent"] is not None]),
        "mean_max_favorable_excursion_percent": _mean_or_none([t["max_favorable_excursion_percent"] for t in stop_trades if t["max_favorable_excursion_percent"] is not None]),
        "mean_entry_candle_range_percent": _mean_or_none([t["entry_candle_range_percent"] for t in stop_trades if t["entry_candle_range_percent"] is not None]),
        "mean_holding_candle_range_percent": _mean_or_none([t["average_holding_candle_range_percent"] for t in stop_trades if t["average_holding_candle_range_percent"] is not None]),
        "mean_stop_distance_in_average_holding_ranges": _mean_or_none([t["stop_distance_in_average_holding_ranges"] for t in stop_trades if t["stop_distance_in_average_holding_ranges"] is not None]),
        "by_entry_probability_band": {
            label: {
                "stop_loss_trade_count": sum(matches(t.get("entry_probability_up")) for t in stop_trades),
                "net_pnl_usdt": float(sum(t["pnl_usdt"] for t in stop_trades if matches(t.get("entry_probability_up")))),
                "mean_holding_candles": _mean_or_none([t["holding_candle_count"] for t in stop_trades if matches(t.get("entry_probability_up"))]),
                "mean_max_adverse_excursion_percent": _mean_or_none([t["max_adverse_excursion_percent"] for t in stop_trades if matches(t.get("entry_probability_up")) and t["max_adverse_excursion_percent"] is not None]),
            }
            for label, matches in probability_bands
        },
        "stop_loss_trade_details": [
            {key: trade.get(key) for key in (
                "entry_timestamp", "exit_timestamp", "entry_price", "exit_price",
                "entry_probability_up", "holding_candle_count",
                "max_adverse_excursion_percent", "max_favorable_excursion_percent",
                "entry_candle_range_percent", "average_holding_candle_range_percent",
                "stop_distance_in_average_holding_ranges", "stop_gap_exit",
                "stop_gap_beyond_level_percent", "gross_price_pnl_usdt",
                "estimated_total_costs_usdt", "pnl_usdt"
            )}
            for trade in stop_trades
        ],
        "interpretation_note": (
            "Descriptive diagnostics on the same chronological walk-forward position simulation. "
            "Entry probability is the model probability recorded at the BUY candle; holding-candle "
            "ranges use high-low divided by entry price. This does not test alternative stop settings "
            "or prove that a wider stop would improve out-of-sample results."
        ),
        "real_money_trading_eligible": False,
    }
    return {
        "real_money_trading_eligible": False,
        "position_strategy_stop_loss_percent": stop_loss_percent,
        "position_strategy_take_profit_percent": take_profit_percent,
        "position_strategy_trade_count": len(trades),
        "position_strategy_maximum_drawdown_usdt": summarize_fixed_stake_drawdown([trade["pnl_usdt"] / STAKE_USDT for trade in trades])["maximum_drawdown_usdt"],
        "position_strategy_closed_trade_pnl_usdt": total_pnl,
        "position_strategy_gross_price_pnl_usdt": float(sum(trade["gross_price_pnl_usdt"] for trade in trades)),
        "position_strategy_estimated_entry_fees_usdt": float(sum(trade["estimated_entry_fee_usdt"] for trade in trades)),
        "position_strategy_estimated_exit_fees_usdt": float(sum(trade["estimated_exit_fee_usdt"] for trade in trades)),
        "position_strategy_estimated_entry_slippage_usdt": float(sum(trade["estimated_entry_slippage_usdt"] for trade in trades)),
        "position_strategy_estimated_exit_slippage_usdt": float(sum(trade["estimated_exit_slippage_usdt"] for trade in trades)),
        "position_strategy_estimated_total_costs_usdt": float(sum(trade["estimated_total_costs_usdt"] for trade in trades)),
        "position_strategy_checked_open_position_candles": checked_position_candles,
        "position_strategy_missing_prediction_candles_checked": missing_prediction_position_candles,
        "position_strategy_stop_level_touch_candle_count": stop_level_touch_candles,
        "position_strategy_target_level_touch_candle_count": target_level_touch_candles,
        "position_strategy_both_levels_touch_candle_count": both_levels_touch_candles,
        "position_strategy_stop_gap_candle_count": stop_gap_candles,
        "position_strategy_exit_price_sample": [
            {
                "entry_timestamp": trade["entry_timestamp"],
                "exit_timestamp": trade["exit_timestamp"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "exit_reason": trade["exit_reason"],
                "net_pnl_usdt": trade["pnl_usdt"],
                "gross_price_pnl_usdt": trade["gross_price_pnl_usdt"],
                "estimated_total_costs_usdt": trade["estimated_total_costs_usdt"],
            }
            for trade in trades[-20:]
        ],
        "position_strategy_average_net_return_percent_per_trade": (
            float(np.mean([trade["net_return_percent"] for trade in trades])) if trades else None
        ),
        "position_strategy_win_rate_percent": (
            float(np.mean([trade["pnl_usdt"] > 0 for trade in trades]) * 100.0) if trades else None
        ),
        "position_strategy_exit_reasons": {
            reason: sum(trade["exit_reason"] == reason for trade in trades)
            for reason in ("STOP_LOSS", "TAKE_PROFIT", "SELL_SIGNAL", "END_OF_TEST")
        },
        "position_strategy_exit_pnl_by_reason": summarize_exit_diagnostics(trades)["exit_pnl_by_reason"],
        "position_strategy_stop_loss_diagnostics": stop_loss_diagnostics,
        "position_strategy_cost_sensitivity": summarize_exit_diagnostics(trades)["cost_sensitivity"],
        "position_strategy_trades_sample": trades[-20:],
        "position_strategy_method": (
            "Long-only Spot simulation, one position at a time. Entry at signal candle close; "
            "every later candle OHLC is checked for stop-loss/take-profit even if a prediction "
            "is missing. If both stop and target are touched in one candle, stop-loss is assumed "
            "first. A candle opening below the stop is filled at its open, not the stop level. "
            "Estimated fees/slippage included; no account balance, exchange execution, or orders."
        ),
    }



def compare_stop_loss_scenarios(usable, predictions, take_profit_percent=3.0):
    """Compare predeclared stop distances on identical signals; descriptive research only."""
    scenarios = []
    # Fixed in advance, symmetric target across scenarios, no selection/tuning.
    for stop_percent in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        result = simulate_position_strategy(
            usable, predictions,
            stop_loss_percent=stop_percent,
            take_profit_percent=take_profit_percent,
        )
        scenarios.append({
            "stop_loss_percent": stop_percent,
            "take_profit_percent": float(take_profit_percent),
            "trade_count": result["position_strategy_trade_count"],
            "net_pnl_usdt": result["position_strategy_closed_trade_pnl_usdt"],
            "gross_price_pnl_usdt": result["position_strategy_gross_price_pnl_usdt"],
            "estimated_total_costs_usdt": result["position_strategy_estimated_total_costs_usdt"],
            "average_net_return_percent_per_trade": result["position_strategy_average_net_return_percent_per_trade"],
            "win_rate_percent": result["position_strategy_win_rate_percent"],
            "maximum_drawdown_usdt": result["position_strategy_maximum_drawdown_usdt"],
            "exit_reasons": result["position_strategy_exit_reasons"],
            "stop_loss_diagnostics": result["position_strategy_stop_loss_diagnostics"],
            "real_money_trading_eligible": False,
        })
    return {
        "report_type": "research-only-controlled-stop-loss-comparison",
        "scenario_design": "Predeclared fixed stop distances 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, and 4.0%; take-profit held at 3.0%; identical chronological out-of-sample BUY/SELL/WAIT signals and cost assumptions.",
        "same_predictions_and_cost_assumptions": True,
        "scenario_selection_performed": False,
        "warning": "These are comparisons on the same evaluation sample, not independent validation. Do not select a winning stop from this report alone; confirm on later unseen data and across disjoint periods.",
        "scenarios": scenarios,
        "real_money_trading_eligible": False,
    }

def summarize_probability_bands(predictions):
    """Summarize out-of-sample prediction quality by probability band after costs."""
    bands = [
        ("below_0_45", 0.0, 0.45),
        ("0_45_to_0_50", 0.45, 0.50),
        ("0_50_to_0_55", 0.50, 0.55),
        ("0_55_to_0_60", 0.55, 0.60),
        ("0_60_to_0_65", 0.60, 0.65),
        ("0_65_and_above", 0.65, 1.0000000001),
    ]
    summaries = []
    for label, lower, upper in bands:
        selected = [
            row for row in predictions
            if lower <= float(row["probability_up"]) < upper
        ]
        net_returns = [float(row["net_return_fraction"]) for row in selected]
        summaries.append({
            "probability_band": label,
            "probability_lower_inclusive": lower,
            "probability_upper_exclusive": upper,
            "prediction_count": len(selected),
            "directional_accuracy": (
                float(np.mean([
                    bool(row["actual_next_close_up"]) == (float(row["probability_up"]) >= 0.5)
                    for row in selected
                ])) if selected else None
            ),
            "average_next_candle_gross_return_percent": (
                float(np.mean([float(row["next_close_return_fraction"]) for row in selected]) * 100.0)
                if selected else None
            ),
            "average_next_candle_net_return_percent_after_costs": (
                float(np.mean(net_returns) * 100.0) if net_returns else None
            ),
            "fixed_stake_net_pnl_usdt": float(sum(net_returns) * STAKE_USDT),
            "profitable_prediction_count_after_costs": sum(value > 0 for value in net_returns),
            "real_money_trading_eligible": False,
        })
    return {
        "method": (
            "Descriptive grouping of chronological out-of-sample predictions. Bands are not "
            "selected using test outcomes; returns represent one-candle hypothetical fixed-stake "
            "trades with estimated entry/exit fees and slippage. Small band counts are noisy."
        ),
        "bands": summaries,
        "real_money_trading_eligible": False,
    }

def compare_bullish_regime_filter(predictions, stake_usdt=STAKE_USDT):
    """Compare fixed-threshold BUYs with the same BUYs filtered by a bullish SMA-gap regime.

    The regime flag is derived from the current candle's sma_gap_20 feature, which
    is available at the signal candle close. This is an exploratory filter, not
    tuned on the outer test period and never authorizes exchange orders.
    """
    baseline = [p for p in predictions if p.get("signal") == "BUY"]
    filtered = [p for p in baseline if p.get("sma_gap_20") is not None and float(p["sma_gap_20"]) > 0.0]

    def summarize(rows):
        returns = [float(p["net_return_fraction"]) for p in rows]
        pnl = float(sum(returns) * float(stake_usdt))
        drawdown = summarize_fixed_stake_drawdown(returns, stake_usdt)
        return {
            "buy_signal_count": len(rows),
            "fixed_stake_pnl_usdt": pnl,
            "average_net_return_percent_per_trade": float(np.mean(returns) * 100.0) if returns else None,
            "maximum_drawdown_usdt": drawdown["maximum_drawdown_usdt"],
            "maximum_drawdown_percent_of_one_stake": drawdown["maximum_drawdown_percent_of_one_stake"],
            "profitable_trade_count": sum(value > 0 for value in returns),
            "estimated_costs_included": True,
        }

    baseline_summary = summarize(baseline)
    filtered_summary = summarize(filtered)
    return {
        "report_type": "experimental-bullish-regime-filter-comparison",
        "filter_definition": "Keep baseline BUY signals only when current-candle sma_gap_20 > 0 (close above the 20-period moving average proxy).",
        "baseline_fixed_threshold": baseline_summary,
        "bullish_regime_filtered": filtered_summary,
        "pnl_difference_filtered_minus_baseline_usdt": filtered_summary["fixed_stake_pnl_usdt"] - baseline_summary["fixed_stake_pnl_usdt"],
        "trade_count_reduction": baseline_summary["buy_signal_count"] - filtered_summary["buy_signal_count"],
        "same_out_of_sample_predictions": True,
        "selection_used_outer_test_returns": False,
        "real_money_trading_eligible": False,
        "interpretation": "Compare after-cost PnL and drawdown as well as trade count. Fewer trades or higher win rate alone is not evidence of a better strategy."
    }


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
    tuned_trade_returns = []
    selected_thresholds = []
    tuned_buy_count = 0
    baseline_returns = []
    directional_actual = []
    directional_predictions = []
    majority_baseline_predictions = []
    buy_count = 0
    sell_count = 0
    wait_count = 0

    # At candle i close, labels for rows before i are known because they depend
    # only on closes through candle i. Candle i itself is never in training.
    # Evaluate every Nth candle to reduce redundant fits; each prediction still
    # targets the immediately following candle, and thresholds are refreshed
    # periodically using only the historical training prefix available at i.
    selected_threshold = BUY_THRESHOLD
    # Small synthetic/unit-test datasets may not contain enough rows for the
    # configured sparse step. Reduce the stride when necessary so the evaluator
    # can still produce its minimum 20 predictions whenever the data allows it.
    available_prediction_rows = len(usable) - 1 - MIN_SAMPLES
    effective_step = min(
        WALK_FORWARD_STEP,
        max(1, available_prediction_rows // 20),
    )
    for i in range(MIN_SAMPLES, len(usable) - 1, effective_step):
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
        if len(predictions) % THRESHOLD_REFRESH_PREDICTIONS == 0:
            selected_threshold = _select_buy_threshold(train, usable)
        selected_thresholds.append(selected_threshold)
        actual_up = int(float(current.iloc[0]["close"]) < float(usable.iloc[i + 1]["close"]))
        predicted_up = int(probability_up >= 0.5)
        directional_actual.append(actual_up)
        directional_predictions.append(predicted_up)
        # Compare each prediction with a baseline learned only from that fold's training data.
        majority_baseline_predictions.append(int(train["target"].mean() >= 0.5))

        if probability_up >= selected_threshold:
            tuned_buy_count += 1
            tuned_trade_returns.append(_net_one_candle_return(
                float(usable.iloc[i]["close"]), float(usable.iloc[i + 1]["close"])
            ))

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
        next_close_return_fraction = (
            float(usable.iloc[i + 1]["close"]) / float(current.iloc[0]["close"]) - 1.0
        )
        predictions.append({
            "timestamp": int(current.iloc[0]["timestamp"]),
            "close": float(current.iloc[0]["close"]),
            "probability_up": probability_up,
            "signal": signal,
            "actual_next_close_up": bool(actual_up),
            "next_close_return_fraction": next_close_return_fraction,
            "net_return_fraction": _net_one_candle_return(
                float(current.iloc[0]["close"]), float(usable.iloc[i + 1]["close"])
            ),
            "sma_gap_20": float(current.iloc[0]["sma_gap_20"]) if np.isfinite(float(current.iloc[0]["sma_gap_20"])) else None,
        })

    if len(predictions) < 20:
        raise ValueError("Walk-forward test produced fewer than 20 predictions.")

    # Fixed-stake PnL makes the assumed exposure explicit. Compounded returns
    # below are illustrative and are not a portfolio-level simulation.
    fixed_stake_pnl_usdt = float(sum(trade_returns) * STAKE_USDT)
    tuned_fixed_stake_pnl_usdt = float(sum(tuned_trade_returns) * STAKE_USDT)
    drawdown_summary = summarize_fixed_stake_drawdown(trade_returns)
    tuned_drawdown_summary = summarize_fixed_stake_drawdown(tuned_trade_returns)
    tuned_average_buy_return = float(np.mean(tuned_trade_returns)) if tuned_trade_returns else None
    average_buy_return = float(np.mean(trade_returns)) if trade_returns else None
    average_market_return = float(np.mean(baseline_returns))
    research_gate = evaluate_research_gate(
        len(predictions), len(trade_returns), average_buy_return,
        average_market_return, fixed_stake_pnl_usdt,
    )
    position_strategy = simulate_position_strategy(usable, predictions)
    stop_loss_comparison = compare_stop_loss_scenarios(usable, predictions)
    holding_period_analysis = analyze_holding_periods(usable, predictions)
    regime_filter_comparison = compare_bullish_regime_filter(predictions)
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
        "walk_forward_majority_baseline_accuracy": float(accuracy_score(directional_actual, majority_baseline_predictions)),
        "directional_accuracy_beats_majority_baseline": bool(accuracy_score(directional_actual, directional_predictions) > accuracy_score(directional_actual, majority_baseline_predictions)),
        "buy_signal_count": buy_count,
        **research_gate,
        **position_strategy,
        "threshold_tuning_method": "At each walk-forward step, compare fixed BUY thresholds on an inner chronological validation slice from that step's training data only. A candidate must have at least five trades, positive average net return after estimated costs, and beat the same-window all-candle baseline after costs; otherwise fall back to ML_BUY_THRESHOLD. The later walk-forward prediction is not used to select its threshold. This remains exploratory and may overfit validation.",
        "configured_buy_threshold": BUY_THRESHOLD,
        "selected_buy_threshold_min": float(min(selected_thresholds)),
        "selected_buy_threshold_max": float(max(selected_thresholds)),
        "selected_buy_threshold_mean": float(np.mean(selected_thresholds)),
        "tuned_buy_signal_count": int(tuned_buy_count),
        "tuned_buy_only_fixed_stake_total_pnl_usdt": tuned_fixed_stake_pnl_usdt,
        "tuned_buy_only_average_net_return_percent_per_trade": tuned_average_buy_return * 100 if tuned_average_buy_return is not None else None,
        "tuned_buy_signals_average_return_beats_all_candles_after_costs": bool(tuned_average_buy_return > average_market_return) if tuned_average_buy_return is not None else False,
        "tuned_threshold_beats_fixed_threshold_after_costs": bool(tuned_average_buy_return > average_buy_return) if tuned_average_buy_return is not None and average_buy_return is not None else False,
        "sell_signal_count": sell_count,
        "wait_signal_count": wait_count,
        "trade_test_method": "Each BUY prediction is a hypothetical one-candle Spot long, entered at that candle close and exited at the next close. Each trade uses a fixed stake; actual portfolio allocation, idle cash, and order execution are not simulated.",
        "stake_per_hypothetical_trade_usdt": STAKE_USDT,
        "fee_rate_each_side": FEE_RATE,
        "slippage_rate_each_side": SLIPPAGE_RATE,
        "estimated_round_trip_cost_percent_approx": float((2 * FEE_RATE + 2 * SLIPPAGE_RATE) * 100),
        "buy_only_trade_count": int(len(trade_returns)),
        "buy_only_fixed_stake_total_pnl_usdt": fixed_stake_pnl_usdt,
        "buy_only_max_drawdown_usdt": drawdown_summary["maximum_drawdown_usdt"],
        "buy_only_max_drawdown_percent_of_one_stake": drawdown_summary["maximum_drawdown_percent_of_one_stake"],
        "buy_only_drawdown_method": drawdown_summary["drawdown_method"],
        "tuned_buy_only_max_drawdown_usdt": tuned_drawdown_summary["maximum_drawdown_usdt"],
        "tuned_buy_only_max_drawdown_percent_of_one_stake": tuned_drawdown_summary["maximum_drawdown_percent_of_one_stake"],
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
            "Position-based stop-loss/take-profit simulation uses candle OHLC; intrabar ordering is unknown and stop-loss-first is assumed when both levels are touched.",
            "No exchange account is accessed and no orders are submitted.",
            "Threshold tuning uses only each fold's training period and rejects candidate thresholds without positive average validation return after estimated costs, but repeated comparisons can still overfit; validate across more symbols and later unseen periods before any live use.",
        ],
        "probability_band_after_cost_analysis": summarize_probability_bands(predictions),
        "holding_period_cost_analysis": holding_period_analysis,
        "bullish_regime_filter_comparison": regime_filter_comparison,
        "controlled_stop_loss_comparison": stop_loss_comparison,
        "sample_predictions": predictions[-10:],
    }



def evaluate_multiple_unseen_periods(candles, period_count=4):
    """Evaluate disjoint chronological periods separately; research only, never authorizes orders."""
    if not isinstance(period_count, int) or period_count < 2:
        raise ValueError("period_count must be an integer of at least 2.")

    frame = build_feature_frame(candles)
    valid = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    minimum_period_candles = MIN_SAMPLES + 30
    if len(valid) < period_count * minimum_period_candles:
        return {
            "report_type": "experimental-multi-period-walk-forward-evaluation",
            "multi_period_evaluation_available": False,
            "period_count_requested": period_count,
            "available_candle_count": int(len(valid)),
            "minimum_candles_required": int(period_count * minimum_period_candles),
            "reason": "Not enough candles to evaluate every disjoint period with a training window and test predictions.",
            "real_money_trading_eligible": False,
        }

    # Rebuild clean OHLCV rows, then split into non-overlapping chronological periods.
    ordered_candles = [
        [
            str(int(row["timestamp"])), str(float(row["open"])),
            str(float(row["high"])), str(float(row["low"])),
            str(float(row["close"])), str(float(row["volume"])), "0",
        ]
        for _, row in valid.iterrows()
    ]
    boundaries = np.linspace(0, len(ordered_candles), period_count + 1, dtype=int)
    period_reports = []
    for period_index in range(period_count):
        period_candles = ordered_candles[boundaries[period_index]:boundaries[period_index + 1]]
        report = walk_forward_evaluate(period_candles)
        average_buy = report["buy_only_average_net_return_percent_per_trade"]
        average_market = report["all_test_candles_average_net_return_percent"]
        checks = {
            "enough_walk_forward_predictions": report["walk_forward_prediction_count"] >= 50,
            "enough_buy_signals": report["buy_signal_count"] >= 5,
            "positive_average_net_buy_return": average_buy is not None and average_buy > 0.0,
            "buy_returns_beat_all_candles_after_costs": (
                average_buy is not None and average_market is not None and average_buy > average_market
            ),
            "positive_fixed_stake_pnl": report["buy_only_fixed_stake_total_pnl_usdt"] > 0.0,
        }
        period_reports.append({
            "period_number": period_index + 1,
            "first_candle_timestamp": int(period_candles[0][0]),
            "last_candle_timestamp": int(period_candles[-1][0]),
            "walk_forward_prediction_count": report["walk_forward_prediction_count"],
            "first_test_timestamp": report["first_test_timestamp"],
            "last_test_timestamp": report["last_test_timestamp"],
            "directional_accuracy": report["directional_accuracy"],
            "majority_baseline_accuracy": report["walk_forward_majority_baseline_accuracy"],
            "directional_accuracy_beats_majority_baseline": report["directional_accuracy_beats_majority_baseline"],
            "directional_accuracy_edge_over_majority_baseline_percentage_points": (
                report["directional_accuracy"] - report["walk_forward_majority_baseline_accuracy"]
            ) * 100.0,
            "buy_signal_count": report["buy_signal_count"],
            "buy_only_fixed_stake_total_pnl_usdt": report["buy_only_fixed_stake_total_pnl_usdt"],
            "buy_only_max_drawdown_usdt": report["buy_only_max_drawdown_usdt"],
            "buy_only_max_drawdown_percent_of_one_stake": report["buy_only_max_drawdown_percent_of_one_stake"],
            "buy_only_average_net_return_percent_per_trade": average_buy,
            "all_test_candles_average_net_return_percent": average_market,
            "buy_return_edge_over_all_candles_percentage_points": (
                average_buy - average_market if average_buy is not None and average_market is not None else None
            ),
            "buy_strategy_beats_one_candle_baseline_after_costs": (
                average_buy is not None and average_market is not None and average_buy > average_market
            ),
            "tuned_buy_signal_count": report["tuned_buy_signal_count"],
            "tuned_buy_only_fixed_stake_total_pnl_usdt": report["tuned_buy_only_fixed_stake_total_pnl_usdt"],
            "tuned_buy_only_average_net_return_percent_per_trade": report["tuned_buy_only_average_net_return_percent_per_trade"],
            "tuned_threshold_beats_fixed_threshold_after_costs": report["tuned_threshold_beats_fixed_threshold_after_costs"],
            "position_strategy_trade_count": report["position_strategy_trade_count"],
            "position_strategy_closed_trade_pnl_usdt": report["position_strategy_closed_trade_pnl_usdt"],
            "research_gate_passed": bool(all(checks.values())),
            "research_gate_checks": checks,
            "real_money_trading_eligible": False,
        })

    passing_periods = sum(period["research_gate_passed"] for period in period_reports)
    directional_wins = sum(period["directional_accuracy_beats_majority_baseline"] for period in period_reports)
    cost_adjusted_benchmark_wins = sum(
        period["buy_strategy_beats_one_candle_baseline_after_costs"] for period in period_reports
    )
    positive_pnl_periods = sum(period["buy_only_fixed_stake_total_pnl_usdt"] > 0 for period in period_reports)
    tuned_threshold_wins = sum(period["tuned_threshold_beats_fixed_threshold_after_costs"] for period in period_reports)
    return {
        "report_type": "experimental-multi-period-walk-forward-evaluation",
        "multi_period_evaluation_available": True,
        "period_count_requested": period_count,
        "period_count_evaluated": len(period_reports),
        "periods_are_chronologically_ordered_and_non_overlapping": True,
        "directional_accuracy_beats_majority_baseline_period_count": int(directional_wins),
        "buy_strategy_beats_one_candle_baseline_after_costs_period_count": int(cost_adjusted_benchmark_wins),
        "positive_buy_only_fixed_stake_pnl_period_count": int(positive_pnl_periods),
        "tuned_threshold_beats_fixed_threshold_after_costs_period_count": int(tuned_threshold_wins),
        "positive_research_gate_period_count": int(passing_periods),
        "multi_period_research_gate_passed": bool(len(period_reports) == period_count and passing_periods == period_count),
        "benchmark_interpretation": (
            "A model is more convincing when directional accuracy beats the training-fold majority baseline "
            "and selected BUY returns beat the same-period one-candle always-buy benchmark after estimated "
            "fees and slippage. Counts show consistency across periods, not proof of future performance."
        ),
        "periods": period_reports,
        "method": (
            "The available candle history is split into disjoint chronological periods. Each period is evaluated "
            "with its own expanding walk-forward training and later test predictions; fees and slippage are included. "
            "This is a research screen, not proof of future profitability."
        ),
        "real_money_trading_eligible": False,
        "note": "This report-only evaluation never enables live trading and never submits exchange orders.",
    }



def analyze_holding_periods(usable, predictions, holding_periods=(1, 3, 6, 12)):
    # Research-only diagnostic: never authorizes live order placement.
    """Compare fixed holding horizons using only chronological out-of-sample BUY signals.

    Trades are non-overlapping within each horizon. Each position enters at the
    signal candle close and exits at the close N candles later. Fees and slippage
    are charged on both sides. This is a diagnostic, not a strategy selector.
    """
    index_by_timestamp = {
        int(row["timestamp"]): idx for idx, (_, row) in enumerate(usable.iterrows())
    }
    buy_indices = sorted({
        index_by_timestamp[int(pred["timestamp"])]
        for pred in predictions
        if pred.get("signal") == "BUY"
        and int(pred["timestamp"]) in index_by_timestamp
    })
    summaries = []
    for horizon in holding_periods:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Holding periods must be positive integers.")
        trades = []
        # Do not re-enter on the same candle that closed the prior trade.
        # This keeps holding-period samples strictly non-overlapping at candle boundaries.
        next_available_index = -1
        for entry_index in buy_indices:
            exit_index = entry_index + horizon
            if entry_index <= next_available_index or exit_index >= len(usable):
                continue
            entry_price = float(usable.iloc[entry_index]["close"])
            exit_price = float(usable.iloc[exit_index]["close"])
            entry_execution = entry_price * (1.0 + SLIPPAGE_RATE)
            exit_execution = exit_price * (1.0 - SLIPPAGE_RATE)
            quantity = STAKE_USDT / entry_execution
            entry_fee = STAKE_USDT * FEE_RATE
            exit_fee = quantity * exit_execution * FEE_RATE
            gross_pnl = quantity * (exit_price - entry_price)
            entry_slippage = quantity * (entry_execution - entry_price)
            exit_slippage = quantity * (exit_price - exit_execution)
            total_costs = entry_fee + exit_fee + entry_slippage + exit_slippage
            net_pnl = gross_pnl - total_costs
            trades.append({
                "entry_timestamp": int(usable.iloc[entry_index]["timestamp"]),
                "exit_timestamp": int(usable.iloc[exit_index]["timestamp"]),
                "holding_candles": horizon,
                "gross_price_pnl_usdt": float(gross_pnl),
                "estimated_total_costs_usdt": float(total_costs),
                "net_pnl_usdt": float(net_pnl),
                "net_return_percent": float(net_pnl / STAKE_USDT * 100.0),
            })
            next_available_index = exit_index
        summaries.append({
            "holding_period_candles": horizon,
            "holding_period_minutes": horizon * (5 if INTERVAL == "5m" else 1),
            "trade_count": len(trades),
            "gross_price_pnl_usdt": float(sum(t["gross_price_pnl_usdt"] for t in trades)),
            "estimated_total_costs_usdt": float(sum(t["estimated_total_costs_usdt"] for t in trades)),
            "net_pnl_usdt": float(sum(t["net_pnl_usdt"] for t in trades)),
            "average_net_return_percent_per_trade": (
                float(np.mean([t["net_return_percent"] for t in trades])) if trades else None
            ),
            "profitable_trade_count": sum(t["net_pnl_usdt"] > 0 for t in trades),
            "real_money_trading_eligible": False,
            "sample_trades": trades[-10:],
        })
    return {
        "report_type": "experimental-out-of-sample-holding-period-cost-analysis",
        "method": (
            "Uses only chronological walk-forward out-of-sample BUY signals. For each horizon, "
            "trades are non-overlapping and exit at the close N candles after entry. Entry/exit "
            "fees and slippage are included. No thresholds are tuned on these results; this "
            "diagnostic does not authorize orders."
        ),
        "holding_periods": summaries,
        "real_money_trading_eligible": False,
    }


def main():
    candles = fetch_candles()
    result = walk_forward_evaluate(candles)
    requested_lookback_days = int(os.getenv("CANDLE_LOOKBACK_DAYS", "60"))
    interval_ms_by_name = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1H": 3_600_000, "4H": 14_400_000,
        "6H": 21_600_000, "12H": 43_200_000, "1D": 86_400_000,
    }
    first_timestamp = int(candles[0][0])
    last_timestamp = int(candles[-1][0])
    interval_ms = interval_ms_by_name.get(INTERVAL)
    expected_candles = (
        int(requested_lookback_days * 24 * 60 * 60 * 1000 / interval_ms)
        if interval_ms else None
    )
    result["evaluation_runtime_settings"] = {
        "walk_forward_step_candles_configured": WALK_FORWARD_STEP,
        "walk_forward_step_candles_effective": min(
            WALK_FORWARD_STEP,
            max(1, (len(build_feature_frame(candles).dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])) - 1 - MIN_SAMPLES) // 20),
        ),
        "threshold_refresh_every_predictions": THRESHOLD_REFRESH_PREDICTIONS,
        "note": "Predictions are sampled chronologically every N candles; each prediction still scores the immediately following candle. Threshold tuning is refreshed periodically using only the training prefix available at that point.",
    }
    result["candle_history_coverage"] = {
        "requested_lookback_days": requested_lookback_days,
        "actual_coverage_days": round((last_timestamp - first_timestamp) / 86_400_000, 3),
        "candle_count": len(candles),
        "expected_candle_count_approx": expected_candles,
        "first_candle_utc": datetime.fromtimestamp(first_timestamp / 1000, timezone.utc).isoformat(),
        "last_candle_utc": datetime.fromtimestamp(last_timestamp / 1000, timezone.utc).isoformat(),
        "coverage_validation": "fetch_candles validates each 30-day window and requires at least 95% of expected candle count.",
    }
    multi_period_result = evaluate_multiple_unseen_periods(candles, period_count=4)
    created_at = datetime.now(timezone.utc).isoformat()
    result["multi_period_evaluation"] = multi_period_result
    result["created_at_utc"] = created_at
    multi_period_result["created_at_utc"] = created_at
    with open("walk_forward_report.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with open("multi_period_report.json", "w", encoding="utf-8") as handle:
        json.dump(multi_period_result, handle, indent=2, allow_nan=False)
    print("WALK-FORWARD EVALUATION — public data only; no orders.")
    print("Predictions:", result["walk_forward_prediction_count"])
    print("Directional accuracy:", round(result["directional_accuracy"], 4))
    print("Balanced accuracy:", round(result["directional_balanced_accuracy"], 4))
    print("Walk-forward majority baseline accuracy:", round(result["walk_forward_majority_baseline_accuracy"], 4))
    print("Beats walk-forward majority baseline:", result["directional_accuracy_beats_majority_baseline"])
    print("BUY / SELL / WAIT:", result["buy_signal_count"], "/", result["sell_signal_count"], "/", result["wait_signal_count"])
    print("BUY-only fixed-stake total PnL (USDT):", round(result["buy_only_fixed_stake_total_pnl_usdt"], 4))
    print("BUY-only average net return per trade (%):", result["buy_only_average_net_return_percent_per_trade"])
    print("Tuned threshold range:", result["selected_buy_threshold_min"], "to", result["selected_buy_threshold_max"])
    print("Tuned BUY count / fixed-stake PnL (USDT):", result["tuned_buy_signal_count"], "/", round(result["tuned_buy_only_fixed_stake_total_pnl_usdt"], 4))
    print("Tuned average net return per trade (%):", result["tuned_buy_only_average_net_return_percent_per_trade"])
    print("Tuned threshold beats fixed threshold after costs:", result["tuned_threshold_beats_fixed_threshold_after_costs"])
    print("All-candle average net return (%):", round(result["all_test_candles_average_net_return_percent"], 4))
    print("Multi-period evaluation available:", multi_period_result["multi_period_evaluation_available"])
    if multi_period_result["multi_period_evaluation_available"]:
        print("Distinct periods evaluated:", multi_period_result["period_count_evaluated"])
        print("Periods passing the research gate:", multi_period_result["positive_research_gate_period_count"])
        print("Multi-period research gate passed:", multi_period_result["multi_period_research_gate_passed"])
    else:
        print("Multi-period evaluation note:", multi_period_result["reason"])
    print("Real-money trading eligible:", multi_period_result["real_money_trading_eligible"])
    print("Compounded figures are illustrative only, not portfolio returns.")
    print("Reports written: walk_forward_report.json and multi_period_report.json")
    print("This research report does not control paper trades or place orders.")


if __name__ == "__main__":
    main()
