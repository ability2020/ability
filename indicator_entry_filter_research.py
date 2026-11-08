"""Research-only predeclared entry-filter comparison for the production SMA/RSI strategy.

No exchange account access or order submission. All signals use completed candles;
entries execute at the following candle open through indicator_strategy_backtest.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from indicator_strategy_backtest import (
    FEE_RATE, SLIPPAGE_RATE, STAKE_USDT,
    build_indicator_signals, simulate, summarize,
)
from learning_model import INTERVAL, SYMBOL, fetch_candles

MINIMUM_TRADES_FOR_INTERPRETATION = 10


def add_entry_features(frame):
    result = frame.copy().reset_index(drop=True)
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["sma_20"] = close.rolling(20, min_periods=20).mean()
    result["sma_21_rising_3"] = result["sma_21"] > result["sma_21"].shift(3)
    result["atr_14_percent"] = (
        true_range.rolling(14, min_periods=14).mean() / close
    )
    return result


def entry_filter_mask(frame, candidate):
    if candidate == "baseline":
        return pd.Series(True, index=frame.index)
    if candidate == "price_above_sma20":
        return frame["close"] > frame["sma_20"]
    if candidate == "sma21_rising_3":
        return frame["sma_21_rising_3"].fillna(False)
    if candidate == "trend_confirmation":
        return ((frame["close"] > frame["sma_20"]) &
                frame["sma_21_rising_3"].fillna(False))
    if candidate == "rsi_50_to_65":
        return frame["rsi_14"].between(50, 65, inclusive="both")
    if candidate == "atr_at_least_0_35_percent":
        return frame["atr_14_percent"] >= 0.0035
    if candidate == "trend_and_rsi":
        return ((frame["close"] > frame["sma_20"]) &
                frame["sma_21_rising_3"].fillna(False) &
                frame["rsi_14"].between(50, 65, inclusive="both"))
    raise ValueError(f"Unknown entry filter: {candidate}")


def apply_entry_filter(frame, candidate):
    filtered = frame.copy()
    allowed = entry_filter_mask(filtered, candidate)
    blocked = (filtered["signal"] == "BUY") & ~allowed
    filtered.loc[blocked, "signal"] = "WAIT"
    return filtered, int(blocked.sum())


def evaluate_candidate(frame, candidate, start=0, end=None):
    filtered, _ = apply_entry_filter(frame, candidate)
    end_index = len(frame) if end is None else min(int(end), len(frame))
    start_index = max(0, int(start))
    blocked = int(((frame["signal"] == "BUY") & (filtered["signal"] == "WAIT")).iloc[start_index:end_index].sum())
    trades = simulate(filtered, "immediate_sell", start=start, end=end)
    result = summarize(trades)
    result["buy_signals_blocked"] = blocked
    result["minimum_10_trades_met"] = result["trade_count"] >= MINIMUM_TRADES_FOR_INTERPRETATION
    result["positive_net_pnl"] = result["net_pnl_usdt"] > 0
    result["interpretable"] = result["minimum_10_trades_met"]
    return result


def run_research(candles):
    if STAKE_USDT <= 0 or not np.isfinite(STAKE_USDT):
        raise ValueError("Stake must be positive and finite.")
    frame = add_entry_features(build_indicator_signals(candles))
    candidates = (
        "baseline",
        "price_above_sma20",
        "sma21_rising_3",
        "trend_confirmation",
        "rsi_50_to_65",
        "atr_at_least_0_35_percent",
        "trend_and_rsi",
    )
    results = {name: evaluate_candidate(frame, name) for name in candidates}
    edges = np.linspace(0, len(frame), 4, dtype=int)
    periods = []
    for index in range(3):
        left, right = int(edges[index]), int(edges[index + 1])
        periods.append({
            "period": index + 1,
            "first_timestamp": int(frame.iloc[left]["timestamp"]),
            "last_timestamp": int(frame.iloc[right - 1]["timestamp"]),
            "candidates": {
                name: evaluate_candidate(frame, name, left, right)
                for name in candidates
            },
        })
    return {
        "report_type": "production_indicator_entry_filter_research",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candle_count": len(frame),
        "first_candle_timestamp": int(frame.iloc[0]["timestamp"]),
        "last_candle_timestamp": int(frame.iloc[-1]["timestamp"]),
        "cost_assumptions": {
            "stake_usdt": STAKE_USDT,
            "fee_each_side": FEE_RATE,
            "slippage_each_side": SLIPPAGE_RATE,
            "note": "Estimated costs, not actual Bitget fills.",
        },
        "execution_method": (
            "Completed-candle signals; next-candle-open entries and SELL exits; "
            "same conservative OHLC stop/target simulation as baseline."
        ),
        "candidate_definitions": {
            "baseline": "Unfiltered production SMA9/SMA21/RSI14 BUY signals.",
            "price_above_sma20": "BUY only when close is above SMA20.",
            "sma21_rising_3": "BUY only when SMA21 is higher than three completed candles earlier.",
            "trend_confirmation": "Close above SMA20 and SMA21 rising over three completed candles.",
            "rsi_50_to_65": "RSI14 between 50 and 65 inclusive.",
            "atr_at_least_0_35_percent": "ATR14 divided by close is at least 0.35%; volatility screen only, not a profit forecast.",
            "trend_and_rsi": "Trend confirmation plus RSI14 between 50 and 65 inclusive.",
        },
        "minimum_closed_trades_for_interpretation": MINIMUM_TRADES_FOR_INTERPRETATION,
        "candidates": results,
        "chronological_periods": periods,
        "research_only": True,
        "real_money_trading_eligible": False,
        "orders_submitted": False,
        "selection_rule": (
            "No candidate is promoted from this report. Any positive candidate must "
            "meet the minimum trade count and then pass new untouched historical windows "
            "with positive net PnL after costs."
        ),
        "limitations": [
            "Three chronological slices are descriptive, not independent holdouts.",
            "Candidates are predeclared diagnostics; multiple comparisons can still overfit.",
            "ATR measures volatility, not directional edge or expected profit.",
            "Estimated fees and slippage may differ from actual execution.",
            "No real-money eligibility is inferred from a successful workflow.",
        ],
    }


def main():
    candles = fetch_candles()
    report = run_research(candles)
    with open("indicator_entry_filter_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("INDICATOR ENTRY FILTER RESEARCH — research only; no orders.")
    for name, result in report["candidates"].items():
        print(
            name,
            "trades:", result["trade_count"],
            "gross:", round(result["gross_pnl_usdt"], 6),
            "costs:", round(result["estimated_costs_usdt"], 6),
            "net:", round(result["net_pnl_usdt"], 6),
            "interpretable:", result["interpretable"],
        )
    print("Report written: indicator_entry_filter_report.json")


if __name__ == "__main__":
    main()
