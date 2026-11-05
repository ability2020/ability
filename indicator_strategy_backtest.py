"""Historical replay of the production SMA(9)/SMA(21)/RSI(14) Spot rules.

Research only: public candles, next-candle-open execution, OHLC exits and
estimated two-sided fees/slippage. It never reads account balances or submits orders.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from learning_model import INTERVAL, SYMBOL, fetch_candles, build_feature_frame

STAKE_USDT = float(os.getenv("BACKTEST_STAKE_USDT", "10"))
FEE_RATE = float(os.getenv("BACKTEST_FEE_RATE", "0.001"))
SLIPPAGE_RATE = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "2"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3"))


def build_indicator_signals(candles):
    """Reproduce main.calculate_sma/calculate_rsi/get_signal on each closed bar."""
    frame = build_feature_frame(candles)
    if frame.empty:
        raise ValueError("No valid candles supplied to indicator replay.")
    frame = frame.copy().reset_index(drop=True)
    close = frame["close"].astype(float)
    frame["sma_9"] = close.rolling(9, min_periods=9).mean()
    frame["sma_21"] = close.rolling(21, min_periods=21).mean()
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-changes.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    frame["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    frame.loc[(losses == 0) & (gains > 0), "rsi_14"] = 100.0
    frame.loc[(losses == 0) & (gains == 0), "rsi_14"] = 50.0
    frame["indicator_score"] = 0
    frame.loc[frame["sma_9"] > frame["sma_21"], "indicator_score"] += 1
    frame.loc[frame["sma_9"] < frame["sma_21"], "indicator_score"] -= 1
    frame.loc[(frame["rsi_14"] >= 50) & (frame["rsi_14"] < 70), "indicator_score"] += 1
    frame.loc[(frame["rsi_14"] > 30) & (frame["rsi_14"] <= 50), "indicator_score"] -= 1
    frame["signal"] = "WAIT"
    frame.loc[frame["indicator_score"] >= 2, "signal"] = "BUY"
    frame.loc[frame["indicator_score"] <= -2, "signal"] = "SELL"
    frame.loc[frame[["sma_9", "sma_21", "rsi_14"]].isna().any(axis=1), "signal"] = "WAIT"
    return frame


def _trade_record(position, exit_reference, exit_reason, exit_timestamp):
    qty = position["quantity"]
    exit_execution = exit_reference * (1.0 - SLIPPAGE_RATE)
    entry_fee = STAKE_USDT * FEE_RATE
    exit_fee = qty * exit_execution * FEE_RATE
    entry_slip = qty * (position["entry_execution"] - position["entry_reference"])
    exit_slip = qty * (exit_reference - exit_execution)
    gross = qty * (exit_reference - position["entry_reference"])
    costs = entry_fee + exit_fee + entry_slip + exit_slip
    return {
        "entry_timestamp": int(position["entry_timestamp"]),
        "exit_timestamp": int(exit_timestamp),
        "entry_price": float(position["entry_reference"]),
        "exit_price": float(exit_reference),
        "exit_reason": exit_reason,
        "holding_candles": int(position["holding_candles"]),
        "gross_pnl_usdt": float(gross),
        "estimated_costs_usdt": float(costs),
        "net_pnl_usdt": float(gross - costs),
    }


def simulate(frame, exit_policy="immediate_sell", start=0, end=None):
    """Enter at next bar open; check held-bar OHLC for stops/targets.

    Policies compare production immediate SELL, two consecutive SELL candles,
    and ignoring SELL signals (protective exits only). Alternatives are diagnostics.
    """
    if exit_policy not in {"immediate_sell", "confirmed_sell_2", "protective_only"}:
        raise ValueError("Unknown exit policy.")
    end = len(frame) if end is None else min(int(end), len(frame))
    start = max(0, int(start))
    if end - start < 2:
        return []
    position, trades = None, []
    for i in range(start, end):
        row = frame.iloc[i]
        prior_signal = frame.iloc[i - 1]["signal"] if i > 0 else "WAIT"
        two_sell = (i >= 2 and frame.iloc[i - 1]["signal"] == "SELL"
                    and frame.iloc[i - 2]["signal"] == "SELL")
        exited = False
        if position is not None:
            op, hi, lo = float(row.open), float(row.high), float(row.low)
            stop = position["entry_reference"] * (1 - STOP_LOSS_PERCENT / 100)
            target = position["entry_reference"] * (1 + TAKE_PROFIT_PERCENT / 100)
            if op <= stop:
                trades.append(_trade_record(position, op, "STOP_LOSS_GAP", row.timestamp)); position = None; exited = True
            elif op >= target:
                trades.append(_trade_record(position, target, "TAKE_PROFIT", row.timestamp)); position = None; exited = True
            elif exit_policy == "immediate_sell" and prior_signal == "SELL":
                trades.append(_trade_record(position, op, "SELL_SIGNAL", row.timestamp)); position = None; exited = True
            elif exit_policy == "confirmed_sell_2" and two_sell:
                trades.append(_trade_record(position, op, "CONFIRMED_SELL_SIGNAL", row.timestamp)); position = None; exited = True
            elif lo <= stop:
                trades.append(_trade_record(position, stop, "STOP_LOSS", row.timestamp)); position = None; exited = True
            elif hi >= target:
                trades.append(_trade_record(position, target, "TAKE_PROFIT", row.timestamp)); position = None; exited = True
            else:
                position["holding_candles"] += 1
        # Signal candle i-1 is closed; execute its BUY at candle i open.
        if position is None and not exited and prior_signal == "BUY":
            reference = float(row.open)
            execution = reference * (1 + SLIPPAGE_RATE)
            position = {"entry_timestamp": row.timestamp, "entry_reference": reference,
                        "entry_execution": execution, "quantity": STAKE_USDT / execution,
                        "holding_candles": 0}
            # Since entry occurs at this candle's open, its remaining range can
            # hit a stop/target immediately. Stop-first resolves dual-touch bars.
            stop = reference * (1 - STOP_LOSS_PERCENT / 100)
            target = reference * (1 + TAKE_PROFIT_PERCENT / 100)
            if float(row.low) <= stop:
                trades.append(_trade_record(position, min(float(row.open), stop), "STOP_LOSS", row.timestamp))
                position = None
            elif float(row.high) >= target:
                trades.append(_trade_record(position, target, "TAKE_PROFIT", row.timestamp))
                position = None
    if position is not None:
        last = frame.iloc[end - 1]
        trades.append(_trade_record(position, float(last.close), "END_OF_WINDOW", last.timestamp))
    return trades


def summarize(trades):
    gross = sum(t["gross_pnl_usdt"] for t in trades)
    costs = sum(t["estimated_costs_usdt"] for t in trades)
    net = sum(t["net_pnl_usdt"] for t in trades)
    by_reason = {}
    for reason in sorted({t["exit_reason"] for t in trades}):
        selected = [t for t in trades if t["exit_reason"] == reason]
        by_reason[reason] = {"count": len(selected),
                             "gross_pnl_usdt": sum(t["gross_pnl_usdt"] for t in selected),
                             "estimated_costs_usdt": sum(t["estimated_costs_usdt"] for t in selected),
                             "net_pnl_usdt": sum(t["net_pnl_usdt"] for t in selected)}
    return {"trade_count": len(trades), "gross_pnl_usdt": gross,
            "estimated_costs_usdt": costs, "net_pnl_usdt": net,
            "win_count_after_costs": sum(t["net_pnl_usdt"] > 0 for t in trades),
            "exit_reasons": by_reason, "trades": trades}


def run_research(candles):
    if STAKE_USDT <= 0 or not np.isfinite(STAKE_USDT):
        raise ValueError("BACKTEST_STAKE_USDT must be positive and finite.")
    if not (0 <= FEE_RATE < 0.1 and 0 <= SLIPPAGE_RATE < 0.1):
        raise ValueError("Fee/slippage assumptions must be in [0, 0.1).")
    frame = build_indicator_signals(candles)
    policies = {policy: summarize(simulate(frame, policy))
                for policy in ("immediate_sell", "confirmed_sell_2", "protective_only")}
    # Three chronological slices are descriptive regime diagnostics, not tuned holdouts.
    periods = []
    edges = np.linspace(0, len(frame), 4, dtype=int)
    for n in range(3):
        left, right = int(edges[n]), int(edges[n + 1])
        periods.append({"period": n + 1, "first_timestamp": int(frame.iloc[left].timestamp),
                        "last_timestamp": int(frame.iloc[right - 1].timestamp),
                        **{policy: summarize(simulate(frame, policy, left, right))
                           for policy in ("immediate_sell", "confirmed_sell_2", "protective_only")}})
    return {"report_type": "production_indicator_strategy_historical_replay",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": SYMBOL, "interval": INTERVAL, "candle_count": len(frame),
            "first_candle_timestamp": int(frame.iloc[0].timestamp),
            "last_candle_timestamp": int(frame.iloc[-1].timestamp),
            "signal_counts": {s: int((frame.signal == s).sum()) for s in ("BUY", "SELL", "WAIT")},
            "indicator_rules": "Exact main.py scoring: SMA9>SMA21 +1; SMA9<SMA21 -1; RSI 50-<70 +1; RSI >30-<=50 -1; BUY at score >=2; SELL at score <=-2.",
            "execution_method": "Closed-candle signal; entry/SELL exit at next candle open; OHLC stop/target on held candles; stop-first on dual-touch; final open position marked at final close.",
            "cost_assumptions": {"stake_usdt": STAKE_USDT, "fee_each_side": FEE_RATE,
                                 "slippage_each_side": SLIPPAGE_RATE,
                                 "note": "Estimated costs are not actual Bitget fills."},
            "policies": policies, "chronological_periods": periods,
            "research_only": True, "real_money_trading_eligible": False,
            "caveats": ["No exchange orders or account access.",
                        "Alternative exit policies are diagnostics, not production recommendations.",
                        "Three time slices are descriptive and do not prove future profitability.",
                        "OHLC data cannot reveal intrabar event order; stop-first is conservative."]}


def main():
    candles = fetch_candles()
    report = run_research(candles)
    report["candle_fetch"] = getattr(fetch_candles, "last_diagnostics", {})
    with open("indicator_strategy_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("PRODUCTION INDICATOR STRATEGY REPLAY — research only; no orders.")
    for policy, result in report["policies"].items():
        print(policy, "trades:", result["trade_count"], "gross:", round(result["gross_pnl_usdt"], 6),
              "costs:", round(result["estimated_costs_usdt"], 6), "net:", round(result["net_pnl_usdt"], 6))
    print("Report written: indicator_strategy_report.json")


if __name__ == "__main__":
    main()
