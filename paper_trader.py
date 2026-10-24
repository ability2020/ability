"""Paper-trading simulator. This module never calls Bitget order endpoints."""
import json
import math
import os
import tempfile

PAPER_STATE_FILE = os.getenv("PAPER_STATE_FILE", "paper_state.json")


def default_state():
    return {"version": 1, "position": None, "closed_trades": [], "realized_pnl_usdt": 0.0}


def validate_state(state):
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Invalid paper state version.")
    position = state.get("position")
    if position is not None:
        if not isinstance(position, dict):
            raise ValueError("Paper position must be an object or null.")
        try:
            entry = float(position["entry_price"])
            quantity = float(position["quantity"])
            invested = float(position["invested_usdt"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Paper position fields are invalid.") from error
        if not all(math.isfinite(v) and v > 0 for v in (entry, quantity, invested)):
            raise ValueError("Paper position values must be positive finite numbers.")
    trades = state.get("closed_trades", [])
    if not isinstance(trades, list) or any(not isinstance(t, dict) for t in trades):
        raise ValueError("Paper closed trades must be a list of objects.")
    try:
        pnl = float(state.get("realized_pnl_usdt", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("Paper realized PnL is invalid.") from error
    if not math.isfinite(pnl):
        raise ValueError("Paper realized PnL must be finite.")
    return {
        "version": 1,
        "position": position,
        "closed_trades": trades[-200:],
        "realized_pnl_usdt": pnl,
    }


def load_state(path=PAPER_STATE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return validate_state(json.load(handle))
    except FileNotFoundError:
        return default_state()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Paper state cannot be loaded safely: {error}") from error


def save_state(state, path=PAPER_STATE_FILE):
    clean = validate_state(state)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
            json.dump(clean, handle, separators=(",", ":"), sort_keys=True, allow_nan=False)
            handle.write("\n")
            temp_path = handle.name
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)



def risk_gated_paper_signal(signal, risk_decision):
    """Gate hypothetical BUY entries on the risk engine; preserve paper exits.

    The risk engine checks real account balances, so SELL is still allowed to
    close an existing hypothetical paper position even when the real account
    has no BTC. No exchange order is submitted by this helper.
    """
    if signal not in ("BUY", "SELL", "WAIT"):
        return "WAIT"
    if signal == "BUY" and risk_decision != "BUY_CANDIDATE":
        return "WAIT"
    return signal


def process_cycle(state, signal, price, trade_amount_usdt=10.0,
                  stop_loss_percent=2.0, take_profit_percent=3.0):
    """Update hypothetical state only; never makes a network request."""
    clean = validate_state(state)
    try:
        price = float(price)
        amount = float(trade_amount_usdt)
        stop = float(stop_loss_percent)
        target = float(take_profit_percent)
    except (TypeError, ValueError) as error:
        raise ValueError("Paper trading inputs must be numeric.") from error
    if not all(math.isfinite(v) for v in (price, amount, stop, target)):
        raise ValueError("Paper trading inputs must be finite.")
    if price <= 0 or amount <= 0 or not (0 < stop < 100) or target <= 0:
        raise ValueError("Paper trading price, amount, or risk settings are invalid.")
    if signal not in ("BUY", "SELL", "WAIT"):
        raise ValueError("Signal must be BUY, SELL, or WAIT.")

    position = clean["position"]
    event = "HOLD_CASH" if position is None else "HOLD_POSITION"
    reason = "No qualifying simulated action."
    if position is None and signal == "BUY":
        quantity = amount / price
        clean["position"] = {
            "entry_price": price,
            "quantity": quantity,
            "invested_usdt": amount,
        }
        event, reason = "PAPER_BUY", "Opened hypothetical position; no exchange order sent."
    elif position is not None:
        entry = float(position["entry_price"])
        quantity = float(position["quantity"])
        exit_reason = None
        if price <= entry * (1 - stop / 100):
            exit_reason = "STOP_LOSS"
        elif price >= entry * (1 + target / 100):
            exit_reason = "TAKE_PROFIT"
        elif signal == "SELL":
            exit_reason = "SELL_SIGNAL"
        if exit_reason:
            pnl = (price - entry) * quantity
            trade = {
                "entry_price": entry,
                "exit_price": price,
                "quantity": quantity,
                "pnl_usdt": pnl,
                "exit_reason": exit_reason,
            }
            clean["closed_trades"] = (clean["closed_trades"] + [trade])[-200:]
            clean["realized_pnl_usdt"] += pnl
            clean["position"] = None
            event, reason = "PAPER_SELL", f"Closed hypothetical position: {exit_reason}; no exchange order sent."
    return validate_state(clean), event, reason
