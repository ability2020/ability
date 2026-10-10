import os
import time
import json
import hmac
import base64
import hashlib
import requests
from datetime import datetime, timezone

# ============================================================
# BITGET UTA V3 SPOT DEMO TRADING BOT
# ============================================================

BASE_URL = "https://api.bitget.com"

API_KEY = os.getenv("BITGET_API_KEY", "")
SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

SYMBOL = os.getenv("BITGET_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BITGET_INTERVAL", "5m")

# Keep this TRUE while testing.
DEMO_TRADING = True

# Bot risk controls
TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", "10"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "2"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3"))

# ------------------------------------------------------------
# Authentication
# ------------------------------------------------------------

def make_signature(timestamp, method, request_path, body=""):
    message = (
        str(timestamp)
        + method.upper()
        + request_path
        + body
    )

    digest = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).digest()

    return base64.b64encode(digest).decode()


def private_request(method, path, params=None):
    if not API_KEY or not SECRET_KEY or not PASSPHRASE:
        raise RuntimeError(
            "Missing Bitget API credentials. "
            "Add BITGET_API_KEY, BITGET_SECRET_KEY and "
            "BITGET_PASSPHRASE as environment variables."
        )

    timestamp = str(int(time.time() * 1000))

    body = ""
    url = BASE_URL + path

    if method.upper() == "GET":
        if params:
            query = "&".join(
                f"{key}={params[key]}"
                for key in sorted(params)
            )
            url += "?" + query
            request_path = path + "?" + query
        else:
            request_path = path
    else:
        body = json.dumps(
            params or {},
            separators=(",", ":")
        )
        request_path = path

    signature = make_signature(
        timestamp,
        method,
        request_path,
        body
    )

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }

    # Bitget demo trading
    if DEMO_TRADING:
        headers["paptrading"] = "1"

    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        data=body if method.upper() == "POST" else None,
        timeout=15
    )

    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError(f"Bitget returned non-JSON response (HTTP {response.status_code}).")

    if response.status_code >= 400:
        raise RuntimeError(
            f"Bitget HTTP {response.status_code}: "
            f"{data.get('msg', data)}"
        )

    if data.get("code") != "00000":
        raise RuntimeError(
            f"Bitget API error {data.get('code')}: "
            f"{data.get('msg', 'Unknown error')}"
        )

    return data


# ------------------------------------------------------------
# Account balances
# ------------------------------------------------------------

def get_account_assets():
    return private_request(
        "GET",
        "/api/v3/account/assets"
    )


def get_available_balance(coin):
    response = get_account_assets()
    assets = response.get("data", [])

    # Be defensive: API/account wrappers can return a list, a single
    # asset object, or JSON-encoded asset entries.
    if isinstance(assets, str):
        try:
            assets = json.loads(assets)
        except (TypeError, ValueError):
            print("Balance check: account data was text, not a parsed asset list.")
            return 0.0

    if isinstance(assets, dict):
        # Some account responses wrap assets in a nested list.
        assets = assets.get("assets", assets.get("list", [assets]))

    if not isinstance(assets, list):
        print("Balance check: unexpected account asset response; treating balance as zero.")
        return 0.0

    for asset in assets:
        if isinstance(asset, str):
            try:
                asset = json.loads(asset)
            except (TypeError, ValueError):
                # Never guess a balance from an unstructured string.
                continue

        if not isinstance(asset, dict):
            continue

        asset_coin = str(
            asset.get("coin", asset.get("asset", asset.get("currency", "")))
        ).upper()

        if asset_coin == coin.upper():
            available = asset.get(
                "available",
                asset.get("availableBalance", asset.get("availableAmount", "0"))
            )
            try:
                return float(available or 0)
            except (TypeError, ValueError):
                print(f"Balance check: invalid {coin} available-balance value; treating it as zero.")
                return 0.0

    return 0.0


# ------------------------------------------------------------
# Public market data
# ------------------------------------------------------------

def get_candles():
    params = {
        "category": "SPOT",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": "50"
    }

    response = requests.get(
        BASE_URL + "/api/v3/market/candles",
        params=params,
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    if data.get("code") != "00000":
        raise RuntimeError(
            f"Bitget candles error {data.get('code')}: "
            f"{data.get('msg', 'Unknown error')}"
        )

    return data["data"]


def get_ticker():
    params = {
        "category": "SPOT",
        "symbol": SYMBOL
    }

    response = requests.get(
        BASE_URL + "/api/v3/market/tickers",
        params=params,
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    if data.get("code") != "00000":
        raise RuntimeError(
            f"Bitget ticker error {data.get('code')}: "
            f"{data.get('msg', 'Unknown error')}"
        )

    if not data["data"]:
        raise RuntimeError("No ticker data returned.")

    return data["data"][0]


# ------------------------------------------------------------
# Indicator scoring (signal-only)
# ------------------------------------------------------------

def calculate_sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def calculate_rsi(values, period=14):
    if len(values) <= period:
        return None

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    recent = changes[-period:]
    gains = [change if change > 0 else 0.0 for change in recent]
    losses = [-change if change < 0 else 0.0 for change in recent]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def get_signal():
    candles = get_candles()

    # Bitget candle format: [timestamp, open, high, low, close, volume, turnover]
    if not isinstance(candles, list) or len(candles) < 22:
        return "WAIT", None, {"reason": "Not enough candle data"}

    try:
        closes = [float(candle[4]) for candle in candles if len(candle) > 4]
    except (TypeError, ValueError, IndexError):
        return "WAIT", None, {"reason": "Invalid candle data"}

    if len(closes) < 22:
        return "WAIT", None, {"reason": "Not enough valid closing prices"}

    # Bitget candle endpoints commonly return newest first; sort by timestamp
    # when timestamps are available so indicators use chronological closes.
    try:
        ordered = sorted(candles, key=lambda candle: int(candle[0]))
        closes = [float(candle[4]) for candle in ordered]
    except (TypeError, ValueError, IndexError):
        pass

    fast_sma = calculate_sma(closes, 9)
    slow_sma = calculate_sma(closes, 21)
    rsi = calculate_rsi(closes, 14)
    current_price = closes[-1]

    if fast_sma is None or slow_sma is None or rsi is None:
        return "WAIT", current_price, {"reason": "Indicators unavailable"}

    score = 0
    score += 1 if fast_sma > slow_sma else (-1 if fast_sma < slow_sma else 0)
    score += 1 if 50 <= rsi < 70 else (-1 if 30 < rsi <= 50 else 0)

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "WAIT"

    details = {
        "fast_sma": fast_sma,
        "slow_sma": slow_sma,
        "rsi_14": rsi,
        "indicator_score": score,
        "reason": "Signal only; no orders are sent by this version",
    }
    return signal, current_price, details


# ------------------------------------------------------------
# Risk engine (decision-only; never places orders)
# ------------------------------------------------------------

def evaluate_risk(signal, price, details, usdt_balance, btc_balance):
    """
    Convert an indicator signal into a guarded candidate decision.
    This function is intentionally decision-only: it never submits orders.
    """
    if not DEMO_TRADING:
        return "WAIT", "Demo mode is not enabled; trading is blocked."

    try:
        numeric_price = float(price)
        numeric_usdt = float(usdt_balance)
        numeric_btc = float(btc_balance)
        score = int(details.get("indicator_score"))
    except (TypeError, ValueError, AttributeError):
        return "WAIT", "Risk inputs are missing or invalid."

    # Reject NaN/infinity as well as negative values; malformed API data
    # must never be interpreted as permission to trade.
    import math
    if not all(math.isfinite(value) for value in (numeric_price, numeric_usdt, numeric_btc)):
        return "WAIT", "Risk inputs must be finite numbers."
    if numeric_price <= 0:
        return "WAIT", "Price must be positive."
    if numeric_usdt < 0 or numeric_btc < 0:
        return "WAIT", "Balances cannot be negative."
    if not math.isfinite(TRADE_AMOUNT_USDT) or TRADE_AMOUNT_USDT <= 0:
        return "WAIT", "Configured trade amount must be positive and finite."
    if not (0 < STOP_LOSS_PERCENT < 100):
        return "WAIT", "STOP_LOSS_PERCENT must be greater than 0 and less than 100."
    if not (TAKE_PROFIT_PERCENT > 0 and math.isfinite(TAKE_PROFIT_PERCENT)):
        return "WAIT", "TAKE_PROFIT_PERCENT must be positive and finite."

    if signal == "BUY":
        if score < 2:
            return "WAIT", "BUY signal did not meet the minimum indicator score."
        if numeric_usdt < TRADE_AMOUNT_USDT:
            return "WAIT", "Insufficient available USDT for the configured demo trade."
        return "BUY_CANDIDATE", "Demo-only candidate; order submission remains disabled."

    if signal == "SELL":
        if score > -2:
            return "WAIT", "SELL signal did not meet the minimum indicator score."
        if numeric_btc <= 0:
            return "WAIT", "No available BTC balance to consider selling."
        return "SELL_CANDIDATE", "Demo-only candidate; order submission remains disabled."

    return "WAIT", "No qualifying BUY or SELL signal."


# ------------------------------------------------------------
# Position exit checks (decision-only)
# ------------------------------------------------------------

def evaluate_position_exit(entry_price, current_price):
    """
    Return a decision-only stop-loss/take-profit result for an existing
    position. This function never places an order and does not persist
    entry prices between GitHub Actions runs.
    """
    try:
        entry = float(entry_price)
        current = float(current_price)
    except (TypeError, ValueError):
        return "WAIT", "Entry/current price is missing or invalid."

    import math
    if not math.isfinite(entry) or not math.isfinite(current) or entry <= 0 or current <= 0:
        return "WAIT", "Entry/current price must be positive finite numbers."
    if not (0 < STOP_LOSS_PERCENT < 100) or TAKE_PROFIT_PERCENT <= 0:
        return "WAIT", "Stop-loss/take-profit settings are invalid."

    stop_price = entry * (1 - STOP_LOSS_PERCENT / 100)
    target_price = entry * (1 + TAKE_PROFIT_PERCENT / 100)

    if current <= stop_price:
        return "STOP_LOSS_CANDIDATE", "Stop-loss threshold reached; no order submitted."
    if current >= target_price:
        return "TAKE_PROFIT_CANDIDATE", "Take-profit threshold reached; no order submitted."
    return "HOLD", "Position remains between stop-loss and take-profit thresholds."


def make_client_order_id(side, time_bucket=None):
    """
    Deterministic order ID helper for a time bucket. It supports future
    duplicate-request protection; order submission remains disabled.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    bucket = int(time.time() // 60) if time_bucket is None else int(time_bucket)
    return f"demo_{side}_{SYMBOL.lower()}_{bucket}"


# ------------------------------------------------------------
# Demo order
# ------------------------------------------------------------

def place_market_buy(price):
    # For a spot market BUY, Bitget uses quote currency
    # for qty. Therefore qty is USDT here.

    order = {
        "category": "SPOT",
        "symbol": SYMBOL,
        "side": "buy",
        "orderType": "market",
        "qty": str(TRADE_AMOUNT_USDT),
        "clientOid": make_client_order_id("buy")
    }

    return private_request(
        "POST",
        "/api/v3/trade/place-order",
        order
    )


def place_market_sell(quantity):
    # For a spot market SELL, qty is base currency.
    order = {
        "category": "SPOT",
        "symbol": SYMBOL,
        "side": "sell",
        "orderType": "market",
        "qty": str(quantity),
        "clientOid": make_client_order_id("sell")
    }

    return private_request(
        "POST",
        "/api/v3/trade/place-order",
        order
    )


# ------------------------------------------------------------
# Main loop: signal-only safety mode
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("BITGET UTA V3 SPOT BOT — SIGNAL-ONLY MODE")
    print("=" * 60)
    print("Symbol:", SYMBOL)
    print("Interval:", INTERVAL)
    print("Demo trading header:", DEMO_TRADING)
    print("ORDER PLACEMENT: DISABLED — safety tests only; no orders can be sent by main().")
    print("Stop-loss/take-profit checks are decision-only; position entry price is not persisted.")
    if TRADE_AMOUNT_USDT <= 0:
        raise RuntimeError("TRADE_AMOUNT_USDT must be positive.")
    print("=" * 60)

    while True:
        try:
            signal, price, details = get_signal()
            ticker = get_ticker()
            last_price = ticker.get("lastPrice")
            if last_price is None:
                raise RuntimeError("Ticker response did not contain lastPrice.")
            live_price = float(last_price)

            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{now}] {SYMBOL} price={live_price:.2f} signal={signal}")
            if "indicator_score" in details:
                print(
                    f"SMA9={details['fast_sma']:.4f} "
                    f"SMA21={details['slow_sma']:.4f} "
                    f"RSI14={details['rsi_14']:.2f} "
                    f"score={details['indicator_score']}"
                )
            else:
                print("Signal detail:", details.get("reason", "Indicators unavailable"))

            # Read balances for the risk decision, but never print the values.
            usdt_balance = get_available_balance("USDT")
            btc_balance = get_available_balance("BTC")
            decision, risk_reason = evaluate_risk(
                signal, live_price, details, usdt_balance, btc_balance
            )
            print(f"Risk decision: {decision} — {risk_reason}")
            print("SAFE MODE: no order was placed.")
            time.sleep(60)

        except Exception as error:
            print("ERROR:", error)
            time.sleep(30)


if __name__ == "__main__":
    main()
