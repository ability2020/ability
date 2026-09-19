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

    response.raise_for_status()
    return response.json()


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
        raise RuntimeError(data)

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
        raise RuntimeError(data)

    if not data["data"]:
        raise RuntimeError("No ticker data returned.")

    return data["data"][0]


# ------------------------------------------------------------
# Simple trend calculation
# ------------------------------------------------------------

def calculate_sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def get_signal():
    candles = get_candles()

    # Bitget candle format:
    # [timestamp, open, high, low, close, volume, turnover]

    closes = [
        float(candle[4])
        for candle in candles
    ]

    if len(closes) < 21:
        return "WAIT", None

    fast_sma = calculate_sma(closes, 9)
    slow_sma = calculate_sma(closes, 21)

    current_price = closes[-1]

    if fast_sma > slow_sma:
        return "BUY", current_price

    if fast_sma < slow_sma:
        return "SELL", current_price

    return "WAIT", current_price


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
        "clientOid": "demo_bot_buy_" + str(int(time.time() * 1000))
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
        "clientOid": "demo_bot_sell_" + str(int(time.time() * 1000))
    }

    return private_request(
        "POST",
        "/api/v3/trade/place-order",
        order
    )


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("BITGET UTA V3 SPOT DEMO BOT")
    print("=" * 60)
    print("Symbol:", SYMBOL)
    print("Interval:", INTERVAL)
    print("Demo trading:", DEMO_TRADING)
    print("Trade amount:", TRADE_AMOUNT_USDT, "USDT")
    print("=" * 60)

    if DEMO_TRADING:
        print("SAFE MODE: Bitget demo trading is enabled.")

    last_signal = None

    while True:
        try:
            signal, price = get_signal()
            ticker = get_ticker()

            live_price = float(ticker["lastPrice"])

            now = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

            print(
                f"[{now}] "
                f"{SYMBOL} price={live_price:.2f} "
                f"signal={signal}"
            )

            # ------------------------------------------------
            # IMPORTANT:
            # We only place a demo BUY when the signal changes
            # to BUY. This prevents repeated orders every loop.
            # ------------------------------------------------

            if signal == "BUY" and last_signal != "BUY":
                print("BUY signal detected.")

                if DEMO_TRADING:
                    result = place_market_buy(live_price)
                    print("DEMO BUY RESPONSE:")
                    print(json.dumps(result, indent=2))
                else:
                    print("LIVE trading disabled.")

            elif signal == "SELL":
                print("SELL signal detected.")

                # We do not automatically sell yet because the
                # bot must first know the actual BTC balance.
                print("Automatic SELL is currently disabled.")

            last_signal = signal

            # Check approximately once per minute.
            time.sleep(60)

        except Exception as error:
            print("ERROR:", error)
            time.sleep(30)


if __name__ == "__main__":
    main()
