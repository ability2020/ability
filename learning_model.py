"""Train and evaluate a cautious next-candle direction model from public Bitget Spot candles.

This module never reads account balances and never submits orders. It retrains on
historical candles each run, uses a chronological holdout, and emits ml_report.json.
The model is experimental; its signal is reported separately and does not control
the paper trader until it has been validated over a larger sample.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://api.bitget.com"
SYMBOL = os.getenv("BITGET_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BITGET_INTERVAL", "5m")
CANDLE_LIMIT = 500
MIN_SAMPLES = 120
MODEL_VERSION = "logistic-direction-v1"

FEATURE_COLUMNS = [
    "return_1", "return_3", "return_6",
    "sma_gap_5", "sma_gap_10", "sma_gap_20",
    "volatility_10", "range_pct", "body_pct", "volume_change",
]


def fetch_candles():
    response = requests.get(
        BASE_URL + "/api/v3/market/candles",
        params={
            "category": "SPOT",
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": str(CANDLE_LIMIT),
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "00000":
        raise RuntimeError(
            f"Bitget candle request failed ({payload.get('code')}): "
            f"{payload.get('msg', 'unknown error')}"
        )
    return payload.get("data", [])


def build_feature_frame(candles):
    """Convert Bitget [timestamp, open, high, low, close, volume, turnover] rows."""
    rows = []
    for candle in candles or []:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue
        try:
            timestamp = int(candle[0])
            open_price, high, low, close, volume = map(float, candle[1:6])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (open_price, high, low, close, volume)):
            continue
        if min(open_price, high, low, close) <= 0 or volume < 0:
            continue
        rows.append({
            "timestamp": timestamp, "open": open_price, "high": high,
            "low": low, "close": close, "volume": volume,
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "close", *FEATURE_COLUMNS, "target"])

    frame = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    close = frame["close"]
    frame["return_1"] = close.pct_change(1)
    frame["return_3"] = close.pct_change(3)
    frame["return_6"] = close.pct_change(6)
    for period in (5, 10, 20):
        frame[f"sma_gap_{period}"] = close / close.rolling(period).mean() - 1.0
    frame["volatility_10"] = frame["return_1"].rolling(10).std()
    frame["range_pct"] = (frame["high"] - frame["low"]) / close
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"]
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    # Label at candle t is whether candle t+1 closes above candle t.
    frame["target"] = (close.shift(-1) > close).astype(float)
    frame.loc[close.shift(-1).isna(), "target"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


def train_and_evaluate(candles):
    frame = build_feature_frame(candles)
    if len(frame) < 1:
        raise ValueError("No valid market candles were available.")
    latest = frame.iloc[[-1]]
    labelled = frame.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    if len(labelled) < MIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_SAMPLES} usable labelled candles; got {len(labelled)}."
        )

    split = int(len(labelled) * 0.8)
    if split < 80 or len(labelled) - split < 20:
        raise ValueError("Not enough samples for a reliable chronological holdout.")
    train = labelled.iloc[:split]
    test = labelled.iloc[split:]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
    test_probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    actual = test["target"].astype(int).to_numpy()
    accuracy = float(accuracy_score(actual, test_predictions))
    balanced_accuracy = float(balanced_accuracy_score(actual, test_predictions))
    majority_class = int(train["target"].mean() >= 0.5)
    baseline_accuracy = float(np.mean(actual == majority_class))

    # Latest candle's next-candle label is unknown; prediction is for the next candle.
    if latest[FEATURE_COLUMNS].isna().any(axis=None):
        latest_probability = None
        latest_signal = "WAIT"
        signal_reason = "Latest candle features are incomplete."
    else:
        latest_probability = float(model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])
        if latest_probability >= 0.58:
            latest_signal = "BUY"
            signal_reason = "Experimental model probability crossed the BUY threshold."
        elif latest_probability <= 0.42:
            latest_signal = "SELL"
            signal_reason = "Experimental model probability crossed the SELL threshold."
        else:
            latest_signal = "WAIT"
            signal_reason = "Model probability is too uncertain; no directional signal."

    return {
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candle_count": int(len(frame)),
        "labelled_sample_count": int(len(labelled)),
        "train_sample_count": int(len(train)),
        "test_sample_count": int(len(test)),
        "test_accuracy": accuracy,
        "test_balanced_accuracy": balanced_accuracy,
        "majority_baseline_accuracy": baseline_accuracy,
        "test_beats_baseline": bool(accuracy > baseline_accuracy),
        "latest_candle_timestamp": int(latest.iloc[0]["timestamp"]),
        "latest_close": float(latest.iloc[0]["close"]),
        "next_candle_up_probability": latest_probability,
        "model_signal": latest_signal,
        "signal_reason": signal_reason,
        "evaluation_method": "Chronological 80/20 holdout; features use current/past candles only.",
        "safety_note": (
            "Experimental research output only. Model signal does not control paper trades "
            "or submit exchange orders. Holdout results are not proof of profitability; "
            "fees, slippage and regime changes are not fully modelled."
        ),
    }


def main():
    candles = fetch_candles()
    result = train_and_evaluate(candles)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open("ml_report.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("EXPERIMENTAL MACHINE-LEARNING TRAINING — no orders can be sent.")
    print("Model:", result["model_version"])
    print("Usable labelled candles:", result["labelled_sample_count"])
    print("Train/test samples:", result["train_sample_count"], "/", result["test_sample_count"])
    print("Holdout accuracy:", round(result["test_accuracy"], 4))
    print("Majority baseline accuracy:", round(result["majority_baseline_accuracy"], 4))
    print("Beats baseline:", result["test_beats_baseline"])
    print("Next-candle up probability:", result["next_candle_up_probability"])
    print("Experimental model signal:", result["model_signal"])
    print("Report written: ml_report.json")
    print("This model signal does not control the paper simulator.")


if __name__ == "__main__":
    main()
