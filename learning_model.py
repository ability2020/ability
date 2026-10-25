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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://api.bitget.com"
SYMBOL = os.getenv("BITGET_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BITGET_INTERVAL", "5m")
CANDLE_LIMIT = 1000  # Larger historical sample for more informative research tests
MIN_SAMPLES = 120
MODEL_VERSION = "logistic-direction-v2-expanded-features"

FEATURE_COLUMNS = [
    "return_1", "return_3", "return_6",
    "sma_gap_5", "sma_gap_10", "sma_gap_20",
    "volatility_10", "range_pct", "body_pct", "volume_change",
    "rsi_14", "macd_hist_pct", "atr_pct", "volume_zscore_20",
]


def fetch_candles(lookback_days=None, end_offset_days=None):
    """Fetch and validate Spot candles in overlapping-safe 30-day windows.

    Bitget's v3 historical candles endpoint returns at most 100 rows per request.
    Each 30-day window is paginated backwards using its oldest returned timestamp,
    deduplicated, and checked for minimum time coverage. This function only reads
    public market data.
    """
    import math
    import time

    if lookback_days is None:
        lookback_days = int(os.getenv("CANDLE_LOOKBACK_DAYS", "60"))
    if not isinstance(lookback_days, int) or not 1 <= lookback_days <= 89:
        raise ValueError("CANDLE_LOOKBACK_DAYS must be an integer from 1 to 89.")
    if end_offset_days is None:
        end_offset_days = int(os.getenv("CANDLE_END_OFFSET_DAYS", "0"))
    if not isinstance(end_offset_days, int) or not 0 <= end_offset_days <= 365:
        raise ValueError("CANDLE_END_OFFSET_DAYS must be an integer from 0 to 365.")

    interval_ms_by_name = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1H": 3_600_000, "4H": 14_400_000,
        "6H": 21_600_000, "12H": 43_200_000, "1D": 86_400_000,
    }
    if INTERVAL not in interval_ms_by_name:
        raise ValueError(f"Unsupported candle interval for pagination: {INTERVAL}")

    interval_ms = interval_ms_by_name[INTERVAL]
    page_limit = 100
    now_ms = int(time.time() * 1000)
    latest_aligned_ms = (now_ms // interval_ms) * interval_ms
    end_ms = latest_aligned_ms - end_offset_days * 24 * 60 * 60 * 1000
    start_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000
    window_ms = 30 * 24 * 60 * 60 * 1000
    candles_by_timestamp = {}

    # Query each bounded window separately. Keep the boundary candle in both
    # adjacent requests if the API rounds timestamps; timestamp deduplication
    # below makes that safe.
    window_start = start_ms
    while window_start < end_ms:
        window_end = min(window_start + window_ms, end_ms)
        cursor_end_ms = window_end
        window_timestamps = set()
        max_pages = int(math.ceil((window_end - window_start) / (interval_ms * (page_limit - 1)))) + 4

        for _ in range(max_pages):
            params = {
                "category": "SPOT",
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": str(window_start),
                "endTime": str(cursor_end_ms),
                "limit": str(page_limit),
            }
            # Historical backfills can make many requests. Pace them and retry
            # transient rate limits instead of aborting the entire research run.
            for attempt in range(5):
                response = requests.get(
                    BASE_URL + "/api/v3/market/history-candles",
                    params=params,
                    timeout=20,
                )
                if response.status_code not in (429, 418):
                    response.raise_for_status()
                    break
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
                except (TypeError, ValueError):
                    delay = min(2 ** attempt, 30)
                print(f"Bitget rate limit (HTTP {response.status_code}); retrying in {delay:.1f}s.")
                time.sleep(max(1.0, delay))
            time.sleep(0.2)
            payload = response.json()
            if payload.get("code") != "00000":
                raise RuntimeError(
                    f"Bitget candle request failed ({payload.get('code')}): "
                    f"{payload.get('msg', 'unknown error')}"
                )
            page = payload.get("data", [])
            if not isinstance(page, list) or not page:
                break

            valid_timestamps = []
            for candle in page:
                if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                    continue
                try:
                    timestamp = int(candle[0])
                except (TypeError, ValueError):
                    continue
                if window_start <= timestamp <= window_end:
                    candles_by_timestamp[timestamp] = candle
                    window_timestamps.add(timestamp)
                    valid_timestamps.append(timestamp)

            if not valid_timestamps:
                break
            oldest_timestamp = min(valid_timestamps)
            if oldest_timestamp <= window_start:
                break
            next_cursor = oldest_timestamp - 1
            if next_cursor >= cursor_end_ms:
                raise RuntimeError("Bitget candle pagination made no backward progress.")
            cursor_end_ms = next_cursor

        if not window_timestamps:
            raise RuntimeError(
                f"Bitget returned no valid candles for requested window "
                f"{datetime.fromtimestamp(window_start / 1000, timezone.utc).isoformat()} "
                f"to {datetime.fromtimestamp(window_end / 1000, timezone.utc).isoformat()}."
            )
        earliest = min(window_timestamps)
        latest = max(window_timestamps)
        if earliest > window_start + 2 * interval_ms:
            raise RuntimeError(
                f"Historical candle coverage gap at window start: expected near "
                f"{window_start}, got {earliest}."
            )
        if latest < window_end - 2 * interval_ms:
            raise RuntimeError(
                f"Historical candle coverage gap at window end: expected near "
                f"{window_end}, got {latest}."
            )
        window_start = window_end

    candles = [candles_by_timestamp[key] for key in sorted(candles_by_timestamp)]
    expected_count = max(1, int((end_ms - start_ms) // interval_ms))
    minimum_expected_count = int(expected_count * 0.95)
    if len(candles) < minimum_expected_count:
        raise RuntimeError(
            f"Incomplete candle history: requested {lookback_days} days at {INTERVAL}, "
            f"expected at least {minimum_expected_count} candles but received {len(candles)}."
        )
    print(
        f"Historical coverage verified: {len(candles)} candles, "
        f"{datetime.fromtimestamp(int(candles[0][0]) / 1000, timezone.utc).isoformat()} "
        f"to {datetime.fromtimestamp(int(candles[-1][0]) / 1000, timezone.utc).isoformat()} "
        f"(requested {lookback_days} days)."
    )
    return candles


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

    # Expanded indicators use only the current and earlier candles.
    delta = close.diff()
    average_gain = delta.clip(lower=0).rolling(14).mean()
    average_loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    frame["rsi_14"] = (100.0 - (100.0 / (1.0 + relative_strength))).fillna(50.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_hist_pct"] = (macd - macd_signal) / close

    previous_close = close.shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    frame["atr_pct"] = true_range.rolling(14).mean() / close

    volume_mean = frame["volume"].rolling(20).mean()
    volume_std = frame["volume"].rolling(20).std().replace(0, np.nan)
    frame["volume_zscore_20"] = ((frame["volume"] - volume_mean) / volume_std).fillna(0.0)

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

    if train["target"].nunique() < 2:
        raise ValueError("Training window contains only one direction class; model training is unsafe.")

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
    baseline_predictions = np.full(len(actual), majority_class, dtype=int)
    baseline_accuracy = float(accuracy_score(actual, baseline_predictions))
    baseline_balanced_accuracy = float(
        balanced_accuracy_score(actual, baseline_predictions)
    )
    probability_brier_score = float(brier_score_loss(actual, test_probabilities))
    buy_threshold_predictions = (test_probabilities >= 0.58).astype(int)
    buy_threshold_precision = float(
        precision_score(actual, buy_threshold_predictions, zero_division=0)
    )
    buy_threshold_recall = float(
        recall_score(actual, buy_threshold_predictions, zero_division=0)
    )
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in confusion_matrix(actual, test_predictions, labels=[0, 1]).ravel()
    )

    # Latest candle's next-candle label is unknown; prediction is for the next candle.
    if latest[FEATURE_COLUMNS].isna().to_numpy().any():
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
        "majority_baseline_balanced_accuracy": baseline_balanced_accuracy,
        "test_beats_baseline": bool(accuracy > baseline_accuracy),
        "probability_brier_score": probability_brier_score,
        "brier_score_note": "Lower is better; 0 is perfect and 0.25 is the reference score for constant 0.5 probabilities on a balanced binary target.",
        "buy_threshold_for_diagnostics": 0.58,
        "buy_threshold_precision": buy_threshold_precision,
        "buy_threshold_recall": buy_threshold_recall,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
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
