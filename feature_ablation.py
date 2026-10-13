"""Chronological feature-ablation comparison; offline research only, never places orders."""
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learning_model import FEATURE_COLUMNS, build_feature_frame, fetch_candles

BASELINE_FEATURES = [
    "return_1", "return_3", "return_6",
    "sma_gap_5", "sma_gap_10", "sma_gap_20",
    "volatility_10", "range_pct", "body_pct", "volume_change",
]
EXPANDED_FEATURES = list(FEATURE_COLUMNS)
BUY_THRESHOLD = 0.58
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
STAKE_USDT = 10.0


def _net_return(entry_close, next_close):
    entry = entry_close * (1 + SLIPPAGE_RATE)
    exit_price = next_close * (1 - SLIPPAGE_RATE)
    quantity = STAKE_USDT / entry
    proceeds = quantity * exit_price * (1 - FEE_RATE)
    return (proceeds - STAKE_USDT - STAKE_USDT * FEE_RATE) / STAKE_USDT


def evaluate_feature_set(usable, features, fold_count=3):
    """Expanding-window evaluation: every fold trains strictly before its test window."""
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    n = len(usable)
    first_test = int(n * 0.5)
    remaining = n - first_test
    if first_test < 120 or remaining < fold_count * 20:
        return {"available": False, "reason": "Not enough candles for chronological folds."}

    fold_size = remaining // fold_count
    predictions = []
    for fold in range(fold_count):
        test_start = first_test + fold * fold_size
        test_end = n if fold == fold_count - 1 else first_test + (fold + 1) * fold_size
        train = usable.iloc[:test_start]
        test = usable.iloc[test_start:test_end]
        if len(test) == 0 or train["target"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        model.fit(train[features], train["target"].astype(int))
        probabilities = model.predict_proba(test[features])[:, 1]
        actual = test["target"].astype(int).to_numpy()
        predicted = (probabilities >= 0.5).astype(int)
        buy_returns = [
            _net_return(float(usable.iloc[i]["close"]), float(usable.iloc[i + 1]["close"]))
            for i, probability in zip(test.index, probabilities)
            if probability >= BUY_THRESHOLD and i + 1 < len(usable)
        ]
        predictions.append({
            "fold": fold + 1,
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "test_accuracy": float(accuracy_score(actual, predicted)),
            "test_balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "brier_score": float(brier_score_loss(actual, probabilities)),
            "buy_signal_count": int(len(buy_returns)),
            "average_net_buy_return_percent": (
                float(np.mean(buy_returns) * 100) if buy_returns else None
            ),
            "fixed_stake_net_pnl_usdt": float(np.sum(buy_returns) * STAKE_USDT),
        })

    if not predictions:
        return {"available": False, "reason": "No valid chronological folds were produced."}
    total_test = sum(item["test_count"] for item in predictions)
    total_buys = sum(item["buy_signal_count"] for item in predictions)
    pnl = sum(item["fixed_stake_net_pnl_usdt"] for item in predictions)
    weighted_accuracy = sum(item["test_accuracy"] * item["test_count"] for item in predictions) / total_test
    weighted_balanced = sum(item["test_balanced_accuracy"] * item["test_count"] for item in predictions) / total_test
    weighted_brier = sum(item["brier_score"] * item["test_count"] for item in predictions) / total_test
    all_buy_returns = [
        item["average_net_buy_return_percent"] / 100 * item["buy_signal_count"]
        for item in predictions if item["average_net_buy_return_percent"] is not None
    ]
    average_buy_return = (
        float(sum(all_buy_returns) / total_buys * 100) if total_buys else None
    )
    return {
        "available": True,
        "features": features,
        "fold_count": len(predictions),
        "folds": predictions,
        "test_sample_count": total_test,
        "test_accuracy": float(weighted_accuracy),
        "test_balanced_accuracy": float(weighted_balanced),
        "brier_score": float(weighted_brier),
        "buy_signal_count": int(total_buys),
        "average_net_buy_return_percent": average_buy_return,
        "fixed_stake_net_pnl_usdt": float(pnl),
    }


def compare_feature_sets(candles, fold_count=3):
    frame = build_feature_frame(candles)
    usable = frame.dropna(subset=EXPANDED_FEATURES + ["target"]).reset_index(drop=True)
    if len(usable) < 180:
        return {
            "report_type": "chronological-feature-ablation",
            "available": False,
            "reason": f"Need at least 180 usable candles; got {len(usable)}.",
            "real_money_trading_eligible": False,
        }
    baseline = evaluate_feature_set(usable, BASELINE_FEATURES, fold_count)
    expanded = evaluate_feature_set(usable, EXPANDED_FEATURES, fold_count)
    comparison = {}
    if baseline.get("available") and expanded.get("available"):
        comparison = {
            "expanded_accuracy_improvement": expanded["test_accuracy"] - baseline["test_accuracy"],
            "expanded_balanced_accuracy_improvement": expanded["test_balanced_accuracy"] - baseline["test_balanced_accuracy"],
            "expanded_brier_score_change_lower_is_better": expanded["brier_score"] - baseline["brier_score"],
            "expanded_after_costs_pnl_improvement_usdt": expanded["fixed_stake_net_pnl_usdt"] - baseline["fixed_stake_net_pnl_usdt"],
            "expanded_average_net_buy_return_improvement_percentage_points": (
                expanded["average_net_buy_return_percent"] - baseline["average_net_buy_return_percent"]
                if expanded["average_net_buy_return_percent"] is not None
                and baseline["average_net_buy_return_percent"] is not None else None
            ),
            "expanded_model_wins_on_accuracy": expanded["test_accuracy"] > baseline["test_accuracy"],
            "expanded_model_wins_on_after_costs_pnl": expanded["fixed_stake_net_pnl_usdt"] > baseline["fixed_stake_net_pnl_usdt"],
        }
    return {
        "report_type": "chronological-feature-ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        "interval": "5m",
        "usable_candle_count": int(len(usable)),
        "method": "Expanding chronological training windows; test windows occur strictly after training. Fixed BUY threshold 0.58; estimated 0.10% fee and 0.05% slippage per side; fixed 10 USDT stake.",
        "baseline_features": baseline,
        "expanded_features": expanded,
        "comparison": comparison,
        "interpretation": "Research comparison only. Results are noisy, threshold is not tuned on these test folds, and positive results are not proof of future profitability.",
        "real_money_trading_eligible": False,
        "orders_sent": False,
    }


def main():
    report = compare_feature_sets(fetch_candles())
    with open("feature_ablation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("CHRONOLOGICAL FEATURE ABLATION — offline only, no orders.")
    print("Available:", report["available"])
    if report["available"]:
        print("Baseline accuracy:", round(report["baseline_features"]["test_accuracy"], 4))
        print("Expanded accuracy:", round(report["expanded_features"]["test_accuracy"], 4))
        print("Baseline net PnL (USDT):", round(report["baseline_features"]["fixed_stake_net_pnl_usdt"], 4))
        print("Expanded net PnL (USDT):", round(report["expanded_features"]["fixed_stake_net_pnl_usdt"], 4))
        print("Comparison:", report["comparison"])
    else:
        print("Reason:", report["reason"])
    print("Report written: feature_ablation_report.json")
    print("Real-money trading eligible:", report["real_money_trading_eligible"])


if __name__ == "__main__":
    main()
