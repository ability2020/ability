"""Validation-only model/threshold selection followed by a later untouched holdout.

Research only: uses public historical candles and never submits exchange orders.
"""
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learning_model import FEATURE_COLUMNS, INTERVAL, SYMBOL, build_feature_frame, fetch_candles

BASELINE_FEATURES = [
    "return_1", "return_3", "return_6",
    "sma_gap_5", "sma_gap_10", "sma_gap_20",
    "volatility_10", "range_pct", "body_pct", "volume_change",
]
CANDIDATE_THRESHOLDS = (0.52, 0.55, 0.58, 0.62, 0.65)
DEFAULT_THRESHOLD = 0.58
STAKE_USDT = 10.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
MIN_VALIDATION_TRADES = 5


def net_one_candle_return(entry_close, next_close):
    """Net fractional return on a fixed stake after estimated two-sided costs."""
    entry_execution = float(entry_close) * (1.0 + SLIPPAGE_RATE)
    exit_execution = float(next_close) * (1.0 - SLIPPAGE_RATE)
    quantity = STAKE_USDT / entry_execution
    exit_proceeds = quantity * exit_execution * (1.0 - FEE_RATE)
    return (exit_proceeds - STAKE_USDT - STAKE_USDT * FEE_RATE) / STAKE_USDT


def _fit_model(train, features):
    if train["target"].nunique() < 2:
        raise ValueError("Training slice must contain both target classes.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(train[features], train["target"].astype(int))
    return model


def _returns_for_rows(usable, rows):
    values = []
    for index in rows.index:
        row = usable.iloc[int(index)]
        next_close = row.get("next_close")
        if next_close is not None and np.isfinite(float(next_close)):
            values.append(net_one_candle_return(float(row["close"]), float(next_close)))
        else:
            values.append(None)
    return values


def _candidate_summary(probabilities, returns):
    summaries = []
    for threshold in CANDIDATE_THRESHOLDS:
        selected = [
            float(ret) for probability, ret in zip(probabilities, returns)
            if ret is not None and float(probability) >= threshold
        ]
        summaries.append({
            "threshold": float(threshold),
            "trade_count": len(selected),
            "average_net_return_percent": float(np.mean(selected) * 100.0) if selected else None,
            "fixed_stake_net_pnl_usdt": float(np.sum(selected) * STAKE_USDT),
        })
    return summaries


def evaluate_model_selection(candles):
    frame = build_feature_frame(candles)
    usable = frame.assign(next_close=frame["close"].shift(-1)).dropna(
        subset=FEATURE_COLUMNS + ["target", "next_close"]
    ).reset_index(drop=True)
    n = len(usable)
    if n < 240:
        return {
            "report_type": "validation-selected-model-cost-aware-holdout",
            "available": False,
            "reason": f"Need at least 240 usable labelled candles; got {n}.",
            "real_money_trading_eligible": False,
            "orders_sent": False,
        }

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    train = usable.iloc[:train_end]
    validation = usable.iloc[train_end:validation_end]
    test = usable.iloc[validation_end:]
    if len(validation) < 20 or len(test) < 20 or train["target"].nunique() < 2:
        return {
            "report_type": "validation-selected-model-cost-aware-holdout",
            "available": False,
            "reason": "Chronological train/validation/test slices are too small or training has one class.",
            "real_money_trading_eligible": False,
            "orders_sent": False,
        }

    validation_returns = _returns_for_rows(usable, validation)
    market_validation = [float(v) for v in validation_returns if v is not None]
    market_average = float(np.mean(market_validation)) if market_validation else 0.0
    candidates = []
    model_validation = {}
    for model_name, features in (
        ("baseline_features", BASELINE_FEATURES),
        ("expanded_features", FEATURE_COLUMNS),
    ):
        model = _fit_model(train, features)
        probabilities = model.predict_proba(validation[features])[:, 1]
        summaries = _candidate_summary(probabilities, validation_returns)
        model_validation[model_name] = {
            "thresholds": summaries,
            "validation_accuracy_at_0_5": float(accuracy_score(
                validation["target"].astype(int), (probabilities >= 0.5).astype(int)
            )),
        }
        for item in summaries:
            avg = item["average_net_return_percent"]
            eligible = (
                item["trade_count"] >= MIN_VALIDATION_TRADES
                and avg is not None
                and avg > 0.0
                and avg / 100.0 > market_average
            )
            candidates.append({
                "model": model_name,
                "threshold": item["threshold"],
                "validation_trade_count": item["trade_count"],
                "validation_average_net_return_percent": avg,
                "validation_net_pnl_usdt": item["fixed_stake_net_pnl_usdt"],
                "beats_validation_market_after_costs": bool(
                    avg is not None and avg / 100.0 > market_average
                ),
                "eligible_for_selection": bool(eligible),
            })

    eligible_candidates = [item for item in candidates if item["eligible_for_selection"]]
    if eligible_candidates:
        # Select only on validation data. Deterministic tie-break favours simpler baseline.
        eligible_candidates.sort(key=lambda item: (
            item["validation_net_pnl_usdt"],
            item["validation_average_net_return_percent"],
            item["model"] == "baseline_features",
        ), reverse=True)
        selected = eligible_candidates[0]
        selection_reason = "Best positive after-cost validation PnL among candidates beating the same-window market baseline and meeting minimum trade count."
    else:
        selected = {
            "model": "baseline_features",
            "threshold": DEFAULT_THRESHOLD,
            "validation_trade_count": 0,
            "validation_average_net_return_percent": None,
            "validation_net_pnl_usdt": 0.0,
            "beats_validation_market_after_costs": False,
            "eligible_for_selection": False,
        }
        selection_reason = "No model/threshold met validation profitability, benchmark, and sample-size rules; conservative baseline/default threshold retained for holdout diagnosis only."

    selected_features = BASELINE_FEATURES if selected["model"] == "baseline_features" else FEATURE_COLUMNS
    # Refit only on train+validation after selection is locked. Final test is untouched.
    fit_data = usable.iloc[:validation_end]
    final_model = _fit_model(fit_data, selected_features)
    test_probabilities = final_model.predict_proba(test[selected_features])[:, 1]
    test_returns = _returns_for_rows(usable, test)
    selected_test_returns = [
        float(ret) for probability, ret in zip(test_probabilities, test_returns)
        if ret is not None and float(probability) >= float(selected["threshold"])
    ]
    market_test_returns = [float(v) for v in test_returns if v is not None]
    test_actual = test["target"].astype(int).to_numpy()
    test_predictions = (test_probabilities >= 0.5).astype(int)

    # Diagnostics only: the expanded model is not allowed to change the selection
    # after the final holdout has been seen.
    holdout_diagnostics = {}
    for model_name, features in (
        ("baseline_features", BASELINE_FEATURES),
        ("expanded_features", FEATURE_COLUMNS),
    ):
        diagnostic_model = _fit_model(fit_data, features)
        probs = diagnostic_model.predict_proba(test[features])[:, 1]
        rets = _returns_for_rows(usable, test)
        selected_rets = [
            float(ret) for probability, ret in zip(probs, rets)
            if ret is not None and float(probability) >= DEFAULT_THRESHOLD
        ]
        holdout_diagnostics[model_name] = {
            "accuracy": float(accuracy_score(test_actual, (probs >= 0.5).astype(int))),
            "balanced_accuracy": float(balanced_accuracy_score(test_actual, (probs >= 0.5).astype(int))),
            "fixed_threshold": DEFAULT_THRESHOLD,
            "buy_count": len(selected_rets),
            "average_net_buy_return_percent": float(np.mean(selected_rets) * 100.0) if selected_rets else None,
            "fixed_stake_net_pnl_usdt": float(np.sum(selected_rets) * STAKE_USDT),
            "selection_use": "diagnostic only; not used to change selected model or threshold",
        }

    avg_selected = float(np.mean(selected_test_returns) * 100.0) if selected_test_returns else None
    pnl_selected = float(np.sum(selected_test_returns) * STAKE_USDT)
    avg_market_test = float(np.mean(market_test_returns) * 100.0) if market_test_returns else None
    return {
        "report_type": "validation-selected-model-cost-aware-holdout",
        "available": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "usable_labelled_candle_count": n,
        "split": {
            "training_count": len(train),
            "validation_count": len(validation),
            "untouched_test_count": len(test),
            "chronological_order": "train -> validation -> untouched test",
            "test_used_for_selection": False,
        },
        "cost_assumptions": {
            "stake_usdt": STAKE_USDT,
            "fee_rate_each_side": FEE_RATE,
            "slippage_rate_each_side": SLIPPAGE_RATE,
            "method": "Each hypothetical BUY is closed one candle later; costs are estimated on entry and exit.",
        },
        "validation_market_average_net_return_percent": market_average * 100.0,
        "validation_model_diagnostics": model_validation,
        "selection_candidates": candidates,
        "selected_model": selected["model"],
        "selected_buy_threshold": float(selected["threshold"]),
        "selection_reason": selection_reason,
        "holdout": {
            "prediction_count": len(test),
            "directional_accuracy": float(accuracy_score(test_actual, test_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_actual, test_predictions)),
            "buy_count": len(selected_test_returns),
            "average_net_buy_return_percent": avg_selected,
            "fixed_stake_net_pnl_usdt": pnl_selected,
            "market_average_net_return_percent": avg_market_test,
            "beats_market_after_costs": bool(
                avg_selected is not None and avg_market_test is not None
                and avg_selected > avg_market_test
            ),
            "positive_net_pnl": pnl_selected > 0.0,
            "model_diagnostics_at_fixed_threshold": holdout_diagnostics,
            "note": "This final test period is reported after selection and must not be used to retune this same report.",
        },
        "multi_period_gate_remains_authoritative": True,
        "real_money_trading_eligible": False,
        "orders_sent": False,
    }


def main():
    report = evaluate_model_selection(fetch_candles())
    with open("model_selection_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("COST-AWARE MODEL SELECTION — offline only, no orders.")
    print("Available:", report["available"])
    if report["available"]:
        print("Selected model:", report["selected_model"])
        print("Selected threshold:", report["selected_buy_threshold"])
        print("Holdout net PnL (USDT):", round(report["holdout"]["fixed_stake_net_pnl_usdt"], 6))
        print("Holdout beats market after costs:", report["holdout"]["beats_market_after_costs"])
    else:
        print("Reason:", report["reason"])
    print("Real-money trading eligible:", report["real_money_trading_eligible"])
    print("Orders sent:", report["orders_sent"])


if __name__ == "__main__":
    main()
