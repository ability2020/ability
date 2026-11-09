"""Paper-only workflow cadence and closed-candle freshness diagnostics."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path(os.getenv("MONITOR_STATE_FILE", "monitor_state.json"))
SIGNAL_PATH = Path(os.getenv("PAPER_SIGNAL_FILE", "paper_signal.json"))
REPORT_PATH = Path(os.getenv("MONITOR_REPORT_FILE", "monitoring_report.json"))
EXPECTED_INTERVAL_SECONDS = int(os.getenv("EXPECTED_INTERVAL_SECONDS", "900"))


def build_monitoring_report(signal, previous_state, observed_at):
    """Build cadence diagnostics; observed_at is script observation time, not scheduler start."""
    last_observed = previous_state.get("last_observed_at_utc")
    gap_seconds = None
    if last_observed:
        try:
            prior = datetime.fromisoformat(last_observed.replace("Z", "+00:00"))
            gap_seconds = max(0.0, (observed_at - prior).total_seconds())
        except (TypeError, ValueError):
            gap_seconds = None

    candle_ms = signal.get("last_closed_candle_timestamp")
    candle_age_seconds = None
    if candle_ms is not None:
        try:
            candle_age_seconds = max(0.0, observed_at.timestamp() - float(candle_ms) / 1000.0)
        except (TypeError, ValueError):
            candle_age_seconds = None

    return {
        "observed_at_utc": observed_at.isoformat(),
        "observation_definition": "Timestamp when this diagnostic script ran; not the exact GitHub scheduler start time.",
        "expected_interval_seconds": EXPECTED_INTERVAL_SECONDS,
        "seconds_since_previous_observation": gap_seconds,
        "cadence_status": (
            "FIRST_OBSERVATION" if gap_seconds is None else
            "ON_OR_NEAR_SCHEDULE" if gap_seconds <= EXPECTED_INTERVAL_SECONDS * 1.5 else
            "DELAYED_OR_MISSED_INTERVAL"
        ),
        "latest_closed_candle_timestamp_ms": candle_ms,
        "latest_closed_candle_age_seconds": candle_age_seconds,
        "closed_candle_count": signal.get("closed_candle_count"),
        "signal_record_available": bool(signal),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_number": os.getenv("GITHUB_RUN_NUMBER"),
        "github_event_name": os.getenv("GITHUB_EVENT_NAME"),
        "workflow_schedule": os.getenv("GITHUB_EVENT_SCHEDULE"),
        "state_restored": bool(last_observed),
        "note": "Paper monitoring only. This measures intervals between script observations, not guaranteed scheduler delivery or exchange execution.",
    }


def main():
    try:
        with SIGNAL_PATH.open("r", encoding="utf-8") as handle:
            signal = json.load(handle)
        if not isinstance(signal, dict):
            signal = {}
    except (OSError, json.JSONDecodeError):
        signal = {}

    try:
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            previous_state = json.load(handle)
        if not isinstance(previous_state, dict):
            previous_state = {}
    except (OSError, json.JSONDecodeError):
        previous_state = {}

    observed_at = datetime.now(timezone.utc)
    report = build_monitoring_report(signal, previous_state, observed_at)
    STATE_PATH.write_text(json.dumps({
        "version": 1,
        "last_observed_at_utc": observed_at.isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Monitoring cadence status:", report["cadence_status"])
    print("Seconds since previous observation:", report["seconds_since_previous_observation"])
    print("Latest closed candle age (seconds):", report["latest_closed_candle_age_seconds"])
    print("State restored:", report["state_restored"])


if __name__ == "__main__":
    main()
