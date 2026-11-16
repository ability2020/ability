"""Paper-only workflow cadence and closed-candle freshness diagnostics."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path(os.getenv("MONITOR_STATE_FILE", "monitor_state.json"))
SIGNAL_PATH = Path(os.getenv("PAPER_SIGNAL_FILE", "paper_signal.json"))
REPORT_PATH = Path(os.getenv("MONITOR_REPORT_FILE", "monitoring_report.json"))
EXPECTED_INTERVAL_SECONDS = int(os.getenv("EXPECTED_INTERVAL_SECONDS", "900"))


def _parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def build_monitoring_report(signal, previous_state, observed_at, workflow_started_at=None):
    """Measure cadence from workflow starts; measure candle age at script execution."""
    workflow_started_at = workflow_started_at or observed_at
    last_started = _parse_timestamp(previous_state.get("last_workflow_started_at_utc"))
    gap_seconds = None
    if last_started is not None:
        gap_seconds = max(0.0, (workflow_started_at - last_started).total_seconds())

    candle_ms = signal.get("last_closed_candle_timestamp")
    candle_age_seconds = None
    if candle_ms is not None:
        try:
            candle_age_seconds = max(0.0, observed_at.timestamp() - float(candle_ms) / 1000.0)
        except (TypeError, ValueError):
            candle_age_seconds = None

    cadence_status = (
        "FIRST_OBSERVATION" if gap_seconds is None else
        "ON_OR_NEAR_SCHEDULE" if gap_seconds <= EXPECTED_INTERVAL_SECONDS * 1.5 else
        "DELAYED_OR_MISSED_INTERVAL"
    )
    estimated_missed_intervals = (
        None if gap_seconds is None else
        max(0, int(gap_seconds // EXPECTED_INTERVAL_SECONDS) - 1)
    )
    cadence_warning = (
        "No previous workflow-start timestamp is available; missed intervals cannot be estimated."
        if gap_seconds is None else
        "WARNING: gap between workflow start times exceeded the expected cadence. This indicates delayed or absent workflow starts, but does not identify whether GitHub delayed, dropped, or queued a scheduled trigger."
        if cadence_status == "DELAYED_OR_MISSED_INTERVAL" else
        "No delayed workflow-start gap detected."
    )

    return {
        "observed_at_utc": observed_at.isoformat(),
        "workflow_started_at_utc": workflow_started_at.isoformat(),
        "observation_definition": "Cadence uses GitHub Actions workflow start time; candle freshness uses the time this diagnostic script executed.",
        "expected_interval_seconds": EXPECTED_INTERVAL_SECONDS,
        "seconds_since_previous_workflow_start": gap_seconds,
        "cadence_status": cadence_status,
        "estimated_missed_intervals": estimated_missed_intervals,
        "cadence_warning": cadence_warning,
        "latest_closed_candle_timestamp_ms": candle_ms,
        "latest_closed_candle_age_seconds": candle_age_seconds,
        "closed_candle_count": signal.get("closed_candle_count"),
        "signal_record_available": bool(signal),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_number": os.getenv("GITHUB_RUN_NUMBER"),
        "github_event_name": os.getenv("GITHUB_EVENT_NAME"),
        "workflow_schedule": os.getenv("MONITOR_WORKFLOW_SCHEDULE"),
        "state_restored": bool(last_started),
        "note": "Paper monitoring only. Workflow-start gaps are evidence of run cadence, not proof of the scheduler's internal delivery behavior or exchange execution.",
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
    workflow_started_at = _parse_timestamp(os.getenv("MONITOR_WORKFLOW_STARTED_AT")) or observed_at
    report = build_monitoring_report(signal, previous_state, observed_at, workflow_started_at)
    STATE_PATH.write_text(json.dumps({
        "version": 2,
        "last_workflow_started_at_utc": workflow_started_at.isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Monitoring cadence status:", report["cadence_status"])
    print("Seconds since previous workflow start:", report["seconds_since_previous_workflow_start"])
    print("Estimated missed intervals:", report["estimated_missed_intervals"])
    print("Cadence warning:", report["cadence_warning"])
    print("Latest closed candle age (seconds):", report["latest_closed_candle_age_seconds"])
    print("State restored:", report["state_restored"])


if __name__ == "__main__":
    main()
