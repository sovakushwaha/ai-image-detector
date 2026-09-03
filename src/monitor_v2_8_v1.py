"""Read-only V2-8 execution monitor. Never pushes kernels or switches credentials."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "v2"
KERNEL_SLUG = "sovaakushwaha/v2-clip-lora-generalisation"
START_UTC = datetime(2026, 9, 2, 15, 15, 46, tzinfo=timezone.utc)
LOCK_PATH = RESULTS / "v2_8_execution_lock_v1.json"
LOG_PATH = RESULTS / "v2_8_monitor_log.jsonl"


def run_kaggle_status() -> tuple[str, str]:
    cp = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = ((cp.stdout or "") + (cp.stderr or "")).strip()
    if cp.returncode != 0 and not raw:
        return "UNKNOWN", raw or "empty response"
    m = re.search(r'status "KernelWorkerStatus\.(\w+)"', raw, re.I)
    if m:
        return m.group(1).upper(), raw
    low = raw.lower()
    for label in ("RUNNING", "COMPLETE", "ERROR", "QUEUED", "STARTING", "PENDING"):
        if label.lower() in low:
            return label, raw
    return "UNKNOWN", raw


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    status, raw = run_kaggle_status()
    elapsed_h = (now - START_UTC).total_seconds() / 3600

    entry = {
        "ts_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": "sovaakushwaha",
        "kernel": KERNEL_SLUG,
        "version": 37,
        "status": status,
        "elapsed_hours": round(elapsed_h, 2),
        "elapsed_minutes": round(elapsed_h * 60),
        "action": (
            "SUCCESS_WORKFLOW"
            if status == "COMPLETE"
            else "FAILURE_WORKFLOW"
            if status == "ERROR"
            else "MONITOR_ONLY"
            if status == "RUNNING"
            else "KAGGLE_STATUS_UNKNOWN_NO_LAUNCH"
        ),
        "notes": raw.splitlines()[0] if raw else "",
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    launch_permitted = status in ("COMPLETE", "ERROR", "NONE") and status not in (
        "RUNNING",
        "QUEUED",
        "STARTING",
        "PENDING",
        "UNKNOWN",
    )
    lock = {
        "lock_type": "V2-8_ACTIVE_GPU_JOB",
        "locked": status in ("RUNNING", "QUEUED", "STARTING", "PENDING"),
        "reason": (
            "Production v37 four-fold LoRA run in progress"
            if status == "RUNNING"
            else f"Kernel status {status}"
        ),
        "account": "sovaakushwaha",
        "kernel": KERNEL_SLUG,
        "version": 37,
        "status": status,
        "start_time_utc": START_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "support_dataset": "sovaakushwaha/v2-clip-lora-support",
        "support_version": 6,
        "run_mode": "smoke_then_full",
        "launch_permitted": launch_permitted,
        "updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n")

    print(json.dumps(entry, indent=2))
    if status == "RUNNING":
        print("DECISION: MONITOR_ONLY — do not push, cancel, or migrate")
    elif status == "COMPLETE":
        print("DECISION: SUCCESS_WORKFLOW — download outputs and import")
    elif status == "ERROR":
        print("DECISION: FAILURE_WORKFLOW — download logs, preserve folds, diagnose")
    else:
        print("DECISION: KAGGLE_STATUS_UNKNOWN_NO_LAUNCH — fail closed")


if __name__ == "__main__":
    main()
