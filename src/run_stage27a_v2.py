#!/usr/bin/env python3
"""Stage 27A V2 — full public external evaluation pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquire_external_v2_public import main as acquire_main  # noqa: E402
from audit_external_v2 import main as audit_main  # noqa: E402
from evaluate_external_v2 import main as evaluate_main  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    print("=" * 60)
    print("STAGE 27A V2 — ORCHESTRATED EXTERNAL VALIDATION")
    print("=" * 60)

    print("\n[1/3] Public data acquisition...")
    acq = acquire_main()
    if acq != 0:
        print("Acquisition incomplete; continuing with available data where possible")

    print("\n[2/3] Native + overlap audit + readiness gate...")
    audit = audit_main()
    if audit != 0:
        print("STOP: readiness gate failed before inference")
        return 1

    print("\n[3/3] Frozen external evaluation...")
    evaluate_main()
    print(f"\nPipeline finished at {utc_now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
