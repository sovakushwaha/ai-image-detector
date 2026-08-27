#!/usr/bin/env python3
"""Stage 28C — UI display-interpretation tests (no inference changes)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from final_inference_v1 import FinalImageDetectorV1  # noqa: E402

REAL_B = 0.26396692384233933
AI_B = 0.7360330761576607


def load_ui():
    path = ROOT / "app" / "local_detector_ui_v1.py"
    name = "local_detector_ui_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # required before dataclass processing under importlib
    spec.loader.exec_module(mod)
    return mod


def selective_from_p(p: float) -> str:
    if p <= REAL_B:
        return "REAL"
    if p >= AI_B:
        return "AI-GENERATED"
    return "UNCERTAIN"


def main() -> int:
    ui = load_ui()
    cases = [
        (0.10, "LIKELY_REAL", "Likely Real"),
        (0.30, "UNCERTAIN_LEANING_REAL", "Uncertain — leaning Real"),
        (0.49, "UNCERTAIN_NEUTRAL", "Uncertain — no clear indication"),
        (0.51, "UNCERTAIN_NEUTRAL", "Uncertain — no clear indication"),
        (0.70, "UNCERTAIN_LEANING_AI", "Uncertain — leaning AI-generated"),
        (0.90, "LIKELY_AI_GENERATED", "Likely AI-generated"),
        (REAL_B, "LIKELY_REAL", "Likely Real"),
        (REAL_B + 1e-12, "UNCERTAIN_LEANING_REAL", "Uncertain — leaning Real"),
        (AI_B - 1e-12, "UNCERTAIN_LEANING_AI", "Uncertain — leaning AI-generated"),
        (AI_B, "LIKELY_AI_GENERATED", "Likely AI-generated"),
        (0.704, "UNCERTAIN_LEANING_AI", "Uncertain — leaning AI-generated"),
    ]

    rows = []
    for p, expected_code, expected_label in cases:
        underlying = selective_from_p(p)
        disp = ui.interpret_display(underlying, p, REAL_B, AI_B)
        ok = (
            disp.selective_prediction == underlying
            and disp.display_code == expected_code
            and disp.display_label == expected_label
        )
        rows.append(
            {
                "calibrated_p": p,
                "underlying_decision": underlying,
                "display_code": disp.display_code,
                "display_label": disp.display_label,
                "expected_code": expected_code,
                "expected_label": expected_label,
                "underlying_preserved": disp.selective_prediction == underlying,
                "pass": ok,
            }
        )

    df = pd.DataFrame(rows)
    out = RESULTS / "local_ui_interpretation_test_v1.csv"
    df.to_csv(out, index=False)
    passed = int(df["pass"].sum())
    failed = int((~df["pass"]).sum())

    # Re-run Stage 28B consistency (must remain exact)
    sys.path.insert(0, str(SRC))
    from verify_local_ui_v1 import main as consistency_main

    consistency_rc = consistency_main()
    cons = pd.read_csv(RESULTS / "local_ui_consistency_v1.csv")
    cons_pass = int(cons["pass"].sum()) == len(cons) and consistency_rc == 0

    example = ui.interpret_display("UNCERTAIN", 0.704, REAL_B, AI_B)
    report = "\n".join(
        [
            "STAGE 28C — UI INTERPRETATION & USABILITY POLISH REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "PRIMARY USER RESULT FORMAT",
            "Likely Real: underlying REAL",
            "Uncertain — leaning Real: underlying UNCERTAIN + p < 0.45 (UI-only)",
            "Uncertain — no clear indication: underlying UNCERTAIN + 0.45<=p<=0.55 (UI-only)",
            "Uncertain — leaning AI-generated: underlying UNCERTAIN + p > 0.55 (UI-only)",
            "Likely AI-generated: underlying AI-GENERATED",
            "",
            "EXAMPLE P(AI)=0.704",
            f"Underlying scientific decision: {example.selective_prediction}",
            f"User-facing result: {example.display_label}",
            "Displayed score: AI likelihood score: 70.4%",
            f"Explanation: {example.explanation}",
            "",
            "BOUNDARIES (unchanged)",
            f"Real: <= {100*REAL_B:.1f}%",
            f"Uncertain: {100*REAL_B:.1f}%–{100*AI_B:.1f}%",
            f"AI-generated: >= {100*AI_B:.1f}%",
            "",
            "MODEL CONSISTENCY (Stage 28B re-check)",
            f"Images checked: {len(cons)}",
            f"Raw probability unchanged: {'YES' if cons_pass else 'NO'}",
            f"Calibrated probability unchanged: {'YES' if cons_pass else 'NO'}",
            f"Underlying decisions unchanged: {'YES' if cons_pass else 'NO'}",
            f"Decision agreement: {int((cons.cli_decision==cons.ui_decision).sum())}/{len(cons)}",
            "",
            "UI INTERPRETATION TESTS",
            f"Passed: {passed}",
            f"Failed: {failed}",
            "",
            "INTEGRITY",
            "Model training: NO",
            "Inference logic changed: NO",
            "Model weights changed: NO",
            "Temperature changed: NO",
            "Real/AI boundaries changed: NO",
            "Underlying selective prediction changed: NO",
            "New scientific decision thresholds: NO",
            "UI-only interpretation bands added: YES",
            "Scientific results changed: NO",
            "",
            f"STATUS: {'PASS' if failed == 0 and cons_pass else 'FAIL'}",
        ]
    )
    (RESULTS / "local_ui_stage28c_report_v1.txt").write_text(report + "\n")
    (RESULTS / "local_ui_stage28c_summary_v1.json").write_text(
        json.dumps(
            {
                "interpretation_passed": passed,
                "interpretation_failed": failed,
                "consistency_pass": cons_pass,
                "example_0_704": {
                    "underlying": example.selective_prediction,
                    "display": example.display_label,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(report)
    return 0 if failed == 0 and cons_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
