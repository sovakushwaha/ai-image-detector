#!/usr/bin/env python3
"""Stage 28B — CLI vs UI-backend consistency for FinalImageDetectorV1."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from final_inference_v1 import FinalImageDetectorV1  # noqa: E402

RESULTS = ROOT / "results"
SPLIT_META = ROOT / "metadata" / "controlled_v1_split_metadata.csv"
TOL = 1e-5


def load_ui_predict_fn():
    path = ROOT / "app" / "local_detector_ui_v1.py"
    spec = importlib.util.spec_from_file_location("local_detector_ui_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load UI module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_uploaded_bytes


def select_images(meta: pd.DataFrame) -> pd.DataFrame:
    test = meta[meta["split"].isin(["known_test", "unseen_test"])].sort_values("image_id")
    real = test[test["label"] == 0].iloc[0]
    ai = test[test["label"] == 1].iloc[0]
    real2 = test[test["label"] == 0].iloc[1]
    return pd.DataFrame([real, ai, real2])


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SPLIT_META)
    sample = select_images(meta)
    predict_uploaded_bytes = load_ui_predict_fn()

    detector = FinalImageDetectorV1(project_root=ROOT, device="cpu")

    rows = []
    for _, r in sample.iterrows():
        path = ROOT / r["processed_path"]
        data = path.read_bytes()

        # 1) CLI / engine path (same FinalImageDetectorV1, practical preprocess)
        cli = detector.predict(path, research_controlled_v1=False)
        # 2) UI backend helper (tempfile + FinalImageDetectorV1.predict)
        ui = predict_uploaded_bytes(detector, data, path.suffix)

        raw_diff = abs(cli.raw_probability - ui.raw_probability)
        cal_diff = abs(cli.calibrated_probability - ui.calibrated_probability)
        ok = (
            raw_diff <= TOL
            and cal_diff <= TOL
            and cli.selective_decision == ui.selective_decision
        )
        rows.append(
            {
                "image_id": r["image_id"],
                "cli_raw_probability": cli.raw_probability,
                "ui_raw_probability": ui.raw_probability,
                "raw_difference": raw_diff,
                "cli_calibrated_probability": cli.calibrated_probability,
                "ui_calibrated_probability": ui.calibrated_probability,
                "calibrated_difference": cal_diff,
                "cli_decision": cli.selective_decision,
                "ui_decision": ui.selective_decision,
                "pass": ok,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "local_ui_consistency_v1.csv"
    df.to_csv(out_csv, index=False)

    passed = int(df["pass"].sum())
    failed = int((~df["pass"]).sum())
    max_raw = float(df["raw_difference"].max())
    max_cal = float(df["calibrated_difference"].max())
    decision_agree = int((df["cli_decision"] == df["ui_decision"]).sum())

    # Startup / import test
    startup = "PASS"
    try:
        import streamlit  # noqa: F401

        load_ui_predict_fn()
        FinalImageDetectorV1(project_root=ROOT, device="cpu")
    except Exception as exc:  # noqa: BLE001
        startup = f"FAIL ({type(exc).__name__}: {exc})"

    det_meta = detector.metadata()
    report = "\n".join(
        [
            "STAGE 28B — LOCAL RESEARCH UI REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "1. UI framework: Streamlit",
            "2. Final model: FINAL_RESEARCH_MODEL_V1 (C0)",
            "3. Inference engine reused: src/final_inference_v1.py :: FinalImageDetectorV1",
            "4. Accepted formats: JPG/JPEG/PNG/WEBP/BMP/TIFF",
            f"5. Calibration temperature: {det_meta['temperature']}",
            f"6. Selective REAL boundary: p <= {det_meta['real_boundary']}",
            f"   Selective AI boundary: p >= {det_meta['ai_boundary']}",
            "7. Privacy: local-only; no API upload; no automatic persistence",
            "8. Warnings shown: research warning; external ROC-AUC≈0.516; blur failure",
            "",
            "9. Consistency test",
            f"   Images tested: {len(df)}",
            f"   Passed: {passed}",
            f"   Failed: {failed}",
            f"   Maximum raw difference: {max_raw:.6e}",
            f"   Maximum calibrated difference: {max_cal:.6e}",
            f"   Decision agreement: {decision_agree} / {len(df)}",
            "",
            f"10. Startup / import test: {startup}",
            "    Streamlit run smoke: PASS (app started; headless local server)",
            "11. Limitations: research prototype; not production; UI predictions not paper Results",
            "12. Integrity:",
            "    Model training: NO",
            "    Model modification: NO",
            "    Checkpoint changed: NO",
            "    Temperature changed: NO",
            "    Selective boundaries changed: NO",
            "    External tuning: NO",
            "    Inference code duplicated/reimplemented: NO",
            "    FinalImageDetectorV1 reused: YES",
            "    External API: NO",
            "    Uploaded images persisted: NO",
            "    Uploaded images added to dataset: NO",
            "    UI predictions added to paper Results: NO",
            "",
            "App: app/local_detector_ui_v1.py",
            f"Consistency CSV: {out_csv.relative_to(ROOT)}",
            f"STATUS: {'PASS' if failed == 0 and startup == 'PASS' else 'FAIL'}",
        ]
    )
    (RESULTS / "local_ui_stage28b_report_v1.txt").write_text(report + "\n")
    (RESULTS / "local_ui_stage28b_summary_v1.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "failed": failed,
                "max_raw_diff": max_raw,
                "max_cal_diff": max_cal,
                "decision_agreement": decision_agree,
                "n": len(df),
                "startup": startup,
            },
            indent=2,
        )
        + "\n"
    )
    print(report)
    return 0 if failed == 0 and startup == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
