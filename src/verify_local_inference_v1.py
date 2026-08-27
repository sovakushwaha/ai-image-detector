#!/usr/bin/env python3
"""Stage 28A verification: reproduction, smoke test, latency sanity."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from final_inference_v1 import FinalImageDetectorV1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_CSV = RESULTS / "rq3_A2_test_predictions_v1.csv"
SPLIT_META = ROOT / "metadata" / "controlled_v1_split_metadata.csv"
TOL = 1e-5
N_REPRO = 10
N_LATENCY = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_repro_ids(pred: pd.DataFrame) -> list[str]:
    orig = pred[pred["condition"] == "original"].copy()
    return sorted(orig["source_image_id"].unique())[:N_REPRO]


def run_reproduction() -> dict:
    pred = pd.read_csv(PRED_CSV)
    meta = pd.read_csv(SPLIT_META)
    ids = select_repro_ids(pred)
    meta_idx = meta.set_index("image_id")
    orig = pred[(pred["condition"] == "original") & (pred["source_image_id"].isin(ids))]
    orig = orig.set_index("source_image_id")

    detector = FinalImageDetectorV1(device="cpu")
    rows = []
    for image_id in ids:
        processed = ROOT / meta_idx.loc[image_id, "processed_path"]
        saved = float(orig.loc[image_id, "probability"])
        result = detector.predict(processed, research_controlled_v1=True)
        diff = abs(result.raw_probability - saved)
        rows.append(
            {
                "image_id": image_id,
                "saved_raw_probability": saved,
                "new_raw_probability": result.raw_probability,
                "absolute_difference": diff,
                "device": "cpu",
                "pass": diff <= TOL,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "local_inference_reproduction_v1.csv", index=False)

    passed = int(df["pass"].sum())
    failed = int((~df["pass"]).sum())
    mad = float(df["absolute_difference"].mean())
    mx = float(df["absolute_difference"].max())
    cpu_pass = failed == 0

    # Optional MPS comparison (document FP variation; not authoritative)
    mps_note = "MPS not available"
    mps_rows = []
    try:
        mps_det = FinalImageDetectorV1(device="mps")
        for image_id in ids:
            processed = ROOT / meta_idx.loc[image_id, "processed_path"]
            saved = float(orig.loc[image_id, "probability"])
            r = mps_det.predict(processed, research_controlled_v1=True)
            mps_rows.append(abs(r.raw_probability - saved))
        mps_note = (
            f"MPS vs saved: mean_abs_diff={float(pd.Series(mps_rows).mean()):.3e}, "
            f"max_abs_diff={float(pd.Series(mps_rows).max()):.3e} "
            f"(CPU is authoritative; small FP variation allowed)"
        )
    except Exception as exc:  # noqa: BLE001
        mps_note = f"MPS comparison skipped: {type(exc).__name__}: {exc}"

    report_lines = [
        "STAGE 28A — LOCAL INFERENCE REPRODUCTION REPORT",
        f"Generated: {utc_now()}",
        "",
        f"Source predictions: {PRED_CSV.relative_to(ROOT)}",
        f"Condition: original",
        f"N={N_REPRO}",
        f"Device: cpu (authoritative)",
        f"Tolerance: abs(diff) <= {TOL}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        f"Mean absolute difference: {mad:.6e}",
        f"Maximum absolute difference: {mx:.6e}",
        f"CPU reproduction: {'PASS' if cpu_pass else 'FAIL'}",
        "",
        mps_note,
        "",
        "Note: reproduction loads controlled_v1 224×224 research files with",
        "tensor normalisation only (matches RQ3 A2 frozen evaluation).",
        "Practical CLI inference uses decoded-pixel resize/crop without JPEG q96.",
    ]
    (RESULTS / "local_inference_reproduction_report_v1.txt").write_text("\n".join(report_lines) + "\n")

    if not cpu_pass:
        raise SystemExit(
            "STOP: CPU reproduction failed — do not modify the model to force agreement.\n"
            + "\n".join(report_lines)
        )

    return {
        "n": N_REPRO,
        "passed": passed,
        "failed": failed,
        "mean_abs_diff": mad,
        "max_abs_diff": mx,
        "cpu_pass": cpu_pass,
        "mps_note": mps_note,
        "ids": ids,
    }


def run_smoke(device: str = "auto") -> dict:
    meta = pd.read_csv(SPLIT_META)
    real = meta[(meta["label"] == 0) & (meta["split"].isin(["known_test", "unseen_test"]))].iloc[0]
    ai = meta[(meta["label"] == 1) & (meta["split"].isin(["known_test", "unseen_test"]))].iloc[0]
    detector = FinalImageDetectorV1(device=device)
    # Engineering smoke: practical path on controlled_v1 files still works, but use
    # research mode for already-224 files to avoid double-geometry on smoke assets.
    real_r = detector.predict(ROOT / real["processed_path"], research_controlled_v1=True)
    ai_r = detector.predict(ROOT / ai["processed_path"], research_controlled_v1=True)
    return {
        "device": str(detector.device),
        "known_real": {
            "image_id": real["image_id"],
            "path": real["processed_path"],
            "raw_probability": real_r.raw_probability,
            "calibrated_probability": real_r.calibrated_probability,
            "selective_decision": real_r.selective_decision,
        },
        "known_ai": {
            "image_id": ai["image_id"],
            "path": ai["processed_path"],
            "raw_probability": ai_r.raw_probability,
            "calibrated_probability": ai_r.calibrated_probability,
            "selective_decision": ai_r.selective_decision,
        },
    }


def run_latency(device: str = "auto") -> dict:
    meta = pd.read_csv(SPLIT_META)
    path = ROOT / meta.iloc[0]["processed_path"]
    detector = FinalImageDetectorV1(device=device)
    # warm-up
    for _ in range(3):
        detector.predict(path, research_controlled_v1=True)
    times = []
    for _ in range(N_LATENCY):
        t0 = time.perf_counter()
        detector.predict(path, research_controlled_v1=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    times_s = pd.Series(times)
    return {
        "device": str(detector.device),
        "n_runs": N_LATENCY,
        "median_e2e_ms": float(times_s.median()),
        "mean_e2e_ms": float(times_s.mean()),
        "p95_e2e_ms": float(times_s.quantile(0.95)),
        "note": "Sanity check only; does not replace Stage 26A benchmark",
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    repro = run_reproduction()
    smoke = run_smoke(device="auto")
    latency = run_latency(device="auto")
    detector = FinalImageDetectorV1(device="cpu")
    meta = detector.metadata()

    lines = [
        "=" * 50,
        "STAGE 28A — FROZEN LOCAL INFERENCE ENGINE COMPLETE",
        "=" * 50,
        "",
        f"Model: {meta['model_id']} ({meta['selected_candidate']})",
        f"Checkpoint: {meta['checkpoint']}",
        f"Parameters: {meta['n_parameters']}",
        f"Temperature: {meta['temperature']}",
        f"REAL boundary: calibrated P(AI) <= {meta['real_boundary']}",
        f"AI boundary: calibrated P(AI) >= {meta['ai_boundary']}",
        f"UNCERTAIN interval: ({meta['real_boundary']}, {meta['ai_boundary']})",
        "",
        "REPRODUCTION TEST",
        f"Images: {repro['n']}",
        f"Passed: {repro['passed']}",
        f"Failed: {repro['failed']}",
        f"Mean absolute difference: {repro['mean_abs_diff']:.6e}",
        f"Max absolute difference: {repro['max_abs_diff']:.6e}",
        f"CPU reproduction: {'PASS' if repro['cpu_pass'] else 'FAIL'}",
        f"MPS comparison: {repro['mps_note']}",
        "",
        "SMOKE TEST (engineering only; not paper Results)",
        "Known Real:",
        f"  image_id={smoke['known_real']['image_id']}",
        f"  Raw P(AI)={smoke['known_real']['raw_probability']:.6f}",
        f"  Calibrated model P(AI)={smoke['known_real']['calibrated_probability']:.6f}",
        f"  Decision={smoke['known_real']['selective_decision']}",
        "Known AI:",
        f"  image_id={smoke['known_ai']['image_id']}",
        f"  Raw P(AI)={smoke['known_ai']['raw_probability']:.6f}",
        f"  Calibrated model P(AI)={smoke['known_ai']['calibrated_probability']:.6f}",
        f"  Decision={smoke['known_ai']['selective_decision']}",
        "",
        "LATENCY SANITY",
        f"Device: {latency['device']}",
        f"Median E2E: {latency['median_e2e_ms']:.2f} ms",
        f"Mean E2E: {latency['mean_e2e_ms']:.2f} ms",
        "",
        "PRIVACY",
        "External API: NO",
        "Image upload: NO",
        "Automatic history: NO",
        "",
        "FILES",
        "src/final_inference_v1.py",
        "src/predict_image_v1.py",
        "results/local_inference_reproduction_v1.csv",
        "results/local_inference_reproduction_report_v1.txt",
        "results/local_inference_stage28a_report_v1.txt",
        "",
        "STATUS",
        "Stage 28A: COMPLETE",
        "Frozen pipeline reproduction: PASS",
        "Stage 28B graphical local upload UI: NOT STARTED",
        "",
        f"Generated: {utc_now()}",
    ]
    report = "\n".join(lines) + "\n"
    (RESULTS / "local_inference_stage28a_report_v1.txt").write_text(report)
    summary = {
        "stage": "28A",
        "model": meta,
        "reproduction": repro,
        "smoke": smoke,
        "latency": latency,
        "integrity": {
            "training": False,
            "weights_changed": False,
            "checkpoint_changed": False,
            "temperature_changed": False,
            "selective_policy_changed": False,
            "external_tuning": False,
            "api_calls": False,
            "ui_built": False,
        },
    }
    (RESULTS / "local_inference_stage28a_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
