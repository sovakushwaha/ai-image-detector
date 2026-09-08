#!/usr/bin/env python3
"""V2-12 — Final scientific results synthesis (documentation only).

NO training, inference, threshold changes, calibration, or model modification.
Consolidates existing V1/V2 artifacts into a paper-ready evidence package.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
LOG = PROJECT_ROOT / "paper" / "research_log.md"
FIG = PROJECT_ROOT / "figures" / "v2"


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def exists(rel: str) -> bool:
    return (PROJECT_ROOT / rel).is_file()


def macro_robust(df: pd.DataFrame, head: str | None = None) -> dict[str, Any]:
    d = df.copy()
    if head is not None and "head" in d.columns:
        d = d[d["head"] == head]
    out: dict[str, Any] = {}
    for cond in ["CLEAN", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]:
        sub = d[d["condition"] == cond]
        if sub.empty:
            continue
        g = sub.groupby("fold").mean(numeric_only=True)
        out[cond] = {
            "auc": float(g["roc_auc"].mean()),
            "ap": float(g["ap"].mean()) if "ap" in g else None,
            "ai_recall": float(g["ai_recall"].mean()),
            "real_specificity": float(g["real_specificity"].mean()),
            "balanced_accuracy": float(g["balanced_accuracy"].mean()),
        }
    tfs = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
    out["mean_transformed"] = {
        "auc": float(np.mean([out[t]["auc"] for t in tfs])),
        "ai_recall": float(np.mean([out[t]["ai_recall"] for t in tfs])),
        "real_specificity": float(np.mean([out[t]["real_specificity"] for t in tfs])),
    }
    out["worst_transformed_auc"] = float(min(out[t]["auc"] for t in tfs))
    return out


def audit_artifacts() -> dict[str, Any]:
    required = [
        "paper/research_log.md",
        "models/v2/final_v2_h4_freeze_manifest_v1.json",
        "results/v2/v2_10i_final_model_specification_v1.json",
        "results/v2/v2_10i_development_model_comparison_v1.csv",
        "results/v2/v2_10i_final_model_limitations_v1.txt",
        "results/v2/v2_11_ntire_protocol_lock_v1.json",
        "results/v2/v2_11_ntire_metrics_v1.json",
        "results/v2/v2_11_ntire_bootstrap_v1.json",
        "results/v2/v2_11_external_transfer_summary_v1.txt",
        "results/v2/v2_10a_representation_geometry_v1.json",
        "results/v2/v2_10e_transform_shift_analysis_v1.json",
        "results/v2/v2_10d_robustness_metrics_v1.csv",
        "results/v2/v2_10f_robust_head_metrics_v1.csv",
        "results/v2/v2_10h_metrics_v1.csv",
        "results/v2/v2_10b_head_metrics_v1.csv",
        "results/v2/v2_10c_selective_report_v1.txt",
    ]
    present = {p: exists(p) for p in required}
    missing = [p for p, ok in present.items() if not ok]
    discrepancies = [
        {
            "item": "NTIRE_val_README_vs_labels",
            "severity": (
                "Val README claims no deepfake labels; reserved revision includes "
                "val_labels.csv. V2-11 used train-doc encoding 0=real/1=generated; "
                "documented in research_log and audit."
            ),
            "resolution": "Prefer V2-11 audit + research_log over README marketing text.",
        },
        {
            "item": "V2-10I_limitations_pre_NTIRE_external_unknown",
            "severity": (
                "v2_10i_final_model_limitations_v1.txt still says FINAL_EXTERNAL_PERFORMANCE=UNKNOWN. "
                "That was correct at freeze time; V2-11 later completed NTIRE."
            ),
            "resolution": "Treat V2-10I limitations as freeze-time register; V2-11 is authoritative for external result.",
        },
        {
            "item": "research_log_header_last_updated",
            "severity": "Header 'Last updated' may still mention V2-8 while later stages are appended.",
            "resolution": "Prefer stage sections over header line for chronology.",
        },
    ]
    return {
        "required_present": present,
        "n_missing": len(missing),
        "missing": missing,
        "discrepancies": discrepancies,
        "authority_order": [
            "stage-specific final artifact",
            "paper/research_log.md stage section",
            "freeze manifest / protocol lock",
            "older roadmap only if consistent",
        ],
    }


def build_timeline() -> pd.DataFrame:
    rows = [
        {
            "stage": "V1_foundation",
            "question": "Can a lightweight RGB detector discriminate Real vs AI under controlled development?",
            "intervention": "MobileNet / EfficientNet family; RQ1–RQ4 pipeline",
            "main_result": "Strong internal discrimination; FINAL_RESEARCH_MODEL_V1=C0 (RQ3 A2 robust RGB MobileNet)",
            "decision": "FINAL_RESEARCH_MODEL_V1 = C0",
            "scientific_consequence": "Established resource-aware V1 baseline and controlled robustness/calibration programme",
        },
        {
            "stage": "V1_robustness_calibration_resource",
            "question": "RQ2–RQ5 + resource: transforms, fusion, calibration, selective, cost",
            "intervention": "StrongRobust / transform-aware training; temperature; selective; Stage26 resource",
            "main_result": "RQ2–RQ5 COMPLETE internally; C0 preferred for calibrated/selective + cost vs C1",
            "decision": "C0 selected as V1 final",
            "scientific_consequence": "Internal story looked trustworthy; independent external still needed",
        },
        {
            "stage": "V1_external_Stage27A",
            "question": "Does C0 transfer to an independent public external set?",
            "intervention": "Frozen C0 inference (no retune)",
            "main_result": "Severe external failure ~AUC0.516 / AI_rec0.220 / RealSpec0.806",
            "decision": "EXTERNAL_FAILURE (historical)",
            "scientific_consequence": "Motivates V2 foundation-model / modern-generator programme; CONTEXT_ONLY vs NTIRE",
        },
        {
            "stage": "V2-0",
            "question": "Freeze V1 and lock V2 scientific protocol",
            "intervention": "Protocol lock; V1 immutable",
            "main_result": "V1 frozen; V2 rules locked; NTIRE sealed",
            "decision": "V2_PROTOCOL_LOCKED",
            "scientific_consequence": "Prevents V1 contamination and post-hoc NTIRE peeking during development",
        },
        {
            "stage": "V2-3",
            "question": "Build balanced multi-generator V2 development dataset + folds",
            "intervention": "Smartphone acquisition + multi-generator splits; four folds locked",
            "main_result": "V2 development pool + fold manifests locked",
            "decision": "FOLDS_LOCKED",
            "scientific_consequence": "All later V2 metrics share the same development evaluation units",
        },
        {
            "stage": "V2-5",
            "question": "Does frozen CLIP linear probing beat lightweight V1-style representations on V2 development?",
            "intervention": "open_clip ViT-B-16-quickgelu embeddings + LogReg",
            "main_result": "Macro AUC~0.815; CLIP_BASELINE_PROMISING",
            "decision": "CLIP_BASELINE_PROMISING",
            "scientific_consequence": "Foundation embeddings are a viable V2 starting point",
        },
        {
            "stage": "V2-6",
            "question": "Can a validation-only linear-probe refinement improve reliability?",
            "intervention": "Select C among frozen CLIP LogReg candidates (C=3.0)",
            "main_result": "Macro AUC~0.820; RealSpec improved vs V2-5",
            "decision": "LINEAR_PROBE_REFINED",
            "scientific_consequence": "Modest clean reliability gain without adapting CLIP weights",
        },
        {
            "stage": "V2-7",
            "question": "Does a small nonlinear head help on frozen CLIP?",
            "intervention": "MLP-B on frozen CLIP",
            "main_result": "Macro AUC~0.837; modest gain; Fold4 Real regression",
            "decision": "MLP_MODEST_GAIN",
            "scientific_consequence": "Remaining limit likely in representation → motivates LoRA",
        },
        {
            "stage": "V2-8",
            "question": "Does parameter-efficient LoRA improve development ranking?",
            "intervention": "LoRA last-4 blocks + MLP-B (Kaggle P100)",
            "main_result": "AUC~0.903 / AP~0.917 but AI_rec@0.5~0.386; LORA_MIXED",
            "decision": "LORA_MIXED",
            "scientific_consequence": "Ranking improved; fixed-threshold operating point worsened",
        },
        {
            "stage": "V2-9A",
            "question": "What failure modes explain LoRA+MLP-B low recall?",
            "intervention": "Analysis-only diagnostic",
            "main_result": "Operating-point / score-shift diagnosis; not a solved detector",
            "decision": "ANALYSIS_ONLY",
            "scientific_consequence": "Points to calibration/head operating-point work rather than immediate new LoRA recipe",
        },
        {
            "stage": "V2-9B",
            "question": "Is a leakage-safe post-hoc calibration protocol available on existing sets?",
            "intervention": "Protocol audit",
            "main_result": "CALIBRATION_PROTOCOL_NOT_SAFE",
            "decision": "CALIBRATION_PROTOCOL_NOT_SAFE",
            "scientific_consequence": "Blocked unsafe calibrator fitting on contaminated splits",
        },
        {
            "stage": "V2-9C",
            "question": "Can an independent calibration set be constructed?",
            "intervention": "Feasibility audit + set construction",
            "main_result": "CALIBRATION_SET_FEASIBLE_WITH_LIMITATIONS",
            "decision": "CALIBRATION_SET_FEASIBLE_WITH_LIMITATIONS",
            "scientific_consequence": "Enabled V2-9D under leakage locks",
        },
        {
            "stage": "V2-9D",
            "question": "Does temperature/Platt safely repair LoRA+MLP-B operating point?",
            "intervention": "Fit calibrators on independent calib set only",
            "main_result": "Temperature helps NLL/Brier but preserves .5 decisions; Platt raises recall with severe Real FPs",
            "decision": "CALIBRATION_NOT_HELPFUL",
            "scientific_consequence": "Calibration does not safely solve LoRA+MLP-B operational failure",
        },
        {
            "stage": "V2-10A",
            "question": "Is LoRA R1 representation-limited or head-limited?",
            "intervention": "kNN / geometry on R0/R1/R2",
            "main_result": "R1 k5 purity ~0.966; FNs remain in AI neighbourhood → HEAD_LIMITED",
            "decision": "HEAD_LIMITED",
            "scientific_consequence": "Justifies frozen-R1 head redesign without further LoRA training",
        },
        {
            "stage": "V2-10B",
            "question": "Can a tiny balanced linear head rescue clean operating performance?",
            "intervention": "H1/H2 LogReg on frozen R1; H2 class-balanced",
            "main_result": "H2 CLEAN AUC~0.951 AI_rec~0.576 RealSpec~0.967",
            "decision": "HEAD_RESCUE_PROMISING",
            "scientific_consequence": "Shows representation was underutilized by MLP-B at thr=0.5",
        },
        {
            "stage": "V2-10C",
            "question": "Do temperature + selective prediction make H2 trustworthy?",
            "intervention": "Temperature on H2; selective S90; thr unchanged",
            "main_result": "Temp improves NLL/Brier; 0 classification changes; S90 err_capture~0.16 enrich~2x",
            "decision": "SELECTIVE_PREDICTION_MIXED",
            "scientific_consequence": "Confidence methods informative but do not solve generalisation; not transferred to H4",
        },
        {
            "stage": "V2-10D",
            "question": "Is H2 robust under StrongRobust transforms?",
            "intervention": "jpeg_q50 / resize_112 / blur_sigma2 / screenshot_strong",
            "main_result": "Catastrophic collapses (JPEG AI_rec~0.089; resize RealSpec~0.131)",
            "decision": "ROBUSTNESS_NOT_ACCEPTABLE",
            "scientific_consequence": "Blocks selecting H2 as final despite best clean scores",
        },
        {
            "stage": "V2-10E",
            "question": "Are StrongRobust failures geometry collapse or head/margin shift?",
            "intervention": "Transformed R1 geometry + H2 margin diagnostics",
            "main_result": "k5 purity remains high under transforms; HEAD_SHIFT_DOMINANT all four",
            "decision": "ROBUSTNESS_HEAD_LIMITED",
            "scientific_consequence": "Supports frozen-R1 robust-head intervention (not LoRA retrain)",
        },
        {
            "stage": "V2-10F",
            "question": "Can equal-condition robust LogReg (H3) rescue transforms?",
            "intervention": "H3 on CLEAN+4 StrongRobust TRAIN R1 equal 20%",
            "main_result": "mean tf AUC~0.832 RealSpec~0.887; CLEAN AUC drops to ~0.918",
            "decision": "ROBUST_HEAD_MIXED",
            "scientific_consequence": "Cheap head-level robustness works but over-rotates vs clean",
        },
        {
            "stage": "V2-10G",
            "question": "Why does H3 pay a clean cost / where does it over-rotate?",
            "intervention": "Direction/conflict analysis H2 vs H3",
            "main_result": "H3 over-rotated from clean separation; Fold4/Seedream amplify cost",
            "decision": "ROBUST_HEAD_MIXED_LIMITATION",
            "scientific_consequence": "Motivates single predeclared clean-anchored H4 (no weight grid)",
        },
        {
            "stage": "V2-10H",
            "question": "Does clean-anchored weighting (H4) recover clean without losing robustness?",
            "intervention": "H4 LogReg sample_weight CLEAN=4 / each transform=1",
            "main_result": "CLEAN AUC~0.926; mean tf AUC~0.826 RealSpec~0.883; PROMISING gates missed",
            "decision": "CLEAN_ANCHORED_ROBUST_HEAD_MIXED",
            "scientific_consequence": "Best completed clean/robust compromise among H2/H3/H4",
        },
        {
            "stage": "V2-10I",
            "question": "Freeze final V2 development model before NTIRE",
            "intervention": "SHA256 manifest + protocol lock; no training/inference",
            "main_result": "FINAL_RESEARCH_MODEL_V2=H4; FROZEN_WITH_DOCUMENTED_LIMITATIONS",
            "decision": "DEVELOPMENT_FROZEN",
            "scientific_consequence": "Makes sealed external evaluation scientifically interpretable",
        },
        {
            "stage": "V2-11",
            "question": "Does frozen H4 transfer to sealed NTIRE?",
            "intervention": "Equal-weight 4-fold ensemble; thr=0.5; no retune",
            "main_result": "AUC0.625 / AI_rec0.130 / RealSpec0.956; WEAK_EXTERNAL_TRANSFER",
            "decision": "WEAK_EXTERNAL_TRANSFER; H4 unchanged",
            "scientific_consequence": "Development gains do not imply broad independent transfer; programme closed experimentally",
        },
        {
            "stage": "V2-12",
            "question": "Consolidate evidence for paper writing",
            "intervention": "Documentation synthesis only",
            "main_result": "Authoritative evidence package + claim register + narrative",
            "decision": "RESULTS_SYNTHESIS_COMPLETE",
            "scientific_consequence": "Enables full paper draft without further experiments",
        },
    ]
    return pd.DataFrame(rows)


def build_progression(h2r: dict, h3r: dict, h4r: dict) -> pd.DataFrame:
    cmp_df = pd.read_csv(OUT / "v2_10i_development_model_comparison_v1.csv")
    # Prefer freeze comparison for clean metrics; attach StrongRobust from stage CSVs
    rows = []
    mapping = {
        "V2-5_CLIP_LogReg": ("Frozen CLIP LogReg", None, None, "513", "CLIP_BASELINE_PROMISING"),
        "V2-6_CLIP_LogReg_C3": ("Refined CLIP linear probe (C=3)", None, None, "513", "LINEAR_PROBE_REFINED"),
        "V2-7_CLIP_MLP-B": ("Frozen CLIP + MLP-B", None, None, "147841", "MLP_MODEST_GAIN"),
        "V2-8_LoRA_MLP-B": ("LoRA + MLP-B", None, None, "LoRA+MLP-B", "LORA_MIXED"),
        "H2_R1_balanced_LogReg": ("H2 balanced LogReg on R1", h2r, 513, "513", "HEAD_RESCUE_PROMISING / ROBUSTNESS_NOT_ACCEPTABLE"),
        "H3_equal_condition_robust_LogReg": ("H3 equal-condition robust LogReg", h3r, 513, "513", "ROBUST_HEAD_MIXED"),
        "H4_clean_anchored_robust_LogReg": ("H4 clean-anchored robust LogReg (FINAL)", h4r, 513, "513", "FINAL_RESEARCH_MODEL_V2"),
    }
    for _, r in cmp_df.iterrows():
        mid = r["model"]
        if mid.startswith("V1_"):
            continue  # keep V2-only progression table; V1 marked separately in report
        label, rob, _, params, decision = mapping.get(
            mid, (mid, None, None, str(r.get("head_or_model_params", "")), str(r.get("decision_or_status", "")))
        )
        rows.append(
            {
                "model": label,
                "source_row": mid,
                "clean_auc": r["clean_auc"],
                "clean_ap": r["clean_ap"],
                "ai_recall": r["ai_recall_at_05_or_frozen_thr"],
                "real_specificity": r["real_specificity"],
                "balanced_accuracy": r["balanced_accuracy"],
                "mean_strongrobust_auc": None if rob is None else rob["mean_transformed"]["auc"],
                "mean_strongrobust_real_specificity": None
                if rob is None
                else rob["mean_transformed"]["real_specificity"],
                "head_parameter_count": params if params else r.get("head_or_model_params"),
                "scientific_decision": decision,
                "context_note": "V2 development eval macro; StrongRobust only where evaluated (H2/H3/H4)",
            }
        )
    return pd.DataFrame(rows)


def build_dev_external_gap(h4r: dict, ntire: dict) -> pd.DataFrame:
    p = ntire["primary_ensemble"]
    rows = [
        {
            "setting": "H4_development_CLEAN",
            "dataset": "V2_dev_eval_folds",
            "auc": h4r["CLEAN"]["auc"],
            "ap": h4r["CLEAN"]["ap"],
            "ai_recall": h4r["CLEAN"]["ai_recall"],
            "real_specificity": h4r["CLEAN"]["real_specificity"],
            "balanced_accuracy": h4r["CLEAN"]["balanced_accuracy"],
            "comparability": "DEVELOPMENT_ONLY",
        },
        {
            "setting": "H4_development_StrongRobust_mean",
            "dataset": "V2_dev_eval_folds_transformed",
            "auc": h4r["mean_transformed"]["auc"],
            "ap": None,
            "ai_recall": h4r["mean_transformed"]["ai_recall"],
            "real_specificity": h4r["mean_transformed"]["real_specificity"],
            "balanced_accuracy": None,
            "comparability": "DEVELOPMENT_TRANSFORM_SUITE",
        },
        {
            "setting": "H4_NTIRE_final_external",
            "dataset": "NTIRE-RobustAIGenDetection-val@c762434",
            "auc": p["roc_auc"],
            "ap": p["ap"],
            "ai_recall": p["ai_recall"],
            "real_specificity": p["real_specificity"],
            "balanced_accuracy": p["balanced_accuracy"],
            "comparability": "SEALED_EXTERNAL",
        },
        {
            "setting": "DESCRIPTIVE_GAP_external_minus_dev_CLEAN",
            "dataset": "DATASETS_DIFFER_NOT_PAIRED_NO_DIRECT_STATISTICAL_TEST",
            "auc": p["roc_auc"] - h4r["CLEAN"]["auc"],
            "ap": p["ap"] - h4r["CLEAN"]["ap"],
            "ai_recall": p["ai_recall"] - h4r["CLEAN"]["ai_recall"],
            "real_specificity": p["real_specificity"] - h4r["CLEAN"]["real_specificity"],
            "balanced_accuracy": p["balanced_accuracy"] - h4r["CLEAN"]["balanced_accuracy"],
            "comparability": "DESCRIPTIVE_GAP_ONLY",
        },
        {
            "setting": "DESCRIPTIVE_GAP_external_minus_dev_mean_StrongRobust_AUC_spec",
            "dataset": "DATASETS_DIFFER_NOT_PAIRED_NO_DIRECT_STATISTICAL_TEST",
            "auc": p["roc_auc"] - h4r["mean_transformed"]["auc"],
            "ap": None,
            "ai_recall": p["ai_recall"] - h4r["mean_transformed"]["ai_recall"],
            "real_specificity": p["real_specificity"] - h4r["mean_transformed"]["real_specificity"],
            "balanced_accuracy": None,
            "comparability": "DESCRIPTIVE_GAP_ONLY",
        },
    ]
    return pd.DataFrame(rows)


def write_claims() -> str:
    text = """V2-12 SUPPORTED / UNSUPPORTED CLAIM REGISTER
Authority: existing stage artifacts + research_log. No new experiments.

======================================================================
SUPPORTED CLAIMS
======================================================================

1. V1 established a resource-aware lightweight detector (C0) with strong
   controlled internal evidence but historically poor independent external
   transfer (Stage 27A CONTEXT_ONLY vs NTIRE).

2. Frozen CLIP representations improved V2 development unseen-generator
   discrimination relative to continuing a V1-style lightweight RGB path
   on the V2 development protocol (V2-5/6/7).

3. Parameter-efficient LoRA improved development ranking (V2-8 AUC~0.903 vs
   MLP-B ~0.837) while degrading AI recall@0.5 under the MLP-B head
   (LORA_MIXED).

4. Adapted LoRA R1 retained strong local class structure on clean evaluation
   (V2-10A k5 purity ~0.966), indicating a head-limited failure mode for the
   original MLP-B operating point.

5. Small class-balanced linear heads (~513 parameters) can materially improve
   clean operating performance on frozen R1 (V2-10B H2).

6. Strong social-media-style transformations can induce large systematic score
   / margin shifts without complete local representation collapse (V2-10D/E;
   transform k5 purities remain high; HEAD_SHIFT_DOMINANT).

7. Robustness-aware head training on frozen R1 (H3/H4) can substantially reduce
   catastrophic transform-driven collapses while preserving a ~513-parameter
   deployment head (V2-10F/H).

8. Development robustness and high clean development scores do not guarantee
   sealed independent-benchmark transfer (V2-11 WEAK_EXTERNAL_TRANSFER).

9. Final NTIRE transfer for frozen H4 was weak: ranking only modestly above
   chance with severe AI false negatives at threshold 0.5, while Real
   specificity remained high.

10. Sealed external evaluation after freeze is scientifically informative even
    when negative: it bounds claims that would otherwise be overstated from
    development alone.


======================================================================
UNSUPPORTED / PROHIBITED CLAIMS
======================================================================

1. H4 is state of the art. (No competitive leaderboard comparison performed.)

2. H4 robustly generalises to arbitrary unseen generators. (NTIRE weak; hard
   subset near chance; no generator IDs on val.)

3. H4 solves AI-image detection. (External AI recall ~0.13.)

4. StrongRobust success guarantees general social-media robustness. (Suite is
   limited; physical recapture not tested; NTIRE still weak.)

5. NTIRE weakness was definitively caused by one specific generator. (No
   official per-generator metadata on the val split used.)

6. One threshold adjustment would solve NTIRE. (No threshold sweep performed
   after freeze; would violate protocol and is not evidence-supported.)

7. Calibration would rescue NTIRE. (H4 was evaluated uncalibrated; H2/V2-9
   calibration findings must not be transferred.)

8. V2 significantly beats V1 on the same external benchmark. (Stage27 and
   NTIRE are not established as the same benchmark; CONTEXT_ONLY.)

9. NTIRE failure is proven to be the same H2 margin-shift mechanism. (No NTIRE
   R1 geometry / margin analysis was run.)

10. Best-fold or reweighted aggregation would be a valid primary NTIRE result.
    (Equal-weight ensemble was locked before result inspection.)

11. Post-NTIRE retraining/tuning is allowed. (POST_NTIRE_MODEL_MODIFICATION_ALLOWED=NO.)
"""
    return text


def write_contributions() -> str:
    return """V2-12 FINAL CONTRIBUTIONS (candidate; evidence-grounded; not inflated)

C1. Resource-aware, protocol-locked detector development framework
    Support: V1 Stages 25–28; V2-0 protocol freeze; contamination guard;
    V2-10I SHA256 freeze + NTIRE protocol lock before access.
    Substance: Shows how to run a multi-stage detector programme with
    sealed external validation and explicit resource accounting.

C2. Modern multi-generator development evaluation for unseen-generator risk
    Support: V2-3 dataset/folds; V2-5..V2-10H hard-generator tables
    (Seedream/FLUX residual documented).
    Substance: Development evidence that foundation embeddings + adaptation
    improve controlled unseen-generator discrimination vs lightweight V1 path,
    while exposing residual modern-generator failures.

C3. LoRA representation diagnostic separating ranking gains from operating failure
    Support: V2-8 LORA_MIXED; V2-9A–D; V2-10A HEAD_LIMITED (R1 k5~0.966).
    Substance: Demonstrates that adaptation can improve ranking yet leave a
    head that underutilizes a class-structured representation at a fixed threshold.

C4. Head-limited robustness mechanism under StrongRobust transforms
    Support: V2-10D collapses; V2-10E HEAD_SHIFT_DOMINANT with preserved k5
    purity under JPEG/resize/blur/screenshot.
    Substance: Provides mechanistic development evidence that transform failures
    can be decision-direction / margin failures rather than complete local
    representation collapse. Scope: development StrongRobust; not extrapolated
    as the proven cause of NTIRE failure.

C5. Tiny robustness-aware linear head intervention (H3→H4)
    Support: V2-10F/G/H; ~513-parameter heads; mean tf RealSpec H2~0.636 → H4~0.883
    with catastrophic JPEG/resize collapses largely corrected; H4 clean/robust compromise.
    Substance: Shows offline transform-weighted linear probing can buy substantial
    development robustness without enlarging the deployment head.

C6. Sealed external validation revealing a persistent generalisation gap
    Support: V2-10I freeze; V2-11 NTIRE WEAK_EXTERNAL_TRANSFER (AUC0.625, AI_rec0.130).
    Substance: Establishes that substantial internal development gains are insufficient
    evidence of broad independent transfer; negative sealed result is a core contribution.

Non-contributions / do not claim as contributions:
- SOTA detector performance
- Solved social-media robustness
- Solved NTIRE / wild AI detection
"""


def write_limitations() -> str:
    return """V2-12 FINAL LIMITATIONS REGISTER

1. Sealed NTIRE external transfer is weak (WEAK_EXTERNAL_TRANSFER).
2. Severe external AI false negatives (FN 4352/5000; AI recall@0.5 ≈ 0.130).
3. Hard external diagnostic subset near chance (AUC ≈ 0.551).
4. No official NTIRE generator identities on the val labels used.
5. Final detector is externally conservative / Real-biased (AI mean p≈0.211; 87% AI < 0.5).
6. Development folds cannot represent full future generator diversity.
7. Residual development weaknesses: Seedream (~0.03 recall), FLUX (~0.26), Fold4 weaker.
8. StrongRobust suite limited to selected transforms; physical screen recapture not tested.
9. H4 missed V2-10H PROMISING gates (MIXED freeze with documented limitations).
10. Selective/calibration layers are not part of frozen H4 primary binary output.
11. No post-NTIRE model modification allowed; external result accepted as observed.
12. V1 Stage27 external vs NTIRE are not directly comparable (CONTEXT_ONLY).
13. External evidence comes from one sealed benchmark revision.
14. NTIRE R1 geometry / margin mechanism was NOT analysed; do not assert identical
    head-shift causality for NTIRE failure.
15. Paper must not soften these limitations into success language.
"""


def write_narrative() -> str:
    return """V2-12 PAPER RESULTS NARRATIVE OUTLINE (factual; not Discussion prose)

A. V1 internal success and external failure motivates V2
   - FINAL_RESEARCH_MODEL_V1=C0 selected after RQ1–RQ5 + resource synthesis.
   - Stage27A independent external historically failed (~AUC0.516).
   - Mark Stage27 vs NTIRE as CONTEXT_ONLY unless exact comparability is shown.

B. Frozen CLIP improves development generalisation on V2 protocol
   - V2-5/6 linear probes: AUC~0.815→0.820; CLIP_BASELINE_PROMISING / REFINED.
   - V2-7 MLP-B: modest gain AUC~0.837 (MLP_MODEST_GAIN).

C. LoRA improves ranking but creates poor fixed-threshold behaviour with MLP-B
   - V2-8: AUC~0.903 / AP~0.917 but AI_rec@0.5~0.386 (LORA_MIXED).

D. Calibration does not safely resolve the LoRA+MLP-B operating problem
   - V2-9B protocol unsafe; V2-9D temperature helps proper scores but preserves .5
     decisions; Platt increases recall with unacceptable Real FPs → CALIBRATION_NOT_HELPFUL.

E. Representation analysis shows LoRA R1 is useful but the head is limiting
   - V2-10A: R1 k5 purity ~0.966; HEAD_LIMITED.

F. H2 balanced linear head rescues clean development performance
   - V2-10B: CLEAN AUC~0.951 / AI_rec~0.576 / RealSpec~0.967 (HEAD_RESCUE_PROMISING).
   - V2-10C: temperature improves confidence quality; selective MIXED (~16% errors at S90).

G. StrongRobust exposes transformation-induced collapse for H2
   - V2-10D: JPEG AI_rec~0.089; resize RealSpec~0.131; ROBUSTNESS_NOT_ACCEPTABLE.

H. Geometry shows development robustness remains head-limited
   - V2-10E: transform k5 purities high; HEAD_SHIFT_DOMINANT; do not claim this explains NTIRE.

I. H3/H4 show robustness can be improved cheaply at the head level
   - H3 rescues mean tf AUC/RealSpec; clean cost.
   - H4 clean-anchored compromise: CLEAN AUC~0.926; mean tf AUC~0.826; RealSpec~0.883.

J. H4 frozen as clean/robust development compromise with documented limitations
   - V2-10I: FINAL_RESEARCH_MODEL_V2=H4; PROMISING gates missed; Seedream/FLUX/Fold4 remain.

K. Sealed NTIRE reveals weak external transfer despite development gains
   - V2-11: N=10000; equal-weight ensemble; AUC0.625; AI_rec0.130; RealSpec0.956;
     hard AUC~0.551; WEAK_EXTERNAL_TRANSFER; H4 unchanged.

L. Implication for Results closing sentence
   - Internal unseen-generator and transform-robustness improvements are insufficient
     evidence of broad external generalisation under sealed evaluation.
"""


def write_discussion() -> str:
    return """V2-12 DISCUSSION THEMES (for later writing)
Each theme lists RESULT / INTERPRETATION / SPECULATION separately.

THEME 1 — Representation quality vs decision-head quality
RESULT: R1 k5 purity high; MLP-B recall low; H2 rescues clean; H2 collapses under transforms via margin shift (V2-10A/B/D/E).
INTERPRETATION: In development, useful representations can be underutilized or misaligned by a small head.
SPECULATION: Similar head/repr mismatch might contribute to NTIRE failure (NOT tested; no NTIRE geometry).

THEME 2 — Development robustness vs true domain generalisation
RESULT: H4 StrongRobust much better than H2; NTIRE still weak.
INTERPRETATION: Controlled transform suites and development generators are not substitutes for sealed external benchmarks.
SPECULATION: Broader transform/generator coverage might have predicted NTIRE better (untested).

THEME 3 — Conservative operating behaviour
RESULT: External AI scores low (87% <0.5) with high Real specificity.
INTERPRETATION: At thr=0.5 the frozen ensemble behaves Real-biased on NTIRE.
SPECULATION: Score bias could reflect domain shift, generator mix, distortion mix, or head orientation (causes not identified).

THEME 4 — Resource efficiency
RESULT: Final deployment head ~513 params / ~2.9KB; LoRA adapters historically ~295k trainable; no new V2-11 training.
INTERPRETATION: Large robustness gains in development came from offline head redesign, not enlarging the online head.
SPECULATION: Even smaller backbones might suffice if R1-like structure is available (not tested in final path).

THEME 5 — Limits of benchmark-driven detector development
RESULT: Progressive internal improvements co-existed with sealed external weakness.
INTERPRETATION: Optimising to development folds can create a misleading competence narrative.
SPECULATION: Multiple sealed externals would further stress-test claims (not run; not recommending now).

THEME 6 — Importance of sealed external evaluation
RESULT: NTIRE opened only after freeze; aggregation locked before results; result accepted.
INTERPRETATION: Negative sealed outcomes are scientifically informative and protect claim integrity.
SPECULATION: Without sealing, post-hoc retuning might have inflated reported transfer (counterfactual).

THEME 7 — Why negative external results matter
RESULT: WEAK_EXTERNAL_TRANSFER with tight bootstrap CIs excluding strong ranking.
INTERPRETATION: The paper's contribution includes bounding overclaim from development success.
SPECULATION: Community detectors may share similar hidden transfer gaps (not evaluated here).
"""


def write_figure_table_plan() -> str:
    return """V2-12 CONFERENCE PAPER FIGURE/TABLE SHORTLIST
Prefer a shortlist; do not include every generated figure.

TABLE 1 — Datasets / evaluation protocol
  Content: V2 development pool + folds; StrongRobust suite; sealed NTIRE reservation/revision;
           equal-weight ensemble; thr=0.5; freeze-before-access.
  Sources: results/v2/v2_11_ntire_protocol_lock_v1.json;
           results/v2/v2_10i_final_model_specification_v1.json;
           paper/research_log.md (V2-1/V2-3/V2-10I/V2-11).

TABLE 2 — V2 model progression (clean development)
  Content: CLIP LogReg → refined linear → MLP-B → LoRA+MLP-B → H2 → H3 → H4.
  Source: results/v2/v2_12_v2_model_progression_v1.csv
          (built from v2_10i_development_model_comparison_v1.csv).

TABLE 3 — Robustness H2 vs H3 vs H4 (StrongRobust)
  Content: per-transform AI recall / RealSpec / AUC + mean transformed.
  Sources: results/v2/v2_10d_robustness_metrics_v1.csv;
           results/v2/v2_10f_robust_head_metrics_v1.csv;
           results/v2/v2_10h_metrics_v1.csv.

TABLE 4 — Final NTIRE external result
  Content: primary metrics + 95% bootstrap CIs + confusion; optional clean/distorted strata.
  Sources: results/v2/v2_11_ntire_metrics_v1.json;
           results/v2/v2_11_ntire_bootstrap_v1.json;
           results/v2/v2_12_development_external_gap_v1.csv.

FIGURE 1 — Method / research pipeline
  Content: V1→V2 freeze→CLIP→LoRA→head rescue→robust head→seal→NTIRE.
  Suggested compose from narrative; no single existing figure covers all.

FIGURE 2 — Representation / head diagnostic
  Preferred existing: figures/v2/ related to V2-10A geometry if present;
  else table+text from results/v2/v2_10a_representation_geometry_report_v1.txt
  and v2_10a_representation_geometry_v1.json (k5 purity R0/R1).

FIGURE 3 — Robustness score / margin shift
  Preferred: figures from V2-10E if present under figures/v2/;
  else summarize from results/v2/v2_10e_transform_shift_report_v1.txt
  (Δz JPEG AI strongly negative; resize Real strongly positive).

FIGURE 4 — NTIRE ROC or score distribution
  Existing: figures/v2/v2_11_ntire_roc_v1.png
            figures/v2/v2_11_ntire_pr_v1.png
            figures/v2/v2_11_ntire_score_distribution_v1.png
  Prefer ROC + score distribution for the main paper; PR optional/supplement.

Optional supplement only: fold diagnostics, hard-subset, generator residual tables
(Seedream/FLUX) from V2-10H/I — not primary conference figures.
"""


def write_readiness() -> str:
    return """V2-12 CONFERENCE-PAPER READINESS

EXPERIMENTS_COMPLETE = YES
FINAL_MODEL_FROZEN = YES
SEALED_EXTERNAL_VALIDATION_COMPLETE = YES
RESULTS_SYNTHESIS_COMPLETE = YES
PAPER_READY_FOR_FULL_DRAFT = YES

Interpretation:
  Experimental programme is closed. Evidence package is sufficient to draft full
  Methods/Results/Discussion from authoritative artifacts. Remaining work is
  writing/editing — not experimentation.

REMAINING NON-EXPERIMENTAL TASKS ONLY
- Final literature refresh and related-work positioning
- Venue formatting (LaTeX template, page limits, anonymity if needed)
- Full prose drafting of Introduction/Methods/Results/Discussion/Conclusion
- Citation verification (no hallucinated references)
- Table/figure formatting from shortlist into camera-ready assets
- Supplementary material curation (fold tables, hard-subset, limitations register)
- Proofreading / claim-check against v2_12_supported_claims_v1.txt
- Optional: update paper_draft.md sections to cite V2-11/V2-12 artifacts

DO NOT recommend another experiment.
DO NOT modify H4.
DO NOT retune NTIRE.
"""


def append_research_log(payload: dict[str, Any]) -> None:
    entry = f"""
## Stage V2-12 — Final Scientific Results Synthesis

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}  
**Status:** **COMPLETE — RESULTS_SYNTHESIS_COMPLETE**  
**Mode:** Documentation / evidence consolidation ONLY. No training, inference, thresholding, calibration, or model modification.

**Purpose:** Build the authoritative paper evidence package after experimental closure (V1 frozen; V2 H4 frozen; NTIRE evaluated once).

**Authoritative final state**
- FINAL_RESEARCH_MODEL_V1 = C0
- FINAL_RESEARCH_MODEL_V2 = H4
- DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS
- FINAL_EXTERNAL_VALIDATION = COMPLETE
- External-transfer assessment = WEAK_EXTERNAL_TRANSFER
- POST_NTIRE_MODEL_MODIFICATION_ALLOWED = NO

**Key consolidated finding**
Development produced strong clean and StrongRobust-improved H4 behaviour, but sealed NTIRE transfer remained weak (AUC≈0.625, AI_rec≈0.130, RealSpec≈0.956). Internal gains are insufficient evidence of broad external generalisation.

**Outputs:** `src/build_v2_12_final_synthesis_v1.py`; `results/v2/v2_12_*`.

**PAPER_READY_FOR_FULL_DRAFT = YES** (non-experimental writing tasks remain).

**Integrity:** MODEL_TRAINING=NO; MODEL_INFERENCE=NO; NEW_DATA=NO; NTIRE_REEVALUATED=NO; NTIRE_TUNING=NO; H4_MODIFIED=NO; THRESHOLD_CHANGED=NO; CALIBRATION=NO; NEW_MODEL_SELECTED=NO; NEXT_EXPERIMENT_STARTED=NO.

---
"""
    text = LOG.read_text()
    if "## Stage V2-12 — Final Scientific Results Synthesis" in text:
        print("[V2-12] research_log already contains V2-12; not duplicating", flush=True)
        return
    LOG.write_text(text.rstrip() + "\n" + entry)
    print("[V2-12] Appended research_log.md", flush=True)


def write_report(payload: dict[str, Any], h2r: dict, h3r: dict, h4r: dict, ntire: dict) -> str:
    p = ntire["primary_ensemble"]
    boot = load_json("results/v2/v2_11_ntire_bootstrap_v1.json")
    geom = load_json("results/v2/v2_10a_representation_geometry_v1.json")
    e10 = load_json("results/v2/v2_10e_transform_shift_analysis_v1.json")
    lines = []
    lines.append("V2-12 — FINAL SCIENTIFIC RESULTS SYNTHESIS")
    lines.append(f"Status: COMPLETE — RESULTS_SYNTHESIS_COMPLETE @ {payload['timestamp_utc']}")
    lines.append("FINAL_RESEARCH_MODEL_V1 = C0 | FINAL_RESEARCH_MODEL_V2 = H4")
    lines.append("EXPERIMENTS_COMPLETE=YES | PAPER_READY_FOR_FULL_DRAFT=YES")
    lines.append("")
    lines.append("ARTIFACT CONSISTENCY")
    lines.append(f"  required_missing={payload['artifact_audit']['n_missing']}")
    for d in payload["artifact_audit"]["discrepancies"]:
        lines.append(f"  DISCREPANCY[{d['item']}]: {d['resolution']}")
    lines.append("")
    lines.append("V2 PROGRESSION (clean development macros from freeze comparison)")
    lines.append("  CLIP LogReg ~0.815 → refined ~0.820 → MLP-B ~0.837 → LoRA+MLP-B ~0.903 (rec~0.386)")
    lines.append("  H2 ~0.951 → H3 ~0.918 → H4 ~0.926 (FINAL)")
    lines.append("")
    lines.append("ROBUSTNESS MECHANISM SUMMARY")
    lines.append(
        f"  H2 mean_tf AUC={h2r['mean_transformed']['auc']:.3f} RealSpec={h2r['mean_transformed']['real_specificity']:.3f}"
    )
    lines.append(
        f"    JPEG AI_rec={h2r['jpeg_q50']['ai_recall']:.3f}; resize RealSpec={h2r['resize_112']['real_specificity']:.3f}; "
        f"blur RealSpec={h2r['blur_sigma2']['real_specificity']:.3f}"
    )
    lines.append(
        f"  H4 mean_tf AUC={h4r['mean_transformed']['auc']:.3f} RealSpec={h4r['mean_transformed']['real_specificity']:.3f}"
    )
    lines.append(
        f"    JPEG AI_rec={h4r['jpeg_q50']['ai_recall']:.3f}; resize RealSpec={h4r['resize_112']['real_specificity']:.3f}; "
        f"blur RealSpec={h4r['blur_sigma2']['real_specificity']:.3f}; "
        f"screenshot AUC={h4r['screenshot_strong']['auc']:.3f}"
    )
    lines.append(
        f"  R1 clean k5 purity={geom['global_summary']['R1']['mean_neighbour_label_purity']:.3f}; "
        f"transform purities JPEG/Resize/Blur/Screenshot="
        f"{e10['geometry_macro']['jpeg_q50']['knn_purity']:.3f}/"
        f"{e10['geometry_macro']['resize_112']['knn_purity']:.3f}/"
        f"{e10['geometry_macro']['blur_sigma2']['knn_purity']:.3f}/"
        f"{e10['geometry_macro']['screenshot_strong']['knn_purity']:.3f}"
    )
    lines.append("  Interpretation: development robustness head-limited; NOT proven as NTIRE cause.")
    lines.append("")
    lines.append("CALIBRATION / SELECTIVE")
    lines.append("  V2-9D: temperature improves NLL/Brier, preserves .5 decisions; Platt unsafe Real FPs.")
    lines.append("  V2-10C: selective S90 enrich~2x but err_capture~0.16; most errors remain confident.")
    lines.append("  Do NOT transfer H2 calibration findings to frozen H4.")
    lines.append("")
    lines.append("NTIRE EXTERNAL SYNTHESIS")
    lines.append(
        f"  AUC={p['roc_auc']:.4f} AP={p['ap']:.4f} AI_rec={p['ai_recall']:.4f} "
        f"RealSpec={p['real_specificity']:.4f} BalAcc={p['balanced_accuracy']:.4f}"
    )
    lines.append(
        f"  Bootstrap AUC [{boot['roc_auc']['ci_low']:.3f},{boot['roc_auc']['ci_high']:.3f}]; "
        f"AI_rec [{boot['ai_recall']['ci_low']:.3f},{boot['ai_recall']['ci_high']:.3f}]"
    )
    lines.append("  Failure components (descriptive): RANKING GENERALISATION LOSS; STRONG REAL-WARD SCORE BIAS / LOW AI SCORES.")
    lines.append("  Assessment: WEAK_EXTERNAL_TRANSFER")
    lines.append("")
    lines.append("DEVELOPMENT–EXTERNAL GAP (DESCRIPTIVE; DATASETS DIFFER; NOT PAIRED)")
    lines.append(
        f"  ΔAUC_vs_CLEAN={p['roc_auc']-h4r['CLEAN']['auc']:.3f}; "
        f"ΔAI_rec={p['ai_recall']-h4r['CLEAN']['ai_recall']:.3f}; "
        f"ΔRealSpec={p['real_specificity']-h4r['CLEAN']['real_specificity']:.3f}"
    )
    lines.append("")
    lines.append("INTEGRITY")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[V2-12] Artifact audit ...", flush=True)
    audit = audit_artifacts()
    if audit["n_missing"]:
        raise SystemExit(f"STOP: missing required artifacts: {audit['missing']}")

    h2r = macro_robust(pd.read_csv(OUT / "v2_10d_robustness_metrics_v1.csv"))
    h3r = macro_robust(pd.read_csv(OUT / "v2_10f_robust_head_metrics_v1.csv"), "H3")
    h4r = macro_robust(pd.read_csv(OUT / "v2_10h_metrics_v1.csv"), "H4")
    ntire = load_json("results/v2/v2_11_ntire_metrics_v1.json")
    boot = load_json("results/v2/v2_11_ntire_bootstrap_v1.json")
    geom = load_json("results/v2/v2_10a_representation_geometry_v1.json")
    e10 = load_json("results/v2/v2_10e_transform_shift_analysis_v1.json")

    timeline = build_timeline()
    progression = build_progression(h2r, h3r, h4r)
    gap = build_dev_external_gap(h4r, ntire)

    timeline.to_csv(OUT / "v2_12_final_experiment_timeline_v1.csv", index=False)
    progression.to_csv(OUT / "v2_12_v2_model_progression_v1.csv", index=False)
    gap.to_csv(OUT / "v2_12_development_external_gap_v1.csv", index=False)

    claims = write_claims()
    contrib = write_contributions()
    limits = write_limitations()
    narrative = write_narrative()
    discussion = write_discussion()
    figplan = write_figure_table_plan()
    readiness = write_readiness()

    (OUT / "v2_12_supported_claims_v1.txt").write_text(claims)
    (OUT / "v2_12_final_contributions_v1.txt").write_text(contrib)
    (OUT / "v2_12_final_limitations_v1.txt").write_text(limits)
    (OUT / "v2_12_paper_results_narrative_v1.txt").write_text(narrative)
    (OUT / "v2_12_discussion_themes_v1.txt").write_text(discussion)
    (OUT / "v2_12_paper_figure_table_plan_v1.txt").write_text(figplan)
    (OUT / "v2_12_paper_readiness_v1.txt").write_text(readiness)

    integrity = {
        "MODEL_TRAINING_PERFORMED": "NO",
        "MODEL_INFERENCE_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "NTIRE_REEVALUATED": "NO",
        "NTIRE_TUNING_PERFORMED": "NO",
        "H4_MODIFIED": "NO",
        "THRESHOLD_CHANGED": "NO",
        "CALIBRATION_PERFORMED": "NO",
        "NEW_MODEL_SELECTED": "NO",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_FROZEN": "YES",
        "FINAL_EXTERNAL_VALIDATION_COMPLETE": "YES",
        "POST_NTIRE_MODEL_MODIFICATION_ALLOWED": "NO",
        "NEXT_EXPERIMENT_STARTED": "NO",
    }

    payload = {
        "stage": "V2-12",
        "status": "COMPLETE",
        "timestamp_utc": ts,
        "RESULTS_SYNTHESIS_COMPLETE": "YES",
        "PAPER_READY_FOR_FULL_DRAFT": "YES",
        "FINAL_RESEARCH_MODEL_V1": "C0",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_STATUS": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "EXTERNAL_TRANSFER_ASSESSMENT": "WEAK_EXTERNAL_TRANSFER",
        "artifact_audit": audit,
        "robustness_macros": {"H2": h2r, "H3": h3r, "H4": h4r},
        "geometry": {
            "r1_clean_k5_purity": geom["global_summary"]["R1"]["mean_neighbour_label_purity"],
            "transform_k5_purity": {
                k: e10["geometry_macro"][k]["knn_purity"]
                for k in ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
            },
            "h2_margin_shifts_macro": e10.get("margin_macro"),
            "scope_note": "Development StrongRobust mechanism only; no NTIRE R1 geometry.",
        },
        "ntire_primary": ntire["primary_ensemble"],
        "ntire_bootstrap": boot,
        "ntire_score_distributions": ntire.get("score_distributions"),
        "failure_components_descriptive": [
            "RANKING_GENERALISATION_LOSS",
            "STRONG_REAL_WARD_SCORE_BIAS_LOW_AI_SCORES",
        ],
        "key_statistical_evidence": {
            "ntire_bootstrap_95": {
                "roc_auc": [boot["roc_auc"]["ci_low"], boot["roc_auc"]["ci_high"]],
                "ap": [boot["ap"]["ci_low"], boot["ap"]["ci_high"]],
                "ai_recall": [boot["ai_recall"]["ci_low"], boot["ai_recall"]["ci_high"]],
                "real_specificity": [boot["real_specificity"]["ci_low"], boot["real_specificity"]["ci_high"]],
                "balanced_accuracy": [
                    boot["balanced_accuracy"]["ci_low"],
                    boot["balanced_accuracy"]["ci_high"],
                ],
            },
            "publication_priority": [
                "NTIRE primary metrics + bootstrap CIs (V2-11)",
                "H2 vs H4 StrongRobust paired/macro contrasts (V2-10D/H)",
                "H2 vs H3/H4 clean deltas (V2-10F/H)",
                "Optional: V2-10B H2 vs MLP-B head rescue contrast",
            ],
            "note": "Existing bootstrap JSONs reused; no rerun.",
        },
        "integrity_statement": integrity,
        "outputs": [
            "results/v2/v2_12_final_synthesis_analysis_v1.json",
            "results/v2/v2_12_final_synthesis_report_v1.txt",
            "results/v2/v2_12_final_experiment_timeline_v1.csv",
            "results/v2/v2_12_v2_model_progression_v1.csv",
            "results/v2/v2_12_development_external_gap_v1.csv",
            "results/v2/v2_12_supported_claims_v1.txt",
            "results/v2/v2_12_final_contributions_v1.txt",
            "results/v2/v2_12_final_limitations_v1.txt",
            "results/v2/v2_12_paper_results_narrative_v1.txt",
            "results/v2/v2_12_discussion_themes_v1.txt",
            "results/v2/v2_12_paper_figure_table_plan_v1.txt",
            "results/v2/v2_12_paper_readiness_v1.txt",
            "src/build_v2_12_final_synthesis_v1.py",
            "paper/research_log.md",
        ],
    }

    report = write_report(payload, h2r, h3r, h4r, ntire)
    (OUT / "v2_12_final_synthesis_report_v1.txt").write_text(report)
    (OUT / "v2_12_final_synthesis_analysis_v1.json").write_text(json.dumps(_clean(payload), indent=2) + "\n")
    append_research_log(payload)
    print(report)
    print("RESULTS_SYNTHESIS_COMPLETE = YES")
    print("PAPER_READY_FOR_FULL_DRAFT = YES")


if __name__ == "__main__":
    main()
