#!/usr/bin/env python3
"""V2-10I — Development closure / formal freeze of FINAL_RESEARCH_MODEL_V2 = H4.

NO training. NO inference. NO NTIRE access.
Documentation + SHA256 freeze only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"

FREEZE_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

MANIFEST_OUT = MODELS / "final_v2_h4_freeze_manifest_v1.json"
SPEC_JSON = OUT / "v2_10i_final_model_specification_v1.json"
SPEC_TXT = OUT / "v2_10i_final_model_specification_v1.txt"
CMP_CSV = OUT / "v2_10i_development_model_comparison_v1.csv"
LIM_TXT = OUT / "v2_10i_final_model_limitations_v1.txt"
NTIRE_JSON = OUT / "v2_11_ntire_protocol_lock_v1.json"
NTIRE_TXT = OUT / "v2_11_ntire_protocol_lock_v1.txt"
RQ_CSV = OUT / "v2_10i_research_question_status_v1.csv"
PAPER_TXT = OUT / "v2_10i_paper_selection_summary_v1.txt"
ANALYSIS_JSON = OUT / "v2_10i_freeze_analysis_v1.json"
REPORT_TXT = OUT / "v2_10i_freeze_report_v1.txt"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit() -> dict[str, Any]:
    required = []
    for fold in [1, 2, 3, 4]:
        required.append(
            (
                "H4_HEAD",
                MODELS / f"v2_10h_h4_clean_anchored_robust_logreg_fold{fold}_v1.joblib",
                fold,
            )
        )
        required.append(("LORA_CHECKPOINT", MODELS / f"clip_lora_fold{fold}_best_v1.pt", fold))
    required += [
        ("H4_ANALYSIS", OUT / "v2_10h_analysis_v1.json", None),
        ("H4_REPORT", OUT / "v2_10h_report_v1.txt", None),
        ("H4_METRICS", OUT / "v2_10h_metrics_v1.csv", None),
        ("H4_PREDICTIONS", OUT / "v2_10h_predictions_v1.csv", None),
        ("LORA_CONFIG", PROJECT_ROOT / "kaggle/v2_lora/configs/v2_lora_config_v1.json", None),
        ("SPLIT_MANIFEST", PROJECT_ROOT / "metadata/v2_split_assignments_v1.csv", None),
        ("CONTAMINATION_GUARD", PROJECT_ROOT / "src/v2_final_test_contamination_guard_v1.py", None),
        ("EXTERNAL_TRANSFORMS", PROJECT_ROOT / "src/external_v2_common.py", None),
        ("NTIRE_RESERVATION", PROJECT_ROOT / "results/v2_final_external_benchmark_reservation_v1.json", None),
        ("NTIRE_BACKUP", PROJECT_ROOT / "results/v2_final_external_backup_candidate_v1.json", None),
        ("H4_TRAIN_SCRIPT", PROJECT_ROOT / "src/run_v2_10h_clean_anchored_robust_head_v1.py", None),
        ("LORA_TRAIN_SCRIPT", PROJECT_ROOT / "kaggle/v2_lora/train_v2_clip_lora.py", None),
        ("H2_HEAD_F1", MODELS / "v2_10b_h2_balanced_logreg_fold1_v1.joblib", 1),
        ("H3_HEAD_F1", MODELS / "v2_10f_h3_robust_balanced_logreg_fold1_v1.joblib", 1),
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for _, p, _ in required if not p.is_file()]
    if missing:
        stop(f"required freeze artifacts missing: {missing}")
    return {"ok": True, "n_required": len(required), "missing": []}


def build_manifest() -> list[dict[str, Any]]:
    entries: list[tuple[str, Path, Any, str]] = []
    for fold in [1, 2, 3, 4]:
        entries.append(
            (
                "H4_HEAD",
                MODELS / f"v2_10h_h4_clean_anchored_robust_logreg_fold{fold}_v1.joblib",
                fold,
                "V2-10H",
            )
        )
        entries.append(
            (
                "LORA_CHECKPOINT",
                MODELS / f"clip_lora_fold{fold}_best_v1.pt",
                fold,
                "V2-8",
            )
        )
    # Supporting frozen provenance (not all are "weights", but required to reproduce)
    support = [
        ("LORA_CONFIG", PROJECT_ROOT / "kaggle/v2_lora/configs/v2_lora_config_v1.json", None, "V2-8"),
        ("SPLIT_MANIFEST", PROJECT_ROOT / "metadata/v2_split_assignments_v1.csv", None, "V2-3"),
        ("H4_TRAIN_SCRIPT", PROJECT_ROOT / "src/run_v2_10h_clean_anchored_robust_head_v1.py", None, "V2-10H"),
        ("LORA_TRAIN_SCRIPT", PROJECT_ROOT / "kaggle/v2_lora/train_v2_clip_lora.py", None, "V2-8"),
        ("PREPROCESS_TRANSFORMS", PROJECT_ROOT / "src/external_v2_common.py", None, "RQ3/V2-10D"),
        ("CLIP_LORA_MODEL_DEF", PROJECT_ROOT / "src/analyze_v2_10a_representation_geometry_v1.py", None, "V2-10A"),
        ("CONTAMINATION_GUARD", PROJECT_ROOT / "src/v2_final_test_contamination_guard_v1.py", None, "V2-1"),
        ("H4_ANALYSIS", OUT / "v2_10h_analysis_v1.json", None, "V2-10H"),
        ("H4_METRICS", OUT / "v2_10h_metrics_v1.csv", None, "V2-10H"),
        ("NTIRE_RESERVATION_METADATA", PROJECT_ROOT / "results/v2_final_external_benchmark_reservation_v1.json", None, "V2-1"),
        ("NTIRE_BACKUP_METADATA", PROJECT_ROOT / "results/v2_final_external_backup_candidate_v1.json", None, "V2-1"),
    ]
    entries.extend(support)

    rows = []
    for role, path, fold, stage in entries:
        if not path.is_file():
            stop(f"cannot hash missing file {path}")
        digest = sha256_file(path)
        rows.append(
            {
                "role": role,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "size_bytes": int(path.stat().st_size),
                "fold": fold,
                "model_stage": stage,
                "freeze_timestamp_utc": FREEZE_TS,
                "final_research_model_v2": "H4",
            }
        )
        print(f"[V2-10I] hashed {path.name} {digest[:12]}...", flush=True)
    return rows


def write_spec(h4: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads((PROJECT_ROOT / "kaggle/v2_lora/configs/v2_lora_config_v1.json").read_text())
    spec = {
        "document": "v2_10i_final_model_specification_v1",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_STATUS": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "FINAL_EXTERNAL_VALIDATION": "NOT_STARTED",
        "NTIRE_STATUS": "SEALED",
        "freeze_timestamp_utc": FREEZE_TS,
        "conceptual_stack": {
            "backbone": {
                "library": "open_clip_torch",
                "model_name": "ViT-B-16-quickgelu",
                "pretrained": "openai",
                "config_source": "kaggle/v2_lora/configs/v2_lora_config_v1.json",
            },
            "adaptation": {
                "stage": "V2-8",
                "method": "LoRA",
                "target": "last_4_visual_transformer_blocks",
                "target_modules": ["attn.in_proj", "attn.out_proj"],
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "checkpoints": [
                    f"models/v2/clip_lora_fold{f}_best_v1.pt" for f in [1, 2, 3, 4]
                ],
            },
            "representation": {
                "name": "R1",
                "definition": "LoRA-adapted visual embedding before classification head",
                "D": 512,
                "l2_normalized": True,
            },
            "head": {
                "name": "H4",
                "full_name": "CLEAN_ANCHORED_ROBUST_LOGREG_BALANCED",
                "type": "sklearn.linear_model.LogisticRegression",
                "penalty": "l2",
                "C": 1.0,
                "fit_intercept": True,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 2000,
                "n_params_approx": 513,
                "serialized_size_approx_bytes": 2907,
                "models": [
                    f"models/v2/v2_10h_h4_clean_anchored_robust_logreg_fold{f}_v1.joblib"
                    for f in [1, 2, 3, 4]
                ],
            },
            "threshold": {"binary_threshold": 0.5, "threshold_tuned": False},
            "offline_head_training_condition_weights": {
                "CLEAN": 4.0,
                "jpeg_q50": 1.0,
                "resize_112": 1.0,
                "blur_sigma2": 1.0,
                "screenshot_strong": 1.0,
                "effective_clean_fraction": 0.5,
                "effective_transformed_fraction": 0.5,
                "note": "OFFLINE sample_weight only; deployment inference uses CLEAN image once; transform identity is NOT an input feature.",
            },
            "preprocessing": {
                "inference": "open_clip create_model_and_transforms preprocess for ViT-B-16-quickgelu/openai; EXIF transpose + RGB",
                "source": "kaggle/v2_lora/train_v2_clip_lora.py + analyze_v2_10a ClipLoRAModel",
                "strongrobust_offline_defs": "src/external_v2_common.py TRANSFORM_FNS (training-time only for H4)",
            },
            "calibration_selective": {
                "primary_external_output": "UNCALIBRATED H4 binary p(AI) at threshold 0.5",
                "h2_temperature_selective": "SCIENTIFIC_DEVELOPMENT_FINDING_ONLY_NOT_TRANSFERRED_TO_H4",
                "h4_calibrator_fitted_at_freeze": False,
            },
        },
        "lora_config_echo": cfg.get("clip"),
        "lora_adaptation_echo": cfg.get("lora"),
        "authoritative_h4_clean_macro": h4["macro"]["H4"]["CLEAN"],
        "authoritative_h4_tradeoff": h4["tradeoff"]["H4"],
        "v2_10h_decision": h4["decision"],
    }
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    lines = [
        "V2-10I FINAL MODEL SPECIFICATION",
        f"Freeze timestamp (UTC): {FREEZE_TS}",
        "",
        "FINAL_RESEARCH_MODEL_V2 = H4",
        "DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "FINAL_EXTERNAL_VALIDATION = NOT_STARTED",
        "NTIRE_STATUS = SEALED",
        "",
        "Conceptual stack:",
        "  Frozen CLIP ViT-B-16-quickgelu (OpenAI pretrained, open_clip)",
        "  + V2-8 fold-specific LoRA (last 4 visual blocks; attn.in_proj + attn.out_proj; r=8, alpha=16, dropout=0.05)",
        "  + R1 512-d L2-normalized visual representation",
        "  + H4 class-balanced clean-anchored robust Logistic Regression",
        "",
        "H4 head:",
        "  penalty=L2; C=1.0; fit_intercept=True; class_weight=balanced; solver=lbfgs; max_iter=2000",
        "  binary threshold = 0.5 (not tuned)",
        "  ~513 learned parameters; ~2.9 KB serialized",
        "",
        "Offline TRAIN condition sample weights (LOCKED; no search):",
        "  CLEAN=4, jpeg_q50=1, resize_112=1, blur_sigma2=1, screenshot_strong=1",
        "  => 50% clean / 50% transformed by sample-weight mass",
        "  Deployment does NOT require transformed copies at inference.",
        "",
        "Primary external output:",
        "  UNCALIBRATED H4 binary p(AI) at p=0.5 (+ threshold-free ranking metrics where labels permit).",
        "  Do NOT transfer H2 temperature/selective policies to H4.",
        "",
        "Artifacts:",
    ]
    for f in [1, 2, 3, 4]:
        lines.append(f"  LoRA: models/v2/clip_lora_fold{f}_best_v1.pt")
        lines.append(f"  H4:   models/v2/v2_10h_h4_clean_anchored_robust_logreg_fold{f}_v1.joblib")
    SPEC_TXT.write_text("\n".join(lines) + "\n")
    return spec


def write_comparison() -> pd.DataFrame:
    v25 = json.loads((OUT / "v2_clip_logreg_summary_v1.json").read_text())
    v26 = pd.read_csv(OUT / "v2_clip_logreg_refinement_summary_v1.csv")
    v26s = v26[v26["selected"] == True].iloc[0]
    v27 = json.loads((OUT / "v2_clip_mlp_selected_config_v1.json").read_text())["selected_summary"]
    v28 = json.loads((OUT / "v2_8_analysis_v1.json").read_text())["cross_fold"]
    h = json.loads((OUT / "v2_10h_analysis_v1.json").read_text())
    t = h["tradeoff"]

    rows = [
        {
            "model": "V1_C0_MobileNet_RQ3_A2",
            "context": "V1_controlled_test_INCOMPARABLE_to_V2_dev_eval",
            "clean_auc": 0.850321,
            "clean_ap": 0.868683,
            "ai_recall_at_05_or_frozen_thr": None,
            "real_specificity": None,
            "balanced_accuracy": None,
            "robustness_summary": "V1 StrongRobustTestAUC unseen ~0.804 (historical); external Stage27 later severe failure — see V1 records",
            "head_or_model_params": 1518881,
            "decision_or_status": "FINAL_RESEARCH_MODEL_V1",
            "notes": "Metrics from V1 unseen-original / Stage26 evidence; INCOMPARABLE protocol/threshold/data vs V2; do not pool",
        },
        {
            "model": "V2-5_CLIP_LogReg",
            "context": "V2_dev_eval_macro",
            "clean_auc": float(v25["mean_heldout_auc"]),
            "clean_ap": float(v25["mean_ap"]),
            "ai_recall_at_05_or_frozen_thr": float(v25["mean_ai_recall"]),
            "real_specificity": float(v25["mean_real_specificity"]),
            "balanced_accuracy": float(v25["mean_balanced_accuracy"]),
            "robustness_summary": "not_StrongRobust_evaluated_at_this_stage",
            "head_or_model_params": 513,
            "decision_or_status": "CLIP_LINEAR_BASELINE",
            "notes": "Frozen CLIP embeddings + LogReg",
        },
        {
            "model": "V2-6_CLIP_LogReg_C3",
            "context": "V2_dev_eval_macro",
            "clean_auc": float(v26s["mean_roc_auc"]),
            "clean_ap": float(v26s["mean_ap"]),
            "ai_recall_at_05_or_frozen_thr": float(v26s["mean_ai_recall"]),
            "real_specificity": float(v26s["mean_real_specificity"]),
            "balanced_accuracy": None,
            "robustness_summary": "not_StrongRobust_evaluated_at_this_stage",
            "head_or_model_params": 513,
            "decision_or_status": "REFINED_LINEAR_SELECTED_C3",
            "notes": "C=3.0 selected by V2-6 reliability guards",
        },
        {
            "model": "V2-7_CLIP_MLP-B",
            "context": "V2_dev_eval_macro",
            "clean_auc": float(v27["mean_roc_auc"]),
            "clean_ap": float(v27["mean_ap"]),
            "ai_recall_at_05_or_frozen_thr": float(v27["mean_ai_recall"]),
            "real_specificity": float(v27["mean_real_specificity"]),
            "balanced_accuracy": None,
            "robustness_summary": "not_StrongRobust_evaluated_at_this_stage",
            "head_or_model_params": int(v27["trainable_params"]),
            "decision_or_status": "MLP_MODEST_GAIN",
            "notes": "Frozen CLIP + MLP-B 512-256-64-1",
        },
        {
            "model": "V2-8_LoRA_MLP-B",
            "context": "V2_dev_eval_macro",
            "clean_auc": float(v28["lora_mean_auc"]),
            "clean_ap": float(v28["lora_mean_ap"]),
            "ai_recall_at_05_or_frozen_thr": float(v28["lora_mean_ai_recall"]),
            "real_specificity": float(v28["lora_mean_real_spec"]),
            "balanced_accuracy": None,
            "robustness_summary": "not_StrongRobust_primary_at_V2-8; later H2 showed collapse",
            "head_or_model_params": "LoRA+MLP-B (MLP-B 147841 + LoRA adapters)",
            "decision_or_status": "LORA_MIXED",
            "notes": "Strong ranking; AI recall@0.5 ~0.386",
        },
        {
            "model": "H2_R1_balanced_LogReg",
            "context": "V2_dev_eval_macro",
            "clean_auc": t["H2"]["clean_auc"],
            "clean_ap": t["H2"]["clean_ap"],
            "ai_recall_at_05_or_frozen_thr": t["H2"]["clean_ai_recall"],
            "real_specificity": t["H2"]["clean_real_spec"],
            "balanced_accuracy": t["H2"]["clean_balacc"],
            "robustness_summary": f"mean_tf_AUC={t['H2']['mean_tf_auc']:.3f}; worst_tf_RealSpec={t['H2']['worst_tf_real_spec']:.3f} (ROBUSTNESS_NOT_ACCEPTABLE)",
            "head_or_model_params": 513,
            "decision_or_status": "HEAD_RESCUE_PROMISING / Robustness fail",
            "notes": "Best clean score; catastrophic StrongRobust collapses",
        },
        {
            "model": "H3_equal_condition_robust_LogReg",
            "context": "V2_dev_eval_macro",
            "clean_auc": t["H3"]["clean_auc"],
            "clean_ap": t["H3"]["clean_ap"],
            "ai_recall_at_05_or_frozen_thr": t["H3"]["clean_ai_recall"],
            "real_specificity": t["H3"]["clean_real_spec"],
            "balanced_accuracy": t["H3"]["clean_balacc"],
            "robustness_summary": f"mean_tf_AUC={t['H3']['mean_tf_auc']:.3f}; mean_tf_RealSpec={t['H3']['mean_tf_real_spec']:.3f}",
            "head_or_model_params": 513,
            "decision_or_status": "ROBUST_HEAD_MIXED",
            "notes": "20/80 clean/transform rows; over-rotated per V2-10G",
        },
        {
            "model": "H4_clean_anchored_robust_LogReg",
            "context": "V2_dev_eval_macro",
            "clean_auc": t["H4"]["clean_auc"],
            "clean_ap": t["H4"]["clean_ap"],
            "ai_recall_at_05_or_frozen_thr": t["H4"]["clean_ai_recall"],
            "real_specificity": t["H4"]["clean_real_spec"],
            "balanced_accuracy": t["H4"]["clean_balacc"],
            "robustness_summary": f"mean_tf_AUC={t['H4']['mean_tf_auc']:.3f}; mean_tf_RealSpec={t['H4']['mean_tf_real_spec']:.3f}; worst_tf_AUC={t['H4']['worst_tf_auc']:.3f}",
            "head_or_model_params": 513,
            "decision_or_status": "CLEAN_ANCHORED_ROBUST_HEAD_MIXED → FINAL_RESEARCH_MODEL_V2",
            "notes": "Selected development compromise; PROMISING gates missed; external unknown",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(CMP_CSV, index=False)
    return df


def write_limitations(h4: dict[str, Any]) -> None:
    t = h4["tradeoff"]["H4"]
    seed = h4["seedream"]
    flux = h4["flux"]
    f4 = h4["fold4"]["CLEAN"]["H4"]
    text = f"""V2-10I FINAL MODEL LIMITATION REGISTER
FINAL_RESEARCH_MODEL_V2 = H4
Status: FROZEN_WITH_DOCUMENTED_LIMITATIONS
Freeze timestamp (UTC): {FREEZE_TS}

These limitations remain visible and must not be softened.

1. Seedream-5.0 weakness
   H4 CLEAN recall remains ~{seed['clean_recall_H4']:.3f} (H2 ~{seed['clean_recall_H2']:.3f}; H3 ~{seed['clean_recall_H3']:.3f}).
   Clean-anchored weighting did NOT recover the Seedream collapse.

2. FLUX.2_max weakness
   H4 CLEAN recall ~{flux['clean_recall_H4']:.3f} (H2 ~{flux['clean_recall_H2']:.3f}; H3 ~{flux['clean_recall_H3']:.3f}).
   FLUX remains a primary residual AI weakness under H4.

3. Fold4 weaker behaviour
   Fold4 H4 CLEAN AUC ~{f4['roc_auc']:.3f} (vs H2 ~{h4['fold4']['CLEAN']['H2']['roc_auc']:.3f}).
   Fold4 continues to underperform Folds 1–3 under the frozen detector.

4. Clean / robust trade-off
   H4 CLEAN AUC ~{t['clean_auc']:.3f} is below H2 (~{h4['tradeoff']['H2']['clean_auc']:.3f}).
   H4 was selected as a compromise, not as a clean-optimal detector.

5. StrongRobust AUC remains imperfect
   Mean transformed AUC ~{t['mean_tf_auc']:.3f}; worst transformed AUC ~{t['worst_tf_auc']:.3f}.
   Robustness is substantially improved vs H2 but is NOT solved completely.

6. H4 missed V2-10H PROMISING gates
   Decision label: {h4['decision']}.
   Predeclared PROMISING criteria were not all met; freeze proceeds with documented MIXED status.

7. Selective prediction was MIXED and is not a complete error solution
   V2-10C (H2) selective prediction captured only a minority of errors.
   Selective prediction is NOT part of the mandatory final binary detector.
   H2 temperature/selective findings are NOT transferred to H4.

8. Calibration experiments did not yield a safe global operating rescue for the original MLP-B
   V2-9B: CALIBRATION_PROTOCOL_NOT_SAFE.
   V2-9D: CALIBRATION_NOT_HELPFUL.
   Frozen H4 external evaluation is UNCALIBRATED binary output at p=0.5.

9. Final external transfer remains unknown before NTIRE
   FINAL_EXTERNAL_PERFORMANCE = UNKNOWN.
   Do NOT claim H4 generalises to NTIRE.
   Do NOT predict NTIRE scores.
   NTIRE remains SEALED until V2-11.

Language for papers/reports:
  "selected development compromise"
  "final V2 model with documented robustness/generalisation limitations"
"""
    LIM_TXT.write_text(text)


def write_ntire_protocol() -> dict[str, Any]:
    reservation = json.loads(
        (PROJECT_ROOT / "results/v2_final_external_benchmark_reservation_v1.json").read_text()
    )
    backup = json.loads(
        (PROJECT_ROOT / "results/v2_final_external_backup_candidate_v1.json").read_text()
    )
    proto = {
        "document": "v2_11_ntire_protocol_lock_v1",
        "created_at_utc": FREEZE_TS,
        "created_during_stage": "V2-10I",
        "NTIRE_STATUS_DURING_THIS_DOCUMENT": "SEALED",
        "NTIRE_ACCESSED_WHILE_WRITING": "NO",
        "images_opened": False,
        "labels_opened": False,
        "filenames_inspected_beyond_prior_reservation_metadata": False,
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "active_final_test": {
            "role": reservation.get("dataset_role"),
            "dataset_id": reservation.get("dataset_id"),
            "revision_sha": reservation.get("revision_sha"),
            "FINAL_TEST_LOCKED": reservation.get("FINAL_TEST_LOCKED"),
            "contamination_guard": "src/v2_final_test_contamination_guard_v1.py",
            "label_usability_status_from_reservation": reservation.get("label_usability_status"),
        },
        "backup_if_ntire_unusable": {
            "rule": backup.get("selection_rule"),
            "backup_dataset_id": backup.get("backup_dataset_id"),
            "backup_revision_sha": backup.get("backup_revision_sha"),
            "note": "Use NTIRE if usable at V2-11 regardless of score; do not cherry-pick between tests.",
        },
        "locked_before_opening_ntire": {
            "model": "H4 = frozen CLIP + V2-8 LoRA fold checkpoints + H4 LogReg heads",
            "representation": "R1 D=512 L2-normalized",
            "preprocessing": "open_clip ViT-B-16-quickgelu/openai preprocess; EXIF transpose + RGB; no StrongRobust transforms at inference unless separately declared as evaluation conditions",
            "binary_threshold": 0.5,
            "threshold_tuning_allowed": False,
            "calibration_fitting_allowed": False,
            "selective_policy_retuning_allowed": False,
            "retraining_allowed": False,
            "architecture_changes_allowed": False,
            "sample_exclusion_after_viewing_results_allowed": False,
            "external_result_driven_model_changes_allowed": False,
            "fold_inference_aggregation": {
                "intent": "Use existing four fold-specific LoRA+H4 pairs; aggregation strategy must follow prior project practice for multi-fold V2 detectors",
                "locked_principle": "No development-data retuning of aggregation after seeing NTIRE outcomes",
                "pre_execution_detail": (
                    "Exact fold-routing / probability aggregation (e.g., per-image fold assignment vs "
                    "mean of four fold models) is an implementation detail to be resolved at V2-11 "
                    "from existing V2 development protocol records WITHOUT using NTIRE labels for tuning."
                ),
            },
            "metrics_where_labels_permit": [
                "ROC-AUC",
                "AP",
                "AI_recall@0.5",
                "Real_specificity@0.5",
                "balanced_accuracy",
                "precision",
                "F1",
                "optional_domain_or_generator_breakdowns_if_metadata_available",
            ],
            "primary_score": "UNCALIBRATED H4 p(AI) at threshold 0.5",
            "ranking_metrics": "report threshold-free AUC/AP whenever labels permit",
        },
        "acceptance_rule": "NTIRE (or declared backup) results must be accepted as observed; no post-hoc model surgery.",
        "v2_10i_note": "This protocol is locked without opening NTIRE contents.",
    }
    NTIRE_JSON.write_text(json.dumps(proto, indent=2) + "\n")
    lines = [
        "V2-11 NTIRE PROTOCOL LOCK (written during V2-10I; NTIRE SEALED)",
        f"Created: {FREEZE_TS}",
        "",
        "NTIRE_ACCESSED_WHILE_WRITING = NO",
        "NTIRE_STATUS = SEALED",
        "",
        f"Active final test (from prior reservation only): {reservation.get('dataset_id')}",
        f"Revision: {reservation.get('revision_sha')}",
        f"FINAL_TEST_LOCKED: {reservation.get('FINAL_TEST_LOCKED')}",
        "",
        "Frozen model for V2-11: FINAL_RESEARCH_MODEL_V2 = H4",
        "Threshold: 0.5 (no tuning)",
        "Output: UNCALIBRATED H4 binary p(AI)",
        "No retraining / calibration / selective retune / architecture change after viewing results.",
        "",
        "Backup rule: if NTIRE unusable at V2-11, activate reserved backup per",
        "results/v2_final_external_backup_candidate_v1.json — do not score-shop.",
        "",
        "Fold aggregation exact mechanics: pre-execution implementation detail at V2-11,",
        "resolved from existing protocol records without NTIRE-label tuning.",
        "",
        "Contamination guard remains active until V2-11 explicit authorisation.",
    ]
    NTIRE_TXT.write_text("\n".join(lines) + "\n")
    return proto


def write_rq_status() -> pd.DataFrame:
    # Preserve authoritative V1 RQ numbering from research_log; map V2 evidence alongside.
    rows = [
        {
            "question_id": "RQ1",
            "authoritative_label": "Lightweight Real vs AI discrimination under known/unseen generators",
            "v2_stages_providing_evidence": "V2-5..V2-10B; H2/H3/H4; hard-generator tables",
            "principal_finding": "Frozen CLIP linear/MLP then LoRA R1+linear heads substantially improve V2 development discrimination; H2/H4 reach high clean AUC with modern-generator residual failures (Seedream/FLUX).",
            "unresolved_limitation": "Hard modern generators remain weak; Fold4 composition effects; external confirmation pending.",
            "final_external_validation_required": "YES",
        },
        {
            "question_id": "RQ2",
            "authoritative_label": "Real-domain diversity / specificity (V1 social-media robustness theme; V2 Real domains)",
            "v2_stages_providing_evidence": "V2-5..V2-10H Real-domain tables; StrongRobust V2-10D/F/H",
            "principal_finding": "Smartphone/COCO Real specificity generally high under H4; MLLM Real improved under robust heads vs H2 resize/blur collapse; Tiny residual under some transforms.",
            "unresolved_limitation": "Domain safety under StrongRobust not perfect; physical recapture not tested.",
            "final_external_validation_required": "YES",
        },
        {
            "question_id": "RQ3",
            "authoritative_label": "Transformation-aware robustness (V1) / V2 LoRA+robust-head analogue",
            "v2_stages_providing_evidence": "V2-8 LoRA; V2-10D/E/F/G/H StrongRobust + clean-anchored head",
            "principal_finding": "LoRA representation remains class-structured under transforms (V2-10E); robustness bottleneck was head direction; H3/H4 offline transform-weighted heads rescue catastrophic H2 collapses.",
            "unresolved_limitation": "Mean/worst StrongRobust AUC imperfect; not a complete solve.",
            "final_external_validation_required": "YES",
        },
        {
            "question_id": "V2_LoRA_adaptation",
            "authoritative_label": "Parameter-efficient CLIP adaptation (V2-8)",
            "v2_stages_providing_evidence": "V2-8; V2-10A geometry",
            "principal_finding": "LORA_MIXED: ranking up, recall@0.5 down for MLP-B; R1 geometry strong (k5~0.966) justifying head rescue.",
            "unresolved_limitation": "Original LoRA+MLP-B operating point unsatisfactory without head redesign.",
            "final_external_validation_required": "YES",
        },
        {
            "question_id": "RQ5_analogue",
            "authoritative_label": "Calibration / selective confidence",
            "v2_stages_providing_evidence": "V2-9A..D; V2-10C",
            "principal_finding": "Operating-point shift primary for LoRA MLP-B; calibration protocol not safe / not helpful for that head; H2 selective MIXED and incomplete.",
            "unresolved_limitation": "No safe calibrator transferred to H4; primary detector remains uncalibrated binary.",
            "final_external_validation_required": "YES",
        },
        {
            "question_id": "RESOURCE",
            "authoritative_label": "Parameter / resource efficiency",
            "v2_stages_providing_evidence": "V2-10B/F/H; V2-10I freeze",
            "principal_finding": "H4 deployment head ~513 params / ~2.9KB on frozen LoRA R1; offline transform weighting does not increase deployment size.",
            "unresolved_limitation": "LoRA backbone still larger than V1 MobileNet; resource story is head-light not backbone-free.",
            "final_external_validation_required": "YES",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RQ_CSV, index=False)
    return df


def write_paper_summary(h4: dict[str, Any]) -> None:
    t = h4["tradeoff"]
    text = f"""V2-10I PAPER-READY SELECTION SUMMARY (factual evidence; not Discussion prose)
Freeze timestamp (UTC): {FREEZE_TS}
FINAL_RESEARCH_MODEL_V2 = H4
DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS
FINAL_EXTERNAL_PERFORMANCE = UNKNOWN (NTIRE SEALED)

Why LoRA was retained
- V2-8 LoRA improved mean development AUC (~0.903) vs frozen CLIP MLP-B (~0.837).
- V2-10A showed LoRA R1 geometry is strong (k5 purity ~0.966), indicating the representation remains useful even when the MLP-B operating point failed.

Why MLP-B was rejected as the final head
- Under LoRA+MLP-B, AI recall@0.5 fell to ~0.386 despite strong ranking.
- V2-9 showed primarily an operating-point / calibration-protocol problem rather than a safe calibrator rescue.
- V2-10A–B showed a class-balanced linear head on frozen R1 (H2) restored clean discrimination more effectively for the binary detector objective.

Why H2 was not selected despite best clean score
- H2 CLEAN AUC ~{t['H2']['clean_auc']:.3f} is strongest among H2/H3/H4.
- V2-10D StrongRobust evaluation was ROBUSTNESS_NOT_ACCEPTABLE: e.g. JPEG AI recall ~0.089; resize Real specificity ~0.131.
- V2-10E attributed this to head/decision-direction sensitivity, not R1 collapse.

Why H3 was not selected
- H3 substantially fixed catastrophic transform collapses (mean tf AUC ~{t['H3']['mean_tf_auc']:.3f}; mean tf RealSpec ~{t['H3']['mean_tf_real_spec']:.3f}).
- Clean ranking degraded more strongly (CLEAN AUC ~{t['H3']['clean_auc']:.3f}).
- V2-10G showed over-rotation away from the clean class-separation direction under 80% transformed training rows.

Why H4 was selected
- H4 is the strongest overall clean-versus-robust compromise among completed candidates.
- CLEAN AUC ~{t['H4']['clean_auc']:.3f} modestly recovers vs H3 while retaining StrongRobust behaviour close to H3 (mean tf AUC ~{t['H4']['mean_tf_auc']:.3f}; mean tf RealSpec ~{t['H4']['mean_tf_real_spec']:.3f}).
- Real-domain safety remains strong (especially Smartphone/COCO; MLLM improved vs H2 under resize/blur).
- Deployment head remains ~513 parameters (~2.9 KB); transform weighting is offline only.
- No further condition-weight search is scientifically justified after the single predeclared V2-10H experiment.

Resource cost
- Deployment: frozen LoRA visual pathway + 513-parameter logistic head.
- Offline H4 training used weighted CLEAN/StrongRobust R1 rows; inference needs a single CLEAN forward pass.

Robustness gains vs H2
- Catastrophic one-class collapses under JPEG/resize/blur are largely corrected.
- Mean transformed Real specificity rises from ~{t['H2']['mean_tf_real_spec']:.3f} (H2) to ~{t['H4']['mean_tf_real_spec']:.3f} (H4).

Remaining limitations (must remain visible)
- H4 missed V2-10H PROMISING gates (MIXED).
- Clean AUC below H2; mean/worst StrongRobust AUC imperfect.
- Seedream and FLUX remain weak; Fold4 remains weaker.
- Selective/calibration layers are not part of the frozen primary detector.
- NTIRE external transfer is unknown.

Required wording
- selected development compromise
- final V2 model with documented robustness/generalisation limitations
- Do not claim NTIRE generalisation or predict NTIRE scores.
"""
    PAPER_TXT.write_text(text)


def append_research_log() -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-10I" in text:
        print("[V2-10I] research_log already contains V2-10I; not rewriting.", flush=True)
        return
    entry = f"""
## Stage V2-10I — Development Closure / Formal Freeze (H4)

**Date:** 2026-09-08  
**Status:** **COMPLETE — DEVELOPMENT FROZEN WITH DOCUMENTED LIMITATIONS**  
**Mode:** Documentation + SHA256 freeze only. No training. No inference. No NTIRE access. Next stage NOT STARTED.

**Scientific decision:** Open-ended V2 model/head experimentation is CLOSED. Sequence H2→H3→H4 is sufficient. No H5, no weight grid, no C/threshold/calibrator/selective retune.

**FINAL_RESEARCH_MODEL_V2 = H4**  
**DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS**  
**FINAL_EXTERNAL_VALIDATION = NOT_STARTED**  
**NTIRE_STATUS = SEALED**  
**NO_FURTHER_MODEL_DEVELOPMENT_BEFORE_NTIRE = YES**

**H4 definition:** Frozen CLIP ViT-B-16-quickgelu + V2-8 fold LoRA + R1 (512-d L2) + clean-anchored balanced LogReg (C=1, lbfgs) with locked offline sample weights CLEAN=4 / each StrongRobust transform=1; binary threshold 0.5; ~513 head params.

**Selection rationale (summary):** H2 rejected for StrongRobust collapse; H3 rejected for excessive clean loss / over-rotation; H4 selected as the strongest completed clean/robust development compromise with documented residual limitations (Seedream, FLUX, Fold4, imperfect StrongRobust AUC, missed PROMISING gates).

**SHA freeze manifest:** `models/v2/final_v2_h4_freeze_manifest_v1.json`  
**Spec / limitations / comparison / NTIRE protocol lock / paper summary:** `results/v2/v2_10i_*`, `results/v2/v2_11_ntire_protocol_lock_v1.*`

**Primary external output policy:** UNCALIBRATED H4 binary p(AI) at 0.5. Do not transfer H2 temperature/selective to H4.

**Integrity:** NTIRE=SEALED/not accessed; fal=NO; V1 unmodified; folds unchanged; no model training/inference/weight updates; threshold unchanged; no calibrator; FINAL_RESEARCH_MODEL_V2_SELECTED=YES (H4); DEVELOPMENT_FROZEN=YES; FINAL_EXTERNAL_VALIDATION_STARTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** V2-11 sealed NTIRE external evaluation under the locked protocol — do not auto-start; do not open NTIRE until explicitly authorised.

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10I] Appended research_log.md", flush=True)


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "V2-10I — Development Closure / Formal Freeze",
        f"Status: COMPLETE — {payload['DEVELOPMENT_STATUS']}",
        f"FINAL_RESEARCH_MODEL_V2 = {payload['FINAL_RESEARCH_MODEL_V2']}",
        f"NTIRE_STATUS = {payload['NTIRE_STATUS']}",
        f"Freeze timestamp (UTC): {FREEZE_TS}",
        "",
        "No training. No inference. No NTIRE access.",
        "",
        "H4 artifact audit: OK",
        f"Freeze manifest entries: {payload['n_manifest_entries']}",
        f"Manifest: {MANIFEST_OUT.relative_to(PROJECT_ROOT)}",
        "",
        "Selection rationale:",
        "  H2 NOT selected: StrongRobust collapse despite best clean AUC.",
        "  H3 NOT selected: robustness gained; clean ranking / over-rotation cost too high.",
        "  H4 SELECTED: strongest completed clean/robust development compromise; ~513-param head.",
        "",
        "Primary external output: UNCALIBRATED H4 p(AI) @ 0.5",
        "NTIRE protocol lock written; NTIRE remains SEALED.",
        "",
        "Integrity statement:",
    ]
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    REPORT_TXT.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    print("[V2-10I] Artifact audit ...", flush=True)
    audit()

    h4 = json.loads((OUT / "v2_10h_analysis_v1.json").read_text())
    if h4.get("decision") != "CLEAN_ANCHORED_ROBUST_HEAD_MIXED":
        # Still allow freeze: human/tutor selected H4 regardless; warn only if file missing fields
        pass
    if "macro" not in h4 or "H4" not in h4["macro"]:
        stop("v2_10h_analysis_v1.json lacks H4 macro — cannot freeze")

    print("[V2-10I] SHA256 freeze manifest ...", flush=True)
    manifest_rows = build_manifest()
    manifest = {
        "document": "final_v2_h4_freeze_manifest_v1",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_STATUS": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "freeze_timestamp_utc": FREEZE_TS,
        "n_artifacts": len(manifest_rows),
        "artifacts": manifest_rows,
        "configuration_identifiers": {
            "lora_config": "kaggle/v2_lora/configs/v2_lora_config_v1.json",
            "h4_stage": "V2-10H",
            "threshold": 0.5,
            "sample_weights": {"CLEAN": 4.0, "transforms": 1.0},
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print("[V2-10I] Writing specification / tables / protocol ...", flush=True)
    spec = write_spec(h4)
    write_comparison()
    write_limitations(h4)
    ntire = write_ntire_protocol()
    write_rq_status()
    write_paper_summary(h4)

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "NTIRE_STATUS": "SEALED",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V1_RERUN": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "MODEL_TRAINING_PERFORMED": "NO",
        "MODEL_INFERENCE_PERFORMED": "NO",
        "LORA_WEIGHTS_UPDATED": "NO",
        "H2_UPDATED": "NO",
        "H3_UPDATED": "NO",
        "H4_UPDATED": "NO",
        "THRESHOLD_CHANGED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "FINAL_RESEARCH_MODEL_V2_SELECTED": "YES",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_FROZEN": "YES",
        "FINAL_EXTERNAL_VALIDATION_STARTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
    }

    payload = {
        "stage": "V2-10I",
        "status": "COMPLETE",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_STATUS": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "FINAL_EXTERNAL_VALIDATION": "NOT_STARTED",
        "NTIRE_STATUS": "SEALED",
        "NO_FURTHER_MODEL_DEVELOPMENT_BEFORE_NTIRE": True,
        "freeze_timestamp_utc": FREEZE_TS,
        "n_manifest_entries": len(manifest_rows),
        "manifest_path": str(MANIFEST_OUT.relative_to(PROJECT_ROOT)),
        "selection_rationale": {
            "H2_not_selected": "Best clean AUC but StrongRobust collapse (ROBUSTNESS_NOT_ACCEPTABLE).",
            "H3_not_selected": "Robustness improved but excessive clean loss / over-rotation (V2-10G).",
            "H4_selected": "Strongest completed clean/robust development compromise; 513-param head; documented limitations retained.",
        },
        "confidence_policy": {
            "primary": "UNCALIBRATED H4 binary p(AI) @ 0.5",
            "h2_temperature_selective": "not_transferred",
        },
        "ntire_protocol_lock": {
            "path_json": str(NTIRE_JSON.relative_to(PROJECT_ROOT)),
            "ntire_accessed_while_writing": ntire["NTIRE_ACCESSED_WHILE_WRITING"],
        },
        "integrity_statement": integrity_statement,
        "spec_path": str(SPEC_JSON.relative_to(PROJECT_ROOT)),
    }
    ANALYSIS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload)
    append_research_log()
    print(REPORT_TXT.read_text())
    print("FINAL_RESEARCH_MODEL_V2 = H4")
    print("DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS")


if __name__ == "__main__":
    main()
