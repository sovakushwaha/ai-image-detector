#!/usr/bin/env python3
"""V2-9B — Leakage-Safe Post-Hoc Calibration Evaluation (AUDIT + GATE).

Analysis / tiny post-hoc calibration on EXISTING V2-8 LoRA scores only.
No detector training, no V2-7 replay, no NTIRE/fal, no threshold adoption.

If the leakage gate fails, the script records CALIBRATION_PROTOCOL_NOT_SAFE
and does NOT invent an alternative split or write fake calibrated predictions.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
PRED_DIR = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/predictions"
)
V29A_JSON = OUT / "v2_9a_lora_failure_analysis_v1.json"

JSON_OUT = OUT / "v2_9b_calibration_analysis_v1.json"
REPORT_OUT = OUT / "v2_9b_calibration_report_v1.txt"
PRED_OUT = OUT / "v2_9b_crossfitted_predictions_v1.csv"

EPS = 1e-6
SEED = 42
MIN_FIT_NEG = 50  # scientific floor: need both classes for binary NLL calibration
MIN_FIT_POS = 50
MIN_FIT_TOTAL = 200

HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]


def load_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(PRED_DIR.glob("fold*_predictions_v1.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["image_id"] = df["image_id"].astype(str)
    df["fold"] = df["fold"].astype(int)
    df["y"] = df["y"].astype(int)
    df["p_lora"] = df["p_lora"].astype(float)
    # Explicitly unused invalid in-run baseline
    df = df.drop(columns=["p_mlpb_baseline"], errors="ignore")
    return df


def verify_against_manifest(preds: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    fold_reports = {}
    ok = True
    for fold in [1, 2, 3, 4]:
        col = f"fold_{fold}_role"
        expected = set(
            splits.loc[
                splits[col].isin(["REAL_VALIDATION", "HOLDOUT_VALIDATION"]),
                "image_id",
            ].astype(str)
        )
        sub = preds[preds["fold"] == fold]
        got = set(sub["image_id"])
        merged = sub.merge(
            splits[["image_id", "binary_label", col]], on="image_id", how="left"
        )
        role_counts = merged[col].value_counts(dropna=False).to_dict()
        report = {
            "n_rows": int(len(sub)),
            "missing_vs_manifest": len(expected - got),
            "extra_vs_manifest": len(got - expected),
            "within_fold_duplicates": int(sub["image_id"].duplicated().sum()),
            "label_disagreements": int((merged["y"] != merged["binary_label"]).sum()),
            "role_counts": {str(k): int(v) for k, v in role_counts.items()},
            "roles_are_validation_only": set(role_counts) <= {
                "REAL_VALIDATION",
                "HOLDOUT_VALIDATION",
            },
        }
        report["ok"] = (
            report["missing_vs_manifest"] == 0
            and report["extra_vs_manifest"] == 0
            and report["within_fold_duplicates"] == 0
            and report["label_disagreements"] == 0
            and report["roles_are_validation_only"]
        )
        ok = ok and report["ok"]
        fold_reports[f"fold_{fold}"] = report
    return {"ok": ok, "folds": fold_reports}


def overlap_matrix(id_sets: dict[int, set[str]]) -> dict[str, dict[str, int]]:
    return {
        f"F{i}": {f"F{j}": int(len(id_sets[i] & id_sets[j])) for j in [1, 2, 3, 4]}
        for i in [1, 2, 3, 4]
    }


def leakage_audit(preds: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    # Prediction row roles (established from locked split assignments)
    role_note = (
        "Each V2-8 prediction row is exactly the locked per-fold evaluation set: "
        "REAL_VALIDATION ∪ HOLDOUT_VALIDATION. "
        "REAL_INTERNAL_HOLDOUT, TRAIN, and PROMPT_BLOCKED are not present in "
        "prediction files. Confirmed via metadata/v2_split_assignments_v1.csv."
    )

    all_ids = {f: set(preds.loc[preds.fold == f, "image_id"]) for f in [1, 2, 3, 4]}
    real_ids = {
        f: set(preds.loc[(preds.fold == f) & (preds.y == 0), "image_id"])
        for f in [1, 2, 3, 4]
    }
    ai_ids = {
        f: set(preds.loc[(preds.fold == f) & (preds.y == 1), "image_id"])
        for f in [1, 2, 3, 4]
    }

    # Which AI IDs overlap between fold pairs?
    ai_pair_details = {}
    for i, j in combinations([1, 2, 3, 4], 2):
        inter = sorted(ai_ids[i] & ai_ids[j])
        if not inter:
            ai_pair_details[f"F{i}_F{j}"] = {"n": 0, "generators": {}}
            continue
        gens = (
            preds[(preds.fold == i) & (preds.image_id.isin(inter))]
            .groupby("generator_id")
            .size()
            .to_dict()
        )
        ai_pair_details[f"F{i}_F{j}"] = {
            "n": len(inter),
            "generators": {str(k): int(v) for k, v in gens.items()},
        }

    fit_plans = {}
    protocol_safe = True
    unsafe_reasons = []
    for target in [1, 2, 3, 4]:
        source = preds[preds.fold != target].copy()
        target_id_set = all_ids[target]
        overlap_mask = source.image_id.isin(target_id_set)
        n_overlap_rows = int(overlap_mask.sum())
        n_overlap_ids = int(source.loc[overlap_mask, "image_id"].nunique())
        fit = source.loc[~overlap_mask].copy()
        n_pos = int((fit.y == 1).sum())
        n_neg = int((fit.y == 0).sum())
        n_fit = int(len(fit))
        # Residual Real after removal should be empty under this protocol
        residual_real_in_source_before = int((source.y == 0).sum())
        plan = {
            "target_fold": target,
            "target_eval_rows": int((preds.fold == target).sum()),
            "source_rows_before_overlap_removal": int(len(source)),
            "overlap_rows_removed_from_fit_only": n_overlap_rows,
            "overlap_unique_ids_removed": n_overlap_ids,
            "fit_rows_after_removal": n_fit,
            "fit_positives_ai": n_pos,
            "fit_negatives_real": n_neg,
            "source_real_rows_before_removal": residual_real_in_source_before,
            "scientifically_valid_binary_fit": bool(
                n_pos >= MIN_FIT_POS and n_neg >= MIN_FIT_NEG and n_fit >= MIN_FIT_TOTAL
            ),
        }
        if not plan["scientifically_valid_binary_fit"]:
            protocol_safe = False
            unsafe_reasons.append(
                f"Target fold {target}: after removing overlapping IDs, fit set has "
                f"AI={n_pos}, Real={n_neg}, total={n_fit}. Binary NLL calibration "
                f"requires both classes (min Real>={MIN_FIT_NEG}, AI>={MIN_FIT_POS})."
            )
        fit_plans[f"fold_{target}"] = plan

    # Root cause
    real_all_shared = all(real_ids[i] == real_ids[1] for i in [2, 3, 4]) and len(
        real_ids[1]
    ) == 923
    root_cause = (
        "Under locked V2-3 generator-holdout evaluation, the same 923 "
        "REAL_VALIDATION image IDs appear in EVERY fold's prediction file. "
        "Preferred leave-one-fold-out calibration therefore has source Real "
        "rows that are exact ID duplicates of the target fold Real set. "
        "Leakage gate removes those IDs from FITTING ONLY, which removes "
        "ALL Real negatives from every calibrator fit set, leaving AI-only "
        "scores. Binary post-hoc calibration (temperature / Platt) cannot be "
        "fitted scientifically without Real labels. Protocol forbids inventing "
        "another split, using target-fold labels, or replaying V2-7 / train "
        "inference to obtain alternate scores."
    )

    return {
        "prediction_role_definition": role_note,
        "manifest_verification": verify_against_manifest(preds, splits),
        "n_pred_rows": int(len(preds)),
        "n_unique_ids_across_folds": int(preds.image_id.nunique()),
        "overlap_matrix_all_ids": overlap_matrix(all_ids),
        "overlap_matrix_real": overlap_matrix(real_ids),
        "overlap_matrix_ai": overlap_matrix(ai_ids),
        "ai_pairwise_overlap_details": ai_pair_details,
        "real_validation_fully_shared_across_folds": bool(real_all_shared),
        "n_shared_real_validation_ids": int(len(real_ids[1])),
        "fit_plans_per_target_fold": fit_plans,
        "min_fit_requirements": {
            "min_positives": MIN_FIT_POS,
            "min_negatives": MIN_FIT_NEG,
            "min_total": MIN_FIT_TOTAL,
        },
        "protocol_safe": protocol_safe,
        "unsafe_reasons": unsafe_reasons,
        "root_cause": root_cause,
        "forbidden_alternatives_not_used": [
            "no target-fold self-fit",
            "no ignoring Real ID overlap",
            "no inventing new split",
            "no using REAL_INTERNAL_HOLDOUT without predictions",
            "no V2-7 inference replay",
            "no train-set score generation",
            "no isotonic/spline/beta/neural calibrators",
        ],
    }


def check_v29a_consistency(preds: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if not V29A_JSON.exists():
        return {"checked": False, "reason": "v2_9a json missing"}
    v29a = json.loads(V29A_JSON.read_text())
    # Reproduce macro mean AI recall / Real spec @0.5 and AUC from LoRA preds
    fold_aucs = []
    fold_rec = []
    fold_spec = []
    for fold in [1, 2, 3, 4]:
        sub = preds[preds.fold == fold]
        y = sub.y.to_numpy()
        p = sub.p_lora.to_numpy()
        fold_aucs.append(float(roc_auc_score(y, p)))
        fold_rec.append(float(((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)))
        fold_spec.append(float(((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)))
    obs = {
        "mean_auc": float(np.mean(fold_aucs)),
        "mean_ai_recall_050": float(np.mean(fold_rec)),
        "mean_real_spec_050": float(np.mean(fold_spec)),
        "pooled_ai_mean_p": float(preds.loc[preds.y == 1, "p_lora"].mean()),
        "pooled_real_mean_p": float(preds.loc[preds.y == 0, "p_lora"].mean()),
        "pooled_ai_frac_ge_050": float((preds.loc[preds.y == 1, "p_lora"] >= 0.5).mean()),
    }
    # Soft consistency with published V2-9A numbers
    ok = (
        abs(obs["mean_auc"] - 0.903) < 0.002
        and abs(obs["mean_ai_recall_050"] - 0.386) < 0.002
        and abs(obs["mean_real_spec_050"] - 0.978) < 0.002
    )
    return {
        "checked": True,
        "consistent_with_v29a_summary": ok,
        "observed_from_v28_preds": obs,
        "v29a_classification": v29a.get("synthesis", {}).get(
            "diagnostic_classification"
        ),
    }


def make_audit_figures(audit: dict[str, Any]) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # Overlap heatmap (all IDs)
    mat = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            mat[i, j] = audit["overlap_matrix_all_ids"][f"F{i+1}"][f"F{j+1}"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in [
        (axes[0], "overlap_matrix_all_ids", "All IDs"),
        (axes[1], "overlap_matrix_real", "Real IDs"),
        (axes[2], "overlap_matrix_ai", "AI IDs"),
    ]:
        m = np.array(
            [[audit[key][f"F{i}"][f"F{j}"] for j in range(1, 5)] for i in range(1, 5)]
        )
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([f"F{i}" for i in range(1, 5)])
        ax.set_yticklabels([f"F{i}" for i in range(1, 5)])
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("V2-9B fold-overlap matrices (image IDs in V2-8 predictions)")
    fig.tight_layout()
    p = FIG / "v2_9b_fold_overlap_matrix_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # Fit composition after leakage removal
    folds = [1, 2, 3, 4]
    pos = [audit["fit_plans_per_target_fold"][f"fold_{f}"]["fit_positives_ai"] for f in folds]
    neg = [audit["fit_plans_per_target_fold"][f"fold_{f}"]["fit_negatives_real"] for f in folds]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    ax.bar(x - 0.2, pos, 0.4, label="AI (fit after removal)")
    ax.bar(x + 0.2, neg, 0.4, label="Real (fit after removal)")
    ax.axhline(MIN_FIT_NEG, color="red", ls="--", lw=1, label=f"min Real={MIN_FIT_NEG}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Target F{f}" for f in folds])
    ax.set_ylabel("Fitting rows")
    ax.set_title("V2-9B leakage-safe fit composition (Real count = 0 for all targets)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "v2_9b_calibration_fit_composition_after_leakage_removal_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    return created


def write_report(payload: dict[str, Any]) -> str:
    a = payload["leakage_audit"]
    lines = []
    lines.append("V2-9B — Leakage-Safe Post-Hoc Calibration Evaluation")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Detector training: NO. V2-7 replay: NO. Threshold adoption: NO.")
    lines.append("")
    lines.append("Purpose")
    lines.append(
        "Test whether leakage-safe post-hoc calibration (C1 temperature-only, "
        "C2 affine/Platt) on existing V2-8 LoRA scores can repair the "
        "PRIMARILY_OPERATING_POINT_SHIFT diagnosed in V2-9A without retraining."
    )
    lines.append("")
    lines.append("Predeclared calibrators (NOT fitted due to unsafe protocol)")
    lines.append("  C0 = uncalibrated LoRA probability")
    lines.append("  C1 = z_cal = z / T   (T>0)")
    lines.append("  C2 = z_cal = a*z + b (Platt/affine)")
    lines.append(f"  logit eps = {EPS}")
    lines.append("  NOT used: isotonic, splines, beta, neural, C-grid search")
    lines.append("")
    lines.append("Prediction role (from locked manifests)")
    lines.append(f"  {a['prediction_role_definition']}")
    lines.append(f"  Manifest alignment OK: {a['manifest_verification']['ok']}")
    for fold, fr in a["manifest_verification"]["folds"].items():
        lines.append(
            f"  {fold}: n={fr['n_rows']} roles={fr['role_counts']} "
            f"missing={fr['missing_vs_manifest']} extra={fr['extra_vs_manifest']} ok={fr['ok']}"
        )
    lines.append("")
    lines.append("Fold-overlap matrix (all IDs)")
    for i in [1, 2, 3, 4]:
        row = a["overlap_matrix_all_ids"][f"F{i}"]
        lines.append(
            "  "
            + " ".join(f"F{j}:{row[f'F{j}']:5d}" for j in [1, 2, 3, 4])
        )
    lines.append("Real overlap matrix: every off-diagonal = 923 (full REAL_VALIDATION reuse)")
    lines.append(
        f"AI pairwise overlaps: {json.dumps(a['ai_pairwise_overlap_details'], sort_keys=True)}"
    )
    lines.append("")
    lines.append("Leakage-safe fit plans (preferred LOFO; overlaps removed from FIT only)")
    for fold, plan in a["fit_plans_per_target_fold"].items():
        lines.append(
            f"  {fold}: source_before={plan['source_rows_before_overlap_removal']} "
            f"removed={plan['overlap_rows_removed_from_fit_only']} "
            f"(unique_ids={plan['overlap_unique_ids_removed']}) "
            f"fit_after={plan['fit_rows_after_removal']} "
            f"AI={plan['fit_positives_ai']} Real={plan['fit_negatives_real']} "
            f"valid={plan['scientifically_valid_binary_fit']} "
            f"target_eval={plan['target_eval_rows']}"
        )
    lines.append("")
    lines.append("ROOT CAUSE")
    lines.append(f"  {a['root_cause']}")
    lines.append("")
    lines.append("Unsafe reasons")
    for r in a["unsafe_reasons"]:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Calibration metrics / C1 / C2 / cross-fitted predictions")
    lines.append("  NOT COMPUTED — protocol unsafe; no calibrator fitted.")
    lines.append(f"  Cross-fitted prediction CSV written: {payload['crossfitted_predictions_written']}")
    lines.append("")
    lines.append("Consistency with V2-9A")
    lines.append(f"  {json.dumps(payload['v29a_consistency'], sort_keys=True)}")
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append(
        "Interpretation: The preferred leakage-safe cross-fold calibration design "
        "cannot be executed on existing V2-8 prediction artifacts because Real "
        "validation IDs are fully shared across folds. Fitting temperature/Platt "
        "would require either (a) leaking target Real IDs, (b) self-fitting on "
        "the target fold, or (c) inventing a new split / generating new scores — "
        "all forbidden. Therefore calibration helpfulness for repairing LoRA's "
        "operating-point shift remains UNTESTED under a safe protocol."
    )
    lines.append("")
    lines.append("Answers (constrained by unsafe protocol)")
    for k, v in payload["interpretation"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Next-stage recommendation ONLY (not executed)")
    lines.append(f"  {payload['next_stage_recommendation']}")
    lines.append("")
    lines.append("Figures")
    for f in payload["figures"]:
        lines.append(f"  - {f}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("FINAL_V2_MODEL_SELECTED = NO")
    lines.append("NEXT_STAGE_STARTED = NO")
    return "\n".join(lines) + "\n"


def main() -> None:
    assert PRED_DIR.is_dir(), PRED_DIR
    assert MANIFEST.is_file(), MANIFEST
    splits = pd.read_csv(MANIFEST)
    splits["image_id"] = splits["image_id"].astype(str)
    preds = load_predictions()

    audit = leakage_audit(preds, splits)
    if not audit["manifest_verification"]["ok"]:
        raise SystemExit("STOP: prediction IDs conflict with locked V2-3 manifests.")

    v29a = check_v29a_consistency(preds)
    if v29a.get("checked") and not v29a.get("consistent_with_v29a_summary"):
        raise SystemExit("STOP: existing V2-8 preds contradict V2-9A summary numbers.")

    figures = make_audit_figures(audit)

    if audit["protocol_safe"]:
        # Reserved path — not reached with current artifacts.
        raise SystemExit(
            "Unexpected: protocol marked safe; implement fitting path before proceeding."
        )

    decision = "CALIBRATION_PROTOCOL_NOT_SAFE"
    interpretation = {
        "q1_was_v28_miscalibrated": (
            "Likely yes in the operating-point sense (V2-9A: Real-ward score shift, "
            "AI mean≈0.39 with high AUC), but V2-9B could not measure ECE/NLL under "
            "a leakage-safe calibrator comparison."
        ),
        "q2_temperature_change_at_0.5": (
            "Not evaluated. Theoretically temperature-only (z/T) preserves the sign of z "
            "and therefore generally does not move examples across p=0.5; it was a "
            "predeclared negative control. Fitting was blocked by the leakage gate."
        ),
        "q3_platt_improve_p05": (
            "Not evaluated — affine/Platt fitting blocked by AI-only fit sets after "
            "Real ID overlap removal."
        ),
        "q4_ai_recall_recovered": "Not evaluated (no calibrator fitted).",
        "q5_real_spec_cost": "Not evaluated (no calibrator fitted).",
        "q6_hard_generators_improved": "Not evaluated (no calibrator fitted).",
        "q7_hard_generators_remain_weak": (
            "V2-9A residual FLUX/Seedream AP weakness remains the best available "
            "evidence; V2-9B adds no new calibrated hard-generator results."
        ),
        "q8_retain_lora_candidate": (
            "Yes as a ranking/development candidate (V2-8 LORA_MIXED + V2-9A OP-shift), "
            "but calibrated probability usability at p=0.5 remains unproven under a "
            "safe protocol."
        ),
        "q9_next_direction": (
            "See next_stage_recommendation; do not auto-start."
        ),
    }
    next_rec = (
        "Human/tutor decision required. Options consistent with locks: "
        "(1) design a new locked calibration split that provides Real negatives "
        "without target-ID leakage (e.g., dedicate REAL_INTERNAL_HOLDOUT scores via "
        "a future approved offline score dump — not this stage); "
        "(2) proceed without post-hoc calibration toward selective prediction / "
        "robustness / targeted representation work acknowledging uncalibrated LoRA "
        "probabilities at 0.5 are not trustworthy; "
        "(3) do NOT silently fit calibrators on overlapping Real IDs or on the "
        "target fold. Prefer tutor review before any new PEFT retrain."
    )

    integrity = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "DETECTOR_MODEL_TRAINING_PERFORMED": "NO",
        "CLIP_EMBEDDINGS_REGENERATED": "NO",
        "V2_7_INFERENCE_REPLAYED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "TARGET_FOLD_LABELS_USED_FOR_ITS_CALIBRATOR": "NO",
        "DUPLICATE_TARGET_IDS_USED_FOR_CALIBRATOR_FIT": "NO",
        "THRESHOLD_SWEEP_FOR_SELECTION": "NO",
        "DEPLOYMENT_THRESHOLD_CHANGED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "CALIBRATOR_FITTED": "NO",
    }

    payload = {
        "stage": "V2-9B",
        "status": "COMPLETE",
        "mode": "LEAKAGE_AUDIT_AND_GATE_ONLY",
        "decision": decision,
        "predeclared_calibrators": {
            "C0": "uncalibrated",
            "C1": "temperature_only z/T",
            "C2": "affine_platt a*z+b",
            "eps": EPS,
            "seed": SEED,
            "fitted": False,
        },
        "leakage_audit": audit,
        "v29a_consistency": v29a,
        "calibration_metrics": None,
        "discrimination_metrics": None,
        "operating_point_at_0.5": None,
        "hard_generators": None,
        "real_domain": None,
        "fitted_parameters": None,
        "crossfitted_predictions_written": False,
        "crossfitted_predictions_path": None,
        "figures": figures,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "integrity_statement": integrity,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    REPORT_OUT.write_text(write_report(payload))

    # Explicitly do NOT write fake calibrated predictions
    if PRED_OUT.exists():
        # Should not exist from a prior partial run; leave untouched if present
        pass

    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print("Did NOT write crossfitted predictions (protocol unsafe).")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
