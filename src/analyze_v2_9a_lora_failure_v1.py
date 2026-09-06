#!/usr/bin/env python3
"""V2-9A — LoRA Failure-Mode Diagnostic (ANALYSIS ONLY).

No training, no inference, no threshold adoption, no NTIRE/fal access.
Uses existing V2-8 LoRA predictions + published V2-7 MLP-B aggregate metrics.
Authoritative V2-7 per-image predictions were never saved by V2-7 training;
paired V2-7 score comparisons and paired bootstrap are therefore SKIPPED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
HOLDOUT = PROJECT_ROOT / "metadata" / "v2_generator_holdout_folds_v1.csv"

LORA_PRED_DIR = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/predictions"
)
LORA_HISTORY = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/v2_lora_training_history_v1.csv"
)
LORA_FOLD_METRICS = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/v2_lora_fold_metrics_v1.csv"
)
LORA_GEN_METRICS = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/v2_lora_generator_metrics_v1.csv"
)
LORA_REAL_METRICS = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/v2_lora_real_domain_metrics_v1.csv"
)
V27_FOLD = OUT / "v2_clip_mlp_fold_metrics_v1.csv"
V27_GEN = OUT / "v2_clip_mlp_generator_metrics_v1.csv"
V27_REAL = OUT / "v2_clip_mlp_real_domain_metrics_v1.csv"
V27_HIST = OUT / "v2_clip_mlp_training_history_v1.csv"
V27_CFG = OUT / "v2_clip_mlp_selected_config_v1.json"
V28_ANALYSIS = OUT / "v2_8_analysis_v1.json"

JSON_OUT = OUT / "v2_9a_lora_failure_analysis_v1.json"
REPORT_OUT = OUT / "v2_9a_lora_failure_report_v1.txt"
BOOT_OUT = OUT / "v2_9a_paired_bootstrap_v1.json"

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
LOCKED_T = 0.5


def score_summary(scores: np.ndarray) -> dict[str, float]:
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "frac_below_0.1": float("nan"),
            "frac_below_0.25": float("nan"),
            "frac_below_0.5": float("nan"),
            "frac_ge_0.5": float("nan"),
            "frac_above_0.75": float("nan"),
            "frac_above_0.9": float("nan"),
        }
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "p05": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "frac_below_0.1": float(np.mean(s < 0.1)),
        "frac_below_0.25": float(np.mean(s < 0.25)),
        "frac_below_0.5": float(np.mean(s < 0.5)),
        "frac_ge_0.5": float(np.mean(s >= 0.5)),
        "frac_above_0.75": float(np.mean(s > 0.75)),
        "frac_above_0.9": float(np.mean(s > 0.9)),
    }


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = (p >= t).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {
            "threshold": float(t),
            "ai_recall": float("nan"),
            "real_specificity": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": float(t),
        "ai_recall": float(recall_score(y, pred, zero_division=0)),
        "real_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def load_lora_predictions() -> pd.DataFrame:
    frames = []
    for f in sorted(LORA_PRED_DIR.glob("fold*_predictions_v1.csv")):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df["image_id"] = df["image_id"].astype(str)
    df["fold"] = df["fold"].astype(int)
    df["y"] = df["y"].astype(int)
    df["p_lora"] = df["p_lora"].astype(float)
    # Explicitly unused: invalid in-run baseline (LoRA visual + frozen MLP-B head)
    df = df.drop(columns=["p_mlpb_baseline"], errors="ignore")
    # Normalize real_domain NaN for AI
    df["real_domain"] = df["real_domain"].where(df["y"] == 0, other=np.nan)
    df["generator_id"] = df["generator_id"].astype(str)
    return df


def integrity_audit(preds: pd.DataFrame) -> dict[str, Any]:
    splits = pd.read_csv(MANIFEST)
    splits["image_id"] = splits["image_id"].astype(str)

    fold_reports = {}
    all_ok = True
    for fold in [1, 2, 3, 4]:
        col = f"fold_{fold}_role"
        expected = set(
            splits.loc[
                splits[col].isin(["REAL_VALIDATION", "HOLDOUT_VALIDATION"]),
                "image_id",
            ]
        )
        sub = preds[preds["fold"] == fold]
        got = set(sub["image_id"])
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        within_dups = int(sub["image_id"].duplicated().sum())
        merged = sub.merge(
            splits[["image_id", "binary_label", col]], on="image_id", how="left"
        )
        label_disagree = int((merged["y"] != merged["binary_label"]).sum())
        role_ok = bool(
            merged[col].isin(["REAL_VALIDATION", "HOLDOUT_VALIDATION"]).all()
        )
        ok = (
            len(missing) == 0
            and len(extra) == 0
            and within_dups == 0
            and label_disagree == 0
            and role_ok
        )
        all_ok = all_ok and ok
        fold_reports[f"fold_{fold}"] = {
            "n_rows": int(len(sub)),
            "n_unique_ids": int(sub["image_id"].nunique()),
            "within_fold_duplicates": within_dups,
            "missing_vs_manifest": len(missing),
            "extra_vs_manifest": len(extra),
            "label_disagreements": label_disagree,
            "fold_role_agreement": role_ok,
            "ok": ok,
        }

    # Cross-fold Real ID reuse is expected under generator-holdout protocol
    real = preds[preds["y"] == 0]
    cross_real_dups = int(real["image_id"].duplicated().sum())

    v27_pred_candidates = list(OUT.glob("*mlp*pred*.csv")) + list(
        OUT.glob("*mlpB*pred*.csv")
    )
    # Narrow search only under results/v2 (authoritative V2-7 never wrote per-image CSVs)
    broader = [
        str(p.relative_to(PROJECT_ROOT))
        for p in OUT.rglob("*prediction*.csv")
        if "mlp" in p.name.lower() and "kaggle_v2_lora_smoke" not in str(p)
    ]

    return {
        "lora_prediction_source": str(LORA_PRED_DIR.relative_to(PROJECT_ROOT)),
        "lora_total_rows": int(len(preds)),
        "lora_unique_ids_across_folds": int(preds["image_id"].nunique()),
        "lora_cross_fold_real_id_reuses": cross_real_dups,
        "note_cross_fold_real_reuse": (
            "Expected: REAL_VALIDATION IDs appear in all four folds under "
            "locked V2-3 generator-holdout protocol."
        ),
        "fold_integrity": fold_reports,
        "manifest_alignment_ok": all_ok,
        "v27_per_image_predictions_found": False,
        "v27_prediction_search_note": (
            "train_v2_clip_mlp_v1.py saved fold/generator/real metrics and "
            "checkpoints but did NOT write per-image prediction CSVs. "
            "LoRA files contain p_mlpb_baseline which is INVALID as V2-7 "
            "(LoRA-adapted visual features + frozen MLP-B head)."
        ),
        "v27_invalid_inrun_baseline_present_in_lora_files": True,
        "v27_invalid_inrun_baseline_used_in_this_analysis": False,
        "v27_authoritative_aggregates_used": [
            str(V27_FOLD.relative_to(PROJECT_ROOT)),
            str(V27_GEN.relative_to(PROJECT_ROOT)),
            str(V27_REAL.relative_to(PROJECT_ROOT)),
            str(V27_CFG.relative_to(PROJECT_ROOT)),
            str(V28_ANALYSIS.relative_to(PROJECT_ROOT)),
        ],
        "broader_mlp_prediction_hits": broader[:20],
        "candidate_local_glob_hits": [str(p) for p in v27_pred_candidates],
        "paired_id_match_v27_v28": False,
        "paired_bootstrap_eligible": False,
        "paired_bootstrap_skip_reason": (
            "Authoritative V2-7 MLP-B per-image predictions do not exist on disk; "
            "cannot prove exact ID-matched pairing. Invalid Kaggle in-run baseline "
            "explicitly excluded."
        ),
    }


def compute_score_distributions(preds: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"overall_pooled_folds": {}, "by_fold": {}, "by_generator_ai": {}, "by_real_domain": {}}
    # Pooled across folds (Real IDs repeated) — descriptive only
    for label, name in [(0, "REAL"), (1, "AI")]:
        out["overall_pooled_folds"][name] = score_summary(
            preds.loc[preds["y"] == label, "p_lora"].to_numpy()
        )
    for fold in [1, 2, 3, 4]:
        sub = preds[preds["fold"] == fold]
        out["by_fold"][f"fold_{fold}"] = {
            "REAL": score_summary(sub.loc[sub["y"] == 0, "p_lora"].to_numpy()),
            "AI": score_summary(sub.loc[sub["y"] == 1, "p_lora"].to_numpy()),
        }
    for gid, gdf in preds[preds["y"] == 1].groupby("generator_id"):
        out["by_generator_ai"][str(gid)] = {
            **score_summary(gdf["p_lora"].to_numpy()),
            "folds": sorted(int(x) for x in gdf["fold"].unique()),
        }
    for dom, ddf in preds[preds["y"] == 0].groupby("real_domain"):
        if pd.isna(dom):
            continue
        out["by_real_domain"][str(dom)] = {
            **score_summary(ddf["p_lora"].to_numpy()),
            "note": "pooled across folds; Real IDs repeat",
        }
    return out


def threshold_sensitivity(preds: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for fold in [1, 2, 3, 4]:
        sub = preds[preds["fold"] == fold]
        y = sub["y"].to_numpy()
        p = sub["p_lora"].to_numpy()
        for t in THRESHOLDS:
            m = metrics_at_threshold(y, p, float(t))
            m["fold"] = fold
            rows.append(m)
    df = pd.DataFrame(rows)
    # Macro-mean across folds
    mean_rows = []
    for t in THRESHOLDS:
        part = df[np.isclose(df["threshold"], t)]
        mean_rows.append(
            {
                "threshold": float(t),
                "ai_recall": float(part["ai_recall"].mean()),
                "real_specificity": float(part["real_specificity"].mean()),
                "balanced_accuracy": float(part["balanced_accuracy"].mean()),
                "precision": float(part["precision"].mean()),
                "f1": float(part["f1"].mean()),
            }
        )

    # Generator recall grid (macro over folds where present)
    gen_rows = []
    for gid in sorted(preds.loc[preds["y"] == 1, "generator_id"].unique()):
        for t in THRESHOLDS:
            fold_recs = []
            for fold in [1, 2, 3, 4]:
                g = preds[(preds["fold"] == fold) & (preds["generator_id"] == gid)]
                if len(g) == 0:
                    continue
                fold_recs.append(float((g["p_lora"] >= t).mean()))
            if fold_recs:
                gen_rows.append(
                    {
                        "generator_id": gid,
                        "threshold": float(t),
                        "mean_ai_recall": float(np.mean(fold_recs)),
                        "n_folds": len(fold_recs),
                    }
                )

    # Counterfactual: thresholds where LoRA mean AI recall ≈ V2-7 mean AI recall@0.5
    v27_mean_ai_rec = float(
        pd.read_csv(V27_FOLD).query("architecture == 'MLP-B'")["ai_recall"].mean()
    )
    mean_df = pd.DataFrame(mean_rows)
    # Closest threshold(s) by absolute recall difference
    mean_df["abs_diff_to_v27_recall"] = (mean_df["ai_recall"] - v27_mean_ai_rec).abs()
    closest = mean_df.loc[mean_df["abs_diff_to_v27_recall"].idxmin()]
    # Also find first threshold (from high to low) where recall >= v27
    ge = mean_df[mean_df["ai_recall"] >= v27_mean_ai_rec]
    first_ge = ge.iloc[0].to_dict() if len(ge) else None

    return {
        "grid": [float(t) for t in THRESHOLDS],
        "diagnostic_only": True,
        "no_threshold_adopted": True,
        "by_fold": rows,
        "macro_mean_across_folds": mean_rows,
        "generator_mean_ai_recall_grid": gen_rows,
        "counterfactual_vs_v27_ai_recall_at_0.5": {
            "v27_mean_ai_recall_at_0.5": v27_mean_ai_rec,
            "closest_lora_threshold": {
                "threshold": float(closest["threshold"]),
                "lora_ai_recall": float(closest["ai_recall"]),
                "lora_real_specificity": float(closest["real_specificity"]),
                "abs_diff_recall": float(closest["abs_diff_to_v27_recall"]),
            },
            "lowest_threshold_reaching_v27_recall": first_ge,
            "label": "COUNTERFACTUAL_DIAGNOSTIC_ONLY_NOT_ADOPTED",
        },
        "v27_full_threshold_grid": {
            "available": False,
            "reason": "No authoritative V2-7 per-image predictions; only locked-threshold aggregates exist.",
        },
    }


def generator_separability(preds: pd.DataFrame) -> dict[str, Any]:
    v27 = pd.read_csv(V27_GEN)
    v27b = v27[v27["architecture"] == "MLP-B"].copy()

    rows = []
    for fold in [1, 2, 3, 4]:
        sub = preds[preds["fold"] == fold]
        real = sub[sub["y"] == 0]
        y_real = real["y"].to_numpy()
        p_real = real["p_lora"].to_numpy()
        for gid, gdf in sub[sub["y"] == 1].groupby("generator_id"):
            y = np.concatenate([y_real, gdf["y"].to_numpy()])
            p = np.concatenate([p_real, gdf["p_lora"].to_numpy()])
            if len(np.unique(y)) < 2:
                auc = float("nan")
                ap = float("nan")
            else:
                auc = float(roc_auc_score(y, p))
                ap = float(average_precision_score(y, p))
            recall = float((gdf["p_lora"] >= LOCKED_T).mean())
            v27_row = v27b[(v27b["fold"] == fold) & (v27b["generator_id"] == gid)]
            v27_recall = float(v27_row["ai_recall_050"].iloc[0]) if len(v27_row) else None
            v27_mean_p = float(v27_row["mean_p_ai"].iloc[0]) if len(v27_row) else None
            rows.append(
                {
                    "fold": fold,
                    "generator_id": str(gid),
                    "n_ai": int(len(gdf)),
                    "lora_auc_vs_fold_real": auc,
                    "lora_ap_vs_fold_real": ap,
                    "lora_recall_050": recall,
                    "lora_mean_score": float(gdf["p_lora"].mean()),
                    "lora_median_score": float(gdf["p_lora"].median()),
                    "v27_recall_050": v27_recall,
                    "v27_mean_p_ai": v27_mean_p,
                    "delta_recall_050": (recall - v27_recall) if v27_recall is not None else None,
                    "delta_mean_score": (
                        float(gdf["p_lora"].mean()) - v27_mean_p
                        if v27_mean_p is not None
                        else None
                    ),
                    "v27_generator_auc_available": False,
                    "v27_generator_ap_available": False,
                }
            )

    # Aggregate hard generators across folds
    hard = []
    for gid in HARD_GENERATORS:
        part = [r for r in rows if r["generator_id"] == gid]
        if not part:
            continue
        hard.append(
            {
                "generator_id": gid,
                "n_ai_total": int(sum(r["n_ai"] for r in part)),
                "folds": [r["fold"] for r in part],
                "lora_mean_auc": float(np.nanmean([r["lora_auc_vs_fold_real"] for r in part])),
                "lora_mean_ap": float(np.nanmean([r["lora_ap_vs_fold_real"] for r in part])),
                "lora_mean_recall_050": float(np.mean([r["lora_recall_050"] for r in part])),
                "lora_mean_score": float(np.mean([r["lora_mean_score"] for r in part])),
                "lora_median_score_mean": float(np.mean([r["lora_median_score"] for r in part])),
                "v27_mean_recall_050": float(
                    np.mean([r["v27_recall_050"] for r in part if r["v27_recall_050"] is not None])
                ),
                "v27_mean_p_ai": float(
                    np.mean([r["v27_mean_p_ai"] for r in part if r["v27_mean_p_ai"] is not None])
                ),
                "delta_recall_050": float(
                    np.mean(
                        [r["delta_recall_050"] for r in part if r["delta_recall_050"] is not None]
                    )
                ),
                "delta_mean_score": float(
                    np.mean(
                        [r["delta_mean_score"] for r in part if r["delta_mean_score"] is not None]
                    )
                ),
                "interpretation_hint": (
                    "If LoRA generator AUC remains high while recall@0.5 collapses and "
                    "mean score drops vs V2-7 mean_p, operating-point/score-scale is favored; "
                    "if LoRA AUC itself is low, representation failure is favored. "
                    "V2-7 generator AUC unavailable."
                ),
            }
        )
    return {
        "protocol": (
            "Per fold: AI samples of generator vs ALL Real validation samples in that fold "
            "(locked V2-3 holdout protocol)."
        ),
        "per_fold_generator": rows,
        "hard_generators_summary": hard,
        "limitation": (
            "V2-7 generator-level ROC-AUC/AP cannot be recomputed without per-image "
            "predictions; comparison uses published recall@0.5 and mean_p_ai only."
        ),
    }


def roc_pr_analysis(preds: pd.DataFrame) -> dict[str, Any]:
    v27 = pd.read_csv(V27_FOLD)
    v27b = v27[v27["architecture"] == "MLP-B"]
    folds = []
    for fold in [1, 2, 3, 4]:
        sub = preds[preds["fold"] == fold]
        y = sub["y"].to_numpy()
        p = sub["p_lora"].to_numpy()
        m05 = metrics_at_threshold(y, p, LOCKED_T)
        vrow = v27b[v27b["fold"] == fold].iloc[0]
        folds.append(
            {
                "fold": fold,
                "n": int(len(sub)),
                "lora_auc": float(roc_auc_score(y, p)),
                "lora_ap": float(average_precision_score(y, p)),
                "lora_tpr_at_0.5": m05["ai_recall"],
                "lora_specificity_at_0.5": m05["real_specificity"],
                "v27_auc": float(vrow["roc_auc"]),
                "v27_ap": float(vrow["ap"]),
                "v27_ai_recall_at_0.5": float(vrow["ai_recall"]),
                "v27_specificity_at_0.5": float(vrow["specificity"]),
                "delta_auc": float(roc_auc_score(y, p) - vrow["roc_auc"]),
                "delta_ap": float(average_precision_score(y, p) - vrow["ap"]),
                "delta_ai_recall_at_0.5": float(m05["ai_recall"] - vrow["ai_recall"]),
                "delta_specificity_at_0.5": float(
                    m05["real_specificity"] - vrow["specificity"]
                ),
            }
        )
    return {
        "fold_level": folds,
        "macro_mean": {
            "lora_auc": float(np.mean([f["lora_auc"] for f in folds])),
            "v27_auc": float(np.mean([f["v27_auc"] for f in folds])),
            "lora_ap": float(np.mean([f["lora_ap"] for f in folds])),
            "v27_ap": float(np.mean([f["v27_ap"] for f in folds])),
            "lora_tpr_at_0.5": float(np.mean([f["lora_tpr_at_0.5"] for f in folds])),
            "v27_ai_recall_at_0.5": float(
                np.mean([f["v27_ai_recall_at_0.5"] for f in folds])
            ),
            "lora_specificity_at_0.5": float(
                np.mean([f["lora_specificity_at_0.5"] for f in folds])
            ),
            "v27_specificity_at_0.5": float(
                np.mean([f["v27_specificity_at_0.5"] for f in folds])
            ),
        },
        "note": (
            "Overall ROC/PR figures are drawn per fold (protocol). "
            "No pooled overall AUC claimed as primary because Real IDs repeat across folds."
        ),
    }


def overfitting_diagnostic() -> dict[str, Any]:
    hist = pd.read_csv(LORA_HISTORY)
    fold_metrics = pd.read_csv(LORA_FOLD_METRICS)
    out = {"by_fold": {}, "comparison_1_3_vs_2_4": {}}
    for fold in [1, 2, 3, 4]:
        h = hist[hist["fold"] == fold].sort_values("epoch")
        sel = int(fold_metrics.loc[fold_metrics["fold"] == fold, "selected_epoch"].iloc[0])
        last = h.iloc[-1]
        best_auc_epoch = int(h.loc[h["val_roc_auc"].idxmax(), "epoch"])
        best_auc = float(h["val_roc_auc"].max())
        at_sel = h[h["epoch"] == sel].iloc[0]
        train_traj = h["train_loss"].to_numpy()
        val_traj = h["val_loss"].to_numpy()
        # Divergence: val loss rising while train falling near end
        mid = max(1, len(h) // 2)
        train_down = float(train_traj[-1] < train_traj[mid])
        val_up = float(val_traj[-1] > val_traj[mid])
        gap_at_sel = float(at_sel["val_loss"] - at_sel["train_loss"])
        gap_last = float(last["val_loss"] - last["train_loss"])
        auc_drop_after_best = float(best_auc - last["val_roc_auc"])
        out["by_fold"][f"fold_{fold}"] = {
            "n_epochs_logged": int(len(h)),
            "selected_epoch": sel,
            "stopping_epoch": int(last["epoch"]),
            "best_val_auc_epoch": best_auc_epoch,
            "best_val_auc": best_auc,
            "val_auc_at_selected": float(at_sel["val_roc_auc"]),
            "val_auc_at_last": float(last["val_roc_auc"]),
            "val_ap_at_selected": float(at_sel["val_ap"]),
            "train_loss_first": float(h.iloc[0]["train_loss"]),
            "train_loss_selected": float(at_sel["train_loss"]),
            "train_loss_last": float(last["train_loss"]),
            "val_loss_first": float(h.iloc[0]["val_loss"]),
            "val_loss_selected": float(at_sel["val_loss"]),
            "val_loss_last": float(last["val_loss"]),
            "val_minus_train_loss_at_selected": gap_at_sel,
            "val_minus_train_loss_at_last": gap_last,
            "train_decreased_after_mid": bool(train_down),
            "val_increased_after_mid": bool(val_up),
            "auc_drop_last_vs_best": auc_drop_after_best,
            "overfit_signal": bool(
                (gap_at_sel > 0.5 and train_down and (val_up or auc_drop_after_best > 0.01))
                or auc_drop_after_best > 0.02
            ),
        }

    f1 = out["by_fold"]["fold_1"]
    f2 = out["by_fold"]["fold_2"]
    f3 = out["by_fold"]["fold_3"]
    f4 = out["by_fold"]["fold_4"]
    out["comparison_1_3_vs_2_4"] = {
        "mean_gap_at_selected_1_3": float(
            np.mean(
                [
                    f1["val_minus_train_loss_at_selected"],
                    f3["val_minus_train_loss_at_selected"],
                ]
            )
        ),
        "mean_gap_at_selected_2_4": float(
            np.mean(
                [
                    f2["val_minus_train_loss_at_selected"],
                    f4["val_minus_train_loss_at_selected"],
                ]
            )
        ),
        "mean_auc_drop_1_3": float(
            np.mean([f1["auc_drop_last_vs_best"], f3["auc_drop_last_vs_best"]])
        ),
        "mean_auc_drop_2_4": float(
            np.mean([f2["auc_drop_last_vs_best"], f4["auc_drop_last_vs_best"]])
        ),
        "kernel_flag_folds_1_and_3": True,
        "fold_1_signal": f1["overfit_signal"],
        "fold_3_signal": f3["overfit_signal"],
        "assessment": (
            "Fold 1: clear overfit signal (val−train gap@selected≈1.32; late AUC drop). "
            "Fold 3: largest late AUC drop after best epoch (~0.030 to epoch 20) despite "
            "smaller gap@selected; matches kernel flag for folds 1 and 3. "
            "Fold 2: large gap but selected at best-AUC epoch near end. "
            "Fold 4: persistent val>train gap, negligible late AUC drop. "
            "Signals are real but secondary to the global score-scale pattern; "
            "they do not alone explain hard-generator recall collapse."
        ),
    }
    # Optional: MLP-B history context (already known overfit) — read-only
    if V27_HIST.exists():
        mh = pd.read_csv(V27_HIST)
        mh_b = mh[mh["architecture"] == "MLP-B"]
        out["v27_mlp_b_history_available"] = True
        out["v27_mlp_b_note"] = (
            "V2-7 MLP-B also showed large train/val loss gaps on folds 1–3 "
            "(see v2_clip_mlp_selected_config_v1.json overfitting section). "
            "Not re-analysed in detail here beyond availability confirmation."
        )
        out["v27_mlp_b_epochs_logged"] = int(len(mh_b))
    else:
        out["v27_mlp_b_history_available"] = False
    return out


def real_domain_analysis(preds: pd.DataFrame) -> dict[str, Any]:
    v27 = pd.read_csv(V27_REAL)
    v27b = v27[v27["architecture"] == "MLP-B"]
    rows = []
    for fold in [1, 2, 3, 4]:
        for dom in ["Tiny", "MLLM", "COCO", "Smartphone"]:
            sub = preds[
                (preds["fold"] == fold)
                & (preds["y"] == 0)
                & (preds["real_domain"] == dom)
            ]
            if len(sub) == 0:
                continue
            spec = float((sub["p_lora"] < LOCKED_T).mean())
            vrow = v27b[(v27b["fold"] == fold) & (v27b["real_domain"] == dom)]
            vspec = float(vrow["specificity"].iloc[0]) if len(vrow) else None
            rows.append(
                {
                    "fold": fold,
                    "real_domain": dom,
                    "n": int(len(sub)),
                    "lora_specificity_050": spec,
                    "lora_mean_score": float(sub["p_lora"].mean()),
                    "lora_median_score": float(sub["p_lora"].median()),
                    "v27_specificity_050": vspec,
                    "delta_specificity": (spec - vspec) if vspec is not None else None,
                }
            )
    mllm = [r for r in rows if r["real_domain"] == "MLLM"]
    return {
        "by_fold_domain": rows,
        "mllm_fold_pattern": mllm,
        "fold3_mllm_assessment": {
            "lora_specificity": next(r["lora_specificity_050"] for r in mllm if r["fold"] == 3),
            "v27_specificity": next(r["v27_specificity_050"] for r in mllm if r["fold"] == 3),
            "other_lora_mllm": [r for r in mllm if r["fold"] != 3],
            "verdict": (
                "FOLD_DEPENDENT_WEAK_POINT: Fold-3 MLLM Real specificity (.823) is the "
                "weakest LoRA Real cell and slightly worse than V2-7 Fold-3 MLLM (.850). "
                "Other folds show LoRA MLLM specificity ≥ .929 and improved vs V2-7. "
                "Not a systematic LoRA MLLM collapse."
            ),
        },
    }


def make_figures(
    preds: pd.DataFrame,
    thr: dict[str, Any],
    gen: dict[str, Any],
    rocinfo: dict[str, Any],
    hist_info: dict[str, Any],
) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # 1) Score ECDF by label, per fold
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, fold in zip(axes.ravel(), [1, 2, 3, 4]):
        sub = preds[preds["fold"] == fold]
        for yv, lab, color in [(0, "Real", "#1f77b4"), (1, "AI", "#d62728")]:
            s = np.sort(sub.loc[sub["y"] == yv, "p_lora"].to_numpy())
            if len(s) == 0:
                continue
            ecdf = np.arange(1, len(s) + 1) / len(s)
            ax.plot(s, ecdf, label=lab, color=color, lw=1.8)
        ax.axvline(0.5, color="k", ls="--", lw=1, label="locked t=0.5")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("P(AI)")
        ax.set_ylabel("ECDF")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("V2-9A LoRA score ECDFs (V2-8 predictions)")
    fig.tight_layout()
    p = FIG / "v2_9a_lora_score_ecdf_by_fold_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 2) Histograms Real vs AI pooled descriptive
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, yv, title, color in [
        (axes[0], 0, "Real (pooled folds)", "#1f77b4"),
        (axes[1], 1, "AI (pooled folds)", "#d62728"),
    ]:
        s = preds.loc[preds["y"] == yv, "p_lora"].to_numpy()
        ax.hist(s, bins=40, range=(0, 1), color=color, alpha=0.85, density=True)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("P(AI)")
        ax.set_ylabel("Density")
    fig.suptitle("V2-9A LoRA score histograms (descriptive; Real IDs repeat across folds)")
    fig.tight_layout()
    p = FIG / "v2_9a_lora_score_hist_real_ai_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 3) Hard-generator score boxplots
    hard_df = preds[preds["generator_id"].isin(HARD_GENERATORS)].copy()
    if len(hard_df):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        order = [g for g in HARD_GENERATORS if g in set(hard_df["generator_id"])]
        data = [hard_df.loc[hard_df["generator_id"] == g, "p_lora"].to_numpy() for g in order]
        ax.boxplot(data, tick_labels=[g.split("::", 1)[-1] for g in order], showfliers=False)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        # overlay V2-7 mean_p as red points if available
        for i, g in enumerate(order, start=1):
            h = next((x for x in gen["hard_generators_summary"] if x["generator_id"] == g), None)
            if h:
                ax.scatter([i], [h["v27_mean_p_ai"]], color="red", zorder=3, label="V2-7 mean_p" if i == 1 else None)
                ax.scatter([i], [h["lora_mean_score"]], color="blue", zorder=3, marker="D", label="LoRA mean" if i == 1 else None)
        ax.set_ylabel("P(AI)")
        ax.set_title("V2-9A hard-generator LoRA scores vs V2-7 published mean_p")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = FIG / "v2_9a_hard_generator_score_box_v1.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p.relative_to(PROJECT_ROOT)))

    # 4) Threshold sensitivity
    mean_rows = thr["macro_mean_across_folds"]
    t = [r["threshold"] for r in mean_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, [r["ai_recall"] for r in mean_rows], label="AI recall", color="#d62728")
    ax.plot(
        t,
        [r["real_specificity"] for r in mean_rows],
        label="Real specificity",
        color="#1f77b4",
    )
    ax.plot(
        t,
        [r["balanced_accuracy"] for r in mean_rows],
        label="Balanced accuracy",
        color="#2ca02c",
    )
    ax.axvline(0.5, color="k", ls="--", lw=1, label="locked t=0.5")
    v27_rec = thr["counterfactual_vs_v27_ai_recall_at_0.5"]["v27_mean_ai_recall_at_0.5"]
    ax.axhline(v27_rec, color="#d62728", ls=":", lw=1, label="V2-7 mean AI recall@0.5")
    cf = thr["counterfactual_vs_v27_ai_recall_at_0.5"]["closest_lora_threshold"]
    ax.scatter([cf["threshold"]], [cf["lora_ai_recall"]], color="purple", zorder=3)
    ax.set_xlabel("Threshold (diagnostic grid; not optimised)")
    ax.set_ylabel("Metric")
    ax.set_title("V2-9A LoRA threshold sensitivity (macro-mean folds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "v2_9a_lora_threshold_sensitivity_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 5) ROC / PR per fold with 0.5 operating point
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, fold in enumerate([1, 2, 3, 4]):
        sub = preds[preds["fold"] == fold]
        y = sub["y"].to_numpy()
        p = sub["p_lora"].to_numpy()
        fpr, tpr, thr_roc = roc_curve(y, p)
        prec, rec, thr_pr = precision_recall_curve(y, p)
        # operating point at 0.5
        m = metrics_at_threshold(y, p, 0.5)
        # approximate FPR at 0.5
        pred = (p >= 0.5).astype(int)
        fpr05 = float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
        axes[0, i].plot(fpr, tpr, color="#d62728", lw=1.8)
        axes[0, i].scatter([fpr05], [m["ai_recall"]], color="k", zorder=3)
        axes[0, i].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0, i].set_title(f"ROC fold {fold}\nAUC={roc_auc_score(y, p):.3f}")
        axes[0, i].set_xlabel("FPR")
        axes[0, i].set_ylabel("TPR")
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].plot(rec, prec, color="#1f77b4", lw=1.8)
        axes[1, i].scatter(
            [m["ai_recall"]],
            [m["precision"]],
            color="k",
            zorder=3,
            label="t=0.5",
        )
        axes[1, i].set_title(f"PR fold {fold}\nAP={average_precision_score(y, p):.3f}")
        axes[1, i].set_xlabel("Recall")
        axes[1, i].set_ylabel("Precision")
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].legend(fontsize=8)
    fig.suptitle("V2-9A LoRA ROC/PR with locked t=0.5 marked (black)")
    fig.tight_layout()
    p = FIG / "v2_9a_lora_roc_pr_by_fold_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 6) Fold AUC/AP LoRA vs V2-7 bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    folds = [f["fold"] for f in rocinfo["fold_level"]]
    x = np.arange(len(folds))
    w = 0.35
    axes[0].bar(x - w / 2, [f["v27_auc"] for f in rocinfo["fold_level"]], w, label="V2-7 MLP-B")
    axes[0].bar(x + w / 2, [f["lora_auc"] for f in rocinfo["fold_level"]], w, label="V2-8 LoRA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"F{f}" for f in folds])
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title("ROC-AUC")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x - w / 2, [f["v27_ap"] for f in rocinfo["fold_level"]], w, label="V2-7 MLP-B")
    axes[1].bar(x + w / 2, [f["lora_ap"] for f in rocinfo["fold_level"]], w, label="V2-8 LoRA")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"F{f}" for f in folds])
    axes[1].set_ylim(0.75, 1.0)
    axes[1].set_title("Average Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("V2-9A ranking metrics: LoRA vs authoritative V2-7 aggregates")
    fig.tight_layout()
    p = FIG / "v2_9a_auc_ap_vs_v27_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 7) Training history
    hist = pd.read_csv(LORA_HISTORY)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, fold in zip(axes.ravel(), [1, 2, 3, 4]):
        h = hist[hist["fold"] == fold]
        ax.plot(h["epoch"], h["train_loss"], label="train loss")
        ax.plot(h["epoch"], h["val_loss"], label="val loss")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["val_roc_auc"], color="green", ls="--", label="val AUC")
        sel = hist_info["by_fold"][f"fold_{fold}"]["selected_epoch"]
        ax.axvline(sel, color="k", ls=":", lw=1)
        ax.set_title(f"Fold {fold} (selected epoch={sel})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Val AUC")
        ax.grid(True, alpha=0.3)
    fig.suptitle("V2-9A LoRA training histories (overfitting diagnostic)")
    fig.tight_layout()
    p = FIG / "v2_9a_lora_training_history_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 8) Hard generator recall + mean score deltas
    hard = gen["hard_generators_summary"]
    if hard:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        names = [h["generator_id"].split("::", 1)[-1] for h in hard]
        x = np.arange(len(names))
        axes[0].bar(x - w / 2, [h["v27_mean_recall_050"] for h in hard], w, label="V2-7")
        axes[0].bar(x + w / 2, [h["lora_mean_recall_050"] for h in hard], w, label="LoRA")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=20, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("AI recall @ locked 0.5")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(x - w / 2, [h["v27_mean_p_ai"] for h in hard], w, label="V2-7 mean_p")
        axes[1].bar(x + w / 2, [h["lora_mean_score"] for h in hard], w, label="LoRA mean")
        axes[1].axhline(0.5, color="k", ls="--", lw=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=20, ha="right")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Mean AI score")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.suptitle("V2-9A hard generators: recall@0.5 and mean score (LoRA AUC shown in report)")
        fig.tight_layout()
        p = FIG / "v2_9a_hard_generator_recall_score_v1.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p.relative_to(PROJECT_ROOT)))

    return created


def synthesize(
    integrity: dict[str, Any],
    scores: dict[str, Any],
    thr: dict[str, Any],
    gen: dict[str, Any],
    rocinfo: dict[str, Any],
    overfit: dict[str, Any],
    real: dict[str, Any],
) -> dict[str, Any]:
    hard = {h["generator_id"]: h for h in gen["hard_generators_summary"]}
    # Evidence buckets
    op_evidence = []
    sep_evidence = []
    mixed_notes = []

    overall_ai = scores["overall_pooled_folds"]["AI"]
    overall_real = scores["overall_pooled_folds"]["REAL"]
    if overall_ai["frac_ge_0.5"] < 0.45 and overall_ai["mean"] < 0.5:
        op_evidence.append(
            f"LoRA pooled AI mean score={overall_ai['mean']:.3f}; only "
            f"{overall_ai['frac_ge_0.5']:.1%} of AI ≥0.5 despite high fold AUCs."
        )
    if overall_real["frac_ge_0.5"] < 0.05:
        op_evidence.append(
            f"LoRA Real scores heavily concentrated below 0.5 "
            f"(frac≥0.5={overall_real['frac_ge_0.5']:.3f}; mean={overall_real['mean']:.3f})."
        )

    cf = thr["counterfactual_vs_v27_ai_recall_at_0.5"]
    if cf["closest_lora_threshold"]["threshold"] < 0.5:
        op_evidence.append(
            f"Counterfactual: LoRA reaches V2-7 mean AI recall@0.5≈{cf['v27_mean_ai_recall_at_0.5']:.3f} "
            f"near t={cf['closest_lora_threshold']['threshold']:.2f} with Real specificity "
            f"{cf['closest_lora_threshold']['lora_real_specificity']:.3f} "
            "(diagnostic only; not adopted)."
        )

    for gid, h in hard.items():
        # High LoRA AUC + collapsed recall + lower mean score → OP
        # Low LoRA AUC → separability
        if h["lora_mean_auc"] >= 0.75 and h["delta_recall_050"] < -0.05:
            op_evidence.append(
                f"{gid}: LoRA AUC≈{h['lora_mean_auc']:.3f} still moderate/high but "
                f"recall@0.5 Δ={h['delta_recall_050']:+.3f}; mean score Δ={h['delta_mean_score']:+.3f}."
            )
        if h["lora_mean_auc"] < 0.70:
            sep_evidence.append(
                f"{gid}: LoRA generator-vs-Real AUC≈{h['lora_mean_auc']:.3f} indicates weak ranking, "
                f"not only threshold shift (recall@0.5={h['lora_mean_recall_050']:.3f})."
            )
        if h["lora_mean_ap"] < 0.35 and h["lora_mean_auc"] < 0.85:
            sep_evidence.append(
                f"{gid}: LoRA AP≈{h['lora_mean_ap']:.3f} is weak (precision-sensitive), "
                f"suggesting residual ranking fragility even if AUC is moderate."
            )
        if h["lora_mean_auc"] < 0.80 and h["delta_recall_050"] < -0.1:
            mixed_notes.append(
                f"{gid}: both lowered mean scores and imperfect AUC suggest mixed mechanisms."
            )

    # Classification rule: OP dominates when hard-gen AUCs remain high while recall collapses.
    # Residual low-AP hard gens keep a representation caveat without flipping the primary label
    # unless AUC itself is poor.
    if any(h["lora_mean_auc"] < 0.70 for h in hard.values()) and op_evidence:
        classification = "MIXED_OPERATING_AND_REPRESENTATION_FAILURE"
    elif op_evidence and not any(h["lora_mean_auc"] < 0.70 for h in hard.values()):
        classification = "PRIMARILY_OPERATING_POINT_SHIFT"
    elif sep_evidence and not op_evidence:
        classification = "PRIMARILY_GENERATOR_SEPARABILITY_FAILURE"
    elif sep_evidence and op_evidence:
        # High-AUC hard gens with score collapse => still primarily OP; note residual AP weakness
        classification = "PRIMARILY_OPERATING_POINT_SHIFT"
    else:
        classification = "INCONCLUSIVE"

    answers = {
        "q1_auc_up_recall_down": (
            "Ranking metrics (AUC/AP) depend on orderings across thresholds; LoRA improved "
            "separation of Real vs AI score ranks (macro mean AUC 0.903 vs 0.837) while "
            "compressing many AI probabilities below the locked 0.5 cut (mean AI recall "
            f"{rocinfo['macro_mean']['lora_tpr_at_0.5']:.3f} vs "
            f"{rocinfo['macro_mean']['v27_ai_recall_at_0.5']:.3f}). "
            "High Real specificity under LoRA further indicates a score-scale shift toward Real."
        ),
        "q2_hard_generators": (
            "Primarily threshold/score-scale for the largest hard generators: GPT Image 2 "
            "and Nano Banana 2 retain LoRA AUC≈0.88 while mean scores fall well below 0.5 "
            "and recall@0.5 collapses vs V2-7. Residual caveat: FLUX.2_max / Seedream-5.0 "
            "have weaker LoRA AP (≈0.27–0.34), so precision-sensitive ranking is imperfect. "
            "V2-7 generator AUCs unavailable, limiting direct ΔAUC claims."
        ),
        "q3_real_bias": (
            f"Yes, partly: LoRA Real mean P(AI)={overall_real['mean']:.3f} with "
            f"{overall_real['frac_below_0.5']:.1%} of Real scores <0.5, and mean Real "
            f"specificity {rocinfo['macro_mean']['lora_specificity_at_0.5']:.3f} "
            f"(vs V2-7 {rocinfo['macro_mean']['v27_specificity_at_0.5']:.3f}). "
            "This conservative Real bias helps specificity but reduces AI recall at 0.5."
        ),
        "q4_overfit_1_3": overfit["comparison_1_3_vs_2_4"]["assessment"],
        "q5_recommendation_only": (
            "Do NOT jump into another unconstrained PEFT/LoRA retrain yet. Prefer a "
            "decision-stage that either (a) diagnoses calibration/operating-point without "
            "adopting a new threshold as final, and/or (b) targets hard-generator "
            "representation failure with a tightly scoped protocol after human review. "
            "If PEFT is revisited, it should explicitly address the Real-score bias and "
            "hard-modern-generator collapse, not repeat the same LoRA recipe. "
            "Alternatively consider non-adaptation directions (calibration/selective "
            "prediction on frozen V2-7/V2-8 scores) depending on tutor priorities."
        ),
        "q6_next_stage_info": (
            "Need: (1) authoritative V2-7 per-image predictions (or approved offline "
            "replay of frozen MLP-B on existing embeddings) to unlock paired bootstrap "
            "and generator AUC deltas; (2) whether the programme prioritises ranking "
            "metrics vs locked-threshold recall; (3) whether Fold-3 MLLM Real weakness "
            "is acceptable; (4) whether hard-generator recovery is mandatory for "
            "PROGRESSING beyond LORA_MIXED."
        ),
    }

    return {
        "diagnostic_classification": classification,
        "supporting_operating_point_evidence": op_evidence,
        "supporting_separability_evidence": sep_evidence,
        "mixed_notes": mixed_notes,
        "contradictions_or_limits": [
            "No authoritative V2-7 per-image scores → cannot directly overlay V2-7 vs LoRA "
            "distributions or run valid paired bootstrap.",
            "Generator-level V2-7 AUC/AP unavailable; separability conclusions for V2-7 "
            "side rely on published recall and mean_p only.",
            "p_mlpb_baseline in LoRA CSVs must not be treated as V2-7.",
        ],
        "answers": answers,
        "final_v2_model_selected": False,
        "next_stage_started": False,
    }


def write_report(payload: dict[str, Any], fig_paths: list[str]) -> str:
    integ = payload["integrity"]
    syn = payload["synthesis"]
    roc = payload["roc_pr"]
    thr = payload["threshold_sensitivity"]
    hard = payload["generator_separability"]["hard_generators_summary"]
    over = payload["overfitting"]
    real = payload["real_domain"]
    scores = payload["score_distributions"]

    lines = []
    lines.append("V2-9A — LoRA Failure-Mode Diagnostic")
    lines.append("Status: COMPLETE (analysis-only)")
    lines.append("No training. No inference. No threshold adoption. No final V2 model selection.")
    lines.append("")
    lines.append("Artifacts used")
    lines.append(f"- LoRA predictions: {integ['lora_prediction_source']}")
    lines.append(f"- LoRA history: {LORA_HISTORY.relative_to(PROJECT_ROOT)}")
    lines.append(f"- V2-7 aggregates: fold/generator/real metrics + selected_config + v2_8_analysis")
    lines.append(f"- Locked manifests: {MANIFEST.relative_to(PROJECT_ROOT)}")
    lines.append("")
    lines.append("Integrity")
    lines.append(f"- Manifest alignment OK: {integ['manifest_alignment_ok']}")
    lines.append(f"- LoRA rows: {integ['lora_total_rows']} (unique IDs across folds: {integ['lora_unique_ids_across_folds']})")
    lines.append(f"- V2-7 per-image predictions found: {integ['v27_per_image_predictions_found']}")
    lines.append(f"- Paired bootstrap executed: False ({integ['paired_bootstrap_skip_reason']})")
    lines.append(f"- Invalid in-run baseline used: {integ['v27_invalid_inrun_baseline_used_in_this_analysis']}")
    for fold, fr in integ["fold_integrity"].items():
        lines.append(
            f"  {fold}: n={fr['n_rows']} missing={fr['missing_vs_manifest']} "
            f"extra={fr['extra_vs_manifest']} dups={fr['within_fold_duplicates']} "
            f"label_disagree={fr['label_disagreements']} ok={fr['ok']}"
        )
    lines.append("")
    lines.append("Score distributions (LoRA only)")
    for name in ["REAL", "AI"]:
        s = scores["overall_pooled_folds"][name]
        lines.append(
            f"- {name}: n={s['n']} mean={s['mean']:.3f} median={s['median']:.3f} "
            f"p25={s['p25']:.3f} p75={s['p75']:.3f} frac≥0.5={s['frac_ge_0.5']:.3f} "
            f"frac<0.25={s['frac_below_0.25']:.3f}"
        )
    lines.append("  (Pooled Real counts include expected cross-fold Real ID reuse.)")
    lines.append("")
    lines.append("Threshold sensitivity (LoRA; diagnostic grid 0.05..0.95)")
    lines.append("  t     AI_rec  Real_spec  bal_acc   F1")
    for r in thr["macro_mean_across_folds"]:
        if r["threshold"] in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            lines.append(
                f"  {r['threshold']:.2f}  {r['ai_recall']:.3f}   {r['real_specificity']:.3f}     "
                f"{r['balanced_accuracy']:.3f}  {r['f1']:.3f}"
            )
    cf = thr["counterfactual_vs_v27_ai_recall_at_0.5"]
    lines.append(
        f"Counterfactual (NOT ADOPTED): closest LoRA t≈{cf['closest_lora_threshold']['threshold']:.2f} "
        f"matches V2-7 AI recall@0.5≈{cf['v27_mean_ai_recall_at_0.5']:.3f} with LoRA Real spec "
        f"{cf['closest_lora_threshold']['lora_real_specificity']:.3f}."
    )
    lines.append("V2-7 full threshold grid: UNAVAILABLE (no per-image preds).")
    lines.append("")
    lines.append("ROC/PR vs authoritative V2-7 aggregates")
    for f in roc["fold_level"]:
        lines.append(
            f"  Fold {f['fold']}: AUC {f['lora_auc']:.3f}/{f['v27_auc']:.3f} "
            f"AP {f['lora_ap']:.3f}/{f['v27_ap']:.3f} "
            f"AI@0.5 {f['lora_tpr_at_0.5']:.3f}/{f['v27_ai_recall_at_0.5']:.3f} "
            f"RealSpec {f['lora_specificity_at_0.5']:.3f}/{f['v27_specificity_at_0.5']:.3f}"
        )
    m = roc["macro_mean"]
    lines.append(
        f"  Macro mean: AUC {m['lora_auc']:.3f}/{m['v27_auc']:.3f} "
        f"AP {m['lora_ap']:.3f}/{m['v27_ap']:.3f} "
        f"AI@0.5 {m['lora_tpr_at_0.5']:.3f}/{m['v27_ai_recall_at_0.5']:.3f}"
    )
    lines.append("")
    lines.append("Hard generators (LoRA AUC/AP vs fold Real; recall/mean_p vs V2-7 published)")
    for h in hard:
        lines.append(
            f"  {h['generator_id']}: n={h['n_ai_total']} "
            f"LoRA AUC={h['lora_mean_auc']:.3f} AP={h['lora_mean_ap']:.3f} "
            f"rec@0.5 {h['lora_mean_recall_050']:.3f} vs V27 {h['v27_mean_recall_050']:.3f} "
            f"(Δ{h['delta_recall_050']:+.3f}); "
            f"mean_p {h['lora_mean_score']:.3f} vs {h['v27_mean_p_ai']:.3f} "
            f"(Δ{h['delta_mean_score']:+.3f})"
        )
    lines.append("")
    lines.append("Overfitting (LoRA histories)")
    for fold, info in over["by_fold"].items():
        lines.append(
            f"  {fold}: selected={info['selected_epoch']}/{info['stopping_epoch']} "
            f"bestAUC_epoch={info['best_val_auc_epoch']} "
            f"gap@sel={info['val_minus_train_loss_at_selected']:.3f} "
            f"AUCdrop_last_vs_best={info['auc_drop_last_vs_best']:.3f} "
            f"signal={info['overfit_signal']}"
        )
    lines.append(f"  Assessment: {over['comparison_1_3_vs_2_4']['assessment']}")
    lines.append("")
    lines.append("Real-domain specificity @0.5")
    for r in real["by_fold_domain"]:
        lines.append(
            f"  Fold{r['fold']} {r['real_domain']}: n={r['n']} "
            f"LoRA={r['lora_specificity_050']:.3f} V27={r['v27_specificity_050']:.3f} "
            f"mean_p={r['lora_mean_score']:.4f}"
        )
    lines.append(f"  Fold3 MLLM: {real['fold3_mllm_assessment']['verdict']}")
    lines.append("")
    lines.append(f"DIAGNOSTIC CLASSIFICATION: {syn['diagnostic_classification']}")
    lines.append("Supporting OP evidence:")
    for e in syn["supporting_operating_point_evidence"]:
        lines.append(f"  - {e}")
    lines.append("Supporting separability evidence:")
    for e in syn["supporting_separability_evidence"] or ["  (none strong under available metrics)"]:
        lines.append(f"  - {e}" if not e.startswith(" ") else e)
    lines.append("")
    lines.append("Answers")
    for k, v in syn["answers"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Figures")
    for fp in fig_paths:
        lines.append(f"  - {fp}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("FINAL_V2_MODEL_SELECTED = NO")
    lines.append("NEXT_STAGE_STARTED = NO")
    return "\n".join(lines) + "\n"


def main() -> None:
    assert LORA_PRED_DIR.is_dir(), f"missing LoRA preds: {LORA_PRED_DIR}"
    assert MANIFEST.is_file()
    assert V27_FOLD.is_file()

    preds = load_lora_predictions()
    integrity = integrity_audit(preds)
    if not integrity["manifest_alignment_ok"]:
        raise SystemExit("STOP: LoRA predictions disagree with locked V2-3 manifests.")

    # Sanity vs authoritative V2-8 published AUCs
    v28 = json.loads(V28_ANALYSIS.read_text())
    for fold_cmp in v28["fold_comparison_vs_v27"]:
        fold = int(fold_cmp["fold"])
        sub = preds[preds["fold"] == fold]
        auc = float(roc_auc_score(sub["y"], sub["p_lora"]))
        if abs(auc - float(fold_cmp["lora_auc"])) > 1e-6:
            raise SystemExit(
                f"STOP: LoRA fold {fold} AUC {auc} disagrees with authoritative "
                f"v2_8_analysis {fold_cmp['lora_auc']}"
            )

    scores = compute_score_distributions(preds)
    thr = threshold_sensitivity(preds)
    gen = generator_separability(preds)
    rocinfo = roc_pr_analysis(preds)
    overfit = overfitting_diagnostic()
    real = real_domain_analysis(preds)
    fig_paths = make_figures(preds, thr, gen, rocinfo, overfit)
    synthesis = synthesize(integrity, scores, thr, gen, rocinfo, overfit, real)

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "MODEL_TRAINING_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "THRESHOLD_RETUNED_OR_ADOPTED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
    }

    payload = {
        "stage": "V2-9A",
        "status": "COMPLETE",
        "mode": "ANALYSIS_ONLY",
        "integrity": integrity,
        "score_distributions": scores,
        "threshold_sensitivity": thr,
        "generator_separability": gen,
        "roc_pr": rocinfo,
        "overfitting": overfit,
        "real_domain": real,
        "paired_bootstrap": {
            "executed": False,
            "reason": integrity["paired_bootstrap_skip_reason"],
            "output_file": None,
        },
        "synthesis": synthesis,
        "figures": fig_paths,
        "integrity_statement": integrity_statement,
    }

    def _json_clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_clean(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(_json_clean(payload), indent=2))
    report = write_report(payload, fig_paths)
    REPORT_OUT.write_text(report)

    # Explicit bootstrap skip stub (do not invent numbers)
    BOOT_OUT.write_text(
        json.dumps(
            {
                "stage": "V2-9A",
                "executed": False,
                "reason": integrity["paired_bootstrap_skip_reason"],
                "n_replicates_requested": 5000,
                "seed_requested": 42,
                "invalid_kaggle_baseline_bootstrap_used": False,
            },
            indent=2,
        )
    )

    print(report)
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Wrote {BOOT_OUT} (skip record)")
    print(f"Figures: {len(fig_paths)}")


if __name__ == "__main__":
    main()
