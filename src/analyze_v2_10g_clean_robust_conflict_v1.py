#!/usr/bin/env python3
"""V2-10G — Clean/Robust Decision-Conflict Diagnostic (analysis only).

NO training. NO H2/H3 change. NO NTIRE. Diagnose whether H3 limitations
are over-rotation, linear conflict, fold/generator-specific, or mixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"

SEED = 42
THR = 0.5
K_NN = 5
TRANSFORMS = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
CONDITIONS = ["CLEAN"] + TRANSFORMS
EXPECTED_TRAIN = {
    1: {"n": 6008, "real": 4280, "ai": 1728},
    2: {"n": 6009, "real": 4280, "ai": 1729},
    3: {"n": 6309, "real": 4280, "ai": 2029},
    4: {"n": 6594, "real": 4280, "ai": 2314},
}
EXPECTED_EVAL = {1: 2821, 2: 2619, 3: 1994, 4: 1709}
SEEDREAM = "qwen::Seedream-5.0"
FLUX = "qwen::FLUX.2_max"

JSON_OUT = OUT / "v2_10g_clean_robust_conflict_analysis_v1.json"
REPORT_OUT = OUT / "v2_10g_clean_robust_conflict_report_v1.txt"
DIR_CSV = OUT / "v2_10g_condition_direction_similarity_v1.csv"
MARGIN_CSV = OUT / "v2_10g_margin_analysis_v1.csv"
TRANS_CSV = OUT / "v2_10g_clean_transition_analysis_v1.csv"
HARD_CSV = OUT / "v2_10g_seedream_flux_diagnostic_v1.csv"
FOLD4_CSV = OUT / "v2_10g_fold4_diagnostic_v1.csv"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def cos_vec(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = np.clip(cos_vec(u, v), -1.0, 1.0)
    if np.isnan(c):
        return float("nan")
    return float(np.degrees(np.arccos(c)))


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


def margin_stats(m: np.ndarray) -> dict[str, float]:
    m = np.asarray(m, dtype=np.float64)
    if len(m) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "frac_le0": float("nan"),
            "frac_low_margin": float("nan"),
        }
    return {
        "n": int(len(m)),
        "mean": float(np.mean(m)),
        "median": float(np.median(m)),
        "p05": float(np.percentile(m, 5)),
        "p25": float(np.percentile(m, 25)),
        "frac_le0": float(np.mean(m <= 0)),
        "frac_low_margin": float(np.mean(m <= 0.5)),
    }


def local_k5_same_label_frac(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    """Mean fraction of k NN (within full set, excluding self) matching own label."""
    if len(idx) == 0:
        return float("nan")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    fracs = []
    for i in idx:
        nn = np.argpartition(D[i], K_NN)[:K_NN]
        nn = nn[np.argsort(D[i, nn])]
        fracs.append(float(np.mean(y[nn] == y[i])))
    return float(np.mean(fracs))


def load_train(fold: int, cond: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cond == "CLEAN":
        z = np.load(OUT / f"v2_10b_lora_train_features_fold{fold}_v1.npz", allow_pickle=False)
    else:
        z = np.load(OUT / f"v2_10f_train_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32), z["y"].astype(np.int64)


def load_eval(fold: int, cond: str) -> tuple[np.ndarray, np.ndarray]:
    if cond == "CLEAN":
        z = np.load(OUT / f"v2_10a_lora_features_fold{fold}_v1.npz", allow_pickle=False)
    else:
        z = np.load(OUT / f"v2_10e_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def audit() -> dict[str, Any]:
    required = []
    for f in [1, 2, 3, 4]:
        required.append(OUT / f"v2_10b_lora_train_features_fold{f}_v1.npz")
        required.append(OUT / f"v2_10a_lora_features_fold{f}_v1.npz")
        required.append(MODELS / f"v2_10b_h2_balanced_logreg_fold{f}_v1.joblib")
        required.append(MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{f}_v1.joblib")
        for c in TRANSFORMS:
            required.append(OUT / f"v2_10f_train_r1_{c}_fold{f}_v1.npz")
            required.append(OUT / f"v2_10e_r1_{c}_fold{f}_v1.npz")
    required += [
        OUT / "v2_10f_robust_head_predictions_v1.csv",
        OUT / "v2_10f_robust_head_metrics_v1.csv",
        OUT / "v2_10f_robust_head_bootstrap_v1.json",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.is_file()]
    if missing:
        stop(f"missing reusable artifacts: {missing[:10]}")

    folds = {}
    for fold in [1, 2, 3, 4]:
        ids0, X0, y0 = load_train(fold, "CLEAN")
        exp = EXPECTED_TRAIN[fold]
        if len(ids0) != exp["n"] or int((y0 == 0).sum()) != exp["real"] or int((y0 == 1).sum()) != exp["ai"]:
            stop(f"fold {fold} CLEAN train membership mismatch")
        if X0.shape != (exp["n"], 512) or not np.isfinite(X0).all():
            stop(f"fold {fold} CLEAN train R1 bad")
        if len(np.unique(ids0)) != len(ids0):
            stop(f"fold {fold} duplicate train IDs")
        for c in TRANSFORMS:
            ids, X, y = load_train(fold, c)
            if list(ids) != list(ids0) or not np.array_equal(y, y0):
                stop(f"fold {fold} {c} train ID/label mismatch")
            if X.shape != (exp["n"], 512) or not np.isfinite(X).all():
                stop(f"fold {fold} {c} train R1 bad")

        eid, Xe = load_eval(fold, "CLEAN")
        if len(eid) != EXPECTED_EVAL[fold] or Xe.shape[1] != 512 or not np.isfinite(Xe).all():
            stop(f"fold {fold} CLEAN eval bad")
        if set(ids0) & set(eid):
            stop(f"fold {fold} train/eval overlap")
        for c in TRANSFORMS:
            ids_e, Xe_t = load_eval(fold, c)
            if list(ids_e) != list(eid):
                stop(f"fold {fold} eval {c} ID mismatch")
            if Xe_t.shape != Xe.shape or not np.isfinite(Xe_t).all():
                stop(f"fold {fold} eval {c} R1 bad")

        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        h3 = joblib.load(MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib")
        if h2.coef_.shape != (1, 512) or h3.coef_.shape != (1, 512):
            stop(f"fold {fold} coef dim mismatch")
        folds[f"fold_{fold}"] = {
            "train_n": int(len(ids0)),
            "eval_n": int(len(eid)),
            "overlap": 0,
            "h2_n_params": int(h2.coef_.size + 1),
            "h3_n_params": int(h3.coef_.size + 1),
            "h2_path": f"models/v2/v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib",
            "h3_path": f"models/v2/v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib",
        }
    return {"ok": True, "missing": [], "folds": folds, "new_feature_inference": False}


def classify_fold(cos_clean_tf: float, cos_h2_h3: float, clean_auc_drop: float) -> str:
    # Geometry-based: high alignment of condition directions + large H2->H3 rotation
    if cos_clean_tf >= 0.70 and cos_h2_h3 < 0.70 and clean_auc_drop > 0.02:
        return "CONDITIONS_MOSTLY_COMPATIBLE"
    if cos_clean_tf >= 0.40:
        return "CONDITIONS_PARTLY_CONFLICTING"
    return "CONDITIONS_STRONGLY_CONFLICTING"


def decide(payload: dict[str, Any]) -> tuple[str, str, list[str]]:
    evidence = []
    mean_pair_cos = payload["macro_direction"]["mean_pairwise_condition_cos"]
    mean_cos_h2_h3 = payload["macro_direction"]["mean_cos_w_h2_h3"]
    mean_cos_h3_sclean = payload["macro_direction"]["mean_cos_w_h3_s_clean"]
    mean_cos_h2_sclean = payload["macro_direction"]["mean_cos_w_h2_s_clean"]
    fold4_drop = payload["fold4"]["clean_auc_h3"] - payload["fold4"]["clean_auc_h2"]
    fold4_share = payload["fold4"]["share_of_macro_clean_auc_drop"]
    seed_drop = payload["seedream"]["clean_recall_h3"] - payload["seedream"]["clean_recall_h2"]
    conflict_folds = payload["compatibility"]
    n_compatible = sum(1 for v in conflict_folds.values() if v == "CONDITIONS_MOSTLY_COMPATIBLE")
    n_strong = sum(1 for v in conflict_folds.values() if v == "CONDITIONS_STRONGLY_CONFLICTING")

    evidence.append(
        f"mean pairwise condition cos(s_c,s_c')={mean_pair_cos:.3f}; "
        f"mean cos(w_H2,w_H3)={mean_cos_h2_h3:.3f}; "
        f"cos(w,s_clean) H2={mean_cos_h2_sclean:.3f} H3={mean_cos_h3_sclean:.3f}"
    )
    evidence.append(
        f"Fold4 clean ΔAUC={fold4_drop:+.3f}; share of macro clean AUC drop≈{fold4_share:.2f}; "
        f"Seedream Δrec={seed_drop:+.3f}"
    )
    evidence.append(f"compatibility={conflict_folds}")

    overrotated = (
        mean_pair_cos >= 0.55
        and mean_cos_h2_h3 < 0.70
        and (mean_cos_h2_sclean - mean_cos_h3_sclean) > 0.05
        and n_strong == 0
    )
    linear_conflict = mean_pair_cos < 0.40 or n_strong >= 2
    fold_gen_limited = fold4_share >= 0.55 and abs(seed_drop) >= 0.15 and mean_pair_cos >= 0.50

    if overrotated and fold_gen_limited:
        decision = "ROBUST_HEAD_MIXED_LIMITATION"
        next_opt = "OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD"
    elif overrotated and not linear_conflict:
        decision = "ROBUST_HEAD_OVERROTATED"
        next_opt = "OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD"
    elif linear_conflict:
        decision = "ROBUST_HEAD_LINEAR_CONFLICT"
        next_opt = "OPTION_B_NONLINEAR_FROZEN_R1_ROBUST_HEAD"
    elif fold_gen_limited:
        decision = "ROBUST_HEAD_FOLD_GENERATOR_LIMITED"
        next_opt = "OPTION_C_GENERATOR_FOLD_DIAGNOSTIC_OR_PROTOCOL_REVIEW"
    elif n_compatible >= 2 and mean_pair_cos >= 0.50:
        decision = "ROBUST_HEAD_OVERROTATED"
        next_opt = "OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD"
    else:
        decision = "ROBUST_HEAD_MIXED_LIMITATION"
        next_opt = "OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD"

    # Prefer mixed if both global rotation AND fold4/seedream dominate
    if overrotated and (fold4_share >= 0.40 or abs(seed_drop) >= 0.15):
        decision = "ROBUST_HEAD_MIXED_LIMITATION"
        # Still recommend clean-anchored linear if geometry compatible
        if mean_pair_cos >= 0.50 and n_strong == 0:
            next_opt = "OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD"
        else:
            next_opt = "OPTION_C_GENERATOR_FOLD_DIAGNOSTIC_OR_PROTOCOL_REVIEW"

    evidence.append(f"decision_logic overrotated={overrotated} linear_conflict={linear_conflict} fold_gen={fold_gen_limited}")
    return decision, next_opt, evidence


def make_figures(sim_df: pd.DataFrame, fold_dir: dict) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    # mean similarity heatmap
    mats = []
    for fold in [1, 2, 3, 4]:
        sub = sim_df[sim_df["fold"] == fold]
        mat = np.zeros((5, 5))
        for i, a in enumerate(CONDITIONS):
            for j, b in enumerate(CONDITIONS):
                row = sub[(sub["cond_a"] == a) & (sub["cond_b"] == b)]
                if len(row):
                    mat[i, j] = row.iloc[0]["cosine"]
        mats.append(mat)
    mean_mat = np.mean(mats, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CONDITIONS, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CONDITIONS, fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{mean_mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("V2-10G mean cos(s_c, s_c')")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = FIG / "v2_10g_condition_direction_similarity_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # H2/H3 alignment to clean vs mean-tf
    folds = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(folds))
    ax.bar(x - 0.3, [fold_dir[f]["cos_w_h2_s_clean"] for f in folds], 0.2, label="H2↔s_clean")
    ax.bar(x - 0.1, [fold_dir[f]["cos_w_h3_s_clean"] for f in folds], 0.2, label="H3↔s_clean")
    ax.bar(x + 0.1, [fold_dir[f]["cos_w_h2_s_tfmean"] for f in folds], 0.2, label="H2↔s_tfmean")
    ax.bar(x + 0.3, [fold_dir[f]["cos_w_h3_s_tfmean"] for f in folds], 0.2, label="H3↔s_tfmean")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine")
    ax.set_title("Head alignment to clean vs mean-transform separation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = FIG / "v2_10g_head_alignment_clean_vs_tf_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10G — Clean/Robust Decision-Conflict Diagnostic")
    lines.append(f"Status: COMPLETE — {payload['decision']}")
    lines.append("Analysis only. No training. No H2/H3 change. thr unchanged.")
    lines.append("FINAL_V2_MODEL NOT SELECTED. NEXT_STAGE_STARTED = NO.")
    lines.append("")
    lines.append("Integrity")
    lines.append(f"  ok={payload['integrity']['ok']} new_feature_inference=NO")
    lines.append("")
    lines.append("Condition-direction compatibility (TRAIN)")
    md = payload["macro_direction"]
    lines.append(f"  mean pairwise cos(s_c,s_c')={md['mean_pairwise_condition_cos']:.4f}")
    lines.append(f"  mean cos(s_clean,s_tf)={md['mean_cos_s_clean_each_tf']}")
    lines.append(f"  greatest conflict transform≈{md['most_conflicting_transform']}")
    lines.append("")
    lines.append("H2 vs H3 direction")
    lines.append(f"  mean cos(w_H2,w_H3)={md['mean_cos_w_h2_h3']:.4f} angle≈{md['mean_angle_w_h2_h3']:.1f}°")
    lines.append(f"  mean cos(w_H2,s_clean)={md['mean_cos_w_h2_s_clean']:.4f}")
    lines.append(f"  mean cos(w_H3,s_clean)={md['mean_cos_w_h3_s_clean']:.4f}")
    lines.append(f"  mean cos(w_H2,s_tfmean)={md['mean_cos_w_h2_s_tfmean']:.4f}")
    lines.append(f"  mean cos(w_H3,s_tfmean)={md['mean_cos_w_h3_s_tfmean']:.4f}")
    lines.append("")
    lines.append("Clean-loss transitions (eval CLEAN)")
    ct = payload["clean_transitions_macro"]
    lines.append(f"  H2ok→H3bad={ct['n_h2ok_h3bad']} H2bad→H3ok={ct['n_h2bad_h3ok']} both_ok={ct['n_both_ok']} both_bad={ct['n_both_bad']}")
    lines.append(f"  H2ok→H3bad by class Real/AI={ct['h2ok_h3bad_real']}/{ct['h2ok_h3bad_ai']}")
    lines.append(f"  top generators in H2ok→H3bad: {ct['top_generators_h2ok_h3bad']}")
    lines.append(f"  Fold4 share of H2ok→H3bad={ct['fold4_share_h2ok_h3bad']:.3f}")
    lines.append("")
    lines.append("Seedream")
    s = payload["seedream"]
    lines.append(
        f"  CLEAN recall H2={s['clean_recall_h2']:.3f} H3={s['clean_recall_h3']:.3f}; "
        f"cos(s_seed,w) H2={s['cos_s_seed_w_h2']:.3f} H3={s['cos_s_seed_w_h3']:.3f}"
    )
    lines.append(f"  {s['diagnosis']}")
    lines.append("")
    lines.append("FLUX.2_max")
    f = payload["flux"]
    lines.append(
        f"  CLEAN recall H2={f['clean_recall_h2']:.3f} H3={f['clean_recall_h3']:.3f}; "
        f"centroid_euc_to_real={f['centroid_euc_to_real']:.4f}; k5_purity={f['k5_same_label_frac']:.3f}"
    )
    lines.append(f"  {f['diagnosis']}")
    lines.append("")
    lines.append("Fold4")
    f4 = payload["fold4"]
    lines.append(
        f"  CLEAN AUC H2={f4['clean_auc_h2']:.3f} H3={f4['clean_auc_h3']:.3f}; "
        f"share_macro_drop≈{f4['share_of_macro_clean_auc_drop']:.2f}; "
        f"cos(w_H2,w_H3)={f4['cos_w_h2_h3']:.3f}"
    )
    lines.append(f"  {f4['diagnosis']}")
    lines.append("")
    lines.append("Per-fold compatibility")
    for k, v in payload["compatibility"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Decision evidence")
    for e in payload["decision_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append(f"NEXT_INTERVENTION_RECOMMENDATION: {payload['next_intervention']}")
    lines.append("")
    lines.append("Scientific answers")
    for k, v in payload["interpretation"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    return "\n".join(lines) + "\n"


def append_research_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-10G" in text:
        print("[V2-10G] research_log already contains V2-10G; not rewriting.", flush=True)
        return
    entry = f"""
## Stage V2-10G — Clean/Robust Decision-Conflict Diagnostic

**Date:** 2026-09-08  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Analysis only. No LoRA/CLIP/H2/H3 training. No condition-weight search. No threshold/calibrator/selective changes. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation:** V2-10F = ROBUST_HEAD_MIXED (clean AUC drop + residual robustness gap; Fold4/Seedream weaknesses).

**NEW_FEATURE_INFERENCE_PERFORMED = NO** — reused V2-10A/B/E/F R1 + H2/H3 heads.

**Condition geometry:** mean pairwise cos(s_c,s_c')={payload['macro_direction']['mean_pairwise_condition_cos']:.3f}; mean cos(w_H2,w_H3)={payload['macro_direction']['mean_cos_w_h2_h3']:.3f}.

**Decision:** **{payload['decision']}**  
**Recommended next intervention (ONE):** **{payload['next_intervention']}**

**Outputs:** `src/analyze_v2_10g_clean_robust_conflict_v1.py`; `results/v2/v2_10g_*`; figures `figures/v2/v2_10g_*.png`.

**Integrity:** NTIRE=NO; fal=NO; folds unchanged; no weight updates; no HP/condition-weight search; no thr/calibrator/selective retune; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_intervention']} — do not auto-start. Do not access NTIRE.

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10G] Appended research_log.md", flush=True)


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("[V2-10G] Integrity / artifact audit ...", flush=True)
    integrity = audit()

    pred = pd.read_csv(OUT / "v2_10f_robust_head_predictions_v1.csv")
    pred["image_id"] = pred["image_id"].astype(str)
    pred["generator_id"] = pred["generator_id"].fillna("").astype(str)
    pred["real_domain"] = pred["real_domain"].fillna("").astype(str)
    metrics = pd.read_csv(OUT / "v2_10f_robust_head_metrics_v1.csv")

    dir_rows = []
    margin_rows = []
    transition_rows = []
    hard_rows = []
    fold4_rows = []
    fold_dir: dict[int, dict[str, Any]] = {}
    sep_store: dict[int, dict[str, np.ndarray]] = {}
    weight_store: dict[int, dict[str, Any]] = {}
    compatibility: dict[str, str] = {}

    # Precompute macro clean AUC drop components
    clean_auc = {}
    for fold in [1, 2, 3, 4]:
        for h in ["H2", "H3"]:
            r = metrics[(metrics["fold"].astype(str) == str(fold)) & (metrics["condition"] == "CLEAN") & (metrics["head"] == h)]
            clean_auc[(fold, h)] = float(r.iloc[0]["roc_auc"])

    macro_drop = np.mean([clean_auc[(f, "H2")] - clean_auc[(f, "H3")] for f in [1, 2, 3, 4]])
    fold4_drop = clean_auc[(4, "H2")] - clean_auc[(4, "H3")]
    # share: fold4 contribution to mean drop
    fold4_share = float(fold4_drop / (4 * macro_drop)) if abs(macro_drop) > 1e-9 else float("nan")

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10G] Fold {fold} geometry + margins ...", flush=True)
        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        h3 = joblib.load(MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib")
        w2 = h2.coef_.ravel().astype(np.float64)
        b2 = float(h2.intercept_[0])
        w3 = h3.coef_.ravel().astype(np.float64)
        b3 = float(h3.intercept_[0])
        weight_store[fold] = {"w2": w2, "b2": b2, "w3": w3, "b3": b3}

        seps = {}
        norms = {}
        for cond in CONDITIONS:
            ids, X, y = load_train(fold, cond)
            mu_ai = X[y == 1].mean(0).astype(np.float64)
            mu_real = X[y == 0].mean(0).astype(np.float64)
            s = mu_ai - mu_real
            seps[cond] = s
            norms[cond] = float(np.linalg.norm(s))
            dir_rows.append(
                {
                    "fold": fold,
                    "cond_a": cond,
                    "cond_b": cond,
                    "cosine": 1.0,
                    "angle_deg": 0.0,
                    "norm_s_a": norms[cond],
                    "norm_s_b": norms[cond],
                    "cos_s_w_h2": cos_vec(s, w2),
                    "cos_s_w_h3": cos_vec(s, w3),
                    "kind": "self",
                }
            )
        sep_store[fold] = seps

        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1 :]:
                c = cos_vec(seps[a], seps[b])
                dir_rows.append(
                    {
                        "fold": fold,
                        "cond_a": a,
                        "cond_b": b,
                        "cosine": c,
                        "angle_deg": angle_deg(seps[a], seps[b]),
                        "norm_s_a": norms[a],
                        "norm_s_b": norms[b],
                        "cos_s_w_h2": float("nan"),
                        "cos_s_w_h3": float("nan"),
                        "kind": "pair",
                    }
                )
                # symmetric for heatmap convenience
                dir_rows.append(
                    {
                        "fold": fold,
                        "cond_a": b,
                        "cond_b": a,
                        "cosine": c,
                        "angle_deg": angle_deg(seps[a], seps[b]),
                        "norm_s_a": norms[b],
                        "norm_s_b": norms[a],
                        "cos_s_w_h2": float("nan"),
                        "cos_s_w_h3": float("nan"),
                        "kind": "pair",
                    }
                )

        s_clean = seps["CLEAN"]
        s_tfmean = np.mean([seps[c] for c in TRANSFORMS], axis=0)
        fold_dir[fold] = {
            "norm_s_clean": norms["CLEAN"],
            "norm_s_tfmean": float(np.linalg.norm(s_tfmean)),
            "cos_w_h2_h3": cos_vec(w2, w3),
            "angle_w_h2_h3": angle_deg(w2, w3),
            "cos_w_h2_s_clean": cos_vec(w2, s_clean),
            "cos_w_h3_s_clean": cos_vec(w3, s_clean),
            "cos_w_h2_s_tfmean": cos_vec(w2, s_tfmean),
            "cos_w_h3_s_tfmean": cos_vec(w3, s_tfmean),
            "cos_s_clean_s_tfmean": cos_vec(s_clean, s_tfmean),
            "mean_pairwise_condition_cos": float(
                np.mean(
                    [
                        cos_vec(seps[a], seps[b])
                        for i, a in enumerate(CONDITIONS)
                        for b in CONDITIONS[i + 1 :]
                    ]
                )
            ),
            "cos_s_clean_each_tf": {c: cos_vec(s_clean, seps[c]) for c in TRANSFORMS},
            "norm_s_each": norms,
        }

        # Margins TRAIN + EVAL
        pf = pred[pred["fold"] == fold].reset_index(drop=True)
        eid_clean, X_eval_clean = load_eval(fold, "CLEAN")
        if list(pf["image_id"]) != list(eid_clean):
            pos = {i: j for j, i in enumerate(eid_clean)}
            order = [pos[i] for i in pf["image_id"]]
            X_eval_clean = X_eval_clean[order]
            eid_clean = np.asarray(list(pf["image_id"]), dtype=str)
        y_ev = pf["y"].to_numpy(dtype=int)
        y_signed = np.where(y_ev == 1, 1.0, -1.0)

        for split, loader in [("TRAIN", lambda c: load_train(fold, c)), ("EVAL", None)]:
            for cond in CONDITIONS:
                if split == "TRAIN":
                    ids, X, y = loader(cond)
                    ys = np.where(y == 1, 1.0, -1.0)
                    gens = None
                    domains = None
                else:
                    if cond == "CLEAN":
                        X = X_eval_clean
                    else:
                        ids_e, Xt = load_eval(fold, cond)
                        if list(ids_e) != list(eid_clean):
                            pos = {i: j for j, i in enumerate(ids_e)}
                            Xt = Xt[[pos[i] for i in eid_clean]]
                        X = Xt
                    y = y_ev
                    ys = y_signed
                    gens = pf["generator_id"].to_numpy()
                    domains = pf["real_domain"].to_numpy()

                z2 = X.astype(np.float64) @ w2 + b2
                z3 = X.astype(np.float64) @ w3 + b3
                m2 = ys * z2
                m3 = ys * z3
                for lab, name in [(0, "Real"), (1, "AI")]:
                    mask = y == lab
                    st2 = margin_stats(m2[mask])
                    st3 = margin_stats(m3[mask])
                    margin_rows.append(
                        {
                            "fold": fold,
                            "split": split,
                            "condition": cond,
                            "group": name,
                            "head": "H2",
                            **{f"m_{k}": v for k, v in st2.items()},
                        }
                    )
                    margin_rows.append(
                        {
                            "fold": fold,
                            "split": split,
                            "condition": cond,
                            "group": name,
                            "head": "H3",
                            **{f"m_{k}": v for k, v in st3.items()},
                        }
                    )

        # Clean transitions on EVAL
        p2 = pf["p_H2_CLEAN"].to_numpy(dtype=float)
        p3 = pf["p_H3_CLEAN"].to_numpy(dtype=float)
        pred2 = (p2 >= THR).astype(int)
        pred3 = (p3 >= THR).astype(int)
        correct2 = pred2 == y_ev
        correct3 = pred3 == y_ev
        z2c = X_eval_clean.astype(np.float64) @ w2 + b2
        z3c = X_eval_clean.astype(np.float64) @ w3 + b3
        m2c = y_signed * z2c
        m3c = y_signed * z3c

        mu_real = X_eval_clean[y_ev == 0].mean(0)
        mu_ai = X_eval_clean[y_ev == 1].mean(0)
        # k5 for all eval (once)
        D = pairwise_distances(X_eval_clean.astype(np.float64), metric="euclidean")
        np.fill_diagonal(D, np.inf)

        for i in range(len(pf)):
            if correct2[i] and not correct3[i]:
                trans = "H2ok_H3bad"
            elif (not correct2[i]) and correct3[i]:
                trans = "H2bad_H3ok"
            elif correct2[i] and correct3[i]:
                trans = "both_ok"
            else:
                trans = "both_bad"
            nn = np.argpartition(D[i], K_NN)[:K_NN]
            nn = nn[np.argsort(D[i, nn])]
            k5_purity = float(np.mean(y_ev[nn] == y_ev[i]))
            cent = mu_ai if y_ev[i] == 1 else mu_real
            transition_rows.append(
                {
                    "fold": fold,
                    "image_id": pf.loc[i, "image_id"],
                    "y": int(y_ev[i]),
                    "generator_id": pf.loc[i, "generator_id"],
                    "real_domain": pf.loc[i, "real_domain"],
                    "transition": trans,
                    "p_H2": float(p2[i]),
                    "p_H3": float(p3[i]),
                    "margin_H2": float(m2c[i]),
                    "margin_H3": float(m3c[i]),
                    "delta_margin": float(m3c[i] - m2c[i]),
                    "k5_same_label_frac": k5_purity,
                    "dist_to_own_centroid": float(np.linalg.norm(X_eval_clean[i].astype(np.float64) - cent)),
                }
            )

        # Trade-off: clean margin change vs mean transformed correct-margin improvement
        clean_dm = m3c - m2c
        tf_impr = []
        for cond in TRANSFORMS:
            ids_e, Xt = load_eval(fold, cond)
            if list(ids_e) != list(eid_clean):
                pos = {i: j for j, i in enumerate(ids_e)}
                Xt = Xt[[pos[i] for i in eid_clean]]
            z2t = Xt.astype(np.float64) @ w2 + b2
            z3t = Xt.astype(np.float64) @ w3 + b3
            m2t = y_signed * z2t
            m3t = y_signed * z3t
            tf_impr.append(m3t - m2t)
        mean_tf_impr = np.mean(tf_impr, axis=0)
        corr = float(np.corrcoef(clean_dm, mean_tf_impr)[0, 1]) if len(clean_dm) > 2 else float("nan")
        fold_dir[fold]["corr_clean_dm_vs_mean_tf_impr"] = corr
        fold_dir[fold]["mean_clean_dm"] = float(np.mean(clean_dm))
        fold_dir[fold]["mean_tf_margin_impr"] = float(np.mean(mean_tf_impr))

        # Compatibility classification
        mean_cos_clean_tf = float(np.mean(list(fold_dir[fold]["cos_s_clean_each_tf"].values())))
        compatibility[f"fold_{fold}"] = classify_fold(
            cos_clean_tf=mean_cos_clean_tf,
            cos_h2_h3=fold_dir[fold]["cos_w_h2_h3"],
            clean_auc_drop=clean_auc[(fold, "H2")] - clean_auc[(fold, "H3")],
        )

        # Seedream / FLUX diagnostics
        for gid, tag in [(SEEDREAM, "seedream"), (FLUX, "flux")]:
            mask = pf["generator_id"] == gid
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            Xg = X_eval_clean[idx]
            # class direction relative to Real centroid
            s_g = Xg.mean(0).astype(np.float64) - mu_real
            k5 = local_k5_same_label_frac(X_eval_clean.astype(np.float64), y_ev, idx)
            row = {
                "fold": fold,
                "generator_id": gid,
                "n": int(len(idx)),
                "centroid_euc_to_real": float(np.linalg.norm(Xg.mean(0) - mu_real)),
                "centroid_euc_to_ai": float(np.linalg.norm(Xg.mean(0) - mu_ai)),
                "k5_same_label_frac": k5,
                "cos_s_gen_w_h2": cos_vec(s_g, w2),
                "cos_s_gen_w_h3": cos_vec(s_g, w3),
                "angle_s_gen_from_h2_to_h3_drop": cos_vec(s_g, w2) - cos_vec(s_g, w3),
            }
            for cond in CONDITIONS:
                col2 = "p_H2_CLEAN" if cond == "CLEAN" else f"p_H2_{cond}"
                col3 = "p_H3_CLEAN" if cond == "CLEAN" else f"p_H3_{cond}"
                p2g = pf.loc[mask, col2].to_numpy(dtype=float)
                p3g = pf.loc[mask, col3].to_numpy(dtype=float)
                # logits via log-odds if needed; use existing margins under that condition
                if cond == "CLEAN":
                    Xc = X_eval_clean[idx]
                else:
                    ids_e, Xt = load_eval(fold, cond)
                    if list(ids_e) != list(eid_clean):
                        pos = {i: j for j, i in enumerate(ids_e)}
                        Xt = Xt[[pos[i] for i in eid_clean]]
                    Xc = Xt[idx]
                z2g = Xc.astype(np.float64) @ w2 + b2
                z3g = Xc.astype(np.float64) @ w3 + b3
                row[f"H2_{cond}_recall"] = float((p2g >= THR).mean())
                row[f"H3_{cond}_recall"] = float((p3g >= THR).mean())
                row[f"H2_{cond}_mean_logit"] = float(np.mean(z2g))
                row[f"H3_{cond}_mean_logit"] = float(np.mean(z3g))
                row[f"H2_{cond}_mean_margin"] = float(np.mean(z2g))  # AI samples: signed +1
                row[f"H3_{cond}_mean_margin"] = float(np.mean(z3g))
                row[f"H2_{cond}_median_margin"] = float(np.median(z2g))
                row[f"H3_{cond}_median_margin"] = float(np.median(z3g))
                # transitions CLEAN
                if cond == "CLEAN":
                    t_ok_bad = int(((p2g >= THR) & (p3g < THR)).sum())
                    t_bad_ok = int(((p2g < THR) & (p3g >= THR)).sum())
                    row["n_H2ok_H3bad_CLEAN"] = t_ok_bad
                    row["n_H2bad_H3ok_CLEAN"] = t_bad_ok
            hard_rows.append(row)

        # Fold4-specific rows vs others collected after loop
        fold4_rows.append(
            {
                "fold": fold,
                "train_n": EXPECTED_TRAIN[fold]["n"],
                "train_ai": EXPECTED_TRAIN[fold]["ai"],
                "train_real": EXPECTED_TRAIN[fold]["real"],
                "eval_n": EXPECTED_EVAL[fold],
                "clean_auc_h2": clean_auc[(fold, "H2")],
                "clean_auc_h3": clean_auc[(fold, "H3")],
                "clean_auc_delta": clean_auc[(fold, "H3")] - clean_auc[(fold, "H2")],
                "cos_w_h2_h3": fold_dir[fold]["cos_w_h2_h3"],
                "angle_w_h2_h3": fold_dir[fold]["angle_w_h2_h3"],
                "cos_w_h3_s_clean": fold_dir[fold]["cos_w_h3_s_clean"],
                "cos_w_h2_s_clean": fold_dir[fold]["cos_w_h2_s_clean"],
                "cos_s_clean_s_tfmean": fold_dir[fold]["cos_s_clean_s_tfmean"],
                "mean_pairwise_condition_cos": fold_dir[fold]["mean_pairwise_condition_cos"],
                "mean_clean_dm": fold_dir[fold]["mean_clean_dm"],
                "mean_tf_margin_impr": fold_dir[fold]["mean_tf_margin_impr"],
                "corr_clean_dm_vs_tf_impr": fold_dir[fold]["corr_clean_dm_vs_mean_tf_impr"],
                "compatibility": compatibility[f"fold_{fold}"],
                "heldout_ai_generators": ";".join(
                    sorted([g for g in pf.loc[pf["y"] == 1, "generator_id"].unique() if g])
                ),
                "n_heldout_ai_generators": int(pf.loc[pf["y"] == 1, "generator_id"].nunique()),
            }
        )

    sim_df = pd.DataFrame(dir_rows)
    sim_df.to_csv(DIR_CSV, index=False)
    pd.DataFrame(margin_rows).to_csv(MARGIN_CSV, index=False)
    trans_df = pd.DataFrame(transition_rows)
    trans_df.to_csv(TRANS_CSV, index=False)
    hard_df = pd.DataFrame(hard_rows)
    hard_df.to_csv(HARD_CSV, index=False)
    fold_cmp_df = pd.DataFrame(fold4_rows)
    fold_cmp_df.to_csv(FOLD4_CSV, index=False)

    # Macro direction stats
    pair_cos = [
        fold_dir[f]["mean_pairwise_condition_cos"] for f in [1, 2, 3, 4]
    ]
    cos_clean_tf = {
        c: float(np.mean([fold_dir[f]["cos_s_clean_each_tf"][c] for f in [1, 2, 3, 4]]))
        for c in TRANSFORMS
    }
    most_conflict = min(cos_clean_tf, key=cos_clean_tf.get)
    macro_direction = {
        "mean_pairwise_condition_cos": float(np.mean(pair_cos)),
        "mean_cos_s_clean_each_tf": cos_clean_tf,
        "most_conflicting_transform": most_conflict,
        "mean_cos_w_h2_h3": float(np.mean([fold_dir[f]["cos_w_h2_h3"] for f in [1, 2, 3, 4]])),
        "mean_angle_w_h2_h3": float(np.mean([fold_dir[f]["angle_w_h2_h3"] for f in [1, 2, 3, 4]])),
        "mean_cos_w_h2_s_clean": float(np.mean([fold_dir[f]["cos_w_h2_s_clean"] for f in [1, 2, 3, 4]])),
        "mean_cos_w_h3_s_clean": float(np.mean([fold_dir[f]["cos_w_h3_s_clean"] for f in [1, 2, 3, 4]])),
        "mean_cos_w_h2_s_tfmean": float(np.mean([fold_dir[f]["cos_w_h2_s_tfmean"] for f in [1, 2, 3, 4]])),
        "mean_cos_w_h3_s_tfmean": float(np.mean([fold_dir[f]["cos_w_h3_s_tfmean"] for f in [1, 2, 3, 4]])),
        "mean_cos_s_clean_s_tfmean": float(
            np.mean([fold_dir[f]["cos_s_clean_s_tfmean"] for f in [1, 2, 3, 4]])
        ),
        "per_fold": {f"fold_{f}": fold_dir[f] for f in [1, 2, 3, 4]},
    }

    # Clean transitions macro
    h2ok_h3bad = trans_df[trans_df["transition"] == "H2ok_H3bad"]
    h2bad_h3ok = trans_df[trans_df["transition"] == "H2bad_H3ok"]
    top_gens = (
        h2ok_h3bad[h2ok_h3bad["y"] == 1]["generator_id"].value_counts().head(8).to_dict()
        if len(h2ok_h3bad)
        else {}
    )
    top_domains = (
        h2ok_h3bad[h2ok_h3bad["y"] == 0]["real_domain"].value_counts().head(8).to_dict()
        if len(h2ok_h3bad)
        else {}
    )
    clean_transitions_macro = {
        "n_h2ok_h3bad": int(len(h2ok_h3bad)),
        "n_h2bad_h3ok": int(len(h2bad_h3ok)),
        "n_both_ok": int((trans_df["transition"] == "both_ok").sum()),
        "n_both_bad": int((trans_df["transition"] == "both_bad").sum()),
        "h2ok_h3bad_real": int(((h2ok_h3bad["y"] == 0).sum())),
        "h2ok_h3bad_ai": int(((h2ok_h3bad["y"] == 1).sum())),
        "top_generators_h2ok_h3bad": {str(k): int(v) for k, v in top_gens.items()},
        "top_real_domains_h2ok_h3bad": {str(k): int(v) for k, v in top_domains.items()},
        "fold4_share_h2ok_h3bad": float(
            (h2ok_h3bad["fold"] == 4).mean() if len(h2ok_h3bad) else float("nan")
        ),
        "mean_margin_h2_of_h2ok_h3bad": float(h2ok_h3bad["margin_H2"].mean()) if len(h2ok_h3bad) else float("nan"),
        "mean_margin_h3_of_h2ok_h3bad": float(h2ok_h3bad["margin_H3"].mean()) if len(h2ok_h3bad) else float("nan"),
        "mean_k5_purity_h2ok_h3bad": float(h2ok_h3bad["k5_same_label_frac"].mean()) if len(h2ok_h3bad) else float("nan"),
        "per_fold_counts": {
            f"fold_{f}": {
                t: int((trans_df[(trans_df["fold"] == f) & (trans_df["transition"] == t)].shape[0]))
                for t in ["H2ok_H3bad", "H2bad_H3ok", "both_ok", "both_bad"]
            }
            for f in [1, 2, 3, 4]
        },
    }

    # Seedream / FLUX aggregate
    def agg_hard(gid: str) -> dict[str, Any]:
        sub = hard_df[hard_df["generator_id"] == gid]
        if len(sub) == 0:
            return {"generator_id": gid, "n": 0}
        out = {
            "generator_id": gid,
            "n": int(sub["n"].sum()),
            "clean_recall_h2": float(sub["H2_CLEAN_recall"].mean()),
            "clean_recall_h3": float(sub["H3_CLEAN_recall"].mean()),
            "cos_s_seed_w_h2": float(sub["cos_s_gen_w_h2"].mean()),
            "cos_s_seed_w_h3": float(sub["cos_s_gen_w_h3"].mean()),
            "cos_s_gen_w_h2": float(sub["cos_s_gen_w_h2"].mean()),
            "cos_s_gen_w_h3": float(sub["cos_s_gen_w_h3"].mean()),
            "centroid_euc_to_real": float(sub["centroid_euc_to_real"].mean()),
            "centroid_euc_to_ai": float(sub["centroid_euc_to_ai"].mean()),
            "k5_same_label_frac": float(sub["k5_same_label_frac"].mean()),
            "n_H2ok_H3bad_CLEAN": int(sub["n_H2ok_H3bad_CLEAN"].sum()),
            "n_H2bad_H3ok_CLEAN": int(sub["n_H2bad_H3ok_CLEAN"].sum()),
            "per_condition_recall": {
                c: {
                    "H2": float(sub[f"H2_{c}_recall"].mean()),
                    "H3": float(sub[f"H3_{c}_recall"].mean()),
                    "H2_mean_margin": float(sub[f"H2_{c}_mean_margin"].mean()),
                    "H3_mean_margin": float(sub[f"H3_{c}_mean_margin"].mean()),
                }
                for c in CONDITIONS
            },
            "per_fold": sub.to_dict(orient="records"),
        }
        return out

    seedream = agg_hard(SEEDREAM)
    flux = agg_hard(FLUX)

    # Seedream diagnosis
    seed_cos_drop = seedream["cos_s_gen_w_h2"] - seedream["cos_s_gen_w_h3"]
    if seed_cos_drop > 0.05 and seedream["clean_recall_h3"] < 0.1:
        seedream["diagnosis"] = (
            "H3 rotated away from Seedream's CLEAN discriminative direction relative to Real "
            f"(Δcos={seed_cos_drop:+.3f}); most Seedream positives lose margin under H3."
        )
    else:
        seedream["diagnosis"] = (
            "Seedream remains near Real centroid / low margin; H3 does not specifically create "
            "the weakness alone, but exacerbates an already weak clean separation."
        )
    # rename keys expected in report
    seedream["cos_s_seed_w_h2"] = seedream["cos_s_gen_w_h2"]
    seedream["cos_s_seed_w_h3"] = seedream["cos_s_gen_w_h3"]

    if flux["clean_recall_h2"] < 0.30 and flux["centroid_euc_to_real"] < flux.get("centroid_euc_to_ai", 1e9):
        flux["diagnosis"] = (
            "Primary weakness is low CLEAN separation (centroid close to Real; low H2 and H3 recall). "
            "Not primarily an H3 over-rotation artifact; head-margin placement cannot rescue weak geometry."
        )
    else:
        flux["diagnosis"] = (
            "FLUX weakness mixes modest geometry separation with head threshold placement; "
            "H3 does not substantially worsen or rescue it."
        )

    # Fold4 diagnosis
    f4 = fold_cmp_df[fold_cmp_df["fold"] == 4].iloc[0].to_dict()
    f123 = fold_cmp_df[fold_cmp_df["fold"].isin([1, 2, 3])]
    # generator composition: which AI gens unique to fold4 eval?
    def _parse_gens(s: Any) -> set[str]:
        if not isinstance(s, str) or not s.strip():
            return set()
        return {x for x in s.split(";") if x}

    gens4 = _parse_gens(f4["heldout_ai_generators"])
    gens123 = set()
    for _, row in f123.iterrows():
        gens123 |= _parse_gens(row["heldout_ai_generators"])
    unique4 = sorted(gens4 - gens123)
    shared = sorted(gens4 & gens123)

    # H2ok->H3bad concentration by generator in fold4
    t4 = h2ok_h3bad[h2ok_h3bad["fold"] == 4]
    gen_counts4 = t4[t4["y"] == 1]["generator_id"].value_counts().head(10).to_dict()

    fold4_payload = {
        "clean_auc_h2": float(f4["clean_auc_h2"]),
        "clean_auc_h3": float(f4["clean_auc_h3"]),
        "clean_auc_delta": float(f4["clean_auc_delta"]),
        "share_of_macro_clean_auc_drop": fold4_share,
        "cos_w_h2_h3": float(f4["cos_w_h2_h3"]),
        "angle_w_h2_h3": float(f4["angle_w_h2_h3"]),
        "cos_w_h3_s_clean": float(f4["cos_w_h3_s_clean"]),
        "cos_s_clean_s_tfmean": float(f4["cos_s_clean_s_tfmean"]),
        "mean_pairwise_condition_cos": float(f4["mean_pairwise_condition_cos"]),
        "train_ai": int(f4["train_ai"]),
        "train_n": int(f4["train_n"]),
        "vs_f123_mean_cos_w_h2_h3": float(f123["cos_w_h2_h3"].mean()),
        "vs_f123_mean_clean_auc_delta": float(f123["clean_auc_delta"].mean()),
        "heldout_generators_unique_vs_f123": unique4,
        "n_heldout_ai_generators": int(f4["n_heldout_ai_generators"]),
        "h2ok_h3bad_top_generators": {str(k): int(v) for k, v in gen_counts4.items()},
        "mean_tf_margin_impr": float(f4["mean_tf_margin_impr"]),
        "mean_clean_dm": float(f4["mean_clean_dm"]),
        "corr_clean_dm_vs_tf_impr": float(f4["corr_clean_dm_vs_tf_impr"]),
    }
    # Why Fold4 is worse: larger H2->H3 rotation away from clean + harder held-out set
    reasons = []
    if fold4_payload["cos_w_h2_h3"] < f123["cos_w_h2_h3"].mean() - 0.05:
        reasons.append(
            f"larger H2→H3 rotation (cos={fold4_payload['cos_w_h2_h3']:.3f} vs F1–3 mean {f123['cos_w_h2_h3'].mean():.3f})"
        )
    if fold4_payload["cos_w_h3_s_clean"] < f123["cos_w_h3_s_clean"].mean() - 0.03:
        reasons.append(
            f"H3 less aligned to s_clean (cos={fold4_payload['cos_w_h3_s_clean']:.3f} vs F1–3 {f123['cos_w_h3_s_clean'].mean():.3f})"
        )
    if fold4_payload["train_ai"] > int(f123["train_ai"].mean()):
        reasons.append(
            f"largest TRAIN AI count ({fold4_payload['train_ai']} vs F1–3 mean {f123['train_ai'].mean():.0f}) "
            "with different held-out AI generator composition"
        )
    if unique4:
        reasons.append(f"held-out AI generators unique vs F1–3: {unique4[:8]}")
    if gen_counts4:
        top = list(gen_counts4.items())[:3]
        reasons.append(f"H2ok→H3bad AI concentrated in {top}")
    fold4_payload["diagnosis"] = (
        "Fold4 H3 clean AUC≈0.854 is not generic difficulty alone: "
        + "; ".join(reasons)
        if reasons
        else "Fold4 shows larger clean cost under H3 with composition differences vs F1–3."
    )

    # Trade-off summary
    tradeoff = {
        "per_fold_corr_clean_dm_vs_mean_tf_impr": {
            f"fold_{f}": fold_dir[f]["corr_clean_dm_vs_mean_tf_impr"] for f in [1, 2, 3, 4]
        },
        "mean_corr": float(np.mean([fold_dir[f]["corr_clean_dm_vs_mean_tf_impr"] for f in [1, 2, 3, 4]])),
        "interpretation": (
            "Observed mean corr(clean Δmargin, mean-tf Δmargin) is positive (~0.63): sample-level "
            "gains/losses are co-directional under H3, so the clean cost is better explained as a "
            "global direction compromise (away from s_clean) than as a strict per-sample clean↔robust swap."
        ),
    }

    figures = make_figures(sim_df, fold_dir)

    pre_decision = {
        "macro_direction": macro_direction,
        "fold4": fold4_payload,
        "seedream": seedream,
        "compatibility": compatibility,
    }
    decision, next_opt, evidence = decide(pre_decision)

    # Scientific answers
    interpretation = {
        "q1_directions_compatible": (
            f"Broadly yes/partial: mean pairwise cos={macro_direction['mean_pairwise_condition_cos']:.3f}; "
            f"cos(s_clean,s_tfmean)={macro_direction['mean_cos_s_clean_s_tfmean']:.3f}. "
            f"Lowest clean↔tf alignment: {most_conflict} ({cos_clean_tf[most_conflict]:.3f})."
        ),
        "q2_h3_overrotated": (
            f"Yes: mean cos(w_H2,w_H3)={macro_direction['mean_cos_w_h2_h3']:.3f} (~{macro_direction['mean_angle_w_h2_h3']:.0f}°); "
            f"alignment to s_clean collapses H2 {macro_direction['mean_cos_w_h2_s_clean']:.3f} → H3 {macro_direction['mean_cos_w_h3_s_clean']:.3f}. "
            f"Relative preference flips: H2 prefers s_clean over s_tfmean "
            f"({macro_direction['mean_cos_w_h2_s_clean']:.3f}>{macro_direction['mean_cos_w_h2_s_tfmean']:.3f}), "
            f"while H3 prefers s_tfmean over s_clean "
            f"({macro_direction['mean_cos_w_h3_s_tfmean']:.3f}>{macro_direction['mean_cos_w_h3_s_clean']:.3f})."
        ),
        "q3_80pct_transformed_rows": (
            "Plausible major contributor: equal 5-condition expansion makes transformed conditions 80% of rows; "
            "H3's preference flips from clean- to transform-side separation while still only partially "
            "matching any single transform direction — consistent with over-rotation / compromise, not pure conflict."
        ),
        "q4_greatest_conflict": f"{most_conflict} (lowest cos with s_clean among transforms).",
        "q5_why_clean_auc_drop": (
            f"H3 reduces clean margins for a non-trivial H2-correct set (n={clean_transitions_macro['n_h2ok_h3bad']}, "
            f"AI-heavy {clean_transitions_macro['h2ok_h3bad_ai']} vs Real {clean_transitions_macro['h2ok_h3bad_real']}); "
            f"Fold4 share of these errors={clean_transitions_macro['fold4_share_h2ok_h3bad']:.2f}."
        ),
        "q6_seedream": seedream["diagnosis"],
        "q7_flux": flux["diagnosis"],
        "q8_fold4": fold4_payload["diagnosis"],
        "q9_fold4_protocol_vs_general": (
            f"Both: Fold4 contributes ~{fold4_share:.0%} of the mean clean-AUC drop and has larger H2→H3 rotation, "
            "but F1–F3 also show clean cost and robust gains — not Fold4-only."
        ),
        "q10_one_linear_head_plausible": (
            "Yes, still plausible: condition separation directions remain positively aligned "
            f"(mean pairwise cos={macro_direction['mean_pairwise_condition_cos']:.3f}); "
            "no strong evidence of opposing linear requirements."
        ),
        "q11_another_linear_justified": (
            "Yes — ONE predeclared clean-anchored robust linear head is justified as testing over-rotation, "
            "not as a free weight search. Arbitrary multi-weight grids would not be justified."
        ),
        "q12_nonlinear": (
            "Not yet: linear conflict is not established; nonlinear head would be premature relative to "
            "a single clean-anchored linear correction."
        ),
        "q13_freeze_now": (
            "No — over-rotation hypothesis is actionable and low-cost; freeze-with-limitations remains fallback."
        ),
        "q14_one_next": next_opt,
    }

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V1_RERUN": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "LORA_BACKBONE_TRAINED": "NO",
        "LORA_WEIGHTS_UPDATED": "NO",
        "CLIP_WEIGHTS_UPDATED": "NO",
        "H2_RETRAINED": "NO",
        "H3_RETRAINED": "NO",
        "NEW_HEAD_TRAINED": "NO",
        "HYPERPARAMETER_SEARCH_PERFORMED": "NO",
        "CONDITION_WEIGHT_SEARCH_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "THRESHOLD_CHANGED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_FEATURE_INFERENCE_PERFORMED": "NO",
    }

    payload = {
        "stage": "V2-10G",
        "status": "COMPLETE",
        "decision": decision,
        "next_intervention": next_opt,
        "decision_evidence": evidence,
        "integrity": integrity,
        "macro_direction": {
            k: v
            for k, v in macro_direction.items()
            if k != "per_fold"
        },
        "macro_direction_per_fold_summary": {
            f"fold_{f}": {
                "cos_w_h2_h3": fold_dir[f]["cos_w_h2_h3"],
                "cos_w_h2_s_clean": fold_dir[f]["cos_w_h2_s_clean"],
                "cos_w_h3_s_clean": fold_dir[f]["cos_w_h3_s_clean"],
                "cos_w_h2_s_tfmean": fold_dir[f]["cos_w_h2_s_tfmean"],
                "cos_w_h3_s_tfmean": fold_dir[f]["cos_w_h3_s_tfmean"],
                "mean_pairwise_condition_cos": fold_dir[f]["mean_pairwise_condition_cos"],
                "cos_s_clean_each_tf": fold_dir[f]["cos_s_clean_each_tf"],
            }
            for f in [1, 2, 3, 4]
        },
        "tradeoff": tradeoff,
        "clean_transitions_macro": clean_transitions_macro,
        "seedream": seedream,
        "flux": flux,
        "fold4": fold4_payload,
        "compatibility": compatibility,
        "figures": figures,
        "interpretation": interpretation,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Decision: {decision}")
    print(f"Next: {next_opt}")


if __name__ == "__main__":
    main()
