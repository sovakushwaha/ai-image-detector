#!/usr/bin/env python3
"""V2-10H — Clean-Anchored Robust Linear Head (H4).

H4 = identical LogReg to H2/H3 on frozen LoRA R1 with LOCKED sample weights:
  CLEAN=4.0, jpeg=1.0, resize=1.0, blur=1.0, screenshot=1.0
  => effective 50% clean / 50% transformed (equal across 4 transforms).

NO weight search. NO feature extraction. NO LoRA/CLIP/H2/H3 training.
Threshold=0.5. FINAL_V2_MODEL NOT SELECTED.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"

SEED = 42
N_BOOT = 5000
THR = 0.5
TRANSFORMS = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
CONDITIONS = ["CLEAN"] + TRANSFORMS
# Locked condition sample weights (raw rows unchanged)
COND_WEIGHT = {
    "CLEAN": 4.0,
    "jpeg_q50": 1.0,
    "resize_112": 1.0,
    "blur_sigma2": 1.0,
    "screenshot_strong": 1.0,
}
EXPECTED_TRAIN = {
    1: {"n": 6008, "real": 4280, "ai": 1728, "rows": 30040},
    2: {"n": 6009, "real": 4280, "ai": 1729, "rows": 30045},
    3: {"n": 6309, "real": 4280, "ai": 2029, "rows": 31545},
    4: {"n": 6594, "real": 4280, "ai": 2314, "rows": 32970},
}
EXPECTED_EVAL = {1: 2821, 2: 2619, 3: 1994, 4: 1709}
HARD = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]
SEEDREAM = "qwen::Seedream-5.0"
FLUX = "qwen::FLUX.2_max"

JSON_OUT = OUT / "v2_10h_analysis_v1.json"
REPORT_OUT = OUT / "v2_10h_report_v1.txt"
PRED_OUT = OUT / "v2_10h_predictions_v1.csv"
METRICS_OUT = OUT / "v2_10h_metrics_v1.csv"
BOOT_OUT = OUT / "v2_10h_bootstrap_v1.json"
DIR_CSV = OUT / "v2_10h_direction_analysis_v1.csv"
RESOURCE_OUT = OUT / "v2_10h_resource_comparison_v1.csv"


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


def metrics_at_05(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= THR).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    ai_rec = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    real_spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, p)) if (y == 1).any() else float("nan"),
        "ai_recall": ai_rec,
        "real_specificity": real_spec,
        "balanced_accuracy": float(0.5 * (ai_rec + real_spec)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def paired_boot(y, p_new, p_base, n_boot=N_BOOT, seed=SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new).astype(float)
    p_base = np.asarray(p_base).astype(float)
    real_idx = np.where(y == 0)[0]
    ai_idx = np.where(y == 1)[0]
    keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy"]
    buckets = {k: [] for k in keys}
    for _ in range(n_boot):
        ri = rng.choice(real_idx, size=len(real_idx), replace=True)
        ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
        idx = np.concatenate([ri, ai])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        mn = metrics_at_05(yy, p_new[idx])
        mb = metrics_at_05(yy, p_base[idx])
        for k in keys:
            buckets[k].append(mn[k] - mb[k])
    out: dict[str, Any] = {"n_boot": n_boot, "seed": seed, "stratified": True}
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_valid": int(len(arr)),
        }
    return out


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
    for fold in [1, 2, 3, 4]:
        if not (MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib").is_file():
            stop(f"missing H2 fold {fold}")
        if not (MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib").is_file():
            stop(f"missing H3 fold {fold}")
        ids0, X0, y0 = load_train(fold, "CLEAN")
        exp = EXPECTED_TRAIN[fold]
        if len(ids0) != exp["n"] or int((y0 == 0).sum()) != exp["real"] or int((y0 == 1).sum()) != exp["ai"]:
            stop(f"fold {fold} CLEAN train membership mismatch")
        if X0.shape != (exp["n"], 512) or not np.isfinite(X0).all() or len(np.unique(ids0)) != len(ids0):
            stop(f"fold {fold} CLEAN train R1 integrity fail")
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
            ids_e, Xt = load_eval(fold, c)
            if list(ids_e) != list(eid) or Xt.shape != Xe.shape or not np.isfinite(Xt).all():
                stop(f"fold {fold} eval {c} integrity fail")
    return {"ok": True, "new_feature_inference": False}


def build_h4_matrix(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ids, Xc, y = load_train(fold, "CLEAN")
    blocks = [Xc]
    sw_blocks = [np.full(len(y), COND_WEIGHT["CLEAN"], dtype=np.float64)]
    for c in TRANSFORMS:
        blocks.append(load_train(fold, c)[1])
        sw_blocks.append(np.full(len(y), COND_WEIGHT[c], dtype=np.float64))
    X = np.concatenate(blocks, axis=0)
    yh = np.concatenate([y] * 5, axis=0)
    sw = np.concatenate(sw_blocks, axis=0)
    exp_rows = EXPECTED_TRAIN[fold]["rows"]
    if X.shape != (exp_rows, 512) or len(yh) != exp_rows or len(sw) != exp_rows:
        stop(f"fold {fold} H4 matrix shape mismatch {X.shape}")

    # Document class_weight="balanced" × sample_weight interaction (sklearn):
    # balanced cw_c = n_samples / (n_classes * count_c) using RAW row counts (unweighted),
    # then final per-row weight = sample_weight_i * cw_{y_i}.
    classes = np.array([0, 1])
    cw_arr = compute_class_weight(class_weight="balanced", classes=classes, y=yh)
    cw = {int(c): float(w) for c, w in zip(classes, cw_arr)}
    final_w = sw * np.array([cw[int(yi)] for yi in yh], dtype=np.float64)

    cond_eff = {}
    offset = 0
    n = len(y)
    for c in CONDITIONS:
        sl = slice(offset, offset + n)
        cond_eff[c] = {
            "raw_rows": int(n),
            "sample_weight_value": float(COND_WEIGHT[c]),
            "sum_sample_weight": float(sw[sl].sum()),
            "sum_final_weight": float(final_w[sl].sum()),
            "frac_sample_weight": float(sw[sl].sum() / sw.sum()),
            "frac_final_weight": float(final_w[sl].sum() / final_w.sum()),
        }
        offset += n

    meta = {
        "raw_rows": int(len(yh)),
        "raw_real": int((yh == 0).sum()),
        "raw_ai": int((yh == 1).sum()),
        "sum_sample_weight": float(sw.sum()),
        "sum_final_weight": float(final_w.sum()),
        "sklearn_balanced_class_weights_from_raw_counts": cw,
        "implementation": (
            "LogisticRegression(class_weight='balanced').fit(X,y,sample_weight=sw); "
            "sklearn multiplies balanced class weights (from raw row class counts) by sample_weight. "
            "Locked sw: CLEAN=4, each transform=1 => 50%/50% clean vs transformed by sample_weight mass."
        ),
        "condition_effective_weights": cond_eff,
        "effective_weight_by_class": {
            "Real": float(final_w[yh == 0].sum()),
            "AI": float(final_w[yh == 1].sum()),
            "Real_frac_final": float(final_w[yh == 0].sum() / final_w.sum()),
            "AI_frac_final": float(final_w[yh == 1].sum() / final_w.sum()),
        },
        "locked_condition_sample_weights": COND_WEIGHT,
        "effective_clean_frac_by_sample_weight": 0.5,
        "effective_transformed_frac_by_sample_weight": 0.5,
    }
    return X, yh, sw, meta


def fit_h4(X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> tuple[LogisticRegression, dict[str, Any]]:
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    t0 = time.perf_counter()
    clf.fit(X, y, sample_weight=sw)
    elapsed = time.perf_counter() - t0
    meta = {
        "n_raw": int(len(y)),
        "n_iter": int(clf.n_iter_[0]),
        "converged": bool(clf.n_iter_[0] < 2000),
        "wall_clock_s": float(elapsed),
        "coef_l2_norm": float(np.linalg.norm(clf.coef_.ravel())),
        "intercept": float(clf.intercept_[0]),
        "n_params": int(clf.coef_.size + 1),
        "classes": [int(c) for c in clf.classes_],
    }
    return clf, meta


def decide(bits: dict[str, Any]) -> tuple[str, list[str]]:
    h4c = bits["h4_clean"]
    h2c = bits["h2_clean"]
    h4tf = bits["h4_tf"]
    phone = bits["phone_h4"]
    mllm = bits["mllm_h4"]
    fold_ok = bits["fold_benefit_folds"]
    resources = bits["resources"]
    improved_tradeoff = bits["improved_tradeoff"]

    mean_tf_auc = float(np.mean([h4tf[c]["roc_auc"] for c in TRANSFORMS]))
    worst_tf_auc = float(np.min([h4tf[c]["roc_auc"] for c in TRANSFORMS]))
    mean_tf_spec = float(np.mean([h4tf[c]["real_specificity"] for c in TRANSFORMS]))
    worst_tf_spec = float(np.min([h4tf[c]["real_specificity"] for c in TRANSFORMS]))
    mean_phone = float(np.mean([phone[c] for c in TRANSFORMS]))
    mean_mllm = float(np.mean([mllm[c] for c in TRANSFORMS]))
    jpeg_ok = h4tf["jpeg_q50"]["ai_recall"] >= 0.30
    resize_ok = h4tf["resize_112"]["real_specificity"] >= 0.80
    blur_ok = h4tf["blur_sigma2"]["real_specificity"] >= 0.80
    no_collapse = all(
        h4tf[c]["ai_recall"] >= 0.20 and h4tf[c]["real_specificity"] >= 0.70 for c in TRANSFORMS
    )

    flags = {
        "clean_auc_ge_093": h4c["roc_auc"] >= 0.93,
        "clean_auc_drop_vs_h2_le_002": (h2c["roc_auc"] - h4c["roc_auc"]) <= 0.02,
        "clean_ai_rec_ge_053": h4c["ai_recall"] >= 0.53,
        "clean_spec_ge_094": h4c["real_specificity"] >= 0.94,
        "phone_clean_ge_098": phone["CLEAN"] >= 0.98,
        "mllm_clean_ge_085": mllm["CLEAN"] >= 0.85,
        "mean_tf_auc_ge_084": mean_tf_auc >= 0.84,
        "worst_tf_auc_ge_080": worst_tf_auc >= 0.80,
        "mean_tf_spec_ge_088": mean_tf_spec >= 0.88,
        "worst_tf_spec_ge_080": worst_tf_spec >= 0.80,
        "phone_mean_tf_ge_095": mean_phone >= 0.95,
        "mllm_mean_tf_ge_078": mean_mllm >= 0.78,
        "jpeg_ai_rec_ge_030": jpeg_ok,
        "resize_spec_ge_080": resize_ok,
        "blur_spec_ge_080": blur_ok,
        "no_collapse": no_collapse,
        "fold_benefit_ge_3": fold_ok >= 3,
        "params_ok": resources["h4_params"] == 513,
    }
    evidence = [
        f"flags={flags}",
        (
            f"H4 clean AUC={h4c['roc_auc']:.4f} AI_rec={h4c['ai_recall']:.4f} "
            f"RealSpec={h4c['real_specificity']:.4f}; mean_tf_AUC={mean_tf_auc:.4f} "
            f"worst_tf_AUC={worst_tf_auc:.4f} mean_tf_spec={mean_tf_spec:.4f}"
        ),
        f"phone_mean_tf={mean_phone:.3f} mllm_mean_tf={mean_mllm:.3f} fold_benefit={fold_ok}",
    ]
    if all(flags.values()):
        return "CLEAN_ANCHORED_ROBUST_HEAD_PROMISING", evidence
    if improved_tradeoff:
        return "CLEAN_ANCHORED_ROBUST_HEAD_MIXED", evidence
    return "CLEAN_ANCHORED_ROBUST_HEAD_NOT_BETTER", evidence


def make_figures(macro: dict, trade: dict, dir_macro: dict) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    heads = ["H2", "H3", "H4"]
    x = np.arange(len(CONDITIONS))
    w = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(
        axes,
        ["roc_auc", "ai_recall", "real_specificity"],
        ["ROC-AUC", "AI recall@0.5", "Real specificity@0.5"],
    ):
        for i, h in enumerate(heads):
            ax.bar(x + (i - 1) * w, [macro[h][c][key] for c in CONDITIONS], width=w, label=h)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("V2-10H H2/H3/H4")
    fig.tight_layout()
    p = FIG / "v2_10h_h2_h3_h4_metrics_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for h, marker in [("H2", "o"), ("H3", "s"), ("H4", "D")]:
        ax.scatter(
            trade[h]["clean_auc"],
            trade[h]["mean_tf_auc"],
            s=80,
            marker=marker,
            label=h,
        )
        ax.annotate(h, (trade[h]["clean_auc"], trade[h]["mean_tf_auc"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("CLEAN AUC")
    ax.set_ylabel("mean StrongRobust AUC")
    ax.set_title("Clean vs robust trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = FIG / "v2_10h_clean_vs_robust_tradeoff_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = ["s_clean", "s_tfmean"]
    for i, h in enumerate(heads):
        vals = [dir_macro[f"cos_w_{h.lower()}_s_clean"], dir_macro[f"cos_w_{h.lower()}_s_tfmean"]]
        ax.bar(np.arange(2) + (i - 1) * 0.25, vals, width=0.25, label=h)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine")
    ax.set_title("Head alignment to clean / mean-tf separation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "v2_10h_direction_alignment_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10H — Clean-Anchored Robust Linear Head (H4)")
    lines.append(f"Status: COMPLETE — {payload['decision']}")
    lines.append("Only H4 trained. Locked sample weights CLEAN=4 / transforms=1. No weight search.")
    lines.append("FINAL_V2_MODEL NOT SELECTED. NEXT_STAGE_STARTED = NO.")
    lines.append("")
    lines.append("Weighting")
    lines.append(f"  {payload['weighting_summary']}")
    lines.append("")
    lines.append("H4 training")
    for fold, tm in payload["h4_train"].items():
        lines.append(
            f"  {fold}: raw_N={tm['n_raw']} iters={tm['n_iter']} time={tm['wall_clock_s']:.2f}s "
            f"params={tm['n_params']} ||w||={tm['coef_l2_norm']:.3f} b={tm['intercept']:+.3f}"
        )
    lines.append("")
    lines.append("CLEAN macro H2/H3/H4")
    for h in ["H2", "H3", "H4"]:
        m = payload["macro"][h]["CLEAN"]
        lines.append(
            f"  {h}: AUC={m['roc_auc']:.4f} AP={m['ap']:.4f} AI_rec={m['ai_recall']:.4f} "
            f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
        )
    lines.append(f"  H4-H2: {payload['deltas']['H4_minus_H2']['CLEAN']}")
    lines.append(f"  H4-H3: {payload['deltas']['H4_minus_H3']['CLEAN']}")
    lines.append("")
    lines.append("StrongRobust macro")
    for c in TRANSFORMS:
        for h in ["H2", "H3", "H4"]:
            m = payload["macro"][h][c]
            lines.append(
                f"  {h}/{c}: AUC={m['roc_auc']:.4f} AI_rec={m['ai_recall']:.4f} "
                f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
            )
    lines.append("")
    lines.append("Trade-off summary")
    for h, t in payload["tradeoff"].items():
        lines.append(f"  {h}: {t}")
    lines.append(f"  characterization: {payload['tradeoff_characterization']}")
    lines.append("")
    lines.append("Direction")
    for k, v in payload["direction_macro"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Seedream / FLUX")
    lines.append(f"  Seedream: {payload['seedream']}")
    lines.append(f"  FLUX: {payload['flux']}")
    lines.append("")
    lines.append("Fold4")
    lines.append(f"  {payload['fold4']}")
    lines.append("")
    lines.append("Decision evidence")
    for e in payload["decision_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append(f"Recommended development candidate: {payload['recommended_development_candidate']}")
    lines.append(f"Next-stage recommendation ONLY: {payload['next_stage_recommendation']}")
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
    if "## Stage V2-10H" in text:
        print("[V2-10H] research_log already contains V2-10H; not rewriting.", flush=True)
        return
    m = payload["macro"]
    entry = f"""
## Stage V2-10H — Clean-Anchored Robust Linear Head (H4)

**Date:** 2026-09-08  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Train only predeclared H4 on frozen LoRA R1 with locked sample weights CLEAN=4 / each transform=1 (50/50 clean vs transformed). No weight search. No feature extraction. No LoRA/CLIP/H2/H3 training. Threshold=0.5. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation:** V2-10G = ROBUST_HEAD_MIXED_LIMITATION / OPTION_A_CLEAN_ANCHORED_ROBUST_LINEAR_HEAD (H3 over-rotated; 80% transformed rows).

**NEW_FEATURE_INFERENCE_PERFORMED = NO**

**CLEAN macro:** H2 AUC={m['H2']['CLEAN']['roc_auc']:.4f}; H3={m['H3']['CLEAN']['roc_auc']:.4f}; H4={m['H4']['CLEAN']['roc_auc']:.4f}.

**Decision:** **{payload['decision']}**  
**Recommended development candidate (not FINAL):** {payload['recommended_development_candidate']}

**Outputs:** `src/run_v2_10h_clean_anchored_robust_head_v1.py`; `results/v2/v2_10h_*`; `models/v2/v2_10h_h4_*`; figures `figures/v2/v2_10h_*.png`.

**Integrity:** NTIRE=NO; fal=NO; folds unchanged; LoRA/CLIP/H2/H3 not updated; only H4 trained; no weight/HP search; no thr/calibrator/selective retune; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10H] Appended research_log.md", flush=True)


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("[V2-10H] Artifact / integrity audit ...", flush=True)
    integrity = audit()

    # Align eval labels/meta from V2-10F predictions
    pred_f = pd.read_csv(OUT / "v2_10f_robust_head_predictions_v1.csv")
    pred_f["image_id"] = pred_f["image_id"].astype(str)
    pred_f["generator_id"] = pred_f["generator_id"].fillna("").astype(str)
    pred_f["real_domain"] = pred_f["real_domain"].fillna("").astype(str)

    pred_rows = []
    metric_rows = []
    dir_rows = []
    fold_metrics: dict[int, dict] = {}
    h4_train = {}
    weighting_per_fold = {}
    boot_folds: dict[str, Any] = {}
    margin_delta_rows = []
    weights = {}

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10H] Fold {fold}: train H4 ...", flush=True)
        X, y, sw, wmeta = build_h4_matrix(fold)
        weighting_per_fold[f"fold_{fold}"] = wmeta
        h4, tm = fit_h4(X, y, sw)
        if not tm["converged"]:
            stop(f"fold {fold} H4 failed to converge (n_iter={tm['n_iter']})")
        path = MODELS / f"v2_10h_h4_clean_anchored_robust_logreg_fold{fold}_v1.joblib"
        joblib.dump(h4, path)
        tm["serialized_path"] = str(path.relative_to(PROJECT_ROOT))
        tm["serialized_bytes"] = int(path.stat().st_size)
        tm["sum_sample_weight"] = wmeta["sum_sample_weight"]
        h4_train[f"fold_{fold}"] = tm

        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        h3 = joblib.load(MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib")
        w2 = h2.coef_.ravel().astype(np.float64)
        b2 = float(h2.intercept_[0])
        w3 = h3.coef_.ravel().astype(np.float64)
        b3 = float(h3.intercept_[0])
        w4 = h4.coef_.ravel().astype(np.float64)
        b4 = float(h4.intercept_[0])
        weights[fold] = {"w2": w2, "b2": b2, "w3": w3, "b3": b3, "w4": w4, "b4": b4}

        # Separation vectors on TRAIN
        seps = {}
        for cond in CONDITIONS:
            _, Xt, yt = load_train(fold, cond)
            seps[cond] = Xt[yt == 1].mean(0).astype(np.float64) - Xt[yt == 0].mean(0).astype(np.float64)
        s_tfmean = np.mean([seps[c] for c in TRANSFORMS], axis=0)
        dir_rows.append(
            {
                "fold": fold,
                "cos_w_h4_h2": cos_vec(w4, w2),
                "cos_w_h4_h3": cos_vec(w4, w3),
                "cos_w_h2_h3": cos_vec(w2, w3),
                "angle_h4_h2": angle_deg(w4, w2),
                "angle_h4_h3": angle_deg(w4, w3),
                "cos_w_h2_s_clean": cos_vec(w2, seps["CLEAN"]),
                "cos_w_h3_s_clean": cos_vec(w3, seps["CLEAN"]),
                "cos_w_h4_s_clean": cos_vec(w4, seps["CLEAN"]),
                "cos_w_h2_s_tfmean": cos_vec(w2, s_tfmean),
                "cos_w_h3_s_tfmean": cos_vec(w3, s_tfmean),
                "cos_w_h4_s_tfmean": cos_vec(w4, s_tfmean),
                **{f"cos_w_h4_s_{c}": cos_vec(w4, seps[c]) for c in CONDITIONS},
                **{f"cos_w_h2_s_{c}": cos_vec(w2, seps[c]) for c in CONDITIONS},
                **{f"cos_w_h3_s_{c}": cos_vec(w3, seps[c]) for c in CONDITIONS},
            }
        )

        pf = pred_f[pred_f["fold"] == fold].reset_index(drop=True)
        eid, Xc = load_eval(fold, "CLEAN")
        if list(pf["image_id"]) != list(eid):
            pos = {i: j for j, i in enumerate(eid)}
            order = [pos[i] for i in pf["image_id"]]
            Xc = Xc[order]
            eid = np.asarray(list(pf["image_id"]), dtype=str)
        y_ev = pf["y"].to_numpy(dtype=int)
        gens = pf["generator_id"].to_numpy()
        domains = pf["real_domain"].to_numpy()

        cls1 = {
            "H2": list(h2.classes_).index(1),
            "H3": list(h3.classes_).index(1),
            "H4": list(h4.classes_).index(1),
        }
        models = {"H2": h2, "H3": h3, "H4": h4}
        ws = {"H2": (w2, b2), "H3": (w3, b3), "H4": (w4, b4)}

        fold_metrics[fold] = {}
        store = {}
        z_clean = {}
        for h in ["H2", "H3", "H4"]:
            store[h] = {}
            z_clean[h] = Xc.astype(np.float64) @ ws[h][0] + ws[h][1]

        for cond in CONDITIONS:
            if cond == "CLEAN":
                X = Xc
            else:
                ids_t, Xt = load_eval(fold, cond)
                if list(ids_t) != list(eid):
                    pos = {i: j for j, i in enumerate(ids_t)}
                    Xt = Xt[[pos[i] for i in eid]]
                X = Xt
            fold_metrics[fold][cond] = {}
            for h in ["H2", "H3", "H4"]:
                p = models[h].predict_proba(X)[:, cls1[h]]
                store[h][cond] = p
                m = metrics_at_05(y_ev, p)
                fold_metrics[fold][cond][h] = m
                metric_rows.append({"fold": fold, "condition": cond, "head": h, **m})
                if cond != "CLEAN":
                    z = X.astype(np.float64) @ ws[h][0] + ws[h][1]
                    dz = z - z_clean[h]
                    real = y_ev == 0
                    ai = y_ev == 1
                    margin_delta_rows.append(
                        {
                            "fold": fold,
                            "condition": cond,
                            "head": h,
                            "delta_z_real": float(dz[real].mean()),
                            "delta_z_ai": float(dz[ai].mean()),
                            "delta_z_overall": float(dz.mean()),
                        }
                    )

        # predictions row
        for j in range(len(eid)):
            row = {
                "fold": fold,
                "image_id": eid[j],
                "y": int(y_ev[j]),
                "generator_id": gens[j],
                "real_domain": domains[j],
            }
            for cond in CONDITIONS:
                for h in ["H2", "H3", "H4"]:
                    row[f"p_{h}_{cond}"] = float(store[h][cond][j])
            pred_rows.append(row)

        # bootstrap H4-H2 and H4-H3
        boot_folds[f"fold_{fold}"] = {}
        for cond in CONDITIONS:
            boot_folds[f"fold_{fold}"][cond] = {
                "H4_minus_H2": paired_boot(y_ev, store["H4"][cond], store["H2"][cond]),
                "H4_minus_H3": paired_boot(y_ev, store["H4"][cond], store["H3"][cond]),
            }

        print(
            f"[V2-10H] Fold {fold} CLEAN AUC H2/H3/H4="
            f"{fold_metrics[fold]['CLEAN']['H2']['roc_auc']:.4f}/"
            f"{fold_metrics[fold]['CLEAN']['H3']['roc_auc']:.4f}/"
            f"{fold_metrics[fold]['CLEAN']['H4']['roc_auc']:.4f}",
            flush=True,
        )

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_OUT, index=False)
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(DIR_CSV, index=False)

    # Macro
    macro = {h: {} for h in ["H2", "H3", "H4"]}
    for cond in CONDITIONS:
        for h in ["H2", "H3", "H4"]:
            keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]
            macro[h][cond] = {
                k: float(np.mean([fold_metrics[f][cond][h][k] for f in [1, 2, 3, 4]])) for k in keys
            }
            metric_rows.append({"fold": "macro", "condition": cond, "head": h, **macro[h][cond]})
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)

    def delta_block(a: str, b: str) -> dict:
        out = {}
        for cond in CONDITIONS:
            out[cond] = {
                k: macro[a][cond][k] - macro[b][cond][k]
                for k in macro[a][cond]
            }
        return out

    deltas = {"H4_minus_H2": delta_block("H4", "H2"), "H4_minus_H3": delta_block("H4", "H3")}

    tradeoff = {}
    for h in ["H2", "H3", "H4"]:
        tradeoff[h] = {
            "clean_auc": macro[h]["CLEAN"]["roc_auc"],
            "clean_ap": macro[h]["CLEAN"]["ap"],
            "clean_ai_recall": macro[h]["CLEAN"]["ai_recall"],
            "clean_real_spec": macro[h]["CLEAN"]["real_specificity"],
            "clean_balacc": macro[h]["CLEAN"]["balanced_accuracy"],
            "mean_tf_auc": float(np.mean([macro[h][c]["roc_auc"] for c in TRANSFORMS])),
            "worst_tf_auc": float(np.min([macro[h][c]["roc_auc"] for c in TRANSFORMS])),
            "mean_tf_ap": float(np.mean([macro[h][c]["ap"] for c in TRANSFORMS])),
            "mean_tf_ai_recall": float(np.mean([macro[h][c]["ai_recall"] for c in TRANSFORMS])),
            "mean_tf_real_spec": float(np.mean([macro[h][c]["real_specificity"] for c in TRANSFORMS])),
            "worst_tf_real_spec": float(np.min([macro[h][c]["real_specificity"] for c in TRANSFORMS])),
            "mean_tf_balacc": float(np.mean([macro[h][c]["balanced_accuracy"] for c in TRANSFORMS])),
            "worst_tf_balacc": float(np.min([macro[h][c]["balanced_accuracy"] for c in TRANSFORMS])),
        }

    # Characterize Pareto position
    c_auc = [tradeoff[h]["clean_auc"] for h in ["H2", "H3", "H4"]]
    r_auc = [tradeoff[h]["mean_tf_auc"] for h in ["H2", "H3", "H4"]]
    # H4 intermediate if between H2 and H3 on both axes (H2 clean high, H3 robust high)
    if (
        min(tradeoff["H3"]["clean_auc"], tradeoff["H2"]["clean_auc"])
        <= tradeoff["H4"]["clean_auc"]
        <= max(tradeoff["H3"]["clean_auc"], tradeoff["H2"]["clean_auc"])
        and min(tradeoff["H3"]["mean_tf_auc"], tradeoff["H2"]["mean_tf_auc"])
        <= tradeoff["H4"]["mean_tf_auc"]
        <= max(tradeoff["H3"]["mean_tf_auc"], tradeoff["H2"]["mean_tf_auc"])
    ):
        trade_char = "genuine_intermediate_Pareto_compromise"
    elif abs(tradeoff["H4"]["clean_auc"] - tradeoff["H2"]["clean_auc"]) < abs(
        tradeoff["H4"]["clean_auc"] - tradeoff["H3"]["clean_auc"]
    ):
        trade_char = "closer_to_H2_clean_behaviour"
    else:
        trade_char = "closer_to_H3_robust_behaviour"

    # Direction macro
    direction_macro = {
        "mean_cos_w_h4_h2": float(dir_df["cos_w_h4_h2"].mean()),
        "mean_cos_w_h4_h3": float(dir_df["cos_w_h4_h3"].mean()),
        "mean_angle_h4_h2": float(dir_df["angle_h4_h2"].mean()),
        "mean_angle_h4_h3": float(dir_df["angle_h4_h3"].mean()),
        "cos_w_h2_s_clean": float(dir_df["cos_w_h2_s_clean"].mean()),
        "cos_w_h3_s_clean": float(dir_df["cos_w_h3_s_clean"].mean()),
        "cos_w_h4_s_clean": float(dir_df["cos_w_h4_s_clean"].mean()),
        "cos_w_h2_s_tfmean": float(dir_df["cos_w_h2_s_tfmean"].mean()),
        "cos_w_h3_s_tfmean": float(dir_df["cos_w_h3_s_tfmean"].mean()),
        "cos_w_h4_s_tfmean": float(dir_df["cos_w_h4_s_tfmean"].mean()),
        **{
            f"cos_w_h4_s_{c}": float(dir_df[f"cos_w_h4_s_{c}"].mean())
            for c in CONDITIONS
        },
    }

    # Margin drift macro
    md = pd.DataFrame(margin_delta_rows)
    margin_macro = {}
    for c in TRANSFORMS:
        margin_macro[c] = {}
        for h in ["H2", "H3", "H4"]:
            sub = md[(md["condition"] == c) & (md["head"] == h)]
            margin_macro[c][h] = {
                "delta_z_real": float(sub["delta_z_real"].mean()),
                "delta_z_ai": float(sub["delta_z_ai"].mean()),
            }

    # Real domains / phone / mllm
    def domain_spec(head: str, cond: str, domain: str) -> float:
        col = f"p_{head}_{cond}"
        specs = []
        for fold in [1, 2, 3, 4]:
            sub = pred_df[
                (pred_df["fold"] == fold)
                & (pred_df["y"] == 0)
                & (pred_df["real_domain"] == domain)
            ]
            if len(sub) == 0:
                continue
            specs.append(float((sub[col].to_numpy() < THR).mean()))
        return float(np.mean(specs)) if specs else float("nan")

    real_domains = []
    phone_h4 = {}
    mllm_h4 = {}
    for dom in REAL_DOMAINS:
        row = {"real_domain": dom}
        for cond in CONDITIONS:
            for h in ["H2", "H3", "H4"]:
                col = f"p_{h}_{cond}"
                specs, means, meds = [], [], []
                for fold in [1, 2, 3, 4]:
                    sub = pred_df[
                        (pred_df["fold"] == fold)
                        & (pred_df["y"] == 0)
                        & (pred_df["real_domain"] == dom)
                    ]
                    if len(sub) == 0:
                        continue
                    p = sub[col].to_numpy(dtype=float)
                    specs.append(float((p < THR).mean()))
                    means.append(float(np.mean(p)))
                    meds.append(float(np.median(p)))
                val = float(np.mean(specs)) if specs else float("nan")
                row[f"{h}_{cond}_specificity"] = val
                row[f"{h}_{cond}_fpr"] = float(1.0 - val) if not np.isnan(val) else float("nan")
                row[f"{h}_{cond}_mean_p"] = float(np.mean(means)) if means else float("nan")
                row[f"{h}_{cond}_median_p"] = float(np.mean(meds)) if meds else float("nan")
                if h == "H4":
                    if dom == "Smartphone":
                        phone_h4[cond] = val
                    if dom == "MLLM":
                        mllm_h4[cond] = val
        real_domains.append(row)

    fold3_mllm = {}
    sub = pred_df[(pred_df["fold"] == 3) & (pred_df["y"] == 0) & (pred_df["real_domain"] == "MLLM")]
    if len(sub):
        fold3_mllm["n"] = int(len(sub))
        for cond in CONDITIONS:
            for h in ["H2", "H3", "H4"]:
                col = f"p_{h}_{cond}"
                fold3_mllm[f"{h}_{cond}"] = float((sub[col].to_numpy() < THR).mean())

    # Hard generators
    hard_out = []
    for gid in HARD:
        row = {"generator_id": gid, "n": int(((pred_df["generator_id"] == gid) & (pred_df["fold"].isin([1, 2, 3, 4]))).sum())}
        # n is sum over folds; each fold has its held-out count
        ns = []
        for fold in [1, 2, 3, 4]:
            ns.append(int(((pred_df["fold"] == fold) & (pred_df["generator_id"] == gid)).sum()))
        row["n"] = int(sum(ns))
        for cond in CONDITIONS:
            for h in ["H2", "H3", "H4"]:
                col = f"p_{h}_{cond}"
                recalls, means, meds = [], [], []
                for fold in [1, 2, 3, 4]:
                    sub = pred_df[(pred_df["fold"] == fold) & (pred_df["generator_id"] == gid)]
                    if len(sub) == 0:
                        continue
                    p = sub[col].to_numpy(dtype=float)
                    recalls.append(float((p >= THR).mean()))
                    means.append(float(np.mean(p)))
                    meds.append(float(np.median(p)))
                row[f"{h}_{cond}_recall"] = float(np.mean(recalls)) if recalls else float("nan")
                row[f"{h}_{cond}_mean_p"] = float(np.mean(means)) if means else float("nan")
                row[f"{h}_{cond}_median_p"] = float(np.mean(meds)) if meds else float("nan")
        hard_out.append(row)

    seedream = next(h for h in hard_out if h["generator_id"] == SEEDREAM)
    flux = next(h for h in hard_out if h["generator_id"] == FLUX)

    # Fold4
    fold4 = {"CLEAN": {}, "transforms": {}}
    for h in ["H2", "H3", "H4"]:
        fold4["CLEAN"][h] = fold_metrics[4]["CLEAN"][h]
        fold4["transforms"][h] = {c: fold_metrics[4][c][h] for c in TRANSFORMS}
    # Fold4 generator-level clean recall for hard gens present
    fold4_gens = []
    pf4 = pred_df[pred_df["fold"] == 4]
    for gid in sorted(pf4.loc[pf4["y"] == 1, "generator_id"].unique()):
        sub = pf4[pf4["generator_id"] == gid]
        fold4_gens.append(
            {
                "generator_id": gid,
                "n": int(len(sub)),
                **{
                    f"{h}_CLEAN_recall": float((sub[f"p_{h}_CLEAN"].to_numpy() >= THR).mean())
                    for h in ["H2", "H3", "H4"]
                },
            }
        )
    fold4["generator_clean_recall"] = fold4_gens

    # Fold benefit: H4 improves clean OR (tf balacc vs H2) without catastrophic clean loss vs H3
    fold_benefit = 0
    for fold in [1, 2, 3, 4]:
        clean_better_than_h3 = (
            fold_metrics[fold]["CLEAN"]["H4"]["roc_auc"] > fold_metrics[fold]["CLEAN"]["H3"]["roc_auc"]
        )
        tf_better_than_h2 = (
            np.mean([fold_metrics[fold][c]["H4"]["balanced_accuracy"] for c in TRANSFORMS])
            > np.mean([fold_metrics[fold][c]["H2"]["balanced_accuracy"] for c in TRANSFORMS])
        )
        if clean_better_than_h3 and tf_better_than_h2:
            fold_benefit += 1

    resources = {
        "h2_params": 513,
        "h3_params": 513,
        "h4_params": 513,
        "h4_mean_serialized": float(np.mean([h4_train[f"fold_{f}"]["serialized_bytes"] for f in [1, 2, 3, 4]])),
        "h4_mean_train_s": float(np.mean([h4_train[f"fold_{f}"]["wall_clock_s"] for f in [1, 2, 3, 4]])),
        "h4_mean_sum_sample_weight": float(
            np.mean([weighting_per_fold[f"fold_{f}"]["sum_sample_weight"] for f in [1, 2, 3, 4]])
        ),
        "note": "Clean anchoring changes offline sample_weight only; deployment head remains ~513 params.",
    }
    pd.DataFrame(
        [
            {"head": "H2", "params": 513},
            {"head": "H3", "params": 513},
            {
                "head": "H4",
                "params": 513,
                "mean_serialized_bytes": resources["h4_mean_serialized"],
                "mean_train_s": resources["h4_mean_train_s"],
                "mean_sum_sample_weight": resources["h4_mean_sum_sample_weight"],
            },
        ]
    ).to_csv(RESOURCE_OUT, index=False)

    # Improved tradeoff heuristic for MIXED
    improved = (
        tradeoff["H4"]["clean_auc"] >= tradeoff["H3"]["clean_auc"] + 0.01
        and tradeoff["H4"]["mean_tf_auc"] >= tradeoff["H2"]["mean_tf_auc"] + 0.04
        and tradeoff["H4"]["mean_tf_real_spec"] >= 0.80
    ) or (
        tradeoff["H4"]["clean_auc"] >= 0.92
        and tradeoff["H4"]["mean_tf_auc"] >= 0.80
        and macro["H4"]["jpeg_q50"]["ai_recall"] >= 0.25
        and macro["H4"]["resize_112"]["real_specificity"] >= 0.75
    )

    decision, evidence = decide(
        {
            "h4_clean": macro["H4"]["CLEAN"],
            "h2_clean": macro["H2"]["CLEAN"],
            "h4_tf": {c: macro["H4"][c] for c in TRANSFORMS},
            "phone_h4": phone_h4,
            "mllm_h4": mllm_h4,
            "fold_benefit_folds": fold_benefit,
            "resources": resources,
            "improved_tradeoff": improved,
        }
    )

    # Recommended candidate
    if decision == "CLEAN_ANCHORED_ROBUST_HEAD_PROMISING":
        rec_cand = "H4"
        next_rec = (
            "Authorize V2 development freeze packaging around H4 as primary clean/robust compromise "
            "(still not FINAL_V2 without tutor approval). Do not run another weight experiment. "
            "Do not auto-start. Do not access NTIRE."
        )
    elif decision == "CLEAN_ANCHORED_ROBUST_HEAD_MIXED":
        # pick best compromise descriptively
        if tradeoff["H4"]["clean_auc"] > tradeoff["H3"]["clean_auc"] and tradeoff["H4"]["mean_tf_auc"] > tradeoff["H2"]["mean_tf_auc"]:
            rec_cand = "H4 (with documented residual gates)"
        else:
            rec_cand = "H3 or H4 depending on clean-vs-robust priority (tutor chooses); default lean H4 if clean recovery dominates"
        next_rec = (
            "Do NOT search another condition weight. Prefer development freeze-with-limitations around the "
            "best clean/robust compromise among H2/H3/H4, or a non-weight residual diagnostic. Do not auto-start."
        )
    else:
        rec_cand = "H2 (clean) or H3 (robust) — H4 not a better compromise"
        next_rec = (
            "H4 did not improve the trade-off; do not grid weights. Tutor may freeze with limitations "
            "on H2/H3 or authorize a different tightly scoped non-weight intervention. Do not auto-start."
        )

    interpretation = {
        "q1_recover_clean_auc": (
            f"H3={macro['H3']['CLEAN']['roc_auc']:.4f} → H4={macro['H4']['CLEAN']['roc_auc']:.4f} "
            f"(Δ vs H3={deltas['H4_minus_H3']['CLEAN']['roc_auc']:+.4f})."
        ),
        "q2_retain_h2_clean": (
            f"H4 retains H2 clean AUC fraction; Δ vs H2={deltas['H4_minus_H2']['CLEAN']['roc_auc']:+.4f} "
            f"(H2={macro['H2']['CLEAN']['roc_auc']:.4f})."
        ),
        "q3_retain_h3_robust": (
            f"H4 mean_tf_AUC={tradeoff['H4']['mean_tf_auc']:.4f} vs H3={tradeoff['H3']['mean_tf_auc']:.4f} "
            f"H2={tradeoff['H2']['mean_tf_auc']:.4f}."
        ),
        "q4_jpeg": f"JPEG AI_rec H2/H3/H4={macro['H2']['jpeg_q50']['ai_recall']:.3f}/{macro['H3']['jpeg_q50']['ai_recall']:.3f}/{macro['H4']['jpeg_q50']['ai_recall']:.3f}.",
        "q5_resize": f"resize RealSpec H2/H3/H4={macro['H2']['resize_112']['real_specificity']:.3f}/{macro['H3']['resize_112']['real_specificity']:.3f}/{macro['H4']['resize_112']['real_specificity']:.3f}.",
        "q6_blur": f"blur RealSpec H2/H3/H4={macro['H2']['blur_sigma2']['real_specificity']:.3f}/{macro['H3']['blur_sigma2']['real_specificity']:.3f}/{macro['H4']['blur_sigma2']['real_specificity']:.3f}.",
        "q7_screenshot": f"screenshot AUC H2/H3/H4={macro['H2']['screenshot_strong']['roc_auc']:.3f}/{macro['H3']['screenshot_strong']['roc_auc']:.3f}/{macro['H4']['screenshot_strong']['roc_auc']:.3f}.",
        "q8_seedream": (
            f"CLEAN recall H2/H3/H4={seedream['H2_CLEAN_recall']:.3f}/{seedream['H3_CLEAN_recall']:.3f}/{seedream['H4_CLEAN_recall']:.3f}."
        ),
        "q9_flux": (
            f"CLEAN recall H2/H3/H4={flux['H2_CLEAN_recall']:.3f}/{flux['H3_CLEAN_recall']:.3f}/{flux['H4_CLEAN_recall']:.3f}."
        ),
        "q10_fold4": (
            f"Fold4 CLEAN AUC H2/H3/H4="
            f"{fold_metrics[4]['CLEAN']['H2']['roc_auc']:.3f}/"
            f"{fold_metrics[4]['CLEAN']['H3']['roc_auc']:.3f}/"
            f"{fold_metrics[4]['CLEAN']['H4']['roc_auc']:.3f}."
        ),
        "q11_direction_between": (
            f"cos(w_H4,w_H2)={direction_macro['mean_cos_w_h4_h2']:.3f}, "
            f"cos(w_H4,w_H3)={direction_macro['mean_cos_w_h4_h3']:.3f}; "
            f"cos(w_H4,s_clean)={direction_macro['cos_w_h4_s_clean']:.3f} "
            f"(H2={direction_macro['cos_w_h2_s_clean']:.3f}, H3={direction_macro['cos_w_h3_s_clean']:.3f})."
        ),
        "q12_validates_overrotation": (
            "Yes if H4 restores clean alignment toward H2 while retaining more transform robustness than H2."
        ),
        "q13_linear_sufficient": f"decision={decision}; trade_char={trade_char}.",
        "q14_strongest_compromise": rec_cand,
        "q15_another_weight_experiment": "NO — V2-10H was the single predeclared clean-anchor test; do not grid weights.",
        "q16_freeze_recommendation": (
            f"Recommend development candidate {rec_cand} for freeze-with-limitations consideration; "
            "do NOT formally select FINAL_V2_MODEL in this stage."
        ),
        "q17_next": next_rec,
    }

    figures = make_figures(macro, tradeoff, direction_macro)

    boot_payload = {
        "protocol": "fold_wise_paired_stratified_bootstrap_5000_seed42",
        "comparisons": ["H4_minus_H2", "H4_minus_H3"],
        "pooled_forbidden": True,
        "folds": boot_folds,
        "macro_deltas": deltas,
    }
    BOOT_OUT.write_text(json.dumps(_clean(boot_payload), indent=2))

    weighting_summary = (
        "Locked sample_weight CLEAN=4.0, each StrongRobust transform=1.0 on identical 5× TRAIN rows; "
        "effective sample-weight mass 50% clean / 50% transformed. "
        "sklearn class_weight='balanced' uses raw row class counts then multiplies by sample_weight."
    )

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
        "H4_HEAD_TRAINING_PERFORMED": "YES",
        "ONLY_PREDECLARED_H4_TRAINED": "YES",
        "CONDITION_WEIGHT_SEARCH_PERFORMED": "NO",
        "HYPERPARAMETER_SEARCH_PERFORMED": "NO",
        "NEW_FEATURE_INFERENCE_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "EVAL_LABELS_USED_FOR_TRAINING": "NO",
        "V2_9C_CALIBRATION_DATA_USED_FOR_TRAINING": "NO",
        "PROMPT_BLOCKED_USED_FOR_TRAINING": "NO",
        "THRESHOLD_CHANGED": "NO",
        "THRESHOLD_SWEEP_PERFORMED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
    }

    payload = {
        "stage": "V2-10H",
        "status": "COMPLETE",
        "decision": decision,
        "decision_evidence": evidence,
        "integrity": integrity,
        "weighting_summary": weighting_summary,
        "weighting_per_fold": weighting_per_fold,
        "h4_train": h4_train,
        "macro": macro,
        "deltas": deltas,
        "tradeoff": tradeoff,
        "tradeoff_characterization": trade_char,
        "direction_macro": direction_macro,
        "margin_macro": margin_macro,
        "hard_generators": hard_out,
        "seedream": {
            "clean_recall_H2": seedream["H2_CLEAN_recall"],
            "clean_recall_H3": seedream["H3_CLEAN_recall"],
            "clean_recall_H4": seedream["H4_CLEAN_recall"],
            "detail": seedream,
        },
        "flux": {
            "clean_recall_H2": flux["H2_CLEAN_recall"],
            "clean_recall_H3": flux["H3_CLEAN_recall"],
            "clean_recall_H4": flux["H4_CLEAN_recall"],
            "detail": flux,
        },
        "real_domains": real_domains,
        "fold3_mllm": fold3_mllm,
        "fold4": fold4,
        "fold_benefit_folds": fold_benefit,
        "bootstrap": {"macro_deltas": deltas, "protocol": boot_payload["protocol"]},
        "resources": resources,
        "figures": figures,
        "recommended_development_candidate": rec_cand,
        "next_stage_recommendation": next_rec,
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


if __name__ == "__main__":
    main()
