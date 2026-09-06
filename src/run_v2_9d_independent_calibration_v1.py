#!/usr/bin/env python3
"""V2-9D — Independent Post-Hoc Calibration Evaluation.

Fit C1 (temperature) and C2 (affine/Platt) on V2-9C independent calibration
scores only; apply to untouched V2-8 evaluation predictions.
No detector training, no eval re-inference, no threshold adoption.
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
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rq5_calibration_utils_v1 import (
    apply_temperature,
    compute_brier,
    compute_ece15,
    compute_nll,
    fit_temperature,
    reliability_curve,
    safe_ap,
    safe_auc,
    sigmoid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"

CALIB_DIR = OUT
EVAL_DIR = OUT / "kaggle_v2_lora_v37/v2_lora_outputs/predictions"

EPS = 1e-6
SEED = 42
LOCKED_T = 0.5
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
EXPECTED_CALIB = {1: 2123, 2: 2323, 3: 2223, 4: 2223}
EXPECTED_EVAL = {1: 2821, 2: 2619, 3: 1994, 4: 1709}

JSON_OUT = OUT / "v2_9d_calibration_analysis_v1.json"
REPORT_OUT = OUT / "v2_9d_calibration_report_v1.txt"
PRED_OUT = OUT / "v2_9d_calibrated_eval_predictions_v1.csv"
PARAMS_OUT = OUT / "v2_9d_calibrator_params_v1.csv"


def to_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def from_logit(z: np.ndarray) -> np.ndarray:
    return sigmoid(np.asarray(z, dtype=np.float64))


def operating_metrics(y: np.ndarray, p: np.ndarray, thr: float = LOCKED_T) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "ai_recall": float(recall_score(y, pred, zero_division=0)),
        "real_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "real_fpr": float(fp / (tn + fp)) if (tn + fp) else float("nan"),
    }


def full_metrics(y: np.ndarray, p: np.ndarray, z: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if z is None:
        z = to_logit(p)
    m = {
        "nll": compute_nll(z, y),
        "brier": compute_brier(p, y),
        "ece15": compute_ece15(p, y),
        "roc_auc": safe_auc(y, p),
        "ap": safe_ap(y, p),
    }
    m.update(operating_metrics(y, p))
    return m


def ece_bin_counts(probs: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, 16)
    rows = []
    n = len(labels)
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < 14 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "frac": float(count / n) if n else 0.0,
                "mean_p": float(probs[mask].mean()) if count else float("nan"),
                "emp_freq": float(labels[mask].mean()) if count else float("nan"),
            }
        )
    return rows


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    """Fit z_cal = a*z + b by binary NLL (sklearn LogisticRegression, no C-grid)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    # Very weak regularisation numerically for stability; not a C-grid search.
    # Use C large enough to approximate unregularised MLE (Platt).
    clf = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
        random_state=SEED,
    )
    clf.fit(z, y)
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    meta = {
        "solver": "sklearn_LogisticRegression_lbfgs",
        "C": 1e12,
        "converged": bool(clf.n_iter_[0] < 1000),
        "n_iter": int(clf.n_iter_[0]),
        "a_nonpositive_flag": bool(a <= 0),
    }
    return a, b, meta


def apply_platt(z: np.ndarray, a: float, b: float) -> np.ndarray:
    return from_logit(a * np.asarray(z, dtype=np.float64) + b)


def load_and_audit() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[str, Any]]:
    calibs: dict[int, pd.DataFrame] = {}
    evals: dict[int, pd.DataFrame] = {}
    report: dict[str, Any] = {"folds": {}, "ok": True}

    for fold in [1, 2, 3, 4]:
        cpath = CALIB_DIR / f"v2_9c_calibration_scores_fold{fold}_v1.csv"
        epath = EVAL_DIR / f"fold{fold}_predictions_v1.csv"
        c = pd.read_csv(cpath)
        e = pd.read_csv(epath)
        c["image_id"] = c["image_id"].astype(str)
        e["image_id"] = e["image_id"].astype(str)
        c["label"] = c["label"].astype(int)
        e["y"] = e["y"].astype(int)
        c["p_ai"] = c["p_ai"].astype(float)
        e["p_lora"] = e["p_lora"].astype(float)

        issues = []
        if c["image_id"].duplicated().any():
            issues.append("calib_duplicate_ids")
        if e["image_id"].duplicated().any():
            issues.append("eval_duplicate_ids")
        overlap = set(c["image_id"]) & set(e["image_id"])
        if overlap:
            issues.append(f"calib_eval_overlap_n={len(overlap)}")
        if len(c) != EXPECTED_CALIB[fold]:
            issues.append(f"calib_n={len(c)} expected={EXPECTED_CALIB[fold]}")
        if len(e) != EXPECTED_EVAL[fold]:
            issues.append(f"eval_n={len(e)} expected={EXPECTED_EVAL[fold]}")
        if not np.isfinite(c["p_ai"]).all() or not ((c["p_ai"] >= 0) & (c["p_ai"] <= 1)).all():
            issues.append("calib_probs_invalid")
        if not np.isfinite(e["p_lora"]).all() or not ((e["p_lora"] >= 0) & (e["p_lora"] <= 1)).all():
            issues.append("eval_probs_invalid")
        if set(c["label"].unique()) != {0, 1}:
            issues.append("calib_labels_not_binary_both_classes")
        if int(c["fold_model"].nunique()) != 1 or int(c["fold_model"].iloc[0]) != fold:
            issues.append("calib_fold_mismatch")
        if int(e["fold"].nunique()) != 1 or int(e["fold"].iloc[0]) != fold:
            issues.append("eval_fold_mismatch")
        roles = sorted(c["original_role"].unique().tolist())
        if roles != ["PROMPT_BLOCKED", "REAL_INTERNAL_HOLDOUT"]:
            issues.append(f"unexpected_calib_roles={roles}")

        fr = {
            "calib_n": int(len(c)),
            "calib_real": int((c["label"] == 0).sum()),
            "calib_ai": int((c["label"] == 1).sum()),
            "eval_n": int(len(e)),
            "eval_real": int((e["y"] == 0).sum()),
            "eval_ai": int((e["y"] == 1).sum()),
            "overlap": int(len(overlap)),
            "roles": {str(k): int(v) for k, v in c["original_role"].value_counts().items()},
            "issues": issues,
            "ok": len(issues) == 0,
        }
        report["folds"][f"fold_{fold}"] = fr
        report["ok"] = report["ok"] and fr["ok"]
        calibs[fold] = c
        evals[fold] = e

    if not report["ok"]:
        raise SystemExit(f"STOP: input integrity failed: {json.dumps(report, indent=2)}")
    return calibs, evals, report


def generator_metrics(df: pd.DataFrame, p_col: str, gid: str, label_col: str = "label") -> dict[str, Any]:
    """AI generator vs fold Real under given probability column."""
    real = df[df[label_col] == 0]
    ai = df[df["generator_id"] == gid]
    if len(ai) == 0:
        return {"n": 0}
    y = np.concatenate([real[label_col].to_numpy(), np.ones(len(ai), dtype=int)])
    p = np.concatenate([real[p_col].to_numpy(), ai[p_col].to_numpy()])
    return {
        "n": int(len(ai)),
        "mean_p": float(ai[p_col].mean()),
        "median_p": float(ai[p_col].median()),
        "recall_050": float((ai[p_col] >= LOCKED_T).mean()),
        "auc_vs_fold_real": safe_auc(y, p),
        "ap_vs_fold_real": safe_ap(y, p),
    }


def real_domain_metrics(
    df: pd.DataFrame, p_col: str, domain: str, label_col: str = "label"
) -> dict[str, Any]:
    sub = df[(df[label_col] == 0) & (df["real_domain"] == domain)]
    if len(sub) == 0:
        return {"n": 0}
    return {
        "n": int(len(sub)),
        "specificity_050": float((sub[p_col] < LOCKED_T).mean()),
        "mean_p": float(sub[p_col].mean()),
        "median_p": float(sub[p_col].median()),
        "frac_ge_050": float((sub[p_col] >= LOCKED_T).mean()),
    }


def decide(eval_summary: dict[str, Any], hard: list[dict], real_dom: dict) -> str:
    """Apply predeclared decision gate; do not alter after seeing results."""
    c0 = eval_summary["macro_mean"]["C0"]
    c1 = eval_summary["macro_mean"]["C1"]
    c2 = eval_summary["macro_mean"]["C2"]

    # Prefer C2 for PROMISING; C1 expected not to move 0.5 boundary much
    cal_improve = (
        c2["nll"] < c0["nll"] - 1e-4
        and c2["brier"] < c0["brier"] - 1e-4
        and c2["ece15"] < c0["ece15"] - 1e-4
    )
    recall_gain = c2["ai_recall"] - c0["ai_recall"]
    recall_substantial = recall_gain >= 0.05
    auc_preserved = c2["roc_auc"] >= c0["roc_auc"] - 0.01
    ap_preserved = c2["ap"] >= c0["ap"] - 0.01
    real_ok = c2["real_specificity"] >= 0.90
    phone_ok = True
    mllm_ok = True
    for fold in [1, 2, 3, 4]:
        phone = real_dom["by_fold"][f"fold_{fold}"]["C2"].get("Smartphone", {})
        mllm = real_dom["by_fold"][f"fold_{fold}"]["C2"].get("MLLM", {})
        if phone.get("n", 0) and phone.get("specificity_050", 1) < 0.95:
            phone_ok = False
        if mllm.get("n", 0) and mllm.get("specificity_050", 1) < 0.70:
            mllm_ok = False
    # multi-fold recall improvement
    folds_improved = sum(
        1
        for fold in [1, 2, 3, 4]
        if eval_summary["by_fold"][f"fold_{fold}"]["C2"]["ai_recall"]
        > eval_summary["by_fold"][f"fold_{fold}"]["C0"]["ai_recall"] + 0.02
    )
    multi_fold = folds_improved >= 2

    severe_real_damage = c2["real_specificity"] < 0.85 or (not phone_ok)

    if (
        cal_improve
        and recall_substantial
        and auc_preserved
        and ap_preserved
        and real_ok
        and phone_ok
        and mllm_ok
        and multi_fold
        and not severe_real_damage
    ):
        return "CALIBRATION_PROMISING"

    if severe_real_damage and recall_gain > 0:
        return "CALIBRATION_NOT_HELPFUL"

    if cal_improve or recall_gain > 0.02:
        # improvements but trade-offs / limitations remain
        return "CALIBRATION_MIXED"

    if (not cal_improve) and recall_gain < 0.02:
        return "CALIBRATION_NOT_HELPFUL"

    return "CALIBRATION_MIXED"


def make_figures(
    eval_all: pd.DataFrame,
    eval_summary: dict[str, Any],
    hard_rows: list[dict],
) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # Reliability diagrams pooled
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, col, title in [
        (axes[0], "p_uncalibrated", "C0 uncalibrated"),
        (axes[1], "p_temperature", "C1 temperature"),
        (axes[2], "p_platt", "C2 Platt"),
    ]:
        p = eval_all[col].to_numpy()
        y = eval_all["label"].to_numpy()
        centers, accs = reliability_curve(p, y)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(centers, accs, "o-", color="#d62728")
        ax.set_title(f"{title}\nECE={compute_ece15(p, y):.3f}")
        ax.set_xlabel("Mean predicted p")
        ax.set_ylabel("Empirical AI frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("V2-9D pooled evaluation reliability (15 equal-width bins)")
    fig.tight_layout()
    pth = FIG / "v2_9d_reliability_pooled_v1.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    # Per-fold reliability for C0 vs C2
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    for i, fold in enumerate([1, 2, 3, 4]):
        sub = eval_all[eval_all["fold"] == fold]
        for row, col, title in [
            (0, "p_uncalibrated", "C0"),
            (1, "p_platt", "C2"),
        ]:
            ax = axes[row, i]
            p = sub[col].to_numpy()
            y = sub["label"].to_numpy()
            centers, accs = reliability_curve(p, y)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.plot(centers, accs, "o-", ms=3)
            ax.set_title(f"Fold {fold} {title}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    fig.suptitle("V2-9D per-fold reliability C0 vs C2")
    fig.tight_layout()
    pth = FIG / "v2_9d_reliability_by_fold_c0_c2_v1.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    # Operating point bars
    methods = ["C0", "C1", "C2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(3)
    axes[0].bar(
        x,
        [eval_summary["macro_mean"][m]["ai_recall"] for m in methods],
        color=["#7f7f7f", "#1f77b4", "#d62728"],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Macro-mean AI recall @0.5")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(
        x,
        [eval_summary["macro_mean"][m]["real_specificity"] for m in methods],
        color=["#7f7f7f", "#1f77b4", "#d62728"],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylim(0.7, 1.0)
    axes[1].set_title("Macro-mean Real specificity @0.5")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("V2-9D locked operating point (evaluation)")
    fig.tight_layout()
    pth = FIG / "v2_9d_operating_point_recall_spec_v1.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    # Hard generator recall C0 vs C2
    # Aggregate means across folds present
    from collections import defaultdict

    agg = defaultdict(lambda: {"C0": [], "C2": []})
    for r in hard_rows:
        if r["n"] == 0:
            continue
        agg[r["generator_id"]]["C0"].append(r["C0"]["recall_050"])
        agg[r["generator_id"]]["C2"].append(r["C2"]["recall_050"])
    gids = [g for g in HARD_GENERATORS if g in agg]
    if gids:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(gids))
        w = 0.35
        ax.bar(x - w / 2, [float(np.mean(agg[g]["C0"])) for g in gids], w, label="C0")
        ax.bar(x + w / 2, [float(np.mean(agg[g]["C2"])) for g in gids], w, label="C2")
        ax.set_xticks(x)
        ax.set_xticklabels([g.split("::", 1)[-1] for g in gids], rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AI recall @0.5")
        ax.set_title("V2-9D hard-generator recall: C0 vs C2 Platt")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        pth = FIG / "v2_9d_operating_point_hard_generator_recall_v1.png"
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        created.append(str(pth.relative_to(PROJECT_ROOT)))

    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-9D — Independent Post-Hoc Calibration Evaluation")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Detector training: NO. Eval re-inference: NO. Threshold adoption: NO.")
    lines.append("")
    lines.append("Design (from V2-9C)")
    lines.append("  Per Fold F: fit on REAL_INTERNAL_HOLDOUT + Fold-F PROMPT_BLOCKED scores")
    lines.append("  produced by Fold-F frozen LoRA checkpoint; apply to untouched eval preds.")
    lines.append("  Limitations preserved: fold-specific AI; PB≠hard holdout; IH role reassignment;")
    lines.append("  no shared unused AI; CAP leftovers excluded.")
    lines.append("")
    lines.append("Input integrity")
    for f, fr in payload["input_integrity"]["folds"].items():
        lines.append(
            f"  {f}: calib={fr['calib_n']} (R{fr['calib_real']}/A{fr['calib_ai']}) "
            f"eval={fr['eval_n']} overlap={fr['overlap']} ok={fr['ok']}"
        )
    lines.append("")
    lines.append("Fitted parameters")
    for row in payload["fitted_parameters"]:
        lines.append(
            f"  Fold {row['fold']}: T={row['T']:.6f}  a={row['a']:.6f}  b={row['b']:.6f} "
            f"a<=0={row['a_nonpositive_flag']} platt_iters={row['platt_n_iter']}"
        )
    lines.append("")
    lines.append("Calibration-set metrics (descriptive)")
    for fold in [1, 2, 3, 4]:
        cm = payload["calibration_set_metrics"][f"fold_{fold}"]
        lines.append(
            f"  F{fold} C0 NLL/Brier/ECE/AUC={cm['C0']['nll']:.4f}/{cm['C0']['brier']:.4f}/"
            f"{cm['C0']['ece15']:.4f}/{cm['C0']['roc_auc']:.4f} | "
            f"C1 {cm['C1']['nll']:.4f}/{cm['C1']['brier']:.4f}/{cm['C1']['ece15']:.4f}/{cm['C1']['roc_auc']:.4f} | "
            f"C2 {cm['C2']['nll']:.4f}/{cm['C2']['brier']:.4f}/{cm['C2']['ece15']:.4f}/{cm['C2']['roc_auc']:.4f}"
        )
    lines.append("")
    lines.append("Untouched evaluation macro-mean")
    for m in ["C0", "C1", "C2"]:
        s = payload["evaluation_metrics"]["macro_mean"][m]
        lines.append(
            f"  {m}: NLL={s['nll']:.4f} Brier={s['brier']:.4f} ECE={s['ece15']:.4f} "
            f"AUC={s['roc_auc']:.4f} AP={s['ap']:.4f} "
            f"AI@0.5={s['ai_recall']:.4f} RealSpec={s['real_specificity']:.4f} "
            f"balAcc={s['balanced_accuracy']:.4f} F1={s['f1']:.4f}"
        )
    lines.append("")
    lines.append("Per-fold evaluation @0.5 (AI recall / Real spec)")
    for fold in [1, 2, 3, 4]:
        bf = payload["evaluation_metrics"]["by_fold"][f"fold_{fold}"]
        lines.append(
            f"  F{fold}: C0 {bf['C0']['ai_recall']:.3f}/{bf['C0']['real_specificity']:.3f}  "
            f"C1 {bf['C1']['ai_recall']:.3f}/{bf['C1']['real_specificity']:.3f}  "
            f"C2 {bf['C2']['ai_recall']:.3f}/{bf['C2']['real_specificity']:.3f}"
        )
    lines.append("")
    lines.append(
        f"C1 vs C0 classification changes @0.5: "
        f"{payload['temperature_expectation']['n_changed_total']} / "
        f"{payload['temperature_expectation']['n_eval_total']} "
        f"(by fold {payload['temperature_expectation']['n_changed_by_fold']})"
    )
    lines.append("")
    lines.append("Hard generators (mean across folds where present)")
    # summarise
    from collections import defaultdict

    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in payload["hard_generators"]["per_fold"]:
        if r["n"] == 0:
            continue
        gid = r["generator_id"]
        for m in ["C0", "C1", "C2"]:
            agg[gid][f"{m}_recall"].append(r[m]["recall_050"])
            agg[gid][f"{m}_auc"].append(r[m]["auc_vs_fold_real"])
            agg[gid][f"{m}_mean_p"].append(r[m]["mean_p"])
    for gid in HARD_GENERATORS:
        if gid not in agg:
            continue
        a = agg[gid]
        lines.append(
            f"  {gid}: recall C0/C1/C2="
            f"{np.mean(a['C0_recall']):.3f}/{np.mean(a['C1_recall']):.3f}/{np.mean(a['C2_recall']):.3f}  "
            f"AUC {np.mean(a['C0_auc']):.3f}/{np.mean(a['C2_auc']):.3f}  "
            f"mean_p {np.mean(a['C0_mean_p']):.3f}->{np.mean(a['C2_mean_p']):.3f}"
        )
    lines.append("")
    lines.append("Real-domain specificity @0.5 (macro over folds)")
    for dom in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        vals = {m: [] for m in ["C0", "C1", "C2"]}
        for fold in [1, 2, 3, 4]:
            for m in ["C0", "C1", "C2"]:
                d = payload["real_domain"]["by_fold"][f"fold_{fold}"][m].get(dom, {})
                if d.get("n", 0):
                    vals[m].append(d["specificity_050"])
        lines.append(
            f"  {dom}: C0={np.mean(vals['C0']):.3f} C1={np.mean(vals['C1']):.3f} "
            f"C2={np.mean(vals['C2']):.3f}"
        )
    f3m = payload["real_domain"]["by_fold"]["fold_3"]
    lines.append(
        f"  Fold3 MLLM: C0={f3m['C0']['MLLM']['specificity_050']:.3f} "
        f"C2={f3m['C2']['MLLM']['specificity_050']:.3f}"
    )
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append("")
    lines.append("Interpretation")
    for k, v in payload["interpretation"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Next-stage recommendation ONLY")
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
    calibs, evals, integrity = load_and_audit()

    fitted_rows = []
    calib_metrics: dict[str, Any] = {}
    eval_rows = []
    eval_by_fold: dict[str, Any] = {}
    hard_per_fold = []
    real_by_fold: dict[str, Any] = {}
    c1_changes = {}
    reliability_bins = {}

    for fold in [1, 2, 3, 4]:
        c = calibs[fold]
        e = evals[fold]
        y_c = c["label"].to_numpy()
        p_c0 = c["p_ai"].to_numpy()
        z_c = to_logit(p_c0)

        # Fit C1
        T, t_meta = fit_temperature(z_c, y_c)
        # Fit C2
        a, b, p_meta = fit_platt(z_c, y_c)
        if p_meta["a_nonpositive_flag"]:
            raise SystemExit(
                f"STOP: fold {fold} Platt slope a={a} <= 0 (rank-reversing); not accepted silently"
            )

        fitted_rows.append(
            {
                "fold": fold,
                "T": T,
                "a": a,
                "b": b,
                "a_nonpositive_flag": p_meta["a_nonpositive_flag"],
                "platt_n_iter": p_meta["n_iter"],
                "platt_converged": p_meta["converged"],
                "temp_calibrated_nll_on_calib": t_meta["calibrated_nll"],
                "temp_raw_nll_on_calib": t_meta["raw_nll"],
            }
        )

        # Calibration-set metrics
        p_c1 = apply_temperature(z_c, T)
        p_c2 = apply_platt(z_c, a, b)
        z_c1 = z_c / T
        z_c2 = a * z_c + b
        calib_metrics[f"fold_{fold}"] = {
            "C0": full_metrics(y_c, p_c0, z_c),
            "C1": full_metrics(y_c, p_c1, z_c1),
            "C2": full_metrics(y_c, p_c2, z_c2),
        }

        # Apply to evaluation
        y_e = e["y"].to_numpy()
        p_e0 = e["p_lora"].to_numpy()
        z_e = to_logit(p_e0)
        p_e1 = apply_temperature(z_e, T)
        p_e2 = apply_platt(z_e, a, b)
        z_e1 = z_e / T
        z_e2 = a * z_e + b

        pred0 = (p_e0 >= LOCKED_T).astype(int)
        pred1 = (p_e1 >= LOCKED_T).astype(int)
        n_changed = int((pred0 != pred1).sum())
        c1_changes[f"fold_{fold}"] = n_changed

        fold_eval = {
            "C0": full_metrics(y_e, p_e0, z_e),
            "C1": full_metrics(y_e, p_e1, z_e1),
            "C2": full_metrics(y_e, p_e2, z_e2),
        }
        eval_by_fold[f"fold_{fold}"] = fold_eval

        reliability_bins[f"fold_{fold}"] = {
            "C0": ece_bin_counts(p_e0, y_e),
            "C1": ece_bin_counts(p_e1, y_e),
            "C2": ece_bin_counts(p_e2, y_e),
        }

        # Build row frame
        edf = e.copy()
        edf["label"] = edf["y"]
        edf["p_uncalibrated"] = p_e0
        edf["p_temperature"] = p_e1
        edf["p_platt"] = p_e2
        # drop invalid baseline
        edf = edf.drop(columns=["p_mlpb_baseline", "y", "p_lora"], errors="ignore")
        eval_rows.append(edf)

        # Hard generators
        for gid in HARD_GENERATORS:
            hard_per_fold.append(
                {
                    "fold": fold,
                    "generator_id": gid,
                    "C0": generator_metrics(edf, "p_uncalibrated", gid),
                    "C1": generator_metrics(edf, "p_temperature", gid),
                    "C2": generator_metrics(edf, "p_platt", gid),
                    "n": int((edf["generator_id"] == gid).sum()),
                }
            )

        # Real domains
        real_by_fold[f"fold_{fold}"] = {}
        for m, col in [
            ("C0", "p_uncalibrated"),
            ("C1", "p_temperature"),
            ("C2", "p_platt"),
        ]:
            real_by_fold[f"fold_{fold}"][m] = {
                dom: real_domain_metrics(edf, col, dom)
                for dom in ["Tiny", "MLLM", "COCO", "Smartphone"]
            }

    eval_all = pd.concat(eval_rows, ignore_index=True)
    # Macro-mean across folds (equal fold weight)
    macro = {"C0": {}, "C1": {}, "C2": {}}
    keys = list(eval_by_fold["fold_1"]["C0"].keys())
    for m in ["C0", "C1", "C2"]:
        for k in keys:
            vals = [eval_by_fold[f"fold_{f}"][m][k] for f in [1, 2, 3, 4]]
            macro[m][k] = float(np.nanmean(vals))

    # Also pooled metrics (note Real IDs repeat across folds)
    pooled = {}
    for m, col, zfun in [
        ("C0", "p_uncalibrated", lambda p: to_logit(p)),
        ("C1", "p_temperature", lambda p: to_logit(p)),  # approx; better use exact z
        ("C2", "p_platt", lambda p: to_logit(p)),
    ]:
        # Recompute exact logits from stored probs for pooled descriptive metrics
        y = eval_all["label"].to_numpy()
        p = eval_all[col].to_numpy()
        pooled[m] = full_metrics(y, p, to_logit(p))
    pooled_note = (
        "Pooled metrics concatenate folds; Real IDs repeat across folds. "
        "Primary reporting uses equal-weight fold macro-mean."
    )

    eval_summary = {
        "by_fold": eval_by_fold,
        "macro_mean": macro,
        "pooled_descriptive": pooled,
        "pooled_note": pooled_note,
    }

    n_changed_total = int(sum(c1_changes.values()))
    temp_expect = {
        "n_changed_by_fold": c1_changes,
        "n_changed_total": n_changed_total,
        "n_eval_total": int(len(eval_all)),
        "expected": (
            "Temperature scaling preserves logit sign for T>0, so p=0.5 side "
            "should be unchanged except numerical edge cases near 0.5 after clipping."
        ),
        "substantive_change": n_changed_total > 0,
    }
    # Investigate if any changes
    if n_changed_total > 0:
        # find them
        changed_examples = []
        for fold in [1, 2, 3, 4]:
            sub = eval_all[eval_all["fold"] == fold]
            p0 = sub["p_uncalibrated"].to_numpy()
            p1 = sub["p_temperature"].to_numpy()
            ch = (p0 >= LOCKED_T) != (p1 >= LOCKED_T)
            if ch.any():
                s = sub.loc[ch, ["image_id", "p_uncalibrated", "p_temperature", "label"]].head(5)
                changed_examples.append({"fold": fold, "rows": s.to_dict(orient="records")})
        temp_expect["examples"] = changed_examples
        temp_expect["investigation"] = (
            "Unexpected under exact math for T>0 without clipping; check values exactly 0.5 "
            "or numerical logit clip effects."
        )

    hard_payload = {"per_fold": hard_per_fold}
    real_payload = {"by_fold": real_by_fold}

    decision = decide(eval_summary, hard_per_fold, real_payload)

    figures = make_figures(eval_all, eval_summary, hard_per_fold)

    # Interpretation
    c0m, c1m, c2m = macro["C0"], macro["C1"], macro["C2"]
    interpretation = {
        "q1_miscalibrated": (
            f"Yes on independent calib and untouched eval: C0 ECE≈{c0m['ece15']:.3f}, "
            f"AI recall@0.5≈{c0m['ai_recall']:.3f} despite AUC≈{c0m['roc_auc']:.3f}, "
            f"consistent with V2-9A operating-point shift."
        ),
        "q2_temperature_calibration": (
            f"C1 T fitted per fold; calib/eval NLL {c0m['nll']:.3f}→{c1m['nll']:.3f}, "
            f"ECE {c0m['ece15']:.3f}→{c1m['ece15']:.3f}. Temperature can improve proper "
            "scoring without moving the 0.5 boundary."
        ),
        "q3_temperature_p05_changes": (
            f"C1 changed {n_changed_total}/{len(eval_all)} evaluation classifications at 0.5."
        ),
        "q4_platt_operating_point": (
            f"C2 AI recall {c0m['ai_recall']:.3f}→{c2m['ai_recall']:.3f} "
            f"(Δ={c2m['ai_recall']-c0m['ai_recall']:+.3f}); "
            f"Real spec {c0m['real_specificity']:.3f}→{c2m['real_specificity']:.3f} "
            f"(Δ={c2m['real_specificity']-c0m['real_specificity']:+.3f})."
        ),
        "q5_ai_recall_recovered": (
            f"Macro-mean AI recall recovered by {c2m['ai_recall']-c0m['ai_recall']:+.3f} absolute."
        ),
        "q6_real_spec_cost": (
            f"Macro-mean Real specificity change {c2m['real_specificity']-c0m['real_specificity']:+.3f}."
        ),
        "q7_hard_improved": "See hard-generator section; improvements where mean_p crosses 0.5 under C2.",
        "q8_hard_separability_limited": (
            "Generators with persistently low AUC/AP after C2 remain separability-limited "
            "(notably weak-AP cases such as FLUX.2_max / Seedream from V2-9A)."
        ),
        "q9_fold3_mllm": (
            f"Fold3 MLLM Real spec C0={real_by_fold['fold_3']['C0']['MLLM']['specificity_050']:.3f} "
            f"C2={real_by_fold['fold_3']['C2']['MLLM']['specificity_050']:.3f}."
        ),
        "q10_lora_stronger_candidate": (
            "Depends on decision label: calibration addresses operating-point usability of "
            "LoRA scores without changing the representation; limitations of PB≠holdout AI "
            "and IH reassignment remain."
        ),
        "q11_next": "See next_stage_recommendation.",
    }

    # Enrich hard-generator interpretation after data known
    from collections import defaultdict

    hard_summ = []
    agg = defaultdict(lambda: defaultdict(list))
    for r in hard_per_fold:
        if r["n"] == 0:
            continue
        gid = r["generator_id"]
        for m in ["C0", "C1", "C2"]:
            agg[gid][f"{m}_recall"].append(r[m]["recall_050"])
            agg[gid][f"{m}_auc"].append(r[m]["auc_vs_fold_real"])
            agg[gid][f"{m}_ap"].append(r[m]["ap_vs_fold_real"])
            agg[gid][f"{m}_mean_p"].append(r[m]["mean_p"])
    improved, limited = [], []
    for gid in HARD_GENERATORS:
        if gid not in agg:
            continue
        a = agg[gid]
        drec = float(np.mean(a["C2_recall"]) - np.mean(a["C0_recall"]))
        auc2 = float(np.mean(a["C2_auc"]))
        ap2 = float(np.mean(a["C2_ap"]))
        entry = {
            "generator_id": gid,
            "mean_recall_C0": float(np.mean(a["C0_recall"])),
            "mean_recall_C2": float(np.mean(a["C2_recall"])),
            "delta_recall": drec,
            "mean_auc_C2": auc2,
            "mean_ap_C2": ap2,
            "mean_p_C0": float(np.mean(a["C0_mean_p"])),
            "mean_p_C2": float(np.mean(a["C2_mean_p"])),
        }
        hard_summ.append(entry)
        if drec >= 0.05 and auc2 >= 0.75:
            improved.append(gid)
        if ap2 < 0.40 or auc2 < 0.80:
            limited.append(gid)
    interpretation["q7_hard_improved"] = (
        f"Improved recall under C2 (Δ≥0.05, AUC still decent): {improved or ['none']}."
    )
    interpretation["q8_hard_separability_limited"] = (
        f"Still weak AP/AUC after C2: {limited or ['none']}."
    )

    if decision == "CALIBRATION_PROMISING":
        next_rec = (
            "Tutor may treat Platt-calibrated LoRA as a stronger development candidate "
            "for probability-aware use at p=0.5, then decide among selective prediction, "
            "robustness, or remaining hard-generator representation work. Do not auto-start. "
            "Do not select FINAL_RESEARCH_MODEL_V2 yet."
        )
    elif decision == "CALIBRATION_MIXED":
        next_rec = (
            "Calibration helps some metrics but trade-offs remain. Prefer a human/tutor "
            "decision among: (a) keep Platt as a diagnostic/operating-point tool without "
            "declaring V2 final; (b) selective prediction on calibrated scores; "
            "(c) targeted representation work for residual hard generators. Do not auto-start."
        )
    else:
        next_rec = (
            "Independent post-hoc calibration did not deliver useful operating-point repair "
            "under locks. Prefer non-calibration directions (selective prediction / robustness / "
            "targeted representation) or accept uncalibrated LoRA as ranking-only. Do not auto-start."
        )

    interpretation["q10_lora_stronger_candidate"] = (
        f"Decision={decision}. "
        + (
            "Yes as a calibrated development candidate (not final)."
            if decision == "CALIBRATION_PROMISING"
            else "Partially; ranking remains strong but probability usability remains qualified."
            if decision == "CALIBRATION_MIXED"
            else "Calibration did not sufficiently strengthen LoRA for p=0.5 use."
        )
    )

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "DETECTOR_TRAINING_PERFORMED": "NO",
        "MODEL_WEIGHTS_UPDATED": "NO",
        "CLIP_EMBEDDINGS_REGENERATED": "NO",
        "V2_7_REPLAYED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "EVAL_INFERENCE_RERUN": "NO",
        "CALIBRATOR_FITTED": "YES",
        "TARGET_EVAL_LABELS_USED_FOR_CALIBRATOR": "NO",
        "TRAIN_IDS_USED_AS_CALIBRATION": "NO",
        "EVAL_IDS_USED_AS_CALIBRATION": "NO",
        "THRESHOLD_SWEEP_FOR_SELECTION": "NO",
        "DEPLOYMENT_THRESHOLD_CHANGED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
    }

    # Save predictions / params
    OUT.mkdir(parents=True, exist_ok=True)
    # Standardize prediction columns
    out_pred = eval_all[
        [
            "image_id",
            "fold",
            "label",
            "generator_id",
            "real_domain",
            "p_uncalibrated",
            "p_temperature",
            "p_platt",
        ]
    ].copy()
    out_pred.to_csv(PRED_OUT, index=False)
    pd.DataFrame(fitted_rows).to_csv(PARAMS_OUT, index=False)

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
        return obj

    payload = {
        "stage": "V2-9D",
        "status": "COMPLETE",
        "decision": decision,
        "eps": EPS,
        "ece_bins": 15,
        "locked_threshold": LOCKED_T,
        "limitations_preserved": [
            "AI calibration IDs differ by fold",
            "PROMPT_BLOCKED AI ≠ hard holdout generators",
            "REAL_INTERNAL_HOLDOUT role reassignment for calibration",
            "No shared unused AI across folds",
            "CAP leftovers excluded",
        ],
        "input_integrity": integrity,
        "fitted_parameters": fitted_rows,
        "calibration_set_metrics": calib_metrics,
        "evaluation_metrics": eval_summary,
        "temperature_expectation": temp_expect,
        "hard_generators": {
            "per_fold": hard_per_fold,
            "summary": hard_summ,
            "improved": improved,
            "separability_limited": limited,
        },
        "real_domain": real_payload,
        "reliability_bin_counts_eval": {
            "pooled_C0": ece_bin_counts(
                eval_all["p_uncalibrated"].to_numpy(), eval_all["label"].to_numpy()
            ),
            "pooled_C1": ece_bin_counts(
                eval_all["p_temperature"].to_numpy(), eval_all["label"].to_numpy()
            ),
            "pooled_C2": ece_bin_counts(
                eval_all["p_platt"].to_numpy(), eval_all["label"].to_numpy()
            ),
            "by_fold": reliability_bins,
        },
        "figures": figures,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "artifacts": {
            "calibrated_eval_predictions": str(PRED_OUT.relative_to(PROJECT_ROOT)),
            "calibrator_params": str(PARAMS_OUT.relative_to(PROJECT_ROOT)),
        },
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Wrote {PRED_OUT}")
    print(f"Wrote {PARAMS_OUT}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
