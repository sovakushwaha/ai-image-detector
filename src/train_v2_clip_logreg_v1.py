"""V2-5: Frozen CLIP + Logistic Regression linear probe on four generator holdout folds.

No C tuning, no threshold tuning, no calibration, no NTIRE, no CLIP fine-tuning.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CAP = 300
OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
NPZ = OUT / "v2_clip_embeddings_v1.npz"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_clip_embedding_manifest_v1.csv"
CAP_PLAN = PROJECT_ROOT / "metadata" / "v2_generator_train_cap_plan_v1.csv"


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def metrics_at_050(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "roc_auc": float("nan"),
        "ap": float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "ai_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float("nan"),
    }
    # specificity / FPR on Real subset
    real = y_true == 0
    if real.any():
        out["specificity"] = float((y_pred[real] == 0).mean())
        out["fpr"] = float((y_pred[real] == 1).mean())
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    return out


def load_data():
    assert_path_not_final_external_test(str(NPZ), str(MANIFEST))
    stop_if(not NPZ.exists(), "embeddings missing; run extract_v2_clip_embeddings_v1.py first")
    data = np.load(NPZ, allow_pickle=False)
    emb = data["embeddings"].astype(np.float32)
    ids = data["image_ids"].astype(str)
    man = pd.read_csv(MANIFEST)
    stop_if(len(man) != len(ids), "manifest/embedding length mismatch")
    stop_if(list(man["image_id"].astype(str)) != list(ids), "manifest/embedding order mismatch")
    stop_if(emb.shape[1] != 512, f"dim {emb.shape[1]}")
    stop_if(np.isnan(emb).any() or np.isinf(emb).any(), "NaN/Inf embeddings")
    man["embedding_row"] = man["embedding_row"].astype(int)
    man["binary_label"] = man["binary_label"].astype(int)
    return man, emb


def train_indices_for_fold(man: pd.DataFrame, fold: int, rng: np.random.Generator) -> np.ndarray:
    """AI: TRAIN generators only, deterministic cap 300/generator (sorted image_id).

    Real: exact V2-3 TRAIN allocation (no resampling).
    """
    del rng  # reserved; cap selection is fully deterministic via sorted image_id
    col = f"fold_{fold}_role"
    # AI train with authoritative V2-3 cap (300 / generator / fold)
    ai = man[(man["binary_label"] == 1) & (man[col] == "TRAIN")].copy()
    keep_ai = []
    for gid, gdf in ai.groupby("generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep_ai.extend(gdf["embedding_row"].tolist())

    # Real train (locked V2-3 roles)
    real = man[(man["binary_label"] == 0) & (man[col] == "TRAIN")]
    keep_real = real["embedding_row"].tolist()
    idx = np.array(sorted(set(keep_ai + keep_real)), dtype=int)
    return idx


def val_indices_for_fold(man: pd.DataFrame, fold: int) -> np.ndarray:
    col = f"fold_{fold}_role"
    # Primary held-out: HOLDOUT_VALIDATION AI + REAL_VALIDATION
    mask = man[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])
    return man.loc[mask, "embedding_row"].to_numpy(dtype=int)


def fit_fold(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict, float]:
    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())
    ratio = max(n_real, n_ai) / max(1, min(n_real, n_ai))
    cw = "balanced" if ratio > 1.25 else None
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=SEED,
        class_weight=cw,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    dt = time.perf_counter() - t0
    meta = {"n_real": n_real, "n_ai": n_ai, "ratio_max_min": float(ratio), "class_weight": cw or "None"}
    return clf, meta, dt


def smartphone_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    # y should all be 0
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(p)),
        "specificity": float((pred == 0).mean()) if len(p) else float("nan"),
        "fpr": float((pred == 1).mean()) if len(p) else float("nan"),
        "mean_p_ai": float(p.mean()) if len(p) else float("nan"),
        "median_p_ai": float(np.median(p)) if len(p) else float("nan"),
        "p50": float(np.percentile(p, 50)) if len(p) else float("nan"),
        "p75": float(np.percentile(p, 75)) if len(p) else float("nan"),
        "p90": float(np.percentile(p, 90)) if len(p) else float("nan"),
        "p95": float(np.percentile(p, 95)) if len(p) else float("nan"),
        "frac_pred_ai_050": float((pred == 1).mean()) if len(p) else float("nan"),
    }


def paired_bootstrap_auc(y, p_a, p_b, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    if len(np.unique(y)) < 2:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    diffs = np.asarray(diffs, dtype=float)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_valid_boot": int(len(diffs)),
    }


def load_v1_predictions_for_ids(image_ids: list[str]) -> dict[str, float]:
    """Map image_id -> calibrated P(AI) from existing Stage27/RQ5 files + optional smartphone run."""
    out: dict[str, float] = {}

    # Tiny: RQ5 C0 test calibrated uses source_image_id
    c0 = pd.read_csv(PROJECT_ROOT / "results" / "rq5_C0_test_calibrated_predictions_v1.csv")
    c0 = c0[c0["condition"] == "original"]
    for _, r in c0.iterrows():
        out[str(r["source_image_id"])] = float(r["calibrated_probability"])

    mllm = pd.read_csv(PROJECT_ROOT / "results" / "external_v2_mllm_predictions_v1.csv")
    mllm = mllm[mllm["condition"] == "original"]
    for _, r in mllm.iterrows():
        out[str(r["image_id"])] = float(r["calibrated_probability"])

    coco = pd.read_csv(PROJECT_ROOT / "results" / "external_v2_coco_predictions_v1.csv")
    for _, r in coco.iterrows():
        out[str(r["image_id"])] = float(r["calibrated_probability"])

    qwen = pd.read_csv(PROJECT_ROOT / "results" / "external_v2_qwen_predictions_v1.csv")
    for _, r in qwen.iterrows():
        out[str(r["image_id"])] = float(r["calibrated_probability"])

    # Smartphone validation predictions if present
    sp = OUT / "v2_v1_smartphone_validation_predictions_v1.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        for _, r in sdf.iterrows():
            out[str(r["image_id"])] = float(r["calibrated_probability"])
    return out


def ensure_v1_smartphone_predictions(man: pd.DataFrame) -> None:
    """Run frozen V1 C0 on smartphone REAL_VALIDATION if missing."""
    out_csv = OUT / "v2_v1_smartphone_validation_predictions_v1.csv"
    if out_csv.exists():
        print("V1 smartphone predictions already exist")
        return

    phone = man[(man["real_domain"] == "Smartphone") & (man["fold_1_role"] == "REAL_VALIDATION")].copy()
    # smartphone roles are stable across folds
    stop_if(len(phone) != 500, f"expected 500 smartphone val, got {len(phone)}")

    # Use final inference engine
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from final_inference_v1 import FinalImageDetectorV1

    det = FinalImageDetectorV1()
    # resolve paths from split assignments
    split = pd.read_csv(PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv")
    id_to_path = dict(zip(split["image_id"].astype(str), split["path"].astype(str)))

    rows = []
    t0 = time.perf_counter()
    for _, r in phone.iterrows():
        iid = str(r["image_id"])
        path = PROJECT_ROOT / id_to_path[iid]
        assert_path_not_final_external_test(str(path))
        res = det.predict(path)
        rows.append(
            {
                "image_id": iid,
                "raw_probability": float(res.raw_probability),
                "calibrated_probability": float(res.calibrated_probability),
                "source": "FINAL_RESEARCH_MODEL_V1",
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote V1 smartphone val preds n={len(rows)} in {time.perf_counter()-t0:.1f}s")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    man, emb = load_data()
    ensure_v1_smartphone_predictions(man)
    v1_map = load_v1_predictions_for_ids(man["image_id"].astype(str).tolist())

    fold_rows = []
    gen_rows = []
    real_rows = []
    phone_rows = []
    same_sample_rows = []
    boot_rows = []
    fold_models_meta = {}

    rng = np.random.default_rng(SEED)

    for fold in range(1, 5):
        print(f"=== FOLD {fold} ===")
        tr_idx = train_indices_for_fold(man, fold, rng)
        va_idx = val_indices_for_fold(man, fold)
        Xtr, ytr = emb[tr_idx], man.iloc[tr_idx]["binary_label"].to_numpy()
        Xva, yva = emb[va_idx], man.iloc[va_idx]["binary_label"].to_numpy()
        clf, meta, train_t = fit_fold(Xtr, ytr)
        model_path = MODELS / f"clip_logreg_fold{fold}_v1.joblib"
        joblib.dump({"model": clf, "meta": meta, "fold": fold, "C": 1.0}, model_path)

        t_inf0 = time.perf_counter()
        pva = clf.predict_proba(Xva)[:, 1]
        inf_t = time.perf_counter() - t_inf0
        m = metrics_at_050(yva, pva)

        # Smartphone validation (stable REAL_VALIDATION + Smartphone domain)
        phone_mask = (man["real_domain"] == "Smartphone") & (man[f"fold_{fold}_role"] == "REAL_VALIDATION")
        phone_idx = man.loc[phone_mask, "embedding_row"].to_numpy()
        p_phone = clf.predict_proba(emb[phone_idx])[:, 1]
        sm = smartphone_metrics(np.zeros(len(p_phone), dtype=int), p_phone)
        sm["fold"] = fold
        phone_rows.append(sm)

        fold_rows.append(
            {
                "fold": fold,
                **m,
                "smartphone_specificity": sm["specificity"],
                "smartphone_fpr": sm["fpr"],
                "smartphone_mean_p_ai": sm["mean_p_ai"],
                "train_n_real": meta["n_real"],
                "train_n_ai": meta["n_ai"],
                "class_weight": meta["class_weight"],
                "train_ratio_max_min": meta["ratio_max_min"],
                "train_seconds": train_t,
                "val_infer_seconds": inf_t,
                "train_n_total": int(len(tr_idx)),
                "val_n_total": int(len(va_idx)),
            }
        )
        fold_models_meta[fold] = {"train_seconds": train_t, "class_weight": meta["class_weight"], **meta}

        # Real domain breakdown on REAL_VALIDATION
        for domain in ["Tiny", "MLLM", "COCO", "Smartphone"]:
            dmask = (man["real_domain"] == domain) & (man[f"fold_{fold}_role"] == "REAL_VALIDATION")
            didx = man.loc[dmask, "embedding_row"].to_numpy()
            if len(didx) == 0:
                continue
            pp = clf.predict_proba(emb[didx])[:, 1]
            pred = (pp >= 0.5).astype(int)
            real_rows.append(
                {
                    "fold": fold,
                    "real_domain": domain,
                    "n": int(len(didx)),
                    "specificity": float((pred == 0).mean()),
                    "fpr": float((pred == 1).mean()),
                    "mean_p_ai": float(pp.mean()),
                    "median_p_ai": float(np.median(pp)),
                }
            )

        # Held-out generator breakdown
        hold = man[(man["binary_label"] == 1) & (man[f"fold_{fold}_role"] == "HOLDOUT_VALIDATION")]
        for gid, gdf in hold.groupby("generator_id"):
            gidx = gdf["embedding_row"].to_numpy()
            pp = clf.predict_proba(emb[gidx])[:, 1]
            pred = (pp >= 0.5).astype(int)
            gen_rows.append(
                {
                    "fold": fold,
                    "generator_id": gid,
                    "n": int(len(gidx)),
                    "ai_recall_050": float((pred == 1).mean()),
                    "mean_p_ai": float(pp.mean()),
                    "median_p_ai": float(np.median(pp)),
                }
            )

        # Same-sample V1 comparison on validation ids that have V1 preds
        va_ids = man.iloc[va_idx]["image_id"].astype(str).to_numpy()
        y_align = []
        p_clip = []
        p_v1 = []
        domains = []
        for i, iid in enumerate(va_ids):
            if iid in v1_map:
                y_align.append(int(yva[i]))
                p_clip.append(float(pva[i]))
                p_v1.append(float(v1_map[iid]))
                domains.append(str(man.iloc[va_idx[i]]["real_domain"]))
        if len(y_align) >= 50 and len(set(y_align)) > 1:
            y_align = np.asarray(y_align)
            p_clip = np.asarray(p_clip)
            p_v1 = np.asarray(p_v1)
            m_clip = metrics_at_050(y_align, p_clip)
            m_v1 = metrics_at_050(y_align, p_v1)
            # smartphone subset
            phone_sel = np.array([d == "Smartphone" for d in domains])
            if phone_sel.any():
                sp_clip = smartphone_metrics(np.zeros(phone_sel.sum(), dtype=int), p_clip[phone_sel])
                sp_v1 = smartphone_metrics(np.zeros(phone_sel.sum(), dtype=int), p_v1[phone_sel])
            else:
                sp_clip = {"specificity": float("nan")}
                sp_v1 = {"specificity": float("nan")}
            same_sample_rows.append(
                {
                    "fold": fold,
                    "n_aligned": int(len(y_align)),
                    "clip_auc": m_clip["roc_auc"],
                    "v1_auc": m_v1["roc_auc"],
                    "auc_diff_clip_minus_v1": m_clip["roc_auc"] - m_v1["roc_auc"],
                    "clip_ap": m_clip["ap"],
                    "v1_ap": m_v1["ap"],
                    "clip_real_specificity": m_clip["specificity"],
                    "v1_real_specificity": m_v1["specificity"],
                    "clip_ai_recall": m_clip["ai_recall"],
                    "v1_ai_recall": m_v1["ai_recall"],
                    "clip_smartphone_specificity": sp_clip["specificity"],
                    "v1_smartphone_specificity": sp_v1["specificity"],
                    "smartphone_spec_diff_clip_minus_v1": sp_clip["specificity"] - sp_v1["specificity"],
                }
            )
            boot = paired_bootstrap_auc(y_align, p_clip, p_v1)
            boot_rows.append({"fold": fold, "metric": "roc_auc_clip_minus_v1", **boot})
            if phone_sel.any():
                # bootstrap specificity diff on smartphone
                rngb = np.random.default_rng(SEED)
                diffs = []
                pc, pv = p_clip[phone_sel], p_v1[phone_sel]
                n = len(pc)
                for _ in range(2000):
                    idx = rngb.integers(0, n, n)
                    sc = float(((pc[idx] < 0.5).mean()))
                    sv = float(((pv[idx] < 0.5).mean()))
                    diffs.append(sc - sv)
                diffs = np.asarray(diffs)
                boot_rows.append(
                    {
                        "fold": fold,
                        "metric": "smartphone_specificity_clip_minus_v1",
                        "mean_diff": float(diffs.mean()),
                        "ci_low": float(np.percentile(diffs, 2.5)),
                        "ci_high": float(np.percentile(diffs, 97.5)),
                        "n_valid_boot": 2000,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    gen_df = pd.DataFrame(gen_rows)
    real_df = pd.DataFrame(real_rows)
    phone_df = pd.DataFrame(phone_rows)
    same_df = pd.DataFrame(same_sample_rows)
    boot_df = pd.DataFrame(boot_rows)

    fold_df.to_csv(OUT / "v2_clip_logreg_fold_metrics_v1.csv", index=False)
    gen_df.to_csv(OUT / "v2_clip_logreg_generator_metrics_v1.csv", index=False)
    real_df.to_csv(OUT / "v2_clip_logreg_real_domain_metrics_v1.csv", index=False)
    phone_df.to_csv(OUT / "v2_clip_logreg_smartphone_metrics_v1.csv", index=False)
    if len(same_df):
        same_df.to_csv(OUT / "v2_clip_vs_v1_same_sample_v1.csv", index=False)
    if len(boot_df):
        boot_df.to_csv(OUT / "v2_clip_vs_v1_bootstrap_v1.csv", index=False)

    # Summary + decision
    summary = {
        "mean_heldout_auc": float(fold_df["roc_auc"].mean()),
        "std_heldout_auc": float(fold_df["roc_auc"].std(ddof=1)),
        "min_heldout_auc": float(fold_df["roc_auc"].min()),
        "max_heldout_auc": float(fold_df["roc_auc"].max()),
        "mean_ap": float(fold_df["ap"].mean()),
        "std_ap": float(fold_df["ap"].std(ddof=1)),
        "min_ap": float(fold_df["ap"].min()),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_ai_recall": float(fold_df["ai_recall"].mean()),
        "mean_real_specificity": float(fold_df["specificity"].mean()),
        "mean_smartphone_specificity": float(fold_df["smartphone_specificity"].mean()),
        "min_smartphone_specificity": float(fold_df["smartphone_specificity"].min()),
        "best_fold_auc": int(fold_df.loc[fold_df["roc_auc"].idxmax(), "fold"]),
        "worst_fold_auc": int(fold_df.loc[fold_df["roc_auc"].idxmin(), "fold"]),
        "same_sample_available": bool(len(same_df) > 0),
    }
    if len(same_df):
        summary["v1_mean_auc"] = float(same_df["v1_auc"].mean())
        summary["clip_mean_auc_aligned"] = float(same_df["clip_auc"].mean())
        summary["auc_diff_mean"] = float(same_df["auc_diff_clip_minus_v1"].mean())
        summary["v1_mean_smartphone_specificity"] = float(same_df["v1_smartphone_specificity"].mean())
        summary["clip_mean_smartphone_specificity_aligned"] = float(
            same_df["clip_smartphone_specificity"].mean()
        )
        summary["smartphone_spec_diff_mean"] = float(
            same_df["smartphone_spec_diff_clip_minus_v1"].mean()
        )

    # Decision gate (evidence-based, not a single hard cutoff)
    decision = "CLIP_BASELINE_MIXED"
    evidence = []
    mean_auc = summary["mean_heldout_auc"]
    worst_auc = summary["min_heldout_auc"]
    mean_sp = summary["mean_smartphone_specificity"]
    worst_sp = summary["min_smartphone_specificity"]
    # V1 reference context from same-sample if available
    improved_rank = False
    improved_phone = False
    if summary["same_sample_available"]:
        improved_rank = summary["auc_diff_mean"] >= 0.05
        improved_phone = summary["smartphone_spec_diff_mean"] >= 0.05
        evidence.append(
            f"same-sample AUC CLIP-V1={summary['auc_diff_mean']:.3f}; "
            f"smartphone spec CLIP-V1={summary['smartphone_spec_diff_mean']:.3f}"
        )
    evidence.append(
        f"mean AUC={mean_auc:.3f} worst={worst_auc:.3f}; "
        f"mean phone spec={mean_sp:.3f} worst={worst_sp:.3f}"
    )
    # Absolute floors relative to V1 external failure (~0.516) and V1 phone/COCO FPR issues
    if mean_auc >= 0.70 and worst_auc >= 0.60 and mean_sp >= 0.80 and worst_sp >= 0.70:
        if (not summary["same_sample_available"]) or improved_rank or improved_phone:
            decision = "CLIP_BASELINE_PROMISING"
    if mean_auc < 0.60 or worst_auc < 0.55 or mean_sp < 0.60:
        decision = "CLIP_BASELINE_NOT_PROMISING"
    # If strong on one axis weak on another -> MIXED unless already NOT
    if decision == "CLIP_BASELINE_PROMISING":
        if (worst_auc < 0.60) or (worst_sp < 0.65):
            decision = "CLIP_BASELINE_MIXED"
    if summary["same_sample_available"]:
        if summary["auc_diff_mean"] < 0 and summary["smartphone_spec_diff_mean"] < 0:
            decision = "CLIP_BASELINE_NOT_PROMISING"
        elif summary["auc_diff_mean"] < 0.02 and summary["smartphone_spec_diff_mean"] < 0.02:
            if decision == "CLIP_BASELINE_PROMISING":
                decision = "CLIP_BASELINE_MIXED"

    summary["decision_gate"] = decision
    summary["decision_evidence"] = " | ".join(evidence)
    summary["folds"] = fold_rows
    summary["C"] = 1.0
    summary["threshold"] = 0.50
    summary["clip_frozen"] = True
    summary["ntire_accessed"] = False
    (OUT / "v2_clip_logreg_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Fold {f}" for f in fold_df["fold"]], fold_df["roc_auc"], color="#4c78a8")
    ax.axhline(summary["mean_heldout_auc"], color="black", ls="--", lw=1, label="mean")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title("V2 CLIP+LogReg held-out AUC by fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_logreg_fold_auc_v1.png", dpi=150)
    plt.close()

    # Real specificity by domain (mean across folds)
    real_mean = real_df.groupby("real_domain")["specificity"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(real_mean.index), list(real_mean.values), color="#59a14f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Specificity")
    ax.set_title("V2 CLIP+LogReg Real validation specificity by domain")
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_logreg_real_specificity_v1.png", dpi=150)
    plt.close()

    # Smartphone probability — use fold1 as representative histogram overlay all folds mean p
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold in range(1, 5):
        phone_mask = (man["real_domain"] == "Smartphone") & (man[f"fold_{fold}_role"] == "REAL_VALIDATION")
        phone_idx = man.loc[phone_mask, "embedding_row"].to_numpy()
        clf = joblib.load(MODELS / f"clip_logreg_fold{fold}_v1.joblib")["model"]
        pp = clf.predict_proba(emb[phone_idx])[:, 1]
        ax.hist(pp, bins=20, alpha=0.35, label=f"fold{fold}", range=(0, 1))
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("P(AI)")
    ax.set_title("Smartphone Real validation P(AI) by fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_logreg_smartphone_probability_v1.png", dpi=150)
    plt.close()

    # Generator recall (mean across folds where present)
    gmean = gen_df.groupby("generator_id")["ai_recall_050"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(gmean)), gmean.values)
    ax.set_yticks(range(len(gmean)))
    ax.set_yticklabels(list(gmean.index), fontsize=7)
    ax.set_xlabel("AI recall @0.50")
    ax.set_xlim(0, 1)
    ax.set_title("Held-out generator recall (mean over folds where held out)")
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_logreg_generator_recall_v1.png", dpi=150)
    plt.close()

    if len(same_df):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(same_df))
        ax.bar(x - 0.15, same_df["v1_auc"], width=0.3, label="V1 AUC")
        ax.bar(x + 0.15, same_df["clip_auc"], width=0.3, label="CLIP AUC")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in same_df["fold"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Same-sample V1 vs CLIP+LogReg held-out AUC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "v2_clip_vs_v1_v1.png", dpi=150)
        plt.close()

    # Text report
    lines = [
        "V2-5 Frozen CLIP + Logistic Regression baseline",
        f"Decision: {decision}",
        f"Evidence: {summary['decision_evidence']}",
        "",
        fold_df.to_string(index=False),
        "",
        "Cross-fold:",
        json.dumps({k: summary[k] for k in summary if k not in ('folds',)}, indent=2),
        "",
        "Real domain mean specificity:",
        real_df.groupby("real_domain")[["specificity", "fpr", "mean_p_ai"]].mean().to_string(),
        "",
        "Smartphone by fold:",
        phone_df.to_string(index=False),
    ]
    if len(same_df):
        lines += ["", "Same-sample V1 comparison:", same_df.to_string(index=False)]
    (OUT / "v2_clip_logreg_report_v1.txt").write_text("\n".join(lines) + "\n")
    print("DECISION", decision)
    print(fold_df[["fold", "roc_auc", "ap", "ai_recall", "specificity", "smartphone_specificity"]])
    if len(same_df):
        print(same_df)


if __name__ == "__main__":
    main()
