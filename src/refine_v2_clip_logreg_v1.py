"""V2-6: Validation-only frozen CLIP linear-probe C-grid refinement.

No CLIP re-extraction, fine-tuning, MLP, LoRA, threshold tuning, calibration, or NTIRE.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CAP = 300
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
EXPECTED_SHA = "cba0cf3176fd8e61d828a102505edb991d5093c7979eb900f61031fe63acd7d0"
EXPECTED_N = 11377

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
NPZ = OUT / "v2_clip_embeddings_v1.npz"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_clip_embedding_manifest_v1.csv"
REGISTRY = PROJECT_ROOT / "metadata" / "v2_generator_registry_v1.csv"
SPLIT = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
MLLM_MAN = PROJECT_ROOT / "metadata" / "external_mllm_manifest_v2.csv"

HARD_GENS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def metrics_at_050(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "roc_auc": float("nan"),
        "ap": float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ai_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float("nan"),
    }
    real = y_true == 0
    if real.any():
        out["specificity"] = float((y_pred[real] == 0).mean())
        out["fpr"] = float((y_pred[real] == 1).mean())
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    return out


def train_indices_for_fold(man: pd.DataFrame, fold: int) -> np.ndarray:
    col = f"fold_{fold}_role"
    ai = man[(man["binary_label"] == 1) & (man[col] == "TRAIN")].copy()
    keep_ai = []
    for _, gdf in ai.groupby("generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep_ai.extend(gdf["embedding_row"].tolist())
    real = man[(man["binary_label"] == 0) & (man[col] == "TRAIN")]
    return np.array(sorted(set(keep_ai + real["embedding_row"].tolist())), dtype=int)


def val_indices_for_fold(man: pd.DataFrame, fold: int) -> np.ndarray:
    col = f"fold_{fold}_role"
    mask = man[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])
    return man.loc[mask, "embedding_row"].to_numpy(dtype=int)


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float) -> tuple[LogisticRegression, dict]:
    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())
    ratio = max(n_real, n_ai) / max(1, min(n_real, n_ai))
    cw = "balanced" if ratio > 1.25 else None
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=3000,
        random_state=SEED,
        class_weight=cw,
    )
    clf.fit(X, y)
    return clf, {
        "n_real": n_real,
        "n_ai": n_ai,
        "ratio_max_min": float(ratio),
        "class_weight": cw or "None",
    }


def domain_specificity(man: pd.DataFrame, emb: np.ndarray, clf: LogisticRegression, fold: int, domain: str) -> dict:
    mask = (man["real_domain"] == domain) & (man[f"fold_{fold}_role"] == "REAL_VALIDATION")
    idx = man.loc[mask, "embedding_row"].to_numpy()
    if len(idx) == 0:
        return {"n": 0, "specificity": float("nan"), "fpr": float("nan"), "mean_p_ai": float("nan")}
    p = clf.predict_proba(emb[idx])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(idx)),
        "specificity": float((pred == 0).mean()),
        "fpr": float((pred == 1).mean()),
        "mean_p_ai": float(p.mean()),
    }


def load_verified() -> tuple[pd.DataFrame, np.ndarray, dict]:
    assert_path_not_final_external_test(str(NPZ), str(MANIFEST))
    stop_if(not NPZ.exists(), "embeddings missing")
    data = np.load(NPZ, allow_pickle=False)
    emb = data["embeddings"].astype(np.float32)
    ids = data["image_ids"].astype(str)
    sha = hashlib.sha256(emb.tobytes()).hexdigest()
    stop_if(sha != EXPECTED_SHA, f"SHA mismatch {sha} != {EXPECTED_SHA}")
    stop_if(emb.shape != (EXPECTED_N, 512), f"shape {emb.shape}")
    stop_if(bool(np.isnan(emb).any() or np.isinf(emb).any()), "NaN/Inf")
    norms = np.linalg.norm(emb, axis=1)
    stop_if(float(np.abs(norms - 1.0).max()) > 1e-3, "L2 norms not ~1")
    man = pd.read_csv(MANIFEST)
    stop_if(len(man) != EXPECTED_N, "manifest length")
    stop_if(list(man["image_id"].astype(str)) != list(ids), "id order mismatch")
    man["embedding_row"] = man["embedding_row"].astype(int)
    man["binary_label"] = man["binary_label"].astype(int)
    meta = {
        "sha256": sha,
        "n": int(emb.shape[0]),
        "d": int(emb.shape[1]),
        "norm_mean": float(norms.mean()),
        "nan": 0,
        "inf": 0,
    }
    return man, emb, meta


def score_dist(p: np.ndarray) -> dict:
    if len(p) == 0:
        return {k: float("nan") for k in ["n", "mean", "median", "p10", "p25", "p75", "p90"]}
    return {
        "n": int(len(p)),
        "mean": float(np.mean(p)),
        "median": float(np.median(p)),
        "p10": float(np.percentile(p, 10)),
        "p25": float(np.percentile(p, 25)),
        "p75": float(np.percentile(p, 75)),
        "p90": float(np.percentile(p, 90)),
    }


def paired_boot_metric_diff(y, p_a, p_b, metric: str, n_boot=5000, seed=SEED):
    """metric in {roc_auc, ap}. Returns mean_diff and 95% CI for p_a - p_b."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    diffs = []
    n = len(y)
    # stratified: sample Real and AI separately with replacement
    real_idx = np.where(y == 0)[0]
    ai_idx = np.where(y == 1)[0]
    for _ in range(n_boot):
        if len(real_idx) == 0 or len(ai_idx) == 0:
            continue
        ri = rng.choice(real_idx, size=len(real_idx), replace=True)
        ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
        idx = np.concatenate([ri, ai])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc_auc":
            diffs.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        else:
            diffs.append(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx]))
    diffs = np.asarray(diffs, dtype=float)
    return {
        "mean_diff": float(diffs.mean()) if len(diffs) else float("nan"),
        "ci_low": float(np.percentile(diffs, 2.5)) if len(diffs) else float("nan"),
        "ci_high": float(np.percentile(diffs, 97.5)) if len(diffs) else float("nan"),
        "n_valid_boot": int(len(diffs)),
    }


def paired_boot_specificity_diff(p_a, p_b, n_boot=5000, seed=SEED):
    """Specificity = mean(p < 0.5) on Real samples. Δ = selected - baseline."""
    rng = np.random.default_rng(seed)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(p_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sa = float((p_a[idx] < 0.5).mean())
        sb = float((p_b[idx] < 0.5).mean())
        diffs.append(sa - sb)
    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_valid_boot": n_boot,
    }


def select_c(agg: pd.DataFrame) -> tuple[float, str, pd.DataFrame]:
    base = agg.loc[agg["C"] == 1.0].iloc[0]
    elig_rows = []
    for _, r in agg.iterrows():
        reasons = []
        ok = True
        if r["mean_smartphone_specificity"] < 0.95:
            ok = False
            reasons.append(f"mean phone spec {r['mean_smartphone_specificity']:.4f} < 0.95")
        if r["min_smartphone_specificity"] < 0.94:
            ok = False
            reasons.append(f"worst phone spec {r['min_smartphone_specificity']:.4f} < 0.94")
        if r["mean_mllm_specificity"] < base["mean_mllm_specificity"] - 0.02:
            ok = False
            reasons.append(
                f"MLLM spec {r['mean_mllm_specificity']:.4f} < baseline-0.02 "
                f"({base['mean_mllm_specificity'] - 0.02:.4f})"
            )
        if r["mean_real_specificity"] < base["mean_real_specificity"] - 0.02:
            ok = False
            reasons.append(
                f"Real spec {r['mean_real_specificity']:.4f} < baseline-0.02 "
                f"({base['mean_real_specificity'] - 0.02:.4f})"
            )
        if ok:
            reasons.append("PASS all reliability guards")
        elig_rows.append(
            {
                "C": float(r["C"]),
                "eligible": ok,
                "reason": "; ".join(reasons),
            }
        )
    elig_df = pd.DataFrame(elig_rows)
    eligible = agg[agg["C"].isin(elig_df.loc[elig_df["eligible"], "C"])].copy()
    stop_if(len(eligible) == 0, "no C passed reliability guards")

    # Primary: highest mean AUC
    best_auc = float(eligible["mean_roc_auc"].max())
    near = eligible[eligible["mean_roc_auc"] >= best_auc - 0.003].copy()
    rule_bits = [f"primary max mean AUC among eligible (best={best_auc:.6f})"]

    if len(near) == 1:
        selected = float(near.iloc[0]["C"])
        rule = rule_bits[0] + f"; unique best C={selected}"
        return selected, rule, elig_df

    # Tie / practical equivalence within 0.003
    rule_bits.append(f"{len(near)} C within 0.003 of best mean AUC")
    # 1) higher worst-fold AUC
    max_worst = float(near["min_roc_auc"].max())
    near2 = near[near["min_roc_auc"] == max_worst]
    if len(near2) < len(near):
        rule_bits.append(f"prefer higher worst-fold AUC ({max_worst:.6f})")
    near = near2
    if len(near) == 1:
        selected = float(near.iloc[0]["C"])
        return selected, "; ".join(rule_bits) + f"; selected C={selected}", elig_df

    # 2) higher mean MLLM Real specificity
    max_mllm = float(near["mean_mllm_specificity"].max())
    near2 = near[near["mean_mllm_specificity"] == max_mllm]
    if len(near2) < len(near):
        rule_bits.append(f"prefer higher mean MLLM specificity ({max_mllm:.6f})")
    near = near2
    if len(near) == 1:
        selected = float(near.iloc[0]["C"])
        return selected, "; ".join(rule_bits) + f"; selected C={selected}", elig_df

    # 3) higher mean smartphone specificity
    max_phone = float(near["mean_smartphone_specificity"].max())
    near2 = near[near["mean_smartphone_specificity"] == max_phone]
    if len(near2) < len(near):
        rule_bits.append(f"prefer higher mean smartphone specificity ({max_phone:.6f})")
    near = near2
    if len(near) == 1:
        selected = float(near.iloc[0]["C"])
        return selected, "; ".join(rule_bits) + f"; selected C={selected}", elig_df

    # 4) smaller C (stronger regularization)
    selected = float(near["C"].min())
    rule_bits.append(f"prefer smaller C (stronger regularization); selected C={selected}")
    return selected, "; ".join(rule_bits), elig_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    man, emb, emb_meta = load_verified()
    print("Embeddings verified:", emb_meta)

    reg = pd.read_csv(REGISTRY)
    legacy_map = dict(zip(reg["canonical_generator_id"], reg["legacy_or_modern"]))

    # Precompute train/val indices once
    fold_idx = {}
    for fold in range(1, 5):
        fold_idx[fold] = {
            "train": train_indices_for_fold(man, fold),
            "val": val_indices_for_fold(man, fold),
        }

    fold_rows = []
    gen_rows = []
    real_rows = []
    # Store predictions: preds[C][fold] = {ids, y, p, domains, generators, roles}
    preds: dict[float, dict[int, dict]] = {}

    t0 = time.perf_counter()
    for C in C_GRID:
        preds[C] = {}
        for fold in range(1, 5):
            tr = fold_idx[fold]["train"]
            va = fold_idx[fold]["val"]
            Xtr, ytr = emb[tr], man.iloc[tr]["binary_label"].to_numpy()
            Xva, yva = emb[va], man.iloc[va]["binary_label"].to_numpy()
            clf, meta = fit_logreg(Xtr, ytr, C)
            pva = clf.predict_proba(Xva)[:, 1]
            m = metrics_at_050(yva, pva)

            phone = domain_specificity(man, emb, clf, fold, "Smartphone")
            mllm = domain_specificity(man, emb, clf, fold, "MLLM")
            coco = domain_specificity(man, emb, clf, fold, "COCO")
            tiny = domain_specificity(man, emb, clf, fold, "Tiny")

            fold_rows.append(
                {
                    "C": C,
                    "fold": fold,
                    **m,
                    "smartphone_specificity": phone["specificity"],
                    "mllm_specificity": mllm["specificity"],
                    "coco_specificity": coco["specificity"],
                    "tiny_specificity": tiny["specificity"],
                    "smartphone_mean_p_ai": phone["mean_p_ai"],
                    "mllm_mean_p_ai": mllm["mean_p_ai"],
                    "class_weight": meta["class_weight"],
                    "train_n_real": meta["n_real"],
                    "train_n_ai": meta["n_ai"],
                }
            )

            for domain in ["Tiny", "MLLM", "COCO", "Smartphone"]:
                d = domain_specificity(man, emb, clf, fold, domain)
                real_rows.append({"C": C, "fold": fold, "real_domain": domain, **d})

            hold = man[(man["binary_label"] == 1) & (man[f"fold_{fold}_role"] == "HOLDOUT_VALIDATION")]
            for gid, gdf in hold.groupby("generator_id"):
                gidx = gdf["embedding_row"].to_numpy()
                pp = clf.predict_proba(emb[gidx])[:, 1]
                pred = (pp >= 0.5).astype(int)
                gen_rows.append(
                    {
                        "C": C,
                        "fold": fold,
                        "generator_id": gid,
                        "legacy_or_modern": legacy_map.get(gid, "unknown"),
                        "n": int(len(gidx)),
                        "ai_recall_050": float((pred == 1).mean()),
                        "mean_p_ai": float(pp.mean()),
                        "median_p_ai": float(np.median(pp)),
                    }
                )

            va_ids = man.iloc[va]["image_id"].astype(str).to_numpy()
            preds[C][fold] = {
                "clf": clf,
                "meta": meta,
                "ids": va_ids,
                "y": yva,
                "p": pva,
                "domains": man.iloc[va]["real_domain"].astype(str).to_numpy(),
                "generators": man.iloc[va]["generator_id"].astype(str).to_numpy(),
                "labels": yva,
            }
        print(f"C={C} done")

    grid_time = time.perf_counter() - t0
    fold_df = pd.DataFrame(fold_rows)
    gen_df = pd.DataFrame(gen_rows)
    real_df = pd.DataFrame(real_rows)
    fold_df.to_csv(OUT / "v2_clip_logreg_c_grid_v1.csv", index=False)
    gen_df.to_csv(OUT / "v2_clip_logreg_generator_c_grid_v1.csv", index=False)
    real_df.to_csv(OUT / "v2_clip_logreg_real_domain_c_grid_v1.csv", index=False)

    # Aggregate per C
    agg_rows = []
    for C in C_GRID:
        sub = fold_df[fold_df["C"] == C]
        agg_rows.append(
            {
                "C": C,
                "mean_roc_auc": float(sub["roc_auc"].mean()),
                "std_roc_auc": float(sub["roc_auc"].std(ddof=1)),
                "min_roc_auc": float(sub["roc_auc"].min()),
                "max_roc_auc": float(sub["roc_auc"].max()),
                "mean_ap": float(sub["ap"].mean()),
                "std_ap": float(sub["ap"].std(ddof=1)),
                "min_ap": float(sub["ap"].min()),
                "mean_ai_recall": float(sub["ai_recall"].mean()),
                "mean_real_specificity": float(sub["specificity"].mean()),
                "mean_smartphone_specificity": float(sub["smartphone_specificity"].mean()),
                "min_smartphone_specificity": float(sub["smartphone_specificity"].min()),
                "mean_mllm_specificity": float(sub["mllm_specificity"].mean()),
                "min_mllm_specificity": float(sub["mllm_specificity"].min()),
                "mean_coco_specificity": float(sub["coco_specificity"].mean()),
                "mean_tiny_specificity": float(sub["tiny_specificity"].mean()),
            }
        )
    agg = pd.DataFrame(agg_rows)
    selected_C, selection_rule, elig_df = select_c(agg)
    print("SELECTED C", selected_C)
    print("RULE", selection_rule)
    print(elig_df.to_string(index=False))

    # Decision
    base_row = agg.loc[agg["C"] == 1.0].iloc[0]
    sel_row = agg.loc[agg["C"] == selected_C].iloc[0]
    if abs(selected_C - 1.0) < 1e-12:
        decision = "LINEAR_PROBE_BASELINE_RETAINED"
        decision_explain = "C=1.0 selected by locked rule (best or practically equivalent under reliability guards)."
    else:
        # Did refinement address remaining weaknesses?
        mllm_gain = float(sel_row["mean_mllm_specificity"] - base_row["mean_mllm_specificity"])
        auc_gain = float(sel_row["mean_roc_auc"] - base_row["mean_roc_auc"])
        # Check hard generators mean recall change
        hard_delta = []
        for gid in HARD_GENS:
            b = gen_df[(gen_df["C"] == 1.0) & (gen_df["generator_id"] == gid)]["ai_recall_050"]
            s = gen_df[(gen_df["C"] == selected_C) & (gen_df["generator_id"] == gid)]["ai_recall_050"]
            if len(b) and len(s):
                hard_delta.append(float(s.mean() - b.mean()))
        hard_mean_delta = float(np.mean(hard_delta)) if hard_delta else 0.0
        if auc_gain < 0.003 and mllm_gain < 0.02 and hard_mean_delta < 0.02:
            decision = "LINEAR_PROBE_LIMIT_REACHED"
            decision_explain = (
                f"Selected C={selected_C} by rule, but gains vs C=1.0 are negligible on AUC/MLLM/"
                f"hard-generator recall (ΔAUC={auc_gain:.4f}, ΔMLLM={mllm_gain:.4f}, "
                f"Δhard_recall≈{hard_mean_delta:.4f}); linear probe likely near useful limit."
            )
        else:
            decision = "LINEAR_PROBE_REFINED"
            decision_explain = (
                f"Selected C={selected_C} by locked rule with material development evidence "
                f"(ΔAUC={auc_gain:.4f}, ΔMLLM={mllm_gain:.4f}, Δhard_recall≈{hard_mean_delta:.4f})."
            )

    # Save refined fold models (even if C=1.0 — separate from V2-5 filenames)
    for fold in range(1, 5):
        clf = preds[selected_C][fold]["clf"]
        meta = preds[selected_C][fold]["meta"]
        joblib.dump(
            {
                "model": clf,
                "meta": meta,
                "fold": fold,
                "C": selected_C,
                "stage": "V2-6",
                "decision": decision,
            },
            MODELS / f"clip_logreg_refined_fold{fold}_v1.joblib",
        )

    # Bootstrap selected vs C=1.0
    boot_rows = []
    if abs(selected_C - 1.0) < 1e-12:
        bootstrap_note = "BASELINE_RETAINED — no self-comparison bootstrap"
    else:
        bootstrap_note = "paired stratified bootstrap 5000× seed=42 (selected − C=1.0)"
        for fold in range(1, 5):
            ps = preds[selected_C][fold]
            pb = preds[1.0][fold]
            y = ps["y"]
            for metric in ["roc_auc", "ap"]:
                b = paired_boot_metric_diff(y, ps["p"], pb["p"], metric=metric, n_boot=5000, seed=SEED)
                boot_rows.append({"fold": fold, "metric": f"{metric}_selected_minus_C1", **b})

            # Real subsets
            for domain_name, mask in [
                ("overall_Real", y == 0),
                ("MLLM_Real", (y == 0) & (ps["domains"] == "MLLM")),
                ("Smartphone_Real", (y == 0) & (ps["domains"] == "Smartphone")),
            ]:
                if mask.sum() == 0:
                    continue
                b = paired_boot_specificity_diff(ps["p"][mask], pb["p"][mask], n_boot=5000, seed=SEED)
                boot_rows.append(
                    {
                        "fold": fold,
                        "metric": f"specificity_{domain_name}_selected_minus_C1",
                        **b,
                    }
                )
        pd.DataFrame(boot_rows).to_csv(OUT / "v2_clip_logreg_refinement_bootstrap_v1.csv", index=False)

    # Error overlap selected vs C=1.0
    overlap_rows = []
    for fold in range(1, 5):
        ps = preds[selected_C][fold]
        pb = preds[1.0][fold]
        y = ps["y"]
        pred_s = (ps["p"] >= 0.5).astype(int)
        pred_b = (pb["p"] >= 0.5).astype(int)
        correct_s = pred_s == y
        correct_b = pred_b == y
        for subset, mask in [("AI", y == 1), ("Real", y == 0)]:
            both_wrong = mask & (~correct_s) & (~correct_b)
            base_wrong_sel_ok = mask & (~correct_b) & correct_s
            base_ok_sel_wrong = mask & correct_b & (~correct_s)
            for tag, msk in [
                ("both_wrong", both_wrong),
                ("baseline_wrong_selected_correct", base_wrong_sel_ok),
                ("baseline_correct_selected_wrong", base_ok_sel_wrong),
            ]:
                idx = np.where(msk)[0]
                if subset == "AI":
                    for gid, cnt in pd.Series(ps["generators"][idx]).value_counts().items():
                        overlap_rows.append(
                            {
                                "fold": fold,
                                "subset": subset,
                                "pattern": tag,
                                "group": gid,
                                "n": int(cnt),
                            }
                        )
                else:
                    for dom, cnt in pd.Series(ps["domains"][idx]).value_counts().items():
                        overlap_rows.append(
                            {
                                "fold": fold,
                                "subset": subset,
                                "pattern": tag,
                                "group": dom,
                                "n": int(cnt),
                            }
                        )
                overlap_rows.append(
                    {
                        "fold": fold,
                        "subset": subset,
                        "pattern": tag,
                        "group": "TOTAL",
                        "n": int(msk.sum()),
                    }
                )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUT / "v2_clip_logreg_error_overlap_v1.csv", index=False)

    # Score distributions for selected C (pool across folds carefully: Real val is shared;
    # use fold 1 Real domains + per-fold holdout AI averaged descriptively via fold1 for Real,
    # and all holdout AI scores pooled with fold tags)
    dist_rows = []
    # Real: stable REAL_VALIDATION — use fold 1 scores (same samples every fold may differ by model)
    # Report per-fold mean of distributions for Real domains
    for domain in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        for fold in range(1, 5):
            ps = preds[selected_C][fold]
            mask = (ps["y"] == 0) & (ps["domains"] == domain)
            d = score_dist(ps["p"][mask])
            dist_rows.append({"C": selected_C, "fold": fold, "group": f"Real::{domain}", **d})
    # AI legacy / modern held-out
    for fold in range(1, 5):
        ps = preds[selected_C][fold]
        for era in ["legacy", "modern"]:
            mask = np.array(
                [
                    (ps["y"][i] == 1) and (legacy_map.get(ps["generators"][i], "unknown") == era)
                    for i in range(len(ps["y"]))
                ]
            )
            d = score_dist(ps["p"][mask])
            dist_rows.append({"C": selected_C, "fold": fold, "group": f"AI::{era}_heldout", **d})
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT / "v2_clip_logreg_score_distributions_v1.csv", index=False)

    # Embedding-space diagnostic (analysis only)
    # Mean embedding per Real domain + cosine similarity between domain means and AI group means
    emb_diag = {}
    real_means = {}
    for domain in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        idx = man.loc[(man["binary_label"] == 0) & (man["real_domain"] == domain), "embedding_row"].to_numpy()
        real_means[domain] = emb[idx].mean(axis=0)
        real_means[domain] = real_means[domain] / (np.linalg.norm(real_means[domain]) + 1e-12)
    ai_legacy_idx = man.loc[
        (man["binary_label"] == 1) & (man["generator_id"].map(lambda g: legacy_map.get(g) == "legacy")),
        "embedding_row",
    ].to_numpy()
    ai_modern_idx = man.loc[
        (man["binary_label"] == 1) & (man["generator_id"].map(lambda g: legacy_map.get(g) == "modern")),
        "embedding_row",
    ].to_numpy()
    ai_leg = emb[ai_legacy_idx].mean(axis=0)
    ai_mod = emb[ai_modern_idx].mean(axis=0)
    ai_leg = ai_leg / (np.linalg.norm(ai_leg) + 1e-12)
    ai_mod = ai_mod / (np.linalg.norm(ai_mod) + 1e-12)
    cos_rows = []
    for domain, v in real_means.items():
        cos_rows.append(
            {
                "real_domain": domain,
                "cos_sim_to_legacy_AI_mean": float(np.dot(v, ai_leg)),
                "cos_sim_to_modern_AI_mean": float(np.dot(v, ai_mod)),
                "cos_dist_to_legacy_AI_mean": float(1.0 - np.dot(v, ai_leg)),
                "cos_dist_to_modern_AI_mean": float(1.0 - np.dot(v, ai_mod)),
            }
        )
    # pairwise Real domain cosine
    domains = list(real_means.keys())
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            emb_diag[f"cos_sim_{a}_vs_{b}"] = float(np.dot(real_means[a], real_means[b]))
    emb_diag["real_vs_ai_cosine"] = cos_rows

    # Optional PCA visualization (descriptive only; fit on all development embeddings)
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(8, 6))
    # subsample for clarity
    rng = np.random.default_rng(SEED)
    plot_idx = rng.choice(len(emb), size=min(4000, len(emb)), replace=False)
    labels_plot = man.iloc[plot_idx]["binary_label"].to_numpy()
    domains_plot = man.iloc[plot_idx]["real_domain"].astype(str).to_numpy()
    for lab, name, color in [(1, "AI", "#e15759"), (0, "Real", "#4c78a8")]:
        msk = labels_plot == lab
        ax.scatter(xy[plot_idx][msk, 0], xy[plot_idx][msk, 1], s=6, alpha=0.35, c=color, label=name)
    # highlight MLLM Real
    mllm_mask = (labels_plot == 0) & (domains_plot == "MLLM")
    ax.scatter(
        xy[plot_idx][mllm_mask, 0],
        xy[plot_idx][mllm_mask, 1],
        s=14,
        alpha=0.7,
        c="#f28e2b",
        label="MLLM Real",
        edgecolors="k",
        linewidths=0.2,
    )
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)")
    ax.set_title("Descriptive PCA of frozen CLIP embeddings (not used for training)")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_embedding_pca_v1.png", dpi=150)
    plt.close()

    # MLLM Real failure analysis under selected C (use fold with MLLM in REAL_VALIDATION — all folds)
    mllm_fp_summary = {"identifiable_subgroup": "NO", "evidence": "", "folds": []}
    if MLLM_MAN.exists():
        mllm_meta = pd.read_csv(MLLM_MAN)
        mllm_meta = mllm_meta[mllm_meta["label"] == 0].copy()
        meta_by_id = mllm_meta.set_index("image_id")
    else:
        meta_by_id = None

    for fold in range(1, 5):
        ps = preds[selected_C][fold]
        mask = (ps["y"] == 0) & (ps["domains"] == "MLLM")
        ids = ps["ids"][mask]
        p = ps["p"][mask]
        pred = (p >= 0.5).astype(int)
        fp = pred == 1
        tn = pred == 0
        fold_info = {
            "fold": fold,
            "n": int(mask.sum()),
            "fpr": float(fp.mean()) if mask.sum() else float("nan"),
            "fp_n": int(fp.sum()),
            "tn_n": int(tn.sum()),
            "fp_score": score_dist(p[fp]),
            "tn_score": score_dist(p[tn]),
        }
        if meta_by_id is not None and mask.sum():
            rows = []
            for iid, pp, is_fp in zip(ids, p, fp):
                if iid not in meta_by_id.index:
                    continue
                r = meta_by_id.loc[iid]
                rows.append(
                    {
                        "image_id": iid,
                        "is_fp": bool(is_fp),
                        "p_ai": float(pp),
                        "width": int(r["width"]) if pd.notna(r["width"]) else None,
                        "height": int(r["height"]) if pd.notna(r["height"]) else None,
                        "format": r["format"],
                        "mode": r["mode"],
                        "domain": r["domain"],
                    }
                )
            if rows:
                rdf = pd.DataFrame(rows)
                fold_info["fp_vs_tn_format"] = {
                    "fp": rdf.loc[rdf["is_fp"], "format"].value_counts().to_dict(),
                    "tn": rdf.loc[~rdf["is_fp"], "format"].value_counts().to_dict(),
                }
                fold_info["fp_vs_tn_domain"] = {
                    "fp": rdf.loc[rdf["is_fp"], "domain"].value_counts().to_dict(),
                    "tn": rdf.loc[~rdf["is_fp"], "domain"].value_counts().to_dict(),
                }
                fold_info["fp_mean_min_side"] = float(
                    rdf.loc[rdf["is_fp"], ["width", "height"]].min(axis=1).mean()
                ) if rdf["is_fp"].any() else float("nan")
                fold_info["tn_mean_min_side"] = float(
                    rdf.loc[~rdf["is_fp"], ["width", "height"]].min(axis=1).mean()
                ) if (~rdf["is_fp"]).any() else float("nan")
        mllm_fp_summary["folds"].append(fold_info)

    # Assess identifiable subgroup across folds (descriptive)
    # If FP concentrated in one domain class or format relative to TN
    try:
        fp_doms = []
        tn_doms = []
        for fi in mllm_fp_summary["folds"]:
            if "fp_vs_tn_domain" in fi:
                fp_doms.append(fi["fp_vs_tn_domain"]["fp"])
                tn_doms.append(fi["fp_vs_tn_domain"]["tn"])
        if fp_doms:
            # Use fold1 as representative (same Real val set → same labels; preds differ)
            f0 = mllm_fp_summary["folds"][0]
            evidence_parts = [
                f"mean FPR≈{float(np.mean([f['fpr'] for f in mllm_fp_summary['folds']])):.3f}",
                f"FP score mean≈{float(np.mean([f['fp_score']['mean'] for f in mllm_fp_summary['folds'] if f['fp_n']])):.3f}",
                f"TN score mean≈{float(np.mean([f['tn_score']['mean'] for f in mllm_fp_summary['folds'] if f['tn_n']])):.3f}",
            ]
            if "fp_vs_tn_domain" in f0:
                evidence_parts.append(f"FP domains fold1={f0['fp_vs_tn_domain']['fp']}")
                evidence_parts.append(f"TN domains fold1={f0['fp_vs_tn_domain']['tn']}")
            if "fp_mean_min_side" in f0:
                evidence_parts.append(
                    f"mean min-side FP={f0['fp_mean_min_side']:.0f} vs TN={f0['tn_mean_min_side']:.0f}"
                )
            # Heuristic: if one domain is >70% of FPs and <50% of TNs, call YES
            fp_counts = f0.get("fp_vs_tn_domain", {}).get("fp", {})
            tn_counts = f0.get("fp_vs_tn_domain", {}).get("tn", {})
            fp_tot = sum(fp_counts.values()) or 1
            tn_tot = sum(tn_counts.values()) or 1
            concentrated = False
            for dname, c in fp_counts.items():
                if c / fp_tot >= 0.70 and (tn_counts.get(dname, 0) / tn_tot) < 0.50:
                    concentrated = True
            mllm_fp_summary["identifiable_subgroup"] = "YES" if concentrated else "NO"
            mllm_fp_summary["evidence"] = "; ".join(evidence_parts)
        else:
            mllm_fp_summary["evidence"] = "MLLM metadata join insufficient for subgroup analysis"
    except Exception as e:
        mllm_fp_summary["evidence"] = f"metadata analysis limited: {e}"

    # Figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["C"], agg["mean_roc_auc"], "o-", label="mean AUC")
    ax.fill_between(agg["C"], agg["min_roc_auc"], agg["max_roc_auc"], alpha=0.2, label="min–max")
    ax.axvline(selected_C, color="red", ls="--", lw=1, label=f"selected C={selected_C}")
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Frozen CLIP LogReg: C vs held-out AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_c_vs_auc_v1.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["C"], agg["mean_real_specificity"], "o-", label="mean Real spec")
    ax.plot(agg["C"], agg["mean_mllm_specificity"], "s-", label="mean MLLM Real spec")
    ax.axvline(selected_C, color="red", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("Specificity @0.50")
    ax.set_ylim(0, 1)
    ax.set_title("C vs Real / MLLM specificity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_c_vs_real_specificity_v1.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["C"], agg["mean_smartphone_specificity"], "o-", label="mean phone spec")
    ax.plot(agg["C"], agg["min_smartphone_specificity"], "s--", label="worst-fold phone spec")
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.axvline(selected_C, color="red", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("Smartphone specificity @0.50")
    ax.set_ylim(0, 1)
    ax.set_title("C vs smartphone Real specificity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_c_vs_smartphone_specificity_v1.png", dpi=150)
    plt.close()

    # Real domain score distributions (selected C, fold1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ps = preds[selected_C][1]
    for domain, color in zip(
        ["Tiny", "MLLM", "COCO", "Smartphone"],
        ["#4c78a8", "#f28e2b", "#59a14f", "#b07aa1"],
    ):
        mask = (ps["y"] == 0) & (ps["domains"] == domain)
        ax.hist(ps["p"][mask], bins=25, range=(0, 1), alpha=0.45, label=domain, color=color)
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("P(AI)")
    ax.set_title(f"Real validation P(AI) by domain (selected C={selected_C}, fold 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_real_domain_scores_v1.png", dpi=150)
    plt.close()

    # Generator scores for hard gens under selected C
    fig, ax = plt.subplots(figsize=(8, 4))
    hard_sub = gen_df[(gen_df["C"] == selected_C) & (gen_df["generator_id"].isin(HARD_GENS))]
    if len(hard_sub):
        gmean = hard_sub.groupby("generator_id")["mean_p_ai"].mean().sort_values()
        ax.barh(range(len(gmean)), gmean.values, color="#e15759")
        ax.set_yticks(range(len(gmean)))
        ax.set_yticklabels(list(gmean.index), fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("mean P(AI) on held-out fold(s)")
        ax.set_title(f"Difficult generators mean P(AI) (selected C={selected_C})")
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_generator_scores_v1.png", dpi=150)
    plt.close()

    # Refinement summary CSV
    summary_df = agg.merge(elig_df, on="C")
    summary_df["selected"] = summary_df["C"] == selected_C
    summary_df.to_csv(OUT / "v2_clip_logreg_refinement_summary_v1.csv", index=False)

    sel_folds = fold_df[fold_df["C"] == selected_C].copy()

    # Observed differences selected - C1
    auc_diff = float(sel_row["mean_roc_auc"] - base_row["mean_roc_auc"])
    ap_diff = float(sel_row["mean_ap"] - base_row["mean_ap"])
    mllm_diff = float(sel_row["mean_mllm_specificity"] - base_row["mean_mllm_specificity"])
    phone_diff = float(sel_row["mean_smartphone_specificity"] - base_row["mean_smartphone_specificity"])

    config = {
        "stage": "V2-6",
        "selected_C": selected_C,
        "selection_rule": selection_rule,
        "decision": decision,
        "decision_explanation": decision_explain,
        "baseline_C": 1.0,
        "C_grid": C_GRID,
        "eligibility": elig_df.to_dict(orient="records"),
        "aggregation": agg.to_dict(orient="records"),
        "selected_cross_fold": {
            "mean_auc": float(sel_row["mean_roc_auc"]),
            "worst_auc": float(sel_row["min_roc_auc"]),
            "mean_ap": float(sel_row["mean_ap"]),
            "mean_real_specificity": float(sel_row["mean_real_specificity"]),
            "mean_mllm_specificity": float(sel_row["mean_mllm_specificity"]),
            "mean_smartphone_specificity": float(sel_row["mean_smartphone_specificity"]),
        },
        "selected_vs_C1": {
            "auc_diff": auc_diff,
            "ap_diff": ap_diff,
            "mllm_specificity_diff": mllm_diff,
            "smartphone_specificity_diff": phone_diff,
        },
        "bootstrap_note": bootstrap_note,
        "embedding_verification": emb_meta,
        "embedding_space_diagnostic": emb_diag,
        "mllm_real_analysis": mllm_fp_summary,
        "grid_wall_seconds": grid_time,
        "integrity": {
            "clip_extraction_rerun": False,
            "clip_weights_changed": False,
            "clip_fine_tuning": False,
            "embedding_normalization_changed": False,
            "dataset_split_changed": False,
            "generator_holdouts_changed": False,
            "threshold_tuned": False,
            "calibration_fitted": False,
            "ntire_accessed": False,
            "v1_changed": False,
            "mlp_started": False,
            "lora_started": False,
            "remote_gpu": False,
            "only_logreg_C_varied": True,
        },
    }
    (OUT / "v2_clip_logreg_selected_config_v1.json").write_text(json.dumps(config, indent=2) + "\n")

    # Text report
    lines = [
        "V2-6 Validation-only frozen CLIP linear-probe refinement",
        f"Decision: {decision}",
        f"Selected C: {selected_C}",
        f"Selection rule: {selection_rule}",
        f"Explanation: {decision_explain}",
        "",
        "Eligibility:",
        elig_df.to_string(index=False),
        "",
        "Cross-fold aggregation:",
        agg.to_string(index=False),
        "",
        "Selected fold metrics:",
        sel_folds.to_string(index=False),
        "",
        f"Selected vs C=1.0: ΔAUC={auc_diff:.6f} ΔAP={ap_diff:.6f} "
        f"ΔMLLM_spec={mllm_diff:.6f} Δphone_spec={phone_diff:.6f}",
        f"Bootstrap: {bootstrap_note}",
        "",
        "MLLM Real analysis:",
        json.dumps({k: v for k, v in mllm_fp_summary.items() if k != "folds"}, indent=2),
        "",
        "Hard generators (selected C mean recall / mean P):",
    ]
    for gid in HARD_GENS:
        sub = gen_df[(gen_df["C"] == selected_C) & (gen_df["generator_id"] == gid)]
        if len(sub):
            lines.append(
                f"  {gid}: recall={sub['ai_recall_050'].mean():.4f} meanP={sub['mean_p_ai'].mean():.4f} folds={list(sub['fold'])}"
            )
        else:
            lines.append(f"  {gid}: not held out in any fold under this C grid run")
    lines += [
        "",
        "Integrity: only LogisticRegression C varied; CLIP frozen; NTIRE untouched; no MLP/LoRA.",
    ]
    (OUT / "v2_clip_logreg_refinement_report_v1.txt").write_text("\n".join(lines) + "\n")

    print("DECISION", decision)
    print(agg.to_string(index=False))
    print("selected", selected_C, selection_rule)


if __name__ == "__main__":
    main()
