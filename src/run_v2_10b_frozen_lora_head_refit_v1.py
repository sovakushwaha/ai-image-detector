#!/usr/bin/env python3
"""V2-10B — Frozen-LoRA Head Refit: Linear vs Class-Balanced Linear.

H0 = authoritative V2-8 LoRA + MLP-B (existing predictions).
H1 = LogisticRegression on frozen R1 (unweighted).
H2 = LogisticRegression on frozen R1 (class_weight='balanced').

NO LoRA/CLIP/MLP-B training. NO threshold sweep. NO calibration.
NO hyperparameter search. FINAL_V2_MODEL NOT SELECTED.
"""

from __future__ import annotations

import json
import pickle
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v2_10a_representation_geometry_v1 import (  # noqa: E402
    ClipLoRAModel,
    PathDataset,
    resolve_path,
)

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
PRED_DIR = OUT / "kaggle_v2_lora_v37/v2_lora_outputs/predictions"
CKPT_DIR = MODELS
EVAL_FEAT = OUT  # v2_10a_lora_features_fold{}_v1.npz

SEED = 42
CAP = 300
N_BOOT = 5000
THR = 0.5
EXPECTED_TRAIN = {
    1: {"n": 6008, "real": 4280, "ai": 1728},
    2: {"n": 6009, "real": 4280, "ai": 1729},
    3: {"n": 6309, "real": 4280, "ai": 2029},
    4: {"n": 6594, "real": 4280, "ai": 2314},
}
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]

JSON_OUT = OUT / "v2_10b_head_analysis_v1.json"
REPORT_OUT = OUT / "v2_10b_head_report_v1.txt"
PRED_OUT = OUT / "v2_10b_head_predictions_v1.csv"
METRICS_OUT = OUT / "v2_10b_head_metrics_v1.csv"
BOOT_OUT = OUT / "v2_10b_head_bootstrap_v1.json"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def load_manifest() -> pd.DataFrame:
    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)
    return man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()


def train_membership(man: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Exact V2-8 detector TRAIN membership (AI CAP=300 per generator)."""
    col = f"fold_{fold}_role"
    ai = man[(man["binary_label"] == 1) & (man[col] == "TRAIN")].copy()
    keep = []
    for _, gdf in ai.groupby("canonical_generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep.append(gdf)
    ai_keep = pd.concat(keep) if keep else ai.iloc[:0]
    real = man[(man["binary_label"] == 0) & (man[col] == "TRAIN")].copy()
    out = pd.concat([ai_keep, real], ignore_index=True)
    out = out.sort_values("image_id").reset_index(drop=True)
    return out


def eval_ids_from_pred(fold: int) -> list[str]:
    pred = pd.read_csv(PRED_DIR / f"fold{fold}_predictions_v1.csv")
    return list(pred["image_id"].astype(str))


def artifact_audit() -> dict[str, Any]:
    eval_npz = {f: OUT / f"v2_10a_lora_features_fold{f}_v1.npz" for f in [1, 2, 3, 4]}
    train_npz = {f: OUT / f"v2_10b_lora_train_features_fold{f}_v1.npz" for f in [1, 2, 3, 4]}
    inv = {
        "eval_r1": {
            f"fold_{f}": {
                "path": str(p.relative_to(PROJECT_ROOT)),
                "exists": p.is_file(),
            }
            for f, p in eval_npz.items()
        },
        "train_r1_preexisting": {
            f"fold_{f}": {
                "path": str(p.relative_to(PROJECT_ROOT)),
                "exists": p.is_file(),
            }
            for f, p in train_npz.items()
        },
        "eval_features_reextract_forbidden": True,
        "train_features_need_extraction": not all(p.is_file() for p in train_npz.values()),
    }
    for f, p in eval_npz.items():
        if not p.is_file():
            stop(f"missing V2-10A eval R1 features: {p}")
        z = np.load(p, allow_pickle=False)
        if "r1" not in z.files:
            stop(f"fold {f} eval npz missing r1")
        if z["r1"].shape[1] != 512:
            stop(f"fold {f} eval R1 D={z['r1'].shape[1]} != 512")
    return inv


def extract_train_r1(man: pd.DataFrame) -> dict[str, Any]:
    """Inference-only R1 dump for exact detector TRAIN IDs."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta: dict[str, Any] = {"performed": False, "device": str(device), "folds": {}}
    man_ix = man.set_index("image_id", drop=False)

    for fold in [1, 2, 3, 4]:
        out_npz = OUT / f"v2_10b_lora_train_features_fold{fold}_v1.npz"
        if out_npz.is_file():
            z = np.load(out_npz, allow_pickle=False)
            meta["folds"][f"fold_{fold}"] = {
                "reused": True,
                "checkpoint": f"models/v2/clip_lora_fold{fold}_best_v1.pt",
                "n": int(len(z["image_ids"])),
                "D_r1": int(z["r1"].shape[1]),
                "path": str(out_npz.relative_to(PROJECT_ROOT)),
            }
            continue

        meta["performed"] = True
        tr = train_membership(man, fold)
        exp = EXPECTED_TRAIN[fold]
        if len(tr) != exp["n"]:
            stop(f"fold {fold} train n={len(tr)} != expected {exp['n']}")
        n_real = int((tr["binary_label"] == 0).sum())
        n_ai = int((tr["binary_label"] == 1).sum())
        if n_real != exp["real"] or n_ai != exp["ai"]:
            stop(f"fold {fold} train real/ai={n_real}/{n_ai} != {exp['real']}/{exp['ai']}")

        ev = set(eval_ids_from_pred(fold))
        overlap = set(tr["image_id"].astype(str)) & ev
        if overlap:
            stop(f"fold {fold} train/eval overlap n={len(overlap)}")

        rows = []
        for iid in tr["image_id"].astype(str):
            r = man_ix.loc[iid]
            path = resolve_path(r)
            if not path.is_file():
                stop(f"missing train image {iid}: {path}")
            rows.append({"image_id": iid, "path": str(path)})

        ckpt_path = CKPT_DIR / f"clip_lora_fold{fold}_best_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})

        print(f"[V2-10B] Extracting TRAIN R1 fold {fold} on {device} n={len(rows)} ...", flush=True)
        model = ClipLoRAModel(cfg, device)
        incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
        crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
        if crit:
            stop(f"fold {fold} critical missing keys {crit[:10]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        ds = PathDataset(rows, model.preprocess)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        r1_list, id_list = [], []
        with torch.no_grad():
            for xb, ids in loader:
                xb = xb.to(device)
                r1 = model.encode(xb)
                r1_list.append(r1.detach().cpu().numpy().astype(np.float32))
                id_list.extend(list(ids))

        r1 = np.concatenate(r1_list, axis=0)
        ids_arr = np.asarray(id_list, dtype=str)
        # order to train membership order
        pos = {i: j for j, i in enumerate(ids_arr)}
        order = [pos[i] for i in tr["image_id"].astype(str)]
        r1 = r1[order]
        ids_arr = ids_arr[order]
        y = tr["binary_label"].to_numpy(dtype=np.int64)

        if not np.isfinite(r1).all():
            stop(f"fold {fold} non-finite train R1")
        if r1.shape != (len(tr), 512):
            stop(f"fold {fold} train R1 shape {r1.shape}")
        if list(ids_arr) != list(tr["image_id"].astype(str)):
            stop(f"fold {fold} train ID order mismatch")

        np.savez_compressed(
            out_npz,
            image_ids=ids_arr,
            r1=r1,
            y=y,
            fold=np.array([fold]),
            layer_r1="lora_visual_embedding_l2_normalized",
            membership="DETECTOR_TRAIN_CAP300",
            checkpoint=np.array([str(ckpt_path.relative_to(PROJECT_ROOT))]),
        )
        meta["folds"][f"fold_{fold}"] = {
            "reused": False,
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "n": int(len(ids_arr)),
            "n_real": n_real,
            "n_ai": n_ai,
            "D_r1": 512,
            "path": str(out_npz.relative_to(PROJECT_ROOT)),
            "finite": True,
            "overlap_eval": 0,
        }
        print(f"[V2-10B] Wrote {out_npz.name} n={len(ids_arr)} D=512", flush=True)

    # If all reused, performed stays False unless any new; if any new, True
    if not meta["folds"]:
        stop("no train feature folds recorded")
    if all(v.get("reused") for v in meta["folds"].values()) and not meta["performed"]:
        meta["performed"] = False
    elif any(not v.get("reused") for v in meta["folds"].values()):
        meta["performed"] = True
    return meta


def load_train_r1(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10b_lora_train_features_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32), z["y"].astype(np.int64)


def load_eval_r1(fold: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10a_lora_features_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def fit_head(X: np.ndarray, y: np.ndarray, balanced: bool) -> tuple[LogisticRegression, dict[str, Any]]:
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        class_weight="balanced" if balanced else None,
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    n_params = int(clf.coef_.size + (1 if clf.intercept_ is not None else 0))
    meta = {
        "converged": bool(clf.n_iter_[0] < 2000),
        "n_iter": int(clf.n_iter_[0]),
        "wall_clock_s": float(elapsed),
        "coef_l2_norm": float(np.linalg.norm(clf.coef_.ravel())),
        "intercept": float(clf.intercept_[0]),
        "n_trainable_params": n_params,
        "class_weight_arg": "balanced" if balanced else None,
        "classes": [int(c) for c in clf.classes_],
    }
    if balanced:
        # sklearn formula: n_samples / (n_classes * count)
        n = len(y)
        n_classes = 2
        counts = np.bincount(y, minlength=2)
        w = {c: float(n / (n_classes * counts[c])) for c in [0, 1]}
        meta["sklearn_balanced_weights"] = w
        meta["class_counts"] = {"real_0": int(counts[0]), "ai_1": int(counts[1])}
    else:
        counts = np.bincount(y, minlength=2)
        meta["sklearn_balanced_weights"] = None
        meta["class_counts"] = {"real_0": int(counts[0]), "ai_1": int(counts[1])}
    return clf, meta


def metrics_at_05(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    real_mask = y == 0
    ai_mask = y == 1
    return {
        "n": int(len(y)),
        "n_real": int(real_mask.sum()),
        "n_ai": int(ai_mask.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, p)) if ai_mask.any() else float("nan"),
        "ai_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "real_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_accuracy": float(
            0.5
            * (
                (tp / (tp + fn) if (tp + fn) else 0.0)
                + (tn / (tn + fp) if (tn + fp) else 0.0)
            )
        ),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "ai_pred_pos_frac": float(pred[ai_mask].mean()) if ai_mask.any() else float("nan"),
        "real_pred_pos_frac": float(pred[real_mask].mean()) if real_mask.any() else float("nan"),
    }


def score_dist(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out = {}
    for lab, name in [(0, "real"), (1, "ai")]:
        s = p[y == lab]
        if len(s) == 0:
            out[name] = None
            continue
        out[name] = {
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "p05": float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
        }
    return out


def paired_boot_fold(
    y: np.ndarray, p_new: np.ndarray, p_h0: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """Stratified paired bootstrap within fold: Δ = new − H0."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new).astype(float)
    p_h0 = np.asarray(p_h0).astype(float)
    real_idx = np.where(y == 0)[0]
    ai_idx = np.where(y == 1)[0]
    keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy"]
    buckets: dict[str, list[float]] = {k: [] for k in keys}

    def _metrics(yy, pp):
        pred = (pp >= THR).astype(int)
        tn = int(((yy == 0) & (pred == 0)).sum())
        fp = int(((yy == 0) & (pred == 1)).sum())
        fn = int(((yy == 1) & (pred == 0)).sum())
        tp = int(((yy == 1) & (pred == 1)).sum())
        return {
            "roc_auc": float(roc_auc_score(yy, pp)),
            "ap": float(average_precision_score(yy, pp)),
            "ai_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "real_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "balanced_accuracy": float(
                0.5
                * (
                    (tp / (tp + fn) if (tp + fn) else 0.0)
                    + (tn / (tn + fp) if (tn + fp) else 0.0)
                )
            ),
        }

    for _ in range(n_boot):
        ri = rng.choice(real_idx, size=len(real_idx), replace=True)
        ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
        idx = np.concatenate([ri, ai])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        m_new = _metrics(yy, p_new[idx])
        m_h0 = _metrics(yy, p_h0[idx])
        for k in keys:
            buckets[k].append(m_new[k] - m_h0[k])

    out = {"n_boot": n_boot, "seed": seed, "stratified_by_label": True, "delta_definition": "new_minus_H0"}
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_valid_boot": int(len(arr)),
        }
    return out


def mlp_b_param_count() -> int:
    # 512→256→64→1 with biases
    return 512 * 256 + 256 + 256 * 64 + 64 + 64 * 1 + 1


def decide(macro: dict[str, Any], fold_metrics: list[dict], hard: list[dict], real: list[dict]) -> tuple[str, list[str]]:
    """Predeclared decision gate — do not alter criteria after seeing results."""
    evidence = []
    h0, h1, h2 = macro["H0"], macro["H1"], macro["H2"]

    def check_head(tag: str, m: dict) -> dict[str, bool]:
        recall_gain = m["ai_recall"] - h0["ai_recall"]
        auc_drop = h0["roc_auc"] - m["roc_auc"]
        # domain specs from real summary
        phone = next(r for r in real if r["real_domain"] == "Smartphone")
        mllm = next(r for r in real if r["real_domain"] == "MLLM")
        phone_spec = phone[f"{tag}_specificity"]
        mllm_spec = mllm[f"{tag}_specificity"]
        # fold consistency: improve balanced accuracy or (recall up without huge spec crash)
        fold_improve = 0
        for fm in fold_metrics:
            if fm[f"{tag}_ai_recall"] - fm["H0_ai_recall"] >= 0.10 and fm[f"{tag}_real_specificity"] >= 0.90:
                fold_improve += 1
            elif fm[f"{tag}_balanced_accuracy"] - fm["H0_balanced_accuracy"] >= 0.05:
                fold_improve += 1
        # hard gens recall improvements
        hard_improve = 0
        for hg in hard:
            if hg[f"{tag}_recall"] - hg["H0_recall"] >= 0.10:
                hard_improve += 1
        flags = {
            "substantial_recall": recall_gain >= 0.15,
            "auc_ok": auc_drop <= 0.02 + 1e-9,
            "real_spec_ok": m["real_specificity"] >= 0.90,
            "phone_ok": phone_spec >= 0.95,
            "mllm_ok": mllm_spec >= 0.80,
            "fold_consistency": fold_improve >= 3,
            "hard_gen_improve": hard_improve >= 3,
            "no_catastrophic_real": phone_spec >= 0.95 and mllm_spec >= 0.80 and m["real_specificity"] >= 0.90,
        }
        evidence.append(
            f"{tag}: recall_gain={recall_gain:+.3f} auc_drop={auc_drop:+.4f} "
            f"RealSpec={m['real_specificity']:.3f} Phone={phone_spec:.3f} MLLM={mllm_spec:.3f} "
            f"folds_improve={fold_improve}/4 hard_improve={hard_improve}/5 flags={flags}"
        )
        return flags

    f1 = check_head("H1", h1)
    f2 = check_head("H2", h2)

    def promising(flags: dict[str, bool]) -> bool:
        return all(
            [
                flags["substantial_recall"],
                flags["auc_ok"],
                flags["real_spec_ok"],
                flags["phone_ok"],
                flags["mllm_ok"],
                flags["fold_consistency"],
                flags["hard_gen_improve"],
                flags["no_catastrophic_real"],
            ]
        )

    def material(flags: dict[str, bool], m: dict) -> bool:
        return (m["ai_recall"] - h0["ai_recall"] >= 0.10) or (
            m["balanced_accuracy"] - h0["balanced_accuracy"] >= 0.05
        )

    if promising(f1) or promising(f2):
        return "HEAD_RESCUE_PROMISING", evidence

    # mixed: material improvement but safety/consistency incomplete
    mixed = False
    for flags, m in [(f1, h1), (f2, h2)]:
        if material(flags, m) and (
            not flags["no_catastrophic_real"]
            or not flags["fold_consistency"]
            or not flags["auc_ok"]
            or not flags["hard_gen_improve"]
            or not flags["substantial_recall"]
        ):
            mixed = True
    if mixed:
        return "HEAD_RESCUE_MIXED", evidence

    # also mixed if substantial recall + auc ok but one safety margin barely fails
    for flags, m in [(f1, h1), (f2, h2)]:
        if flags["substantial_recall"] and flags["auc_ok"] and material(flags, m):
            return "HEAD_RESCUE_MIXED", evidence

    return "HEAD_RESCUE_NOT_BETTER", evidence


def make_figures(pred_df: pd.DataFrame, macro: dict, hard: list, real: list) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # 1. Macro metrics bars
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    metrics = ["roc_auc", "ap", "ai_recall", "real_specificity"]
    titles = ["ROC-AUC", "AP", "AI recall@0.5", "Real spec@0.5"]
    for ax, m, t in zip(axes, metrics, titles):
        vals = [macro[h][m] for h in ["H0", "H1", "H2"]]
        ax.bar(["H0", "H1", "H2"], vals, color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_ylim(0, 1.05)
        ax.set_title(t)
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
    fig.suptitle("V2-10B macro H0/H1/H2")
    fig.tight_layout()
    p = FIG / "v2_10b_macro_metrics_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 2. Hard-generator recall
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(hard))
    w = 0.25
    for i, h in enumerate(["H0", "H1", "H2"]):
        ax.bar(x + i * w, [r[f"{h}_recall"] for r in hard], width=w, label=h)
    ax.set_xticks(x + w)
    ax.set_xticklabels([r["generator_id"].split("::")[-1] for r in hard], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AI recall@0.5")
    ax.set_title("Hard-generator recall@0.5")
    ax.legend()
    fig.tight_layout()
    p = FIG / "v2_10b_hard_generator_recall_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 3. Real-domain specificity
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(real))
    for i, h in enumerate(["H0", "H1", "H2"]):
        ax.bar(x + i * w, [r[f"{h}_specificity"] for r in real], width=w, label=h)
    ax.set_xticks(x + w)
    ax.set_xticklabels([r["real_domain"] for r in real])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Real specificity@0.5")
    ax.set_title("Real-domain specificity@0.5")
    ax.legend()
    fig.tight_layout()
    p = FIG / "v2_10b_real_domain_specificity_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 4. Score distributions (pooled descriptive only; not for inference)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, h, col in zip(axes, ["H0", "H1", "H2"], ["p_H0", "p_H1", "p_H2"]):
        for lab, name, c in [(0, "Real", "#4C72B0"), (1, "AI", "#C44E52")]:
            s = pred_df.loc[pred_df["y"] == lab, col]
            ax.hist(s, bins=40, alpha=0.55, label=name, color=c, density=True)
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
        ax.set_title(h)
        ax.set_xlabel("p(AI)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("Score distributions (all folds pooled; descriptive)")
    fig.tight_layout()
    p = FIG / "v2_10b_score_distributions_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10B — Frozen-LoRA Head Refit: Linear vs Class-Balanced Linear")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Only predeclared H1/H2 trained on frozen R1. No LoRA/CLIP/MLP-B training.")
    lines.append("No threshold sweep. No calibration. FINAL_V2_MODEL NOT SELECTED.")
    lines.append("")
    inv = payload["artifact_audit"]
    lines.append("Artifact audit")
    lines.append(f"  Eval R1 reused (V2-10A): all present={all(v['exists'] for v in inv['eval_r1'].values())}")
    lines.append(
        f"  Train R1 preexisting before stage: "
        f"{any(v['exists'] for v in inv['train_r1_preexisting'].values())}"
    )
    inf = payload["train_feature_inference"]
    lines.append(f"NEW_TRAIN_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}")
    for k, v in sorted(inf.get("folds", {}).items()):
        lines.append(
            f"  {k}: ckpt={v['checkpoint']} n={v['n']} D_r1={v['D_r1']} "
            f"reused={v.get('reused')} path={v['path']}"
        )
    lines.append("")
    lines.append("Train/eval integrity")
    for k, v in payload["integrity"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("H1/H2 train meta (macro over folds)")
    for h in ["H1", "H2"]:
        tm = payload["head_train_meta"][h]
        lines.append(
            f"  {h}: mean_n_iter={tm['mean_n_iter']:.1f} mean_time_s={tm['mean_wall_clock_s']:.3f} "
            f"params={tm['n_trainable_params']} mean_coef_l2={tm['mean_coef_l2_norm']:.4f}"
        )
    lines.append("")
    lines.append("Macro metrics @ p=0.5")
    for h in ["H0", "H1", "H2"]:
        m = payload["macro"][h]
        lines.append(
            f"  {h}: AUC={m['roc_auc']:.4f} AP={m['ap']:.4f} "
            f"AI_rec={m['ai_recall']:.4f} RealSpec={m['real_specificity']:.4f} "
            f"BalAcc={m['balanced_accuracy']:.4f} F1={m['f1']:.4f}"
        )
    lines.append("")
    lines.append("Per-fold (AUC / AP / AI_rec / RealSpec / BalAcc)")
    for fm in payload["fold_metrics"]:
        lines.append(f"  Fold {fm['fold']}:")
        for h in ["H0", "H1", "H2"]:
            lines.append(
                f"    {h}: {fm[f'{h}_roc_auc']:.4f} / {fm[f'{h}_ap']:.4f} / "
                f"{fm[f'{h}_ai_recall']:.4f} / {fm[f'{h}_real_specificity']:.4f} / "
                f"{fm[f'{h}_balanced_accuracy']:.4f}"
            )
    lines.append("")
    lines.append("Worst-fold by balanced accuracy")
    for h, w in payload["worst_fold"].items():
        lines.append(f"  {h}: fold={w['fold']} BalAcc={w['balanced_accuracy']:.4f}")
    lines.append("")
    lines.append("Score distributions (macro-mean of fold summaries)")
    for h in ["H0", "H1", "H2"]:
        sd = payload["score_dist_macro"][h]
        lines.append(
            f"  {h} Real mean/med={sd['real']['mean']:.4f}/{sd['real']['median']:.4f} "
            f"AI mean/med={sd['ai']['mean']:.4f}/{sd['ai']['median']:.4f}"
        )
    lines.append("")
    lines.append("Hard generators (macro recall@0.5)")
    for hg in payload["hard_generators"]:
        lines.append(
            f"  {hg['generator_id']}: H0={hg['H0_recall']:.3f} H1={hg['H1_recall']:.3f} "
            f"H2={hg['H2_recall']:.3f}  ΔH1={hg['delta_H1_recall']:+.3f} ΔH2={hg['delta_H2_recall']:+.3f}"
        )
    lines.append("")
    lines.append("Real domains (macro specificity@0.5)")
    for rd in payload["real_domains"]:
        lines.append(
            f"  {rd['real_domain']}: H0={rd['H0_specificity']:.3f} H1={rd['H1_specificity']:.3f} "
            f"H2={rd['H2_specificity']:.3f}"
        )
    if payload.get("fold3_mllm"):
        f3 = payload["fold3_mllm"]
        lines.append(
            f"  Fold3 MLLM: H0={f3['H0_specificity']:.3f} H1={f3['H1_specificity']:.3f} "
            f"H2={f3['H2_specificity']:.3f}"
        )
    lines.append("")
    lines.append("Paired bootstrap (fold-wise; Δ = head − H0; 5000× seed42 stratified)")
    for fold, comps in payload["bootstrap"]["folds"].items():
        lines.append(f"  {fold}:")
        for tag in ["H1_minus_H0", "H2_minus_H0"]:
            b = comps[tag]
            lines.append(
                f"    {tag}: AUC {b['roc_auc']['mean_diff']:+.4f} "
                f"[{b['roc_auc']['ci_low']:+.4f},{b['roc_auc']['ci_high']:+.4f}]; "
                f"AI_rec {b['ai_recall']['mean_diff']:+.4f} "
                f"[{b['ai_recall']['ci_low']:+.4f},{b['ai_recall']['ci_high']:+.4f}]; "
                f"RealSpec {b['real_specificity']['mean_diff']:+.4f} "
                f"[{b['real_specificity']['ci_low']:+.4f},{b['real_specificity']['ci_high']:+.4f}]"
            )
    lines.append("")
    lines.append("Resource analysis")
    for k, v in payload["resource"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Decision evidence")
    for e in payload["decision_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append("")
    lines.append("Interpretation")
    for k, v in payload["interpretation"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Development candidate: {payload['development_candidate']}")
    lines.append(f"Next-stage recommendation ONLY: {payload['next_stage_recommendation']}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("FINAL_V2_MODEL_SELECTED = NO")
    lines.append("NEXT_STAGE_STARTED = NO")
    return "\n".join(lines) + "\n"


def append_research_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-10B" in text:
        print("[V2-10B] research_log already contains V2-10B; not rewriting.", flush=True)
        return
    inf = payload["train_feature_inference"]
    m = payload["macro"]
    folds_line = "; ".join(
        f"F{k.split('_')[1]} n={v['n']} D={v['D_r1']} ckpt={v['checkpoint']} reused={v.get('reused')}"
        for k, v in sorted(inf.get("folds", {}).items())
    )
    hard_lines = "\n".join(
        f"- {h['generator_id']}: recall H0={h['H0_recall']:.3f} H1={h['H1_recall']:.3f} "
        f"H2={h['H2_recall']:.3f}"
        for h in payload["hard_generators"]
    )
    real_lines = "\n".join(
        f"- {r['real_domain']}: spec H0={r['H0_specificity']:.3f} H1={r['H1_specificity']:.3f} "
        f"H2={r['H2_specificity']:.3f}"
        for r in payload["real_domains"]
    )
    entry = f"""
## Stage V2-10B — Frozen-LoRA Head Refit: Linear vs Class-Balanced Linear

**Date:** 2026-09-07  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Head-only refit on frozen V2-8 LoRA R1. No LoRA/CLIP/MLP-B training. No threshold sweep. No calibration. No hyperparameter search. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation (V2-10A):** HEAD_LIMITED — R1 geometry strongly separates Real/AI (kNN purity ~0.966) while MLP-B AI recall@0.5 ~0.386; FNs remain AI-neighbourhood.

**Heads:**  
- H0 = authoritative V2-8 LoRA + MLP-B (`p_lora`)  
- H1 = LogisticRegression L2 C=1.0 unweighted on frozen 512-d R1  
- H2 = same as H1 with `class_weight='balanced'`

**Eval R1:** reused V2-10A `v2_10a_lora_features_fold{{1-4}}_v1.npz` (no re-extraction).  
**NEW_TRAIN_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}**  
{folds_line}

**Macro @ p=0.5:**  
| | AUC | AP | AI_rec | RealSpec | BalAcc |
|--|-----|-----|--------|----------|--------|
| H0 | {m['H0']['roc_auc']:.4f} | {m['H0']['ap']:.4f} | {m['H0']['ai_recall']:.4f} | {m['H0']['real_specificity']:.4f} | {m['H0']['balanced_accuracy']:.4f} |
| H1 | {m['H1']['roc_auc']:.4f} | {m['H1']['ap']:.4f} | {m['H1']['ai_recall']:.4f} | {m['H1']['real_specificity']:.4f} | {m['H1']['balanced_accuracy']:.4f} |
| H2 | {m['H2']['roc_auc']:.4f} | {m['H2']['ap']:.4f} | {m['H2']['ai_recall']:.4f} | {m['H2']['real_specificity']:.4f} | {m['H2']['balanced_accuracy']:.4f} |

**Hard generators:**  
{hard_lines}

**Real domains:**  
{real_lines}

**Bootstrap:** fold-wise paired stratified 5000× seed=42 (Δ = head−H0). No naive 9143 pooled bootstrap.

**Resource:** H0 MLP-B ~{payload['resource']['H0_mlp_b_params']} params; H1/H2 linear ~{payload['resource']['H1_H2_params']} params.

**Decision:** **{payload['decision']}**  
**Development candidate:** {payload['development_candidate']}

**Outputs:** `src/run_v2_10b_frozen_lora_head_refit_v1.py`; `results/v2/v2_10b_*`; `models/v2/v2_10b_h{{1,2}}_*`; figures `figures/v2/v2_10b_*.png`.

**Integrity:** NTIRE=NO; fal=NO; V1 unmodified; folds unchanged; LoRA/CLIP weights not updated; HEAD_TRAINING_PERFORMED=YES (only H1/H2); eval features not re-extracted; eval labels unused for train; V2-9C calib unused; no threshold/calibrator; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10B] Appended research_log.md", flush=True)


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("[V2-10B] Artifact audit ...", flush=True)
    inv = artifact_audit()
    man = load_manifest()

    # Train membership pre-check
    integrity: dict[str, Any] = {"folds": {}}
    for fold in [1, 2, 3, 4]:
        tr = train_membership(man, fold)
        exp = EXPECTED_TRAIN[fold]
        n_real = int((tr["binary_label"] == 0).sum())
        n_ai = int((tr["binary_label"] == 1).sum())
        if len(tr) != exp["n"] or n_real != exp["real"] or n_ai != exp["ai"]:
            stop(f"fold {fold} membership mismatch got {(len(tr), n_real, n_ai)} expected {exp}")
        if tr["image_id"].duplicated().any():
            stop(f"fold {fold} duplicate train IDs")
        ev = set(eval_ids_from_pred(fold))
        overlap = set(tr["image_id"].astype(str)) & ev
        if overlap:
            stop(f"fold {fold} train/eval overlap {len(overlap)}")
        integrity["folds"][f"fold_{fold}"] = {
            "train_n": len(tr),
            "train_real": n_real,
            "train_ai": n_ai,
            "train_dups": 0,
            "overlap_eval": 0,
            "expected_match": True,
        }
    integrity["ok"] = True
    print("[V2-10B] Train membership OK", flush=True)

    train_feat_meta = extract_train_r1(man)

    # Mark created-in-stage provenance if performed
    if train_feat_meta["performed"]:
        for v in train_feat_meta["folds"].values():
            if not v.get("reused"):
                v["created_in_v2_10b"] = True

    all_pred_rows = []
    fold_metrics = []
    fold_score_dists = []
    head_train_per_fold = {"H1": [], "H2": []}
    bootstrap_folds: dict[str, Any] = {}
    hard_rows = []
    real_rows = []
    fold3_mllm = None

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10B] Fold {fold}: fit H1/H2 + evaluate ...", flush=True)
        tr_ids, Xtr, ytr = load_train_r1(fold)
        if Xtr.shape[1] != 512 or not np.isfinite(Xtr).all():
            stop(f"fold {fold} bad train R1")
        ev_ids, Xev = load_eval_r1(fold)
        pred = pd.read_csv(PRED_DIR / f"fold{fold}_predictions_v1.csv")
        pred["image_id"] = pred["image_id"].astype(str)
        if list(ev_ids) != list(pred["image_id"]):
            # align eval features to pred order
            pos = {i: j for j, i in enumerate(ev_ids)}
            try:
                order = [pos[i] for i in pred["image_id"]]
            except KeyError as e:
                stop(f"fold {fold} eval ID missing in V2-10A features: {e}")
            Xev = Xev[order]
            ev_ids = np.asarray(list(pred["image_id"]), dtype=str)
        if set(tr_ids) & set(ev_ids):
            stop(f"fold {fold} feature-level train/eval overlap")

        yev = pred["y"].to_numpy(dtype=int)
        p_h0 = pred["p_lora"].to_numpy(dtype=float)

        h1, m1 = fit_head(Xtr, ytr, balanced=False)
        h2, m2 = fit_head(Xtr, ytr, balanced=True)
        head_train_per_fold["H1"].append(m1)
        head_train_per_fold["H2"].append(m2)

        # persist development heads
        p1 = MODELS / f"v2_10b_h1_logreg_fold{fold}_v1.joblib"
        p2 = MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib"
        joblib.dump(h1, p1)
        joblib.dump(h2, p2)
        m1["serialized_path"] = str(p1.relative_to(PROJECT_ROOT))
        m2["serialized_path"] = str(p2.relative_to(PROJECT_ROOT))
        m1["serialized_bytes"] = int(p1.stat().st_size)
        m2["serialized_bytes"] = int(p2.stat().st_size)

        p_h1 = h1.predict_proba(Xev)[:, list(h1.classes_).index(1)]
        p_h2 = h2.predict_proba(Xev)[:, list(h2.classes_).index(1)]

        for iid, y, a, b, c, rd, gid in zip(
            pred["image_id"],
            yev,
            p_h0,
            p_h1,
            p_h2,
            pred["real_domain"],
            pred["generator_id"],
        ):
            all_pred_rows.append(
                {
                    "fold": fold,
                    "image_id": iid,
                    "y": int(y),
                    "p_H0": float(a),
                    "p_H1": float(b),
                    "p_H2": float(c),
                    "real_domain": rd if pd.notna(rd) else "",
                    "generator_id": gid if pd.notna(gid) else "",
                }
            )

        m0 = metrics_at_05(yev, p_h0)
        m1e = metrics_at_05(yev, p_h1)
        m2e = metrics_at_05(yev, p_h2)
        fold_metrics.append(
            {
                "fold": fold,
                **{f"H0_{k}": m0[k] for k in ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]},
                **{f"H1_{k}": m1e[k] for k in ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]},
                **{f"H2_{k}": m2e[k] for k in ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]},
                "H0_confusion": m0["confusion"],
                "H1_confusion": m1e["confusion"],
                "H2_confusion": m2e["confusion"],
            }
        )
        fold_score_dists.append(
            {
                "fold": fold,
                "H0": score_dist(yev, p_h0),
                "H1": score_dist(yev, p_h1),
                "H2": score_dist(yev, p_h2),
            }
        )

        # hard gens / real domains per fold accumulate later from pred_df
        bootstrap_folds[f"fold_{fold}"] = {
            "H1_minus_H0": paired_boot_fold(yev, p_h1, p_h0),
            "H2_minus_H0": paired_boot_fold(yev, p_h2, p_h0),
        }
        print(f"[V2-10B] Fold {fold} done AUC H0/H1/H2={m0['roc_auc']:.4f}/{m1e['roc_auc']:.4f}/{m2e['roc_auc']:.4f}", flush=True)

    pred_df = pd.DataFrame(all_pred_rows)
    pred_df.to_csv(PRED_OUT, index=False)

    # Macro = mean of fold metrics (not pooled, due to repeated Real IDs)
    def macro_from_folds(prefix: str) -> dict[str, float]:
        keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]
        return {k: float(np.mean([fm[f"{prefix}_{k}"] for fm in fold_metrics])) for k in keys}

    macro = {"H0": macro_from_folds("H0"), "H1": macro_from_folds("H1"), "H2": macro_from_folds("H2")}

    # Hard generators macro
    for gid in HARD_GENERATORS:
        row = {"generator_id": gid}
        ns = []
        for h, col in [("H0", "p_H0"), ("H1", "p_H1"), ("H2", "p_H2")]:
            recalls, aucs, aps, means, meds = [], [], [], [], []
            for fold in [1, 2, 3, 4]:
                sub = pred_df[(pred_df["fold"] == fold) & (pred_df["generator_id"] == gid)]
                if len(sub) == 0:
                    continue
                y = np.ones(len(sub), dtype=int)
                p = sub[col].to_numpy(dtype=float)
                # AUC/AP vs Real in same fold
                real = pred_df[(pred_df["fold"] == fold) & (pred_df["y"] == 0)]
                y_bin = np.concatenate([np.zeros(len(real)), np.ones(len(sub))])
                p_bin = np.concatenate([real[col].to_numpy(dtype=float), p])
                recalls.append(float((p >= THR).mean()))
                aucs.append(float(roc_auc_score(y_bin, p_bin)))
                aps.append(float(average_precision_score(y_bin, p_bin)))
                means.append(float(p.mean()))
                meds.append(float(np.median(p)))
                if h == "H0":
                    ns.append(len(sub))
            row[f"{h}_recall"] = float(np.mean(recalls)) if recalls else float("nan")
            row[f"{h}_auc"] = float(np.mean(aucs)) if aucs else float("nan")
            row[f"{h}_ap"] = float(np.mean(aps)) if aps else float("nan")
            row[f"{h}_mean_p"] = float(np.mean(means)) if means else float("nan")
            row[f"{h}_median_p"] = float(np.mean(meds)) if meds else float("nan")
        row["n_total"] = int(sum(ns))
        row["delta_H1_recall"] = row["H1_recall"] - row["H0_recall"]
        row["delta_H2_recall"] = row["H2_recall"] - row["H0_recall"]
        hard_rows.append(row)

    # Real domains
    for dom in REAL_DOMAINS:
        row = {"real_domain": dom}
        for h, col in [("H0", "p_H0"), ("H1", "p_H1"), ("H2", "p_H2")]:
            specs, fprs, means, meds = [], [], [], []
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
                fprs.append(float((p >= THR).mean()))
                means.append(float(p.mean()))
                meds.append(float(np.median(p)))
            row[f"{h}_specificity"] = float(np.mean(specs)) if specs else float("nan")
            row[f"{h}_fpr"] = float(np.mean(fprs)) if fprs else float("nan")
            row[f"{h}_mean_p"] = float(np.mean(means)) if means else float("nan")
            row[f"{h}_median_p"] = float(np.mean(meds)) if meds else float("nan")
        real_rows.append(row)

    # Fold 3 MLLM
    sub = pred_df[(pred_df["fold"] == 3) & (pred_df["y"] == 0) & (pred_df["real_domain"] == "MLLM")]
    if len(sub):
        fold3_mllm = {
            "n": int(len(sub)),
            "H0_specificity": float((sub["p_H0"] < THR).mean()),
            "H1_specificity": float((sub["p_H1"] < THR).mean()),
            "H2_specificity": float((sub["p_H2"] < THR).mean()),
            "H0_mean_p": float(sub["p_H0"].mean()),
            "H1_mean_p": float(sub["p_H1"].mean()),
            "H2_mean_p": float(sub["p_H2"].mean()),
        }

    # Score dist macro
    score_dist_macro = {}
    for h in ["H0", "H1", "H2"]:
        score_dist_macro[h] = {
            "real": {
                k: float(np.mean([fd[h]["real"][k] for fd in fold_score_dists]))
                for k in ["mean", "median", "p05", "p25", "p75", "p95"]
            },
            "ai": {
                k: float(np.mean([fd[h]["ai"][k] for fd in fold_score_dists]))
                for k in ["mean", "median", "p05", "p25", "p75", "p95"]
            },
        }

    # Worst fold by BalAcc
    worst_fold = {}
    for h in ["H0", "H1", "H2"]:
        worst = min(fold_metrics, key=lambda fm: fm[f"{h}_balanced_accuracy"])
        worst_fold[h] = {
            "fold": worst["fold"],
            "balanced_accuracy": worst[f"{h}_balanced_accuracy"],
            "ai_recall": worst[f"{h}_ai_recall"],
            "real_specificity": worst[f"{h}_real_specificity"],
            "roc_auc": worst[f"{h}_roc_auc"],
        }

    # Head train meta aggregate
    head_train_meta = {}
    for h in ["H1", "H2"]:
        rows = head_train_per_fold[h]
        head_train_meta[h] = {
            "mean_n_iter": float(np.mean([r["n_iter"] for r in rows])),
            "mean_wall_clock_s": float(np.mean([r["wall_clock_s"] for r in rows])),
            "mean_coef_l2_norm": float(np.mean([r["coef_l2_norm"] for r in rows])),
            "n_trainable_params": rows[0]["n_trainable_params"],
            "per_fold": rows,
        }

    # Metrics CSV
    metric_rows = []
    for fm in fold_metrics:
        for h in ["H0", "H1", "H2"]:
            metric_rows.append(
                {
                    "fold": fm["fold"],
                    "head": h,
                    "roc_auc": fm[f"{h}_roc_auc"],
                    "ap": fm[f"{h}_ap"],
                    "ai_recall": fm[f"{h}_ai_recall"],
                    "real_specificity": fm[f"{h}_real_specificity"],
                    "balanced_accuracy": fm[f"{h}_balanced_accuracy"],
                    "precision": fm[f"{h}_precision"],
                    "f1": fm[f"{h}_f1"],
                }
            )
    for h in ["H0", "H1", "H2"]:
        metric_rows.append({"fold": "macro", "head": h, **macro[h]})
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)

    decision, evidence = decide(macro, fold_metrics, hard_rows, real_rows)

    resource = {
        "H0_mlp_b_params": mlp_b_param_count(),
        "H1_H2_params": 513,
        "H1_mean_serialized_bytes": float(np.mean([r["serialized_bytes"] for r in head_train_per_fold["H1"]])),
        "H2_mean_serialized_bytes": float(np.mean([r["serialized_bytes"] for r in head_train_per_fold["H2"]])),
        "H1_mean_train_s": head_train_meta["H1"]["mean_wall_clock_s"],
        "H2_mean_train_s": head_train_meta["H2"]["mean_wall_clock_s"],
        "note": (
            "Linear head ~513 params vs MLP-B ~147841; inference is one dot-product + sigmoid "
            "vs 3-layer MLP. Backbone cost dominates either way."
        ),
    }

    # Interpretation
    h0, h1, h2 = macro["H0"], macro["H1"], macro["H2"]
    gpt = next(h for h in hard_rows if "GPT_Image_2" in h["generator_id"])
    nano = next(h for h in hard_rows if "Nano_Banana" in h["generator_id"])
    flux = next(h for h in hard_rows if "FLUX.2_max" in h["generator_id"])
    seed = next(h for h in hard_rows if "Seedream" in h["generator_id"])
    phone = next(r for r in real_rows if r["real_domain"] == "Smartphone")
    mllm = next(r for r in real_rows if r["real_domain"] == "MLLM")

    best_dev = "NONE"
    if decision == "HEAD_RESCUE_PROMISING":
        # pick safer of promising heads by BalAcc then RealSpec
        cand = []
        for tag, m in [("H1", h1), ("H2", h2)]:
            if m["ai_recall"] - h0["ai_recall"] >= 0.15 and m["real_specificity"] >= 0.90:
                cand.append((tag, m))
        if cand:
            best_dev = max(cand, key=lambda t: (t[1]["balanced_accuracy"], t[1]["real_specificity"]))[0]
    elif decision == "HEAD_RESCUE_MIXED":
        # recommend the better-balanced head as DEVELOPMENT candidate only if RealSpec>=0.90
        ranked = sorted(
            [("H1", h1), ("H2", h2)],
            key=lambda t: (t[1]["balanced_accuracy"], t[1]["real_specificity"]),
            reverse=True,
        )
        for tag, m in ranked:
            if m["real_specificity"] >= 0.90 and m["balanced_accuracy"] > h0["balanced_accuracy"]:
                best_dev = tag
                break
        if best_dev == "NONE" and ranked[0][1]["balanced_accuracy"] > h0["balanced_accuracy"]:
            best_dev = ranked[0][0] + "_CONDITIONAL"

    if decision == "HEAD_RESCUE_PROMISING":
        next_rec = (
            f"Authorize a locked follow-up validating {best_dev} as V2 development candidate "
            "(operating-point / selective-prediction study on frozen R1) — not another LoRA recipe. "
            "Do not auto-start. Do not select FINAL_V2_MODEL."
        )
    elif decision == "HEAD_RESCUE_MIXED":
        next_rec = (
            "Head refit helps but is incomplete: next smallest controlled step is a "
            "predeclared operating-point / selective-prediction diagnostic on the safer linear head "
            "(still frozen R1), OR a narrowly scoped head-objective experiment — not LoRA retraining. "
            "Do not auto-start."
        )
    else:
        next_rec = (
            "Linear heads did not rescue operating behaviour under locks; do not broaden head search. "
            "Next diagnostic should revisit objective/score mapping limitations or selective prediction "
            "without claiming MLP-B is fixed. Do not auto-start LoRA retraining."
        )

    interpretation = {
        "q1_linearly_separable": (
            f"Partially: H1 AUC={h1['roc_auc']:.4f} vs H0={h0['roc_auc']:.4f} "
            f"(Δ={h1['roc_auc']-h0['roc_auc']:+.4f}); linear head recovers ranking on frozen R1."
        ),
        "q2_h1_vs_mlpb": (
            f"H1 vs H0: AUC Δ={h1['roc_auc']-h0['roc_auc']:+.4f}, "
            f"AI_rec Δ={h1['ai_recall']-h0['ai_recall']:+.4f}, "
            f"RealSpec Δ={h1['real_specificity']-h0['real_specificity']:+.4f}, "
            f"BalAcc Δ={h1['balanced_accuracy']-h0['balanced_accuracy']:+.4f}."
        ),
        "q3_h2_recall": (
            f"H2 AI_rec={h2['ai_recall']:.4f} vs H1={h1['ai_recall']:.4f} vs H0={h0['ai_recall']:.4f} "
            f"(H2−H0={h2['ai_recall']-h0['ai_recall']:+.4f})."
        ),
        "q4_h2_real_cost": (
            f"H2 RealSpec={h2['real_specificity']:.4f} vs H0={h0['real_specificity']:.4f} "
            f"(Δ={h2['real_specificity']-h0['real_specificity']:+.4f})."
        ),
        "q5_vs_platt": (
            f"V2-9D Platt RealSpec~0.765; H2 RealSpec={h2['real_specificity']:.4f}; "
            f"Phone H2={phone['H2_specificity']:.3f}; MLLM H2={mllm['H2_specificity']:.3f}."
        ),
        "q6_gpt_nano": (
            f"GPT Image2 recall H0/H1/H2={gpt['H0_recall']:.3f}/{gpt['H1_recall']:.3f}/{gpt['H2_recall']:.3f}; "
            f"Nano Banana2={nano['H0_recall']:.3f}/{nano['H1_recall']:.3f}/{nano['H2_recall']:.3f}."
        ),
        "q7_flux_seedream": (
            f"FLUX.2_max H0/H1/H2={flux['H0_recall']:.3f}/{flux['H1_recall']:.3f}/{flux['H2_recall']:.3f}; "
            f"Seedream={seed['H0_recall']:.3f}/{seed['H1_recall']:.3f}/{seed['H2_recall']:.3f}."
        ),
        "q8_fold3_mllm": (
            f"Fold3 MLLM spec H0/H1/H2="
            f"{fold3_mllm['H0_specificity']:.3f}/{fold3_mllm['H1_specificity']:.3f}/{fold3_mllm['H2_specificity']:.3f}"
            if fold3_mllm
            else "N/A"
        ),
        "q9_imbalance_contributor": (
            f"H2 (balanced) vs H1 (unweighted): AI_rec Δ={h2['ai_recall']-h1['ai_recall']:+.4f}, "
            f"RealSpec Δ={h2['real_specificity']-h1['real_specificity']:+.4f}. "
            "If H2 lifts recall materially beyond H1, imbalance/objective likely contributed to H0 failure."
        ),
        "q10_resource": (
            f"Linear ~{resource['H1_H2_params']} params vs MLP-B ~{resource['H0_mlp_b_params']}; "
            f"mean H1 train {resource['H1_mean_train_s']:.3f}s. Backbone still dominates runtime."
        ),
        "q11_dev_candidate": best_dev,
        "q12_next": next_rec,
    }

    figures = make_figures(pred_df, macro, hard_rows, real_rows)

    boot_payload = {
        "protocol": "fold_wise_paired_stratified_bootstrap_5000_seed42",
        "pooled_9143_forbidden": True,
        "reason_no_pooled": "Real evaluation IDs reused across folds",
        "folds": bootstrap_folds,
        "macro_deltas_descriptive": {
            "H1_minus_H0": {k: h1[k] - h0[k] for k in h0},
            "H2_minus_H0": {k: h2[k] - h0[k] for k in h0},
        },
    }
    BOOT_OUT.write_text(json.dumps(_clean(boot_payload), indent=2))

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "LORA_BACKBONE_TRAINED": "NO",
        "LORA_WEIGHTS_UPDATED": "NO",
        "CLIP_WEIGHTS_UPDATED": "NO",
        "HEAD_TRAINING_PERFORMED": "YES",
        "ONLY_PREDECLARED_H1_H2_TRAINED": "YES",
        "EVAL_FEATURES_REEXTRACTED": "NO",
        "EVAL_LABELS_USED_FOR_TRAINING": "NO",
        "V2_9C_CALIBRATION_DATA_USED_FOR_TRAINING": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "THRESHOLD_TUNED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_TRAIN_FEATURE_INFERENCE_PERFORMED": "YES" if train_feat_meta.get("performed") else "NO",
    }

    payload = {
        "stage": "V2-10B",
        "status": "COMPLETE",
        "decision": decision,
        "decision_evidence": evidence,
        "artifact_audit": inv,
        "train_feature_inference": train_feat_meta,
        "integrity": integrity,
        "head_train_meta": head_train_meta,
        "macro": macro,
        "fold_metrics": fold_metrics,
        "worst_fold": worst_fold,
        "score_dist_macro": score_dist_macro,
        "score_dist_per_fold": fold_score_dists,
        "hard_generators": hard_rows,
        "real_domains": real_rows,
        "fold3_mllm": fold3_mllm,
        "bootstrap": boot_payload,
        "resource": resource,
        "interpretation": interpretation,
        "development_candidate": best_dev,
        "next_stage_recommendation": next_rec,
        "figures": figures,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "locked_config": {
            "H1": {
                "penalty": "l2",
                "C": 1.0,
                "fit_intercept": True,
                "class_weight": None,
                "solver": "lbfgs",
                "max_iter": 2000,
                "seed": SEED,
            },
            "H2": {
                "penalty": "l2",
                "C": 1.0,
                "fit_intercept": True,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 2000,
                "seed": SEED,
            },
            "threshold": THR,
            "n_boot": N_BOOT,
        },
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
