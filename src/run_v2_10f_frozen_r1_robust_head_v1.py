#!/usr/bin/env python3
"""V2-10F — Frozen-R1 Robust Linear Head Intervention (H3).

H3 = LogisticRegression identical to H2 except trained on
CLEAN + jpeg_q50 + resize_112 + blur_sigma2 + screenshot_strong TRAIN R1
(equal conditions; class_weight=balanced).

NO LoRA/CLIP/H2 training. NO hyperparameter search. Threshold=0.5.
FINAL_V2_MODEL NOT SELECTED.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v2_10a_representation_geometry_v1 import ClipLoRAModel, resolve_path  # noqa: E402
from external_v2_common import TRANSFORM_FNS  # noqa: E402

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"

SEED = 42
N_BOOT = 5000
THR = 0.5
TRANSFORMS = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
CONDITIONS = ["CLEAN"] + TRANSFORMS
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

JSON_OUT = OUT / "v2_10f_robust_head_analysis_v1.json"
REPORT_OUT = OUT / "v2_10f_robust_head_report_v1.txt"
PRED_OUT = OUT / "v2_10f_robust_head_predictions_v1.csv"
METRICS_OUT = OUT / "v2_10f_robust_head_metrics_v1.csv"
BOOT_OUT = OUT / "v2_10f_robust_head_bootstrap_v1.json"
MARGIN_OUT = OUT / "v2_10f_margin_comparison_v1.csv"
RESOURCE_OUT = OUT / "v2_10f_resource_comparison_v1.csv"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


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


def score_dist(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out = {}
    for lab, name in [(0, "real"), (1, "ai")]:
        s = np.asarray(p)[np.asarray(y) == lab]
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
    out: dict[str, Any] = {
        "n_boot": n_boot,
        "seed": seed,
        "stratified": True,
        "delta": "H3_minus_H2",
    }
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_valid": int(len(arr)),
        }
    return out


def cos_vec(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


class TransformDataset(Dataset):
    def __init__(self, rows: list[dict], preprocess, transform_fn: Callable | None):
        self.rows = rows
        self.preprocess = preprocess
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(r["path"]) as im:
            try:
                t = ImageOps.exif_transpose(im)
                if t is not None:
                    im = t
            except Exception:
                pass
            rgb = im.convert("RGB")
        if self.transform_fn is not None:
            rgb = self.transform_fn(rgb)
        return self.preprocess(rgb), r["image_id"]


def load_train_clean(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10b_lora_train_features_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32), z["y"].astype(np.int64)


def load_eval_clean(fold: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10a_lora_features_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def load_eval_tf(fold: int, cond: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10e_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def extract_train_transformed(man: pd.DataFrame) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta: dict[str, Any] = {
        "performed": False,
        "device": str(device),
        "folds": {},
    }
    man_ix = man.set_index("image_id", drop=False)

    for fold in [1, 2, 3, 4]:
        ids, Xc, y = load_train_clean(fold)
        exp = EXPECTED_TRAIN[fold]
        if len(ids) != exp["n"] or int((y == 0).sum()) != exp["real"] or int((y == 1).sum()) != exp["ai"]:
            stop(f"fold {fold} clean train membership mismatch")
        if len(np.unique(ids)) != len(ids):
            stop(f"fold {fold} duplicate clean train IDs")

        rows = []
        for iid in ids:
            r = man_ix.loc[iid]
            path = resolve_path(r)
            if not path.is_file():
                stop(f"missing train image {iid}: {path}")
            rows.append({"image_id": iid, "path": str(path)})

        ckpt_path = MODELS / f"clip_lora_fold{fold}_best_v1.pt"
        need = []
        for cond in TRANSFORMS:
            out_npz = OUT / f"v2_10f_train_r1_{cond}_fold{fold}_v1.npz"
            if out_npz.is_file():
                z = np.load(out_npz, allow_pickle=False)
                if list(z["image_ids"].astype(str)) != list(ids):
                    stop(f"existing {out_npz.name} ID mismatch")
                if z["r1"].shape != (exp["n"], 512) or not np.isfinite(z["r1"]).all():
                    stop(f"bad existing {out_npz.name}")
                meta["folds"][f"fold_{fold}_{cond}"] = {
                    "reused": True,
                    "n": exp["n"],
                    "D_r1": 512,
                    "path": str(out_npz.relative_to(PROJECT_ROOT)),
                    "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                }
            else:
                need.append(cond)

        if not need:
            continue

        meta["performed"] = True
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})
        model = ClipLoRAModel(cfg, device)
        incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
        crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
        if crit:
            stop(f"critical missing keys {crit[:5]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        for cond in need:
            out_npz = OUT / f"v2_10f_train_r1_{cond}_fold{fold}_v1.npz"
            print(
                f"[V2-10F] Extracting TRAIN R1 fold{fold} {cond} on {device} n={len(rows)} ...",
                flush=True,
            )
            t0 = time.perf_counter()
            ds = TransformDataset(rows, model.preprocess, TRANSFORM_FNS[cond])
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            r1_list, id_list = [], []
            with torch.no_grad():
                for xb, batch_ids in loader:
                    xb = xb.to(device)
                    r1 = model.encode(xb)
                    r1_list.append(r1.detach().cpu().numpy().astype(np.float32))
                    id_list.extend(list(batch_ids))
            r1 = np.concatenate(r1_list, axis=0)
            ids_arr = np.asarray(id_list, dtype=str)
            pos = {i: j for j, i in enumerate(ids_arr)}
            order = [pos[i] for i in ids]
            r1 = r1[order]
            ids_arr = ids_arr[order]
            if list(ids_arr) != list(ids) or r1.shape != (exp["n"], 512) or not np.isfinite(r1).all():
                stop(f"fold {fold} {cond} train R1 integrity fail")

            np.savez_compressed(
                out_npz,
                image_ids=ids_arr,
                r1=r1,
                y=y,
                fold=np.array([fold]),
                condition=np.array([cond]),
                membership="DETECTOR_TRAIN_CAP300",
                layer_r1="lora_visual_embedding_l2_normalized",
                checkpoint=np.array([str(ckpt_path.relative_to(PROJECT_ROOT))]),
            )
            meta["folds"][f"fold_{fold}_{cond}"] = {
                "reused": False,
                "n": exp["n"],
                "D_r1": 512,
                "path": str(out_npz.relative_to(PROJECT_ROOT)),
                "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                "seconds": float(time.perf_counter() - t0),
            }
            print(f"[V2-10F] Wrote {out_npz.name} in {time.perf_counter()-t0:.1f}s", flush=True)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if any(not v.get("reused", True) for v in meta["folds"].values()):
        meta["performed"] = True
    return meta


def load_train_tf(fold: int, cond: str) -> np.ndarray:
    z = np.load(OUT / f"v2_10f_train_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
    return z["r1"].astype(np.float32)


def fit_h3(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict[str, Any]]:
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
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    counts = np.bincount(y, minlength=2)
    n = len(y)
    w = {c: float(n / (2 * counts[c])) for c in [0, 1]}
    meta = {
        "n": int(n),
        "n_real": int(counts[0]),
        "n_ai": int(counts[1]),
        "n_iter": int(clf.n_iter_[0]),
        "converged": bool(clf.n_iter_[0] < 2000),
        "wall_clock_s": float(elapsed),
        "coef_l2_norm": float(np.linalg.norm(clf.coef_.ravel())),
        "intercept": float(clf.intercept_[0]),
        "n_params": int(clf.coef_.size + 1),
        "sklearn_balanced_weights": w,
        "classes": [int(c) for c in clf.classes_],
    }
    return clf, meta


def decide(payload_bits: dict[str, Any]) -> tuple[str, list[str]]:
    evidence = []
    h3c = payload_bits["h3_clean"]
    h2c = payload_bits["h2_clean"]
    h3tf = payload_bits["h3_tf_macro"]  # per transform
    phone = payload_bits["phone"]
    mllm = payload_bits["mllm"]
    fold_improve = payload_bits["fold_improve"]
    resources = payload_bits["resources"]

    mean_tf_auc = float(np.mean([h3tf[c]["roc_auc"] for c in TRANSFORMS]))
    worst_tf_auc = float(np.min([h3tf[c]["roc_auc"] for c in TRANSFORMS]))
    mean_tf_spec = float(np.mean([h3tf[c]["real_specificity"] for c in TRANSFORMS]))
    mean_phone = float(np.mean([phone[c] for c in TRANSFORMS]))
    mean_mllm = float(np.mean([mllm[c] for c in TRANSFORMS]))
    mean_tf_rec = float(np.mean([h3tf[c]["ai_recall"] for c in TRANSFORMS]))
    mean_rec_drop = h3c["ai_recall"] - mean_tf_rec
    no_spec_lt_075 = all(h3tf[c]["real_specificity"] >= 0.75 for c in TRANSFORMS)
    # jpeg/resize not one-class collapse: AI recall not ~0 and RealSpec not ~0
    jpeg_ok = h3tf["jpeg_q50"]["ai_recall"] >= 0.25 and h3tf["jpeg_q50"]["real_specificity"] >= 0.75
    resize_ok = h3tf["resize_112"]["real_specificity"] >= 0.75 and h3tf["resize_112"]["ai_recall"] <= 0.95
    # fold improve: majority of transforms improve balacc in >=3 folds
    majority_folds = fold_improve >= 3

    flags = {
        "clean_auc_ge_093": h3c["roc_auc"] >= 0.93,
        "clean_auc_drop_le_003": (h2c["roc_auc"] - h3c["roc_auc"]) <= 0.03,
        "clean_spec_ge_093": h3c["real_specificity"] >= 0.93,
        "clean_phone_ge_098": phone["CLEAN"] >= 0.98,
        "clean_mllm_ge_085": mllm["CLEAN"] >= 0.85,
        "clean_ai_rec_ge_050": h3c["ai_recall"] >= 0.50,
        "mean_tf_auc_ge_090": mean_tf_auc >= 0.90,
        "worst_tf_auc_ge_085": worst_tf_auc >= 0.85,
        "mean_tf_spec_ge_090": mean_tf_spec >= 0.90,
        "mean_phone_ge_095": mean_phone >= 0.95,
        "mean_mllm_ge_080": mean_mllm >= 0.80,
        "no_spec_lt_075": no_spec_lt_075,
        "mean_rec_drop_le_010": mean_rec_drop <= 0.10,
        "fold_consistency": majority_folds,
        "jpeg_not_collapse": jpeg_ok,
        "resize_not_collapse": resize_ok,
        "params_ok": resources["h3_params"] == 513,
    }
    evidence.append(f"flags={flags}")
    evidence.append(
        f"H3 clean AUC={h3c['roc_auc']:.4f} AI_rec={h3c['ai_recall']:.4f} RealSpec={h3c['real_specificity']:.4f}; "
        f"mean_tf_AUC={mean_tf_auc:.4f} worst_tf_AUC={worst_tf_auc:.4f} mean_tf_spec={mean_tf_spec:.4f}"
    )
    evidence.append(
        f"phone_mean_tf={mean_phone:.3f} mllm_mean_tf={mean_mllm:.3f} fold_improve={fold_improve} "
        f"jpeg_ok={jpeg_ok} resize_ok={resize_ok}"
    )

    if all(flags.values()):
        return "ROBUST_HEAD_PROMISING", evidence

    # material robustness improvement vs H2?
    h2tf = payload_bits["h2_tf_macro"]
    mean_h2_auc = float(np.mean([h2tf[c]["roc_auc"] for c in TRANSFORMS]))
    mean_h2_spec = float(np.mean([h2tf[c]["real_specificity"] for c in TRANSFORMS]))
    improved = (mean_tf_auc >= mean_h2_auc + 0.05) or (mean_tf_spec >= mean_h2_spec + 0.10)
    clean_okish = h3c["roc_auc"] >= 0.90 and h3c["real_specificity"] >= 0.90
    if improved and clean_okish and not all(flags.values()):
        return "ROBUST_HEAD_MIXED", evidence
    if improved:
        return "ROBUST_HEAD_MIXED", evidence
    return "ROBUST_HEAD_NOT_BETTER", evidence


def make_figures(macro: dict, hard: list, real: list, margin: dict) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    conds = CONDITIONS
    x = np.arange(len(conds))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(
        axes,
        ["roc_auc", "ai_recall", "real_specificity"],
        ["ROC-AUC", "AI recall@0.5", "Real specificity@0.5"],
    ):
        ax.bar(x, [macro["H2"][c][key] for c in conds], width=w, label="H2")
        ax.bar(x + w, [macro["H3"][c][key] for c in conds], width=w, label="H3")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("V2-10F H2 vs H3")
    fig.tight_layout()
    p = FIG / "v2_10f_h2_vs_h3_metrics_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # margin deltas
    fig, ax = plt.subplots(figsize=(8, 4))
    xt = np.arange(len(TRANSFORMS))
    ax.bar(xt, [margin[c]["h2_delta_z_ai"] for c in TRANSFORMS], width=0.2, label="H2 AI Δz")
    ax.bar(xt + 0.2, [margin[c]["h3_delta_z_ai"] for c in TRANSFORMS], width=0.2, label="H3 AI Δz")
    ax.bar(xt + 0.4, [margin[c]["h2_delta_z_real"] for c in TRANSFORMS], width=0.2, label="H2 Real Δz")
    ax.bar(xt + 0.6, [margin[c]["h3_delta_z_real"] for c in TRANSFORMS], width=0.2, label="H3 Real Δz")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xt + 0.3)
    ax.set_xticklabels(TRANSFORMS, rotation=20, ha="right")
    ax.set_ylabel("mean Δz")
    ax.set_title("Margin shift H2 vs H3")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = FIG / "v2_10f_margin_delta_h2_vs_h3_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # hard gen JPEG recall
    fig, ax = plt.subplots(figsize=(9, 4))
    gens = [h["generator_id"].split("::")[-1] for h in hard]
    xg = np.arange(len(gens))
    ax.bar(xg, [h["H2_CLEAN_recall"] for h in hard], width=0.2, label="H2 CLEAN")
    ax.bar(xg + 0.2, [h["H3_CLEAN_recall"] for h in hard], width=0.2, label="H3 CLEAN")
    ax.bar(xg + 0.4, [h["H2_jpeg_q50_recall"] for h in hard], width=0.2, label="H2 JPEG")
    ax.bar(xg + 0.6, [h["H3_jpeg_q50_recall"] for h in hard], width=0.2, label="H3 JPEG")
    ax.set_xticks(xg + 0.3)
    ax.set_xticklabels(gens, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AI recall@0.5")
    ax.set_title("Hard-generator CLEAN/JPEG H2 vs H3")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "v2_10f_hard_generator_h2_vs_h3_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # real domain resize/blur
    fig, ax = plt.subplots(figsize=(8, 4))
    doms = [r["real_domain"] for r in real]
    xd = np.arange(len(doms))
    ax.bar(xd, [r["H2_resize_112_specificity"] for r in real], width=0.2, label="H2 resize")
    ax.bar(xd + 0.2, [r["H3_resize_112_specificity"] for r in real], width=0.2, label="H3 resize")
    ax.bar(xd + 0.4, [r["H2_blur_sigma2_specificity"] for r in real], width=0.2, label="H2 blur")
    ax.bar(xd + 0.6, [r["H3_blur_sigma2_specificity"] for r in real], width=0.2, label="H3 blur")
    ax.set_xticks(xd + 0.3)
    ax.set_xticklabels(doms)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Real specificity@0.5")
    ax.set_title("Real-domain resize/blur H2 vs H3")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "v2_10f_real_domain_h2_vs_h3_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10F — Frozen-R1 Robust Linear Head Intervention (H3)")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Only H3 trained on frozen R1. No LoRA/CLIP/H2 training. thr=0.5.")
    lines.append("FINAL_V2_MODEL NOT SELECTED.")
    lines.append("")
    inf = payload["train_feature_inference"]
    lines.append(
        f"NEW_TRANSFORMED_TRAIN_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}"
    )
    lines.append(f"  device={inf.get('device')} n_entries={len(inf.get('folds', {}))}")
    lines.append("")
    lines.append("Integrity")
    lines.append(f"  {payload['integrity']}")
    lines.append("")
    lines.append("H3 training")
    for fold, tm in payload["h3_train"].items():
        lines.append(
            f"  {fold}: N={tm['n']} Real={tm['n_real']} AI={tm['n_ai']} "
            f"iters={tm['n_iter']} time={tm['wall_clock_s']:.2f}s params={tm['n_params']} "
            f"||w||={tm['coef_l2_norm']:.3f} b={tm['intercept']:+.3f}"
        )
    lines.append("")
    lines.append("CLEAN macro H2 vs H3")
    for h in ["H2", "H3"]:
        m = payload["macro"][h]["CLEAN"]
        lines.append(
            f"  {h}: AUC={m['roc_auc']:.4f} AP={m['ap']:.4f} AI_rec={m['ai_recall']:.4f} "
            f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
        )
    d = payload["deltas_clean"]
    lines.append(
        f"  H3-H2: ΔAUC={d['roc_auc']:+.4f} ΔAI_rec={d['ai_recall']:+.4f} "
        f"ΔRealSpec={d['real_specificity']:+.4f} ΔBalAcc={d['balanced_accuracy']:+.4f}"
    )
    lines.append("")
    lines.append("StrongRobust macro H2 vs H3")
    for c in TRANSFORMS:
        for h in ["H2", "H3"]:
            m = payload["macro"][h][c]
            lines.append(
                f"  {h}/{c}: AUC={m['roc_auc']:.4f} AI_rec={m['ai_recall']:.4f} "
                f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
            )
    lines.append("")
    lines.append("Robustness summary")
    for k, v in payload["robustness_summary"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Margin / direction")
    lines.append(f"  cos(w_H2,w_H3) macro={payload['weight_comparison']['mean_cos_w']:.4f}")
    for c in TRANSFORMS:
        m = payload["margin_macro"][c]
        lines.append(
            f"  {c}: H2 Δz Real/AI={m['h2_delta_z_real']:+.3f}/{m['h2_delta_z_ai']:+.3f}; "
            f"H3={m['h3_delta_z_real']:+.3f}/{m['h3_delta_z_ai']:+.3f}; "
            f"cos(d_real,w) H2/H3={m['h2_cos_d_real']:+.3f}/{m['h3_cos_d_real']:+.3f}"
        )
    lines.append("")
    lines.append("Hard generators (CLEAN / JPEG recall)")
    for h in payload["hard_generators"]:
        lines.append(
            f"  {h['generator_id']}: H2 C/J={h['H2_CLEAN_recall']:.3f}/{h['H2_jpeg_q50_recall']:.3f} "
            f"H3 C/J={h['H3_CLEAN_recall']:.3f}/{h['H3_jpeg_q50_recall']:.3f}"
        )
    lines.append("")
    lines.append("Real domains (CLEAN / resize / blur specificity)")
    for r in payload["real_domains"]:
        lines.append(
            f"  {r['real_domain']}: H2 {r['H2_CLEAN_specificity']:.3f}/"
            f"{r['H2_resize_112_specificity']:.3f}/{r['H2_blur_sigma2_specificity']:.3f}; "
            f"H3 {r['H3_CLEAN_specificity']:.3f}/"
            f"{r['H3_resize_112_specificity']:.3f}/{r['H3_blur_sigma2_specificity']:.3f}"
        )
    if payload.get("fold3_mllm"):
        f3 = payload["fold3_mllm"]
        lines.append(f"  Fold3 MLLM: {f3}")
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
    if "## Stage V2-10F" in text:
        print("[V2-10F] research_log already contains V2-10F; not rewriting.", flush=True)
        return
    m2, m3 = payload["macro"]["H2"], payload["macro"]["H3"]
    entry = f"""
## Stage V2-10F — Frozen-R1 Robust Linear Head Intervention (H3)

**Date:** 2026-09-08  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Train only predeclared H3 on frozen LoRA R1 (CLEAN+4 StrongRobust TRAIN conditions equally). No LoRA/CLIP/H2 training. No C-grid. Threshold=0.5. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation:** V2-10E = ROBUSTNESS_HEAD_LIMITED / OPTION_A_FROZEN_R1_ROBUST_HEAD.

**H3 design:** identical to H2 LogReg (L2, C=1.0, balanced, lbfgs) trained on 5× Fold-F detector TRAIN R1 rows.

**NEW_TRANSFORMED_TRAIN_FEATURE_INFERENCE_PERFORMED = {'YES' if payload['train_feature_inference'].get('performed') else 'NO'}**

**CLEAN macro:** H2 AUC={m2['CLEAN']['roc_auc']:.4f} AI_rec={m2['CLEAN']['ai_recall']:.4f} RealSpec={m2['CLEAN']['real_specificity']:.4f}; H3 AUC={m3['CLEAN']['roc_auc']:.4f} AI_rec={m3['CLEAN']['ai_recall']:.4f} RealSpec={m3['CLEAN']['real_specificity']:.4f}.

**Decision:** **{payload['decision']}**

**Outputs:** `src/run_v2_10f_frozen_r1_robust_head_v1.py`; `results/v2/v2_10f_*`; `models/v2/v2_10f_h3_*`; figures `figures/v2/v2_10f_*.png`.

**Integrity:** NTIRE=NO; fal=NO; folds unchanged; LoRA/CLIP/H2 not updated; only H3 trained; no HP search; no thr/calibrator/selective retune; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10F] Appended research_log.md", flush=True)


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
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("[V2-10F] Membership / artifact audit ...", flush=True)
    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)

    # verify eval artifacts
    for fold in [1, 2, 3, 4]:
        ids, X = load_eval_clean(fold)
        if len(ids) != EXPECTED_EVAL[fold] or X.shape[1] != 512:
            stop(f"eval clean fold {fold} bad")
        for cond in TRANSFORMS:
            ids_t, Xt = load_eval_tf(fold, cond)
            if list(ids_t) != list(ids):
                stop(f"eval tf {cond} fold {fold} ID mismatch vs clean")
        if not (MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib").is_file():
            stop(f"missing H2 fold {fold}")

    # train/eval overlap check via IDs
    integrity = {"folds": {}, "ok": True}
    for fold in [1, 2, 3, 4]:
        tr_ids, _, _ = load_train_clean(fold)
        ev_ids, _ = load_eval_clean(fold)
        ov = set(tr_ids) & set(ev_ids)
        if ov:
            stop(f"fold {fold} train/eval overlap {len(ov)}")
        integrity["folds"][f"fold_{fold}"] = {
            "train_n": int(len(tr_ids)),
            "eval_n": int(len(ev_ids)),
            "overlap": 0,
            "expected_match": True,
        }

    feat_meta = extract_train_transformed(man)

    # verify transform train IDs identical
    for fold in [1, 2, 3, 4]:
        ids0, _, y0 = load_train_clean(fold)
        for cond in TRANSFORMS:
            z = np.load(OUT / f"v2_10f_train_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
            if list(z["image_ids"].astype(str)) != list(ids0):
                stop(f"fold {fold} {cond} train IDs differ from CLEAN")
            if not np.array_equal(z["y"].astype(np.int64), y0):
                stop(f"fold {fold} {cond} labels differ")

    # Train H3 + evaluate
    pred_rows = []
    metric_rows = []
    fold_metrics: dict[int, dict] = {}
    h3_train_meta = {}
    weight_comp = []
    margin_rows = []
    boot_folds = {}
    score_dists = {}

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10F] Fold {fold}: build H3 matrix + train ...", flush=True)
        ids, Xc, y = load_train_clean(fold)
        blocks = [Xc]
        for cond in TRANSFORMS:
            blocks.append(load_train_tf(fold, cond))
        Xh3 = np.concatenate(blocks, axis=0)
        yh3 = np.concatenate([y] * 5, axis=0)
        exp_rows = EXPECTED_TRAIN[fold]["rows"]
        if Xh3.shape != (exp_rows, 512) or len(yh3) != exp_rows:
            stop(f"fold {fold} H3 matrix shape {Xh3.shape} != ({exp_rows},512)")

        h3, tm = fit_h3(Xh3, yh3)
        path = MODELS / f"v2_10f_h3_robust_balanced_logreg_fold{fold}_v1.joblib"
        joblib.dump(h3, path)
        tm["serialized_path"] = str(path.relative_to(PROJECT_ROOT))
        tm["serialized_bytes"] = int(path.stat().st_size)
        h3_train_meta[f"fold_{fold}"] = tm

        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        w2 = h2.coef_.ravel().astype(np.float64)
        b2 = float(h2.intercept_[0])
        w3 = h3.coef_.ravel().astype(np.float64)
        b3 = float(h3.intercept_[0])
        weight_comp.append(
            {
                "fold": fold,
                "norm_w_h2": float(np.linalg.norm(w2)),
                "norm_w_h3": float(np.linalg.norm(w3)),
                "cos_w": cos_vec(w2, w3),
                "intercept_h2": b2,
                "intercept_h3": b3,
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos_vec(w2, w3), -1, 1)))),
            }
        )

        # eval IDs from V2-10B preds for labels/meta
        # Prefer align to clean eval R1 order then map labels from rob/v2_10b
        ev_ids, X_eval_clean = load_eval_clean(fold)
        # get labels from V2-10B predictions
        pred_h2 = pd.read_csv(OUT / "v2_10b_head_predictions_v1.csv")
        pred_h2 = pred_h2[pred_h2["fold"] == fold].copy()
        pred_h2["image_id"] = pred_h2["image_id"].astype(str)
        if list(pred_h2["image_id"]) != list(ev_ids):
            pos = {i: j for j, i in enumerate(ev_ids)}
            order = [pos[i] for i in pred_h2["image_id"]]
            X_eval_clean = X_eval_clean[order]
            ev_ids = np.asarray(list(pred_h2["image_id"]), dtype=str)
        y_ev = pred_h2["y"].to_numpy(dtype=int)
        gens = pred_h2["generator_id"].fillna("").astype(str).to_numpy()
        domains = pred_h2["real_domain"].fillna("").astype(str).to_numpy()

        cls1_h2 = list(h2.classes_).index(1)
        cls1_h3 = list(h3.classes_).index(1)

        fold_metrics[fold] = {}
        score_dists[fold] = {}

        # CLEAN
        p2_c = h2.predict_proba(X_eval_clean)[:, cls1_h2]
        p3_c = h3.predict_proba(X_eval_clean)[:, cls1_h3]
        # sanity vs stored H2
        max_abs = float(np.max(np.abs(p2_c - pred_h2["p_H2"].to_numpy(dtype=float))))
        if max_abs > 1e-5:
            print(f"[V2-10F] WARNING fold{fold} H2 clean p mismatch {max_abs}", flush=True)

        z2_c = X_eval_clean.astype(np.float64) @ w2 + b2
        z3_c = X_eval_clean.astype(np.float64) @ w3 + b3

        for cond in CONDITIONS:
            if cond == "CLEAN":
                X = X_eval_clean
            else:
                ids_t, Xt = load_eval_tf(fold, cond)
                if list(ids_t) != list(ev_ids):
                    pos = {i: j for j, i in enumerate(ids_t)}
                    Xt = Xt[[pos[i] for i in ev_ids]]
                X = Xt
            p2 = h2.predict_proba(X)[:, cls1_h2]
            p3 = h3.predict_proba(X)[:, cls1_h3]
            m2 = metrics_at_05(y_ev, p2)
            m3 = metrics_at_05(y_ev, p3)
            fold_metrics[fold][cond] = {"H2": m2, "H3": m3}
            metric_rows.append({"fold": fold, "condition": cond, "head": "H2", **m2})
            metric_rows.append({"fold": fold, "condition": cond, "head": "H3", **m3})
            score_dists[fold][cond] = {"H2": score_dist(y_ev, p2), "H3": score_dist(y_ev, p3)}

            if cond != "CLEAN":
                z2 = X.astype(np.float64) @ w2 + b2
                z3 = X.astype(np.float64) @ w3 + b3
                dz2 = z2 - z2_c
                dz3 = z3 - z3_c
                delta = X.astype(np.float64) - X_eval_clean.astype(np.float64)
                real = y_ev == 0
                ai = y_ev == 1
                d_real = delta[real].mean(0)
                d_ai = delta[ai].mean(0)
                margin_rows.append(
                    {
                        "fold": fold,
                        "condition": cond,
                        "h2_delta_z_real": float(dz2[real].mean()),
                        "h2_delta_z_ai": float(dz2[ai].mean()),
                        "h3_delta_z_real": float(dz3[real].mean()),
                        "h3_delta_z_ai": float(dz3[ai].mean()),
                        "h2_cos_d_real": cos_vec(d_real, w2),
                        "h2_cos_d_ai": cos_vec(d_ai, w2),
                        "h3_cos_d_real": cos_vec(d_real, w3),
                        "h3_cos_d_ai": cos_vec(d_ai, w3),
                    }
                )

            # store predictions for this condition in long form later via wide row
            if cond == "CLEAN":
                p2_store = {"p_H2_CLEAN": p2, "p_H3_CLEAN": p3}
            else:
                p2_store[f"p_H2_{cond}"] = p2
                p2_store[f"p_H3_{cond}"] = p3

        for j in range(len(ev_ids)):
            row = {
                "fold": fold,
                "image_id": ev_ids[j],
                "y": int(y_ev[j]),
                "generator_id": gens[j],
                "real_domain": domains[j],
            }
            for cond in CONDITIONS:
                if cond == "CLEAN":
                    row["p_H2_CLEAN"] = float(p2_store["p_H2_CLEAN"][j])
                    row["p_H3_CLEAN"] = float(p2_store["p_H3_CLEAN"][j])
                else:
                    row[f"p_H2_{cond}"] = float(p2_store[f"p_H2_{cond}"][j])
                    row[f"p_H3_{cond}"] = float(p2_store[f"p_H3_{cond}"][j])
            pred_rows.append(row)

        # bootstrap CLEAN + each transform
        boot_folds[f"fold_{fold}"] = {}
        for cond in CONDITIONS:
            col2 = "p_H2_CLEAN" if cond == "CLEAN" else f"p_H2_{cond}"
            col3 = "p_H3_CLEAN" if cond == "CLEAN" else f"p_H3_{cond}"
            # get from last stored - rebuild from fold_metrics path
            if cond == "CLEAN":
                p2b, p3b = p2_store["p_H2_CLEAN"], p2_store["p_H3_CLEAN"]
            else:
                p2b, p3b = p2_store[f"p_H2_{cond}"], p2_store[f"p_H3_{cond}"]
            boot_folds[f"fold_{fold}"][cond] = paired_boot(y_ev, p3b, p2b)

        print(
            f"[V2-10F] Fold {fold} done CLEAN AUC H2/H3="
            f"{fold_metrics[fold]['CLEAN']['H2']['roc_auc']:.4f}/"
            f"{fold_metrics[fold]['CLEAN']['H3']['roc_auc']:.4f}",
            flush=True,
        )

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_OUT, index=False)

    # Macro metrics
    macro = {"H2": {}, "H3": {}}
    for cond in CONDITIONS:
        for h in ["H2", "H3"]:
            keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]
            macro[h][cond] = {
                k: float(np.mean([fold_metrics[f][cond][h][k] for f in [1, 2, 3, 4]])) for k in keys
            }
            metric_rows.append({"fold": "macro", "condition": cond, "head": h, **macro[h][cond]})
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)

    deltas_clean = {k: macro["H3"]["CLEAN"][k] - macro["H2"]["CLEAN"][k] for k in macro["H2"]["CLEAN"]}

    # Robustness summary
    rob_sum = {
        "H2_clean_auc": macro["H2"]["CLEAN"]["roc_auc"],
        "H3_clean_auc": macro["H3"]["CLEAN"]["roc_auc"],
        "H2_mean_tf_auc": float(np.mean([macro["H2"][c]["roc_auc"] for c in TRANSFORMS])),
        "H3_mean_tf_auc": float(np.mean([macro["H3"][c]["roc_auc"] for c in TRANSFORMS])),
        "H2_worst_tf_auc": float(np.min([macro["H2"][c]["roc_auc"] for c in TRANSFORMS])),
        "H3_worst_tf_auc": float(np.min([macro["H3"][c]["roc_auc"] for c in TRANSFORMS])),
        "H2_clean_ai_recall": macro["H2"]["CLEAN"]["ai_recall"],
        "H3_clean_ai_recall": macro["H3"]["CLEAN"]["ai_recall"],
        "H2_mean_tf_ai_recall": float(np.mean([macro["H2"][c]["ai_recall"] for c in TRANSFORMS])),
        "H3_mean_tf_ai_recall": float(np.mean([macro["H3"][c]["ai_recall"] for c in TRANSFORMS])),
        "H2_clean_real_spec": macro["H2"]["CLEAN"]["real_specificity"],
        "H3_clean_real_spec": macro["H3"]["CLEAN"]["real_specificity"],
        "H2_mean_tf_real_spec": float(np.mean([macro["H2"][c]["real_specificity"] for c in TRANSFORMS])),
        "H3_mean_tf_real_spec": float(np.mean([macro["H3"][c]["real_specificity"] for c in TRANSFORMS])),
        "H2_worst_tf_real_spec": float(np.min([macro["H2"][c]["real_specificity"] for c in TRANSFORMS])),
        "H3_worst_tf_real_spec": float(np.min([macro["H3"][c]["real_specificity"] for c in TRANSFORMS])),
        "H3_mean_tf_balacc": float(np.mean([macro["H3"][c]["balanced_accuracy"] for c in TRANSFORMS])),
        "H3_worst_tf_balacc_cond": min(TRANSFORMS, key=lambda c: macro["H3"][c]["balanced_accuracy"]),
    }

    # Margin macro
    margin_macro = {}
    for cond in TRANSFORMS:
        sub = [r for r in margin_rows if r["condition"] == cond]
        margin_macro[cond] = {
            k: float(np.mean([r[k] for r in sub]))
            for k in [
                "h2_delta_z_real",
                "h2_delta_z_ai",
                "h3_delta_z_real",
                "h3_delta_z_ai",
                "h2_cos_d_real",
                "h2_cos_d_ai",
                "h3_cos_d_real",
                "h3_cos_d_ai",
            ]
        }
    pd.DataFrame(margin_rows).to_csv(MARGIN_OUT, index=False)

    # Hard generators (pooled within-fold then macro-average where defined)
    hard_out = []
    for gid in HARD:
        row = {"generator_id": gid}
        ns = []
        for cond in CONDITIONS:
            for h, pref in [("H2", "p_H2"), ("H3", "p_H3")]:
                col = f"{pref}_CLEAN" if cond == "CLEAN" else f"{pref}_{cond}"
                recalls, means, meds, aps = [], [], [], []
                for fold in [1, 2, 3, 4]:
                    sub = pred_df[(pred_df["fold"] == fold) & (pred_df["generator_id"] == gid)]
                    if len(sub) == 0:
                        continue
                    if cond == "CLEAN" and h == "H2":
                        ns.append(len(sub))
                    p = sub[col].to_numpy(dtype=float)
                    y = sub["y"].to_numpy(dtype=int)
                    recalls.append(float((p >= THR).mean()))
                    means.append(float(np.mean(p)))
                    meds.append(float(np.median(p)))
                    if (y == 1).any() and (y == 0).any():
                        aps.append(float(average_precision_score(y, p)))
                    elif (y == 1).all():
                        # AI-only subgroup: AP undefined vs Real; keep mean-p focus
                        pass
                row[f"{h}_{cond}_recall"] = float(np.mean(recalls)) if recalls else float("nan")
                row[f"{h}_{cond}_mean_p"] = float(np.mean(means)) if means else float("nan")
                row[f"{h}_{cond}_median_p"] = float(np.mean(meds)) if meds else float("nan")
                row[f"{h}_{cond}_ap"] = float(np.mean(aps)) if aps else float("nan")
        row["n"] = int(sum(ns))
        hard_out.append(row)

    # Real domains
    real_out = []
    phone = {}
    mllm = {}
    for dom in REAL_DOMAINS:
        row = {"real_domain": dom}
        for cond in CONDITIONS:
            for h, pref in [("H2", "p_H2"), ("H3", "p_H3")]:
                col = f"{pref}_CLEAN" if cond == "CLEAN" else f"{pref}_{cond}"
                specs = []
                for fold in [1, 2, 3, 4]:
                    sub = pred_df[
                        (pred_df["fold"] == fold)
                        & (pred_df["y"] == 0)
                        & (pred_df["real_domain"] == dom)
                    ]
                    if len(sub) == 0:
                        continue
                    specs.append(float((sub[col].to_numpy() < THR).mean()))
                val = float(np.mean(specs)) if specs else float("nan")
                row[f"{h}_{cond}_specificity"] = val
                if h == "H3":
                    if dom == "Smartphone":
                        phone[cond] = val
                    if dom == "MLLM":
                        mllm[cond] = val
        real_out.append(row)

    # Fold3 MLLM
    fold3_mllm = {}
    sub = pred_df[(pred_df["fold"] == 3) & (pred_df["y"] == 0) & (pred_df["real_domain"] == "MLLM")]
    if len(sub):
        fold3_mllm["n"] = int(len(sub))
        for cond in CONDITIONS:
            for h, pref in [("H2", "p_H2"), ("H3", "p_H3")]:
                col = f"{pref}_CLEAN" if cond == "CLEAN" else f"{pref}_{cond}"
                fold3_mllm[f"{h}_{cond}"] = float((sub[col].to_numpy() < THR).mean())

    # Fold consistency: for how many folds does H3 improve balacc for majority of transforms?
    folds_with_majority = 0
    for fold in [1, 2, 3, 4]:
        improved = 0
        for cond in TRANSFORMS:
            if (
                fold_metrics[fold][cond]["H3"]["balanced_accuracy"]
                > fold_metrics[fold][cond]["H2"]["balanced_accuracy"]
            ):
                improved += 1
        if improved >= 3:  # majority of 4 transforms
            folds_with_majority += 1

    resources = {
        "h2_params": 513,
        "h3_params": 513,
        "h2_mean_serialized": float(
            np.mean(
                [
                    (MODELS / f"v2_10b_h2_balanced_logreg_fold{f}_v1.joblib").stat().st_size
                    for f in [1, 2, 3, 4]
                ]
            )
        ),
        "h3_mean_serialized": float(np.mean([h3_train_meta[f"fold_{f}"]["serialized_bytes"] for f in [1, 2, 3, 4]])),
        "h3_mean_train_s": float(np.mean([h3_train_meta[f"fold_{f}"]["wall_clock_s"] for f in [1, 2, 3, 4]])),
        "note": "Offline training uses 5× TRAIN R1 rows; deployment head remains ~513 params.",
    }
    pd.DataFrame(
        [
            {"head": "H2", "params": 513, "mean_serialized_bytes": resources["h2_mean_serialized"]},
            {
                "head": "H3",
                "params": 513,
                "mean_serialized_bytes": resources["h3_mean_serialized"],
                "mean_train_s": resources["h3_mean_train_s"],
            },
        ]
    ).to_csv(RESOURCE_OUT, index=False)

    decision, evidence = decide(
        {
            "h3_clean": macro["H3"]["CLEAN"],
            "h2_clean": macro["H2"]["CLEAN"],
            "h3_tf_macro": {c: macro["H3"][c] for c in TRANSFORMS},
            "h2_tf_macro": {c: macro["H2"][c] for c in TRANSFORMS},
            "phone": phone,
            "mllm": mllm,
            "fold_improve": folds_with_majority,
            "resources": resources,
        }
    )

    figures = make_figures(macro, hard_out, real_out, margin_macro)

    boot_payload = {
        "protocol": "fold_wise_paired_stratified_bootstrap_5000_seed42_H3_minus_H2",
        "pooled_forbidden": True,
        "folds": boot_folds,
        "macro_deltas_clean": deltas_clean,
        "macro_deltas_tf": {
            c: {k: macro["H3"][c][k] - macro["H2"][c][k] for k in macro["H2"][c]} for c in TRANSFORMS
        },
    }
    BOOT_OUT.write_text(json.dumps(_clean(boot_payload), indent=2))

    mean_cos_w = float(np.mean([w["cos_w"] for w in weight_comp]))

    if decision == "ROBUST_HEAD_PROMISING":
        next_rec = (
            "Authorize V2 DEVELOPMENT FREEZE packaging for H3 as primary robust development "
            "candidate (still not FINAL_V2 without tutor approval). Optional: selective/calibration "
            "diagnostic on H3. Do not auto-start. Do not access NTIRE."
        )
    elif decision == "ROBUST_HEAD_MIXED":
        next_rec = (
            "Keep H3 as promising robust-head candidate with documented residual failures; "
            "next may be one tightly scoped residual diagnostic (e.g. remaining collapse transform) "
            "or development freeze with caveats — tutor chooses. Do not auto-start LoRA training."
        )
    else:
        next_rec = (
            "H3 did not justify robust-head rescue; revisit V2-10E assumptions or authorize a "
            "different tightly scoped intervention (not a head search). Do not auto-start."
        )

    interpretation = {
        "q1_retain_ranking": f"H3 clean AUC={macro['H3']['CLEAN']['roc_auc']:.4f} vs H2={macro['H2']['CLEAN']['roc_auc']:.4f} (Δ={deltas_clean['roc_auc']:+.4f}).",
        "q2_clean_cost": f"ΔAI_rec={deltas_clean['ai_recall']:+.4f} ΔRealSpec={deltas_clean['real_specificity']:+.4f} ΔBalAcc={deltas_clean['balanced_accuracy']:+.4f}.",
        "q3_jpeg_rescue": f"JPEG AI_rec H2={macro['H2']['jpeg_q50']['ai_recall']:.3f} H3={macro['H3']['jpeg_q50']['ai_recall']:.3f}.",
        "q4_resize_rescue": f"resize RealSpec H2={macro['H2']['resize_112']['real_specificity']:.3f} H3={macro['H3']['resize_112']['real_specificity']:.3f}.",
        "q5_blur_rescue": f"blur RealSpec H2={macro['H2']['blur_sigma2']['real_specificity']:.3f} H3={macro['H3']['blur_sigma2']['real_specificity']:.3f}.",
        "q6_screenshot": f"screenshot AUC H2/H3={macro['H2']['screenshot_strong']['roc_auc']:.3f}/{macro['H3']['screenshot_strong']['roc_auc']:.3f}.",
        "q7_mean_tf_auc": f"H2={rob_sum['H2_mean_tf_auc']:.4f} H3={rob_sum['H3_mean_tf_auc']:.4f}.",
        "q8_worst_tf_auc": f"H2={rob_sum['H2_worst_tf_auc']:.4f} H3={rob_sum['H3_worst_tf_auc']:.4f}.",
        "q9_margin_projection": f"See margin_macro; mean |H3 Real Δz| vs H2 for resize/blur.",
        "q10_direction_rotate": f"mean cos(w_H2,w_H3)={mean_cos_w:.4f}.",
        "q11_hard_gens": "See hard_generators table.",
        "q12_flux": next(h for h in hard_out if "FLUX" in h["generator_id"]),
        "q13_real_domains": "See real_domains table.",
        "q14_fold3": fold3_mllm,
        "q15_validates_head_limited": f"decision={decision}; if H3 improves robustness with frozen R1, validates ROBUSTNESS_HEAD_LIMITED.",
        "q16_freeze_ready": "YES" if decision == "ROBUST_HEAD_PROMISING" else "NO",
        "q17_next": next_rec,
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
        "H3_HEAD_TRAINING_PERFORMED": "YES",
        "ONLY_PREDECLARED_H3_TRAINED": "YES",
        "HYPERPARAMETER_SEARCH_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "EVAL_LABELS_USED_FOR_TRAINING": "NO",
        "V2_9C_CALIBRATION_DATA_USED_FOR_TRAINING": "NO",
        "PROMPT_BLOCKED_USED_FOR_TRAINING": "NO",
        "BINARY_THRESHOLD_CHANGED": "NO",
        "THRESHOLD_SWEEP_PERFORMED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_TRANSFORMED_TRAIN_FEATURE_INFERENCE_PERFORMED": "YES"
        if feat_meta.get("performed")
        else "NO",
    }

    payload = {
        "stage": "V2-10F",
        "status": "COMPLETE",
        "decision": decision,
        "decision_evidence": evidence,
        "train_feature_inference": feat_meta,
        "integrity": integrity,
        "h3_train": h3_train_meta,
        "macro": macro,
        "deltas_clean": deltas_clean,
        "robustness_summary": rob_sum,
        "weight_comparison": {"per_fold": weight_comp, "mean_cos_w": mean_cos_w},
        "margin_macro": margin_macro,
        "hard_generators": hard_out,
        "real_domains": real_out,
        "fold3_mllm": fold3_mllm,
        "fold_consistency": {"folds_with_majority_transform_balacc_improve": folds_with_majority},
        "bootstrap": boot_payload,
        "resources": resources,
        "figures": figures,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "score_distributions_note": "per-fold in analysis JSON under score_dists omitted for size; see predictions CSV",
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
