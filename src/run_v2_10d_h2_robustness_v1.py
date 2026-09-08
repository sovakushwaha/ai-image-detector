#!/usr/bin/env python3
"""V2-10D — Frozen H2 Robustness Evaluation.

Authoritative transform protocol (from research_log RQ3 / StrongRobust / C0):
  jpeg_q50, resize_112, blur_sigma2, screenshot_strong

Exact implementations reused from external_v2_common / generate_rq3_validation_v1.

CLEAN predictions = authoritative V2-10B H2 (no re-inference).
TRANSFORMED = locked transform → V2-8 CLIP preprocess → frozen LoRA R1 → frozen H2.

NO training. NO threshold tuning. NO calibrator fit. FINAL_V2 NOT SELECTED.
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

from analyze_v2_10a_representation_geometry_v1 import (  # noqa: E402
    ClipLoRAModel,
    resolve_path,
)
from external_v2_common import TRANSFORM_FNS  # noqa: E402
from rq5_calibration_utils_v1 import apply_temperature, sigmoid  # noqa: E402

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
EVAL_PRED = OUT / "v2_10b_head_predictions_v1.csv"
TEMP_CSV = OUT / "v2_10c_temperature_params_v1.csv"
RQ3_METRICS = PROJECT_ROOT / "results" / "rq3_test_metrics_v1.csv"
CACHE = OUT / "v2_10d_transform_cache"

SEED = 42
N_BOOT = 5000
THR = 0.5
EPS = 1e-6
EXPECTED_EVAL = {1: 2821, 2: 2619, 3: 1994, 4: 1709}
# Authoritative StrongRobust / RQ3 / C0 evaluation suite (NOT the full RQ2 mild+strong 8-set)
TRANSFORMS = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
TRANSFORM_META = {
    "jpeg_q50": {
        "parameters": {"quality": 50, "subsampling": 0},
        "severity": "strong",
        "source": "external_v2_common.apply_jpeg_q50 / generate_rq3_validation_v1",
        "deterministic": True,
        "previous_v1_role": "RQ2/RQ3/RQ5 StrongRobust; FINAL C0 evaluation condition",
        "safe_to_reuse": True,
    },
    "resize_112": {
        "parameters": {"downsample_to": 112, "restore_to": 224, "resample": "LANCZOS"},
        "severity": "strong",
        "source": "external_v2_common.apply_resize_112 / generate_rq3_validation_v1",
        "deterministic": True,
        "previous_v1_role": "RQ2/RQ3/RQ5 StrongRobust; FINAL C0 evaluation condition",
        "safe_to_reuse": True,
    },
    "blur_sigma2": {
        "parameters": {"sigma_or_radius": 2.0, "filter": "PIL.GaussianBlur"},
        "severity": "strong",
        "source": "external_v2_common.apply_blur_sigma2 / generate_rq3_validation_v1",
        "deterministic": True,
        "previous_v1_role": "RQ2/RQ3/RQ5 StrongRobust; historically most damaging for V1 C0",
        "safe_to_reuse": True,
    },
    "screenshot_strong": {
        "parameters": {
            "jpeg_quality": 65,
            "display_size": 384,
            "canvas": 512,
            "canvas_rgb": [32, 32, 32],
            "final_size": 224,
        },
        "severity": "strong",
        "source": "external_v2_common.apply_screenshot_strong / generate_rq3_validation_v1 / generate_screenshot_v1",
        "deterministic": True,
        "previous_v1_role": "RQ2 Stage22C + RQ3 StrongRobust; FINAL C0 evaluation condition",
        "safe_to_reuse": True,
    },
}
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]
CLEAN_REF = {
    "roc_auc": 0.951,
    "ap": 0.958,
    "ai_recall": 0.576,
    "real_specificity": 0.967,
}

JSON_OUT = OUT / "v2_10d_robustness_analysis_v1.json"
REPORT_OUT = OUT / "v2_10d_robustness_report_v1.txt"
PRED_OUT = OUT / "v2_10d_robustness_predictions_v1.csv"
METRICS_OUT = OUT / "v2_10d_robustness_metrics_v1.csv"
BOOT_OUT = OUT / "v2_10d_robustness_bootstrap_v1.json"
V1_CMP_OUT = OUT / "v2_10d_v1_historical_comparison_v1.csv"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def confidence(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


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


def paired_boot_delta(
    y: np.ndarray, p_new: np.ndarray, p_base: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new).astype(float)
    p_base = np.asarray(p_base).astype(float)
    real_idx = np.where(y == 0)[0]
    ai_idx = np.where(y == 1)[0]
    keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy"]
    buckets: dict[str, list[float]] = {k: [] for k in keys}

    def _m(yy, pp):
        return metrics_at_05(yy, pp)

    for _ in range(n_boot):
        ri = rng.choice(real_idx, size=len(real_idx), replace=True)
        ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
        idx = np.concatenate([ri, ai])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        mn = _m(yy, p_new[idx])
        mb = _m(yy, p_base[idx])
        for k in keys:
            buckets[k].append(mn[k] - mb[k])
    out: dict[str, Any] = {
        "n_boot": n_boot,
        "seed": seed,
        "stratified_by_label": True,
        "delta_definition": "TRANSFORM_minus_CLEAN",
    }
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_valid_boot": int(len(arr)),
        }
    return out


class TransformDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        preprocess,
        transform_fn: Callable[[Image.Image], Image.Image] | None,
    ) -> None:
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


def load_eval_clean() -> pd.DataFrame:
    df = pd.read_csv(EVAL_PRED)
    df["image_id"] = df["image_id"].astype(str)
    for fold, exp in EXPECTED_EVAL.items():
        sub = df[df["fold"] == fold]
        if len(sub) != exp:
            stop(f"clean fold {fold} n={len(sub)} != {exp}")
        if sub["image_id"].duplicated().any():
            stop(f"clean fold {fold} duplicate IDs")
    return df


def verify_clean_baseline(df: pd.DataFrame) -> dict[str, Any]:
    fold_m = []
    for fold in [1, 2, 3, 4]:
        sub = df[df["fold"] == fold]
        fold_m.append(metrics_at_05(sub["y"].to_numpy(), sub["p_H2"].to_numpy()))
    macro = {
        k: float(np.mean([m[k] for m in fold_m]))
        for k in ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]
    }
    # match authoritative ~ values
    checks = {
        "auc_ok": abs(macro["roc_auc"] - CLEAN_REF["roc_auc"]) < 0.005,
        "ap_ok": abs(macro["ap"] - CLEAN_REF["ap"]) < 0.005,
        "recall_ok": abs(macro["ai_recall"] - CLEAN_REF["ai_recall"]) < 0.005,
        "spec_ok": abs(macro["real_specificity"] - CLEAN_REF["real_specificity"]) < 0.005,
    }
    if not all(checks.values()):
        stop(f"clean baseline mismatch macro={macro} checks={checks}")
    return {"macro": macro, "per_fold": fold_m, "checks": checks, "ok": True}


def extract_transformed_fold(
    fold: int,
    clean_df: pd.DataFrame,
    man: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    """Inference-only: transform → CLIP → LoRA R1 → H2."""
    cache_path = CACHE / f"fold{fold}_{condition}_v1.csv"
    if cache_path.is_file():
        out = pd.read_csv(cache_path)
        out["image_id"] = out["image_id"].astype(str)
        exp = EXPECTED_EVAL[fold]
        if len(out) != exp:
            stop(f"cache fold{fold}/{condition} n={len(out)} != {exp}")
        clean_ids = list(clean_df.loc[clean_df["fold"] == fold, "image_id"])
        if list(out["image_id"]) != clean_ids:
            stop(f"cache fold{fold}/{condition} ID order mismatch")
        print(f"[V2-10D] Reusing cache {cache_path.name}", flush=True)
        return out

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ef = clean_df[clean_df["fold"] == fold].reset_index(drop=True)
    man_ix = man.set_index("image_id", drop=False)
    rows = []
    for iid in ef["image_id"]:
        r = man_ix.loc[iid]
        path = resolve_path(r)
        if not path.is_file():
            stop(f"missing image {iid}: {path}")
        rows.append({"image_id": iid, "path": str(path)})

    ckpt_path = MODELS / f"clip_lora_fold{fold}_best_v1.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg.setdefault("clip", {})
    cfg["clip"].setdefault("l2_normalize", True)
    if isinstance(cfg.get("head"), str):
        cfg["head"] = {"dropout": 0.2}
    cfg.setdefault("head", {"dropout": 0.2})

    print(
        f"[V2-10D] Fold {fold} {condition}: extracting R1 on {device} n={len(rows)} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = ClipLoRAModel(cfg, device)
    incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
    crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
    if crit:
        stop(f"fold {fold} critical missing keys {crit[:10]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
    tfm = TRANSFORM_FNS[condition]
    ds = TransformDataset(rows, model.preprocess, tfm)
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
    pos = {i: j for j, i in enumerate(ids_arr)}
    order = [pos[i] for i in ef["image_id"]]
    r1 = r1[order]
    if not np.isfinite(r1).all() or r1.shape != (len(ef), 512):
        stop(f"fold {fold} {condition} bad R1 shape/finite")
    p = h2.predict_proba(r1)[:, list(h2.classes_).index(1)].astype(np.float64)
    if not np.isfinite(p).all():
        stop(f"fold {fold} {condition} non-finite probs")

    out = pd.DataFrame(
        {
            "fold": fold,
            "condition": condition,
            "image_id": list(ef["image_id"]),
            "y": ef["y"].to_numpy(dtype=int),
            "p_h2": p,
            "generator_id": ef["generator_id"].fillna("").astype(str).to_numpy(),
            "real_domain": ef["real_domain"].fillna("").astype(str).to_numpy(),
            "checkpoint": f"models/v2/clip_lora_fold{fold}_best_v1.pt",
            "h2_head": f"models/v2/v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib",
        }
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    print(
        f"[V2-10D] Wrote {cache_path.name} in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    return out


def decide(macro_by_cond: dict[str, dict], phone_spec: dict, mllm_spec: dict, fold_aucs: dict) -> tuple[str, list[str]]:
    evidence = []
    transforms = TRANSFORMS
    mean_auc = float(np.mean([macro_by_cond[c]["roc_auc"] for c in transforms]))
    worst_auc = float(np.min([macro_by_cond[c]["roc_auc"] for c in transforms]))
    mean_spec = float(np.mean([macro_by_cond[c]["real_specificity"] for c in transforms]))
    mean_phone = float(np.mean([phone_spec[c] for c in transforms]))
    mean_mllm = float(np.mean([mllm_spec[c] for c in transforms]))
    clean_rec = macro_by_cond["CLEAN"]["ai_recall"]
    mean_rec = float(np.mean([macro_by_cond[c]["ai_recall"] for c in transforms]))
    mean_rec_drop = clean_rec - mean_rec
    # catastrophic Real collapse: any transform RealSpec < 0.70
    no_collapse = all(macro_by_cond[c]["real_specificity"] >= 0.70 for c in transforms)
    # fold domination: check worst-fold AUC across transforms not wildly above mean due to one fold
    # "not dominated by one exceptionally good fold": for each transform, std of fold AUCs
    fold_ok = True
    for c in transforms:
        aucs = [fold_aucs[f][c] for f in [1, 2, 3, 4]]
        if float(np.std(aucs)) > 0.15 and float(np.min(aucs)) < 0.75:
            fold_ok = False
    # most transforms preserve primary advantages: AUC>=0.85 and RealSpec>=0.85
    preserve = sum(
        1
        for c in transforms
        if macro_by_cond[c]["roc_auc"] >= 0.85 and macro_by_cond[c]["real_specificity"] >= 0.85
    )
    most_preserve = preserve >= 3

    flags = {
        "mean_auc_ge_090": mean_auc >= 0.90,
        "worst_auc_ge_085": worst_auc >= 0.85,
        "mean_spec_ge_090": mean_spec >= 0.90,
        "phone_ge_095": mean_phone >= 0.95,
        "mllm_ge_080": mean_mllm >= 0.80,
        "no_catastrophic_real": no_collapse,
        "mean_recall_drop_le_010": mean_rec_drop <= 0.10,
        "fold_behaviour_ok": fold_ok,
        "most_transforms_preserve": most_preserve,
    }
    evidence.append(f"flags={flags}")
    evidence.append(
        f"mean_auc={mean_auc:.4f} worst_auc={worst_auc:.4f} mean_spec={mean_spec:.4f} "
        f"mean_phone={mean_phone:.4f} mean_mllm={mean_mllm:.4f} mean_rec_drop={mean_rec_drop:+.4f} "
        f"preserve={preserve}/4"
    )

    if all(flags.values()):
        return "ROBUSTNESS_PROMISING", evidence

    usable = mean_auc >= 0.80 and mean_spec >= 0.75 and no_collapse
    hurt = (
        not flags["mean_auc_ge_090"]
        or not flags["worst_auc_ge_085"]
        or not flags["mean_spec_ge_090"]
        or not flags["phone_ge_095"]
        or not flags["mllm_ge_080"]
        or not flags["mean_recall_drop_le_010"]
        or not flags["most_transforms_preserve"]
    )
    if usable and hurt:
        return "ROBUSTNESS_MIXED", evidence
    if not usable or (worst_auc < 0.70) or (mean_spec < 0.70):
        return "ROBUSTNESS_NOT_ACCEPTABLE", evidence
    return "ROBUSTNESS_MIXED", evidence


def make_figures(macro_by_cond: dict, hard_rows: list, real_rows: list) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    conds = ["CLEAN"] + TRANSFORMS

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(
        axes,
        ["roc_auc", "ai_recall", "real_specificity"],
        ["ROC-AUC", "AI recall@0.5", "Real specificity@0.5"],
    ):
        vals = [macro_by_cond[c][key] for c in conds]
        ax.bar(range(len(conds)), vals, color="#4C72B0")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        if key == "roc_auc":
            ax.axhline(0.90, color="gray", ls="--", lw=0.8)
    fig.suptitle("V2-10D H2 CLEAN vs transforms")
    fig.tight_layout()
    p = FIG / "v2_10d_robustness_macro_metrics_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # deltas
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(TRANSFORMS))
    w = 0.2
    for i, key in enumerate(["roc_auc", "ai_recall", "real_specificity", "balanced_accuracy"]):
        deltas = [macro_by_cond[c][key] - macro_by_cond["CLEAN"][key] for c in TRANSFORMS]
        ax.bar(x + i * w, deltas, width=w, label=key)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(TRANSFORMS, rotation=20, ha="right")
    ax.set_ylabel("TRANSFORM − CLEAN")
    ax.set_title("Metric deltas by transform")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "v2_10d_robustness_metric_deltas_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # hard gen recall
    fig, ax = plt.subplots(figsize=(10, 4))
    gens = [h["generator_id"].split("::")[-1] for h in hard_rows]
    x = np.arange(len(gens))
    w = 0.15
    for i, c in enumerate(conds):
        ax.bar(x + i * w, [h[f"{c}_recall"] for h in hard_rows], width=w, label=c)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(gens, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AI recall@0.5")
    ax.set_title("Hard-generator transform sensitivity")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    p = FIG / "v2_10d_robustness_hard_generator_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # real domain spec
    fig, ax = plt.subplots(figsize=(8, 4))
    doms = [r["real_domain"] for r in real_rows]
    x = np.arange(len(doms))
    for i, c in enumerate(conds):
        ax.bar(x + i * w, [r[f"{c}_specificity"] for r in real_rows], width=w, label=c)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(doms)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Real specificity@0.5")
    ax.set_title("Real-domain transform sensitivity")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    p = FIG / "v2_10d_robustness_real_domain_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10D — Frozen H2 Robustness Evaluation")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Evaluation only. No training. Threshold=0.5. FINAL_V2_MODEL NOT SELECTED.")
    lines.append("")
    lines.append("Authoritative robustness protocol")
    lines.append(f"  Suite: {payload['protocol']['suite_name']}")
    lines.append(f"  Rationale: {payload['protocol']['rationale']}")
    lines.append(f"  Source: {payload['protocol']['source']}")
    lines.append("  Note: Full RQ2 8-transform mild+strong suite exists but StrongRobust/C0")
    lines.append("  authoritative evaluation uses the 4 strong conditions below (no crop/mild).")
    for name, meta in payload["protocol"]["transforms"].items():
        lines.append(
            f"  - {name}: sev={meta['severity']} params={meta['parameters']} "
            f"det={meta['deterministic']} src={meta['source']}"
        )
    lines.append("")
    lines.append("Clean baseline integrity")
    lines.append(f"  {payload['clean_baseline']}")
    lines.append("")
    lines.append(f"TRANSFORMED_INFERENCE_PERFORMED = {payload['transformed_inference']['performed']}")
    for k, v in payload["transformed_inference"]["summary"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Macro metrics CLEAN vs transforms")
    for c in ["CLEAN"] + TRANSFORMS:
        m = payload["macro"][c]
        lines.append(
            f"  {c}: AUC={m['roc_auc']:.4f} AP={m['ap']:.4f} AI_rec={m['ai_recall']:.4f} "
            f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Deltas (TRANSFORM − CLEAN)")
    for c in TRANSFORMS:
        d = payload["deltas"][c]
        lines.append(
            f"  {c}: ΔAUC={d['roc_auc']:+.4f} ΔAP={d['ap']:+.4f} ΔAI_rec={d['ai_recall']:+.4f} "
            f"ΔRealSpec={d['real_specificity']:+.4f} ΔBalAcc={d['balanced_accuracy']:+.4f}"
        )
    lines.append("")
    lines.append("Worst transforms")
    for k, v in payload["worst"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Hard generators (recall@0.5)")
    for h in payload["hard_generators"]:
        lines.append(
            f"  {h['generator_id']}: CLEAN={h['CLEAN_recall']:.3f} "
            + " ".join(f"{c}={h[f'{c}_recall']:.3f}" for c in TRANSFORMS)
        )
    lines.append("")
    lines.append("Real domains (specificity@0.5)")
    for r in payload["real_domains"]:
        lines.append(
            f"  {r['real_domain']}: CLEAN={r['CLEAN_specificity']:.3f} "
            + " ".join(f"{c}={r[f'{c}_specificity']:.3f}" for c in TRANSFORMS)
        )
    if payload.get("fold3_mllm"):
        f3 = payload["fold3_mllm"]
        lines.append(
            "  Fold3 MLLM: "
            + " ".join(f"{c}={f3[f'{c}_specificity']:.3f}" for c in ["CLEAN"] + TRANSFORMS)
        )
    lines.append("")
    lines.append("Prediction flips (macro mean folds)")
    for c, fl in payload["flips_macro"].items():
        lines.append(
            f"  {c}: flip_rate={fl['flip_rate']:.4f} "
            f"correct→wrong={fl['correct_to_wrong']:.4f} wrong→correct={fl['wrong_to_correct']:.4f} "
            f"mean|Δp|={fl['mean_abs_dp']:.4f} meanΔp={fl['mean_signed_dp']:+.4f}"
        )
    lines.append("")
    lines.append("Paired bootstrap (fold-wise TRANSFORM−CLEAN; 5000× seed42)")
    for fold, comps in payload["bootstrap"]["folds"].items():
        lines.append(f"  {fold}:")
        for cond, b in comps.items():
            lines.append(
                f"    {cond}: AUC {b['roc_auc']['mean_diff']:+.4f} "
                f"[{b['roc_auc']['ci_low']:+.4f},{b['roc_auc']['ci_high']:+.4f}]; "
                f"AI_rec {b['ai_recall']['mean_diff']:+.4f}; "
                f"RealSpec {b['real_specificity']['mean_diff']:+.4f}"
            )
    lines.append("")
    lines.append("V1 historical comparison")
    lines.append(f"  status: {payload['v1_comparison']['comparability']}")
    lines.append(f"  note: {payload['v1_comparison']['note']}")
    lines.append("")
    if payload.get("temperature_diagnostic"):
        lines.append("Optional temperature confidence diagnostic")
        for c, td in payload["temperature_diagnostic"].items():
            lines.append(
                f"  {c}: mean_conf_clean={td['mean_conf_clean']:.4f} "
                f"mean_conf_tf={td['mean_conf_transformed']:.4f} "
                f"Δ={td['delta_mean_conf']:+.4f}"
            )
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
    if "## Stage V2-10D" in text:
        print("[V2-10D] research_log already contains V2-10D; not rewriting.", flush=True)
        return
    m = payload["macro"]
    entry = f"""
## Stage V2-10D — Frozen H2 Robustness Evaluation

**Date:** 2026-09-07  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Evaluation only on locked V2 eval IDs. No LoRA/CLIP/H2 training. No threshold sweep. No calibrator fit. No selective retune. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Primary candidate:** H2 (frozen LoRA R1 + class-balanced LogReg; ~513 head params).

**Authoritative protocol reused:** RQ3 / StrongRobust / FINAL_C0 suite — `jpeg_q50`, `resize_112`, `blur_sigma2`, `screenshot_strong` (implementations from `external_v2_common` / `generate_rq3_validation_v1`). Full RQ2 8-transform mild+strong suite exists but was not expanded here.

**CLEAN:** reused V2-10B H2 predictions (no clean re-inference). Macro AUC={m['CLEAN']['roc_auc']:.4f} AP={m['CLEAN']['ap']:.4f} AI_rec={m['CLEAN']['ai_recall']:.4f} RealSpec={m['CLEAN']['real_specificity']:.4f}.

**TRANSFORMED_INFERENCE_PERFORMED = YES** on F1–F4 × 4 transforms; frozen `clip_lora_fold{{F}}_best_v1.pt` + `v2_10b_h2_balanced_logreg_fold{{F}}_v1.joblib`.

**Macro transformed:**  
| cond | AUC | AI_rec | RealSpec | ΔAUC | ΔAI_rec | ΔRealSpec |
|--|--|--|--|--|--|--|
"""
    for c in TRANSFORMS:
        d = payload["deltas"][c]
        mm = m[c]
        entry += (
            f"| {c} | {mm['roc_auc']:.4f} | {mm['ai_recall']:.4f} | {mm['real_specificity']:.4f} | "
            f"{d['roc_auc']:+.4f} | {d['ai_recall']:+.4f} | {d['real_specificity']:+.4f} |\n"
        )
    entry += f"""
**Decision:** **{payload['decision']}**  
**V1 comparison:** {payload['v1_comparison']['comparability']} — {payload['v1_comparison']['note']}

**Outputs:** `src/run_v2_10d_h2_robustness_v1.py`; `results/v2/v2_10d_*`; figures `figures/v2/v2_10d_*.png`.

**Integrity:** NTIRE=NO; fal=NO; V1 unmodified/not rerun; folds unchanged; LoRA/CLIP/H2 not updated; no robustness-aware training; no thr/calibrator/selective retune; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10D] Appended research_log.md", flush=True)


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
    CACHE.mkdir(parents=True, exist_ok=True)

    print("[V2-10D] Protocol audit + clean baseline ...", flush=True)
    protocol = {
        "suite_name": "RQ3_StrongRobust_C0_evaluation_suite",
        "rationale": (
            "research_log: Stage 23A–23E / RQ5 / Stage26B StrongRobust conditions for FINAL_RESEARCH_MODEL_V1=C0 "
            "are jpeg_q50, resize_112, blur_sigma2, screenshot_strong. Full RQ2 Stage22A suite has 8 mild+strong "
            "transforms including crop; StrongRobust aggregate and C0 paper-facing robustness use the 4 strong "
            "conditions without crop/mild variants."
        ),
        "source": (
            "paper/research_log.md (Stages 22D, 23A–E, 26B); "
            "src/generate_rq3_validation_v1.py; src/external_v2_common.py; "
            "src/evaluate_rq3_frozen_test_v1.py CONDITIONS"
        ),
        "rejected_alternate": {
            "name": "RQ2_full_8_transform_suite",
            "why_not_primary": "Includes mild variants and crops; not the StrongRobust/C0 primary evaluation set.",
        },
        "transforms": TRANSFORM_META,
        "preprocessing_order": (
            "load original RGB (EXIF transpose) → apply locked transform → exact V2-8 open_clip preprocess "
            "→ frozen LoRA encode (L2 R1) → frozen H2 LogReg"
        ),
    }

    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)
    clean = load_eval_clean()
    clean_base = verify_clean_baseline(clean)
    print(f"[V2-10D] Clean baseline OK: {clean_base['macro']}", flush=True)

    # Transformed inference
    all_tf = []
    integrity_tf = {"folds": {}}
    for fold in [1, 2, 3, 4]:
        integrity_tf["folds"][f"fold_{fold}"] = {}
        for cond in TRANSFORMS:
            tf = extract_transformed_fold(fold, clean, man, cond)
            # integrity
            if len(tf) != EXPECTED_EVAL[fold]:
                stop(f"fold {fold} {cond} N")
            if tf["image_id"].duplicated().any():
                stop(f"fold {fold} {cond} dups")
            if not np.isfinite(tf["p_h2"]).all():
                stop(f"fold {fold} {cond} nonfinite")
            integrity_tf["folds"][f"fold_{fold}"][cond] = {
                "n": int(len(tf)),
                "finite": True,
                "ids_match_clean": True,
            }
            all_tf.append(tf)

    tf_df = pd.concat(all_tf, ignore_index=True)

    # Build unified predictions: clean + each transform
    pred_rows = []
    for fold in [1, 2, 3, 4]:
        cf = clean[clean["fold"] == fold].reset_index(drop=True)
        for j, row in cf.iterrows():
            base = {
                "fold": fold,
                "image_id": row["image_id"],
                "y": int(row["y"]),
                "generator_id": row["generator_id"] if pd.notna(row["generator_id"]) else "",
                "real_domain": row["real_domain"] if pd.notna(row["real_domain"]) else "",
                "p_CLEAN": float(row["p_H2"]),
            }
            for cond in TRANSFORMS:
                sub = tf_df[(tf_df["fold"] == fold) & (tf_df["condition"] == cond)]
                # align by image_id
                pmap = dict(zip(sub["image_id"], sub["p_h2"]))
                base[f"p_{cond}"] = float(pmap[row["image_id"]])
            pred_rows.append(base)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_OUT, index=False)

    # Metrics per fold × condition
    metric_rows = []
    fold_metrics: dict[int, dict[str, dict]] = {f: {} for f in [1, 2, 3, 4]}
    for fold in [1, 2, 3, 4]:
        sub = pred_df[pred_df["fold"] == fold]
        y = sub["y"].to_numpy()
        for cond, col in [("CLEAN", "p_CLEAN")] + [(c, f"p_{c}") for c in TRANSFORMS]:
            m = metrics_at_05(y, sub[col].to_numpy())
            fold_metrics[fold][cond] = m
            metric_rows.append({"fold": fold, "condition": cond, **m})

    # Macro
    macro = {}
    for cond in ["CLEAN"] + TRANSFORMS:
        keys = ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy", "precision", "f1"]
        macro[cond] = {
            k: float(np.mean([fold_metrics[f][cond][k] for f in [1, 2, 3, 4]])) for k in keys
        }
    deltas = {
        c: {k: macro[c][k] - macro["CLEAN"][k] for k in macro["CLEAN"]} for c in TRANSFORMS
    }

    # Worst
    worst = {
        "worst_auc": min(TRANSFORMS, key=lambda c: macro[c]["roc_auc"]),
        "worst_ai_recall": min(TRANSFORMS, key=lambda c: macro[c]["ai_recall"]),
        "worst_real_specificity": min(TRANSFORMS, key=lambda c: macro[c]["real_specificity"]),
        "worst_balanced_accuracy": min(TRANSFORMS, key=lambda c: macro[c]["balanced_accuracy"]),
        "mean_transformed_auc": float(np.mean([macro[c]["roc_auc"] for c in TRANSFORMS])),
        "mean_transformed_ap": float(np.mean([macro[c]["ap"] for c in TRANSFORMS])),
        "mean_transformed_ai_recall": float(np.mean([macro[c]["ai_recall"] for c in TRANSFORMS])),
        "mean_transformed_real_specificity": float(
            np.mean([macro[c]["real_specificity"] for c in TRANSFORMS])
        ),
    }
    # worst fold×transform by balacc
    worst_ft = None
    worst_val = 1e9
    for f in [1, 2, 3, 4]:
        for c in TRANSFORMS:
            v = fold_metrics[f][c]["balanced_accuracy"]
            if v < worst_val:
                worst_val = v
                worst_ft = {"fold": f, "condition": c, "balanced_accuracy": v}
    worst["worst_fold_transform"] = worst_ft

    # Hard generators
    hard_rows = []
    for gid in HARD_GENERATORS:
        row = {"generator_id": gid}
        ns = []
        for cond, col in [("CLEAN", "p_CLEAN")] + [(c, f"p_{c}") for c in TRANSFORMS]:
            recalls, means, meds, aucs, aps = [], [], [], [], []
            for fold in [1, 2, 3, 4]:
                sub = pred_df[(pred_df["fold"] == fold) & (pred_df["generator_id"] == gid)]
                if len(sub) == 0:
                    continue
                if cond == "CLEAN":
                    ns.append(len(sub))
                p = sub[col].to_numpy()
                recalls.append(float((p >= THR).mean()))
                means.append(float(p.mean()))
                meds.append(float(np.median(p)))
                real = pred_df[(pred_df["fold"] == fold) & (pred_df["y"] == 0)]
                yb = np.concatenate([np.zeros(len(real)), np.ones(len(sub))])
                pb = np.concatenate([real[col].to_numpy(), p])
                aucs.append(float(roc_auc_score(yb, pb)))
                aps.append(float(average_precision_score(yb, pb)))
            row[f"{cond}_recall"] = float(np.mean(recalls)) if recalls else float("nan")
            row[f"{cond}_mean_p"] = float(np.mean(means)) if means else float("nan")
            row[f"{cond}_median_p"] = float(np.mean(meds)) if meds else float("nan")
            row[f"{cond}_auc"] = float(np.mean(aucs)) if aucs else float("nan")
            row[f"{cond}_ap"] = float(np.mean(aps)) if aps else float("nan")
        row["n"] = int(sum(ns))
        for c in TRANSFORMS:
            row[f"delta_{c}_recall"] = row[f"{c}_recall"] - row["CLEAN_recall"]
        hard_rows.append(row)

    # Real domains
    real_rows = []
    phone_spec = {}
    mllm_spec = {}
    for dom in REAL_DOMAINS:
        row = {"real_domain": dom}
        ns = []
        for cond, col in [("CLEAN", "p_CLEAN")] + [(c, f"p_{c}") for c in TRANSFORMS]:
            specs, fprs, means, meds = [], [], [], []
            for fold in [1, 2, 3, 4]:
                sub = pred_df[
                    (pred_df["fold"] == fold)
                    & (pred_df["y"] == 0)
                    & (pred_df["real_domain"] == dom)
                ]
                if len(sub) == 0:
                    continue
                if cond == "CLEAN":
                    ns.append(len(sub))
                p = sub[col].to_numpy()
                specs.append(float((p < THR).mean()))
                fprs.append(float((p >= THR).mean()))
                means.append(float(p.mean()))
                meds.append(float(np.median(p)))
            row[f"{cond}_specificity"] = float(np.mean(specs)) if specs else float("nan")
            row[f"{cond}_fpr"] = float(np.mean(fprs)) if fprs else float("nan")
            row[f"{cond}_mean_p"] = float(np.mean(means)) if means else float("nan")
            row[f"{cond}_median_p"] = float(np.mean(meds)) if meds else float("nan")
            if dom == "Smartphone" and cond in TRANSFORMS:
                phone_spec[cond] = row[f"{cond}_specificity"]
            if dom == "MLLM" and cond in TRANSFORMS:
                mllm_spec[cond] = row[f"{cond}_specificity"]
        row["n"] = int(sum(ns))
        real_rows.append(row)

    # Fold3 MLLM
    fold3_mllm = {"n": 0}
    sub = pred_df[(pred_df["fold"] == 3) & (pred_df["y"] == 0) & (pred_df["real_domain"] == "MLLM")]
    if len(sub):
        fold3_mllm["n"] = int(len(sub))
        for cond, col in [("CLEAN", "p_CLEAN")] + [(c, f"p_{c}") for c in TRANSFORMS]:
            p = sub[col].to_numpy()
            fold3_mllm[f"{cond}_specificity"] = float((p < THR).mean())
            fold3_mllm[f"{cond}_mean_p"] = float(p.mean())

    # Flips
    flips_macro = {}
    for cond in TRANSFORMS:
        frates, c2w, w2c, absdp, sigdp = [], [], [], [], []
        for fold in [1, 2, 3, 4]:
            sub = pred_df[pred_df["fold"] == fold]
            y = sub["y"].to_numpy()
            pc = sub["p_CLEAN"].to_numpy()
            pt = sub[f"p_{cond}"].to_numpy()
            pred_c = (pc >= THR).astype(int)
            pred_t = (pt >= THR).astype(int)
            flip = pred_c != pred_t
            correct_c = pred_c == y
            wrong_c = ~correct_c
            frates.append(float(flip.mean()))
            c2w.append(float((correct_c & flip).mean()))
            w2c.append(float((wrong_c & flip).mean()))
            absdp.append(float(np.mean(np.abs(pt - pc))))
            sigdp.append(float(np.mean(pt - pc)))
        flips_macro[cond] = {
            "flip_rate": float(np.mean(frates)),
            "correct_to_wrong": float(np.mean(c2w)),
            "wrong_to_correct": float(np.mean(w2c)),
            "mean_abs_dp": float(np.mean(absdp)),
            "mean_signed_dp": float(np.mean(sigdp)),
        }

    # Bootstrap
    print("[V2-10D] Paired bootstrap ...", flush=True)
    boot_folds = {}
    for fold in [1, 2, 3, 4]:
        sub = pred_df[pred_df["fold"] == fold]
        y = sub["y"].to_numpy()
        pc = sub["p_CLEAN"].to_numpy()
        boot_folds[f"fold_{fold}"] = {}
        for cond in TRANSFORMS:
            boot_folds[f"fold_{fold}"][cond] = paired_boot_delta(
                y, sub[f"p_{cond}"].to_numpy(), pc
            )

    # Optional temperature diagnostic
    temp_diag = {}
    if TEMP_CSV.is_file():
        temps = pd.read_csv(TEMP_CSV).set_index("fold")["T"].to_dict()
        for cond in TRANSFORMS:
            mc_c, mc_t = [], []
            for fold in [1, 2, 3, 4]:
                T = float(temps[fold])
                sub = pred_df[pred_df["fold"] == fold]
                pc = apply_temperature(to_logit(sub["p_CLEAN"].to_numpy()), T)
                pt = apply_temperature(to_logit(sub[f"p_{cond}"].to_numpy()), T)
                mc_c.append(float(confidence(pc).mean()))
                mc_t.append(float(confidence(pt).mean()))
            temp_diag[cond] = {
                "mean_conf_clean": float(np.mean(mc_c)),
                "mean_conf_transformed": float(np.mean(mc_t)),
                "delta_mean_conf": float(np.mean(mc_t) - np.mean(mc_c)),
            }

    # V1 historical comparison
    v1_rows = []
    comparability = "CONTEXT_ONLY"
    note = (
        "V1 C0 (RQ3 A2) StrongRobust metrics are on controlled_v1 known/unseen test with MobileNet "
        "thresholds and 224-controlled pipeline; V2 H2 is on V2 development eval with CLIP+LoRA+LogReg "
        "at p=0.5. Datasets/generators/thresholds differ — compare qualitative degradation patterns only "
        "(e.g. blur_sigma2 historically most damaging to Real specificity for V1)."
    )
    if RQ3_METRICS.is_file():
        rq3 = pd.read_csv(RQ3_METRICS)
        a2u = rq3[(rq3["regime"] == "A2") & (rq3["split"] == "unseen_test")]
        for _, r in a2u.iterrows():
            v1_rows.append(
                {
                    "source": "results/rq3_test_metrics_v1.csv",
                    "regime": "A2(=FINAL_C0)",
                    "split": "unseen_test",
                    "condition": r["condition"],
                    "v1_roc_auc": float(r["roc_auc"]),
                    "v1_ai_recall": float(r["recall"]),
                    "v1_specificity": float(r["specificity"]),
                    "v1_delta_auc": float(r["delta_auc"]),
                    "v2_h2_macro_auc": macro.get(r["condition"], {}).get("roc_auc"),
                    "v2_h2_macro_ai_recall": macro.get(r["condition"], {}).get("ai_recall"),
                    "v2_h2_macro_real_specificity": macro.get(r["condition"], {}).get(
                        "real_specificity"
                    ),
                    "comparability": "CONTEXT_ONLY",
                }
            )
        pd.DataFrame(v1_rows).to_csv(V1_CMP_OUT, index=False)

    # Decision
    fold_aucs = {f: {c: fold_metrics[f][c]["roc_auc"] for c in TRANSFORMS} for f in [1, 2, 3, 4]}
    decision, evidence = decide(macro, phone_spec, mllm_spec, fold_aucs)

    figures = make_figures(macro, hard_rows, real_rows)

    # Metrics CSV
    for cond in ["CLEAN"] + TRANSFORMS:
        metric_rows.append({"fold": "macro", "condition": cond, **macro[cond]})
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)

    boot_payload = {
        "protocol": "fold_wise_paired_stratified_bootstrap_5000_seed42",
        "pooled_forbidden": True,
        "folds": boot_folds,
        "macro_deltas_descriptive": deltas,
    }
    BOOT_OUT.write_text(json.dumps(_clean(boot_payload), indent=2))

    # Interpretation
    worst_auc_c = worst["worst_auc"]
    worst_rec_c = worst["worst_ai_recall"]
    worst_spec_c = worst["worst_real_specificity"]
    flux = next(h for h in hard_rows if "FLUX" in h["generator_id"])
    nano = next(h for h in hard_rows if "Nano_Banana" in h["generator_id"])
    gpt = next(h for h in hard_rows if "GPT_Image_2" in h["generator_id"])
    phone = next(r for r in real_rows if r["real_domain"] == "Smartphone")
    mllm = next(r for r in real_rows if r["real_domain"] == "MLLM")

    # ranking vs threshold: large AUC drop vs recall/spec shift with small AUC drop
    ranking_fail = any(abs(deltas[c]["roc_auc"]) >= 0.05 for c in TRANSFORMS)
    score_shift = any(
        abs(deltas[c]["ai_recall"]) >= 0.05 or abs(deltas[c]["real_specificity"]) >= 0.05
        for c in TRANSFORMS
    )

    if decision == "ROBUSTNESS_PROMISING":
        next_rec = (
            "Authorize formal V2 DEVELOPMENT FREEZE packaging for H2 (documentation / paper-facing "
            "summary) — still NOT FINAL_V2_MODEL selection without tutor approval. Do not auto-start."
        )
        q14 = "A. formal development freeze (recommendation only)"
    elif decision == "ROBUSTNESS_MIXED":
        next_rec = (
            "Keep H2 as PRIMARY V2 development candidate; next may be one tightly scoped robustness "
            "diagnostic/intervention OR development freeze with documented transform limitations — "
            "tutor chooses. Do not auto-start. Do not select FINAL_V2_MODEL."
        )
        q14 = "B or C depending on severity of worst transforms — see recommendation"
    else:
        next_rec = (
            "Do not freeze H2; authorize one tightly scoped robustness intervention or deeper "
            "failure diagnostic for the worst transform. Do not auto-start."
        )
        q14 = "B. one tightly scoped robustness intervention / C. further diagnostic"

    interpretation = {
        "q1_overall": f"Decision={decision}; mean_tf_AUC={worst['mean_transformed_auc']:.4f}; mean_tf_RealSpec={worst['mean_transformed_real_specificity']:.4f}.",
        "q2_worst_auc": f"{worst_auc_c} (AUC={macro[worst_auc_c]['roc_auc']:.4f}, Δ={deltas[worst_auc_c]['roc_auc']:+.4f}).",
        "q3_worst_recall": f"{worst_rec_c} (AI_rec={macro[worst_rec_c]['ai_recall']:.4f}, Δ={deltas[worst_rec_c]['ai_recall']:+.4f}).",
        "q4_worst_spec": f"{worst_spec_c} (RealSpec={macro[worst_spec_c]['real_specificity']:.4f}, Δ={deltas[worst_spec_c]['real_specificity']:+.4f}).",
        "q5_hard_more_sensitive": (
            "See hard-generator table; compare transform recall deltas vs overall AI recall deltas."
        ),
        "q6_flux": f"FLUX CLEAN={flux['CLEAN_recall']:.3f}; transforms="
        + ", ".join(f"{c}={flux[f'{c}_recall']:.3f}" for c in TRANSFORMS),
        "q7_nano": f"Nano CLEAN={nano['CLEAN_recall']:.3f}; transforms="
        + ", ".join(f"{c}={nano[f'{c}_recall']:.3f}" for c in TRANSFORMS),
        "q8_gpt": f"GPT Image2 CLEAN={gpt['CLEAN_recall']:.3f}; transforms="
        + ", ".join(f"{c}={gpt[f'{c}_recall']:.3f}" for c in TRANSFORMS),
        "q9_mllm": "MLLM specs="
        + ", ".join(f"{c}={mllm[f'{c}_specificity']:.3f}" for c in ["CLEAN"] + TRANSFORMS),
        "q10_phone": "Phone specs="
        + ", ".join(f"{c}={phone[f'{c}_specificity']:.3f}" for c in ["CLEAN"] + TRANSFORMS),
        "q11_failure_mode": (
            f"ranking_degradation_present={ranking_fail}; operating_score_shift_present={score_shift}."
        ),
        "q12_vs_v1": f"{comparability}: {note}",
        "q13_h2_primary": (
            "H2 remains the strongest CLEAN-development candidate, but if decision is "
            "ROBUSTNESS_NOT_ACCEPTABLE do not freeze; require a robustness intervention first."
            if decision == "ROBUSTNESS_NOT_ACCEPTABLE"
            else "YES — H2 remains PRIMARY V2 DEVELOPMENT candidate."
        ),
        "q14_next": q14,
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
        "NEW_HEAD_TRAINED": "NO",
        "ROBUSTNESS_AWARE_TRAINING_PERFORMED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "EVALUATION_MEMBERSHIP_CHANGED": "NO",
        "BINARY_THRESHOLD_CHANGED": "NO",
        "THRESHOLD_SWEEP_PERFORMED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "TRANSFORMED_INFERENCE_PERFORMED": "YES",
    }

    payload = {
        "stage": "V2-10D",
        "status": "COMPLETE",
        "decision": decision,
        "decision_evidence": evidence,
        "protocol": protocol,
        "clean_baseline": clean_base,
        "transformed_inference": {
            "performed": "YES",
            "summary": {
                "folds": [1, 2, 3, 4],
                "transforms": TRANSFORMS,
                "n_per_fold": EXPECTED_EVAL,
                "checkpoints": [f"models/v2/clip_lora_fold{f}_best_v1.pt" for f in [1, 2, 3, 4]],
                "h2_heads": [
                    f"models/v2/v2_10b_h2_balanced_logreg_fold{f}_v1.joblib" for f in [1, 2, 3, 4]
                ],
                "integrity": integrity_tf,
            },
        },
        "macro": macro,
        "deltas": deltas,
        "worst": worst,
        "hard_generators": hard_rows,
        "real_domains": real_rows,
        "fold3_mllm": fold3_mllm,
        "flips_macro": flips_macro,
        "bootstrap": boot_payload,
        "temperature_diagnostic": temp_diag,
        "v1_comparison": {
            "comparability": comparability,
            "note": note,
            "artifact": str(V1_CMP_OUT.relative_to(PROJECT_ROOT)) if v1_rows else None,
        },
        "resource_note": {
            "H2_head_params": 513,
            "H2_serialized_kb": 2.9,
            "H0_mlp_b_params": 147841,
            "LoRA_trainable_historical": "~295k during V2-8",
            "note": "Robustness evaluation does not alter deployment parameter count.",
        },
        "figures": figures,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "fold_metrics": {
            f"fold_{f}": {c: fold_metrics[f][c] for c in ["CLEAN"] + TRANSFORMS}
            for f in [1, 2, 3, 4]
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
