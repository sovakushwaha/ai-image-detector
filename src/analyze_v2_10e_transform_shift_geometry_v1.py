#!/usr/bin/env python3
"""V2-10E — Transform-Shift Representation & Margin Diagnostic.

ANALYSIS ONLY. No training. No threshold/calibrator/selective changes.

If transformed R1 not stored (V2-10D cache = predictions only):
inference-only R1 dump via same on-the-fly transforms as V2-10D
(no regenerated image files; reuse locked transform fns + frozen LoRA).
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
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v2_10a_representation_geometry_v1 import (  # noqa: E402
    ClipLoRAModel,
    class_geometry,
    knn_loo_purity,
    resolve_path,
)
from external_v2_common import TRANSFORM_FNS  # noqa: E402

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
CLEAN_PRED = OUT / "v2_10b_head_predictions_v1.csv"
ROB_PRED = OUT / "v2_10d_robustness_predictions_v1.csv"
CACHE10D = OUT / "v2_10d_transform_cache"

SEED = 42
K_NN = 5
THR = 0.5
EPS = 1e-6
EXPECTED = {1: 2821, 2: 2619, 3: 1994, 4: 1709}
TRANSFORMS = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
HARD = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]

JSON_OUT = OUT / "v2_10e_transform_shift_analysis_v1.json"
REPORT_OUT = OUT / "v2_10e_transform_shift_report_v1.txt"
DRIFT_CSV = OUT / "v2_10e_feature_drift_v1.csv"
MARGIN_CSV = OUT / "v2_10e_margin_shift_v1.csv"
GEOM_CSV = OUT / "v2_10e_transform_geometry_v1.csv"
GEN_CSV = OUT / "v2_10e_generator_transform_geometry_v1.csv"
REAL_CSV = OUT / "v2_10e_real_domain_transform_geometry_v1.csv"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {k: float("nan") for k in ["n", "mean", "median", "p25", "p75", "p95"]}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1).clip(min=1e-12)
    nb = np.linalg.norm(b, axis=1).clip(min=1e-12)
    return np.sum(a * b, axis=1) / (na * nb)


def cos_vec(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def local_opp_frac(X: np.ndarray, y: np.ndarray, k: int = K_NN) -> dict[str, float]:
    """Fraction of k NN with opposite label (LOO); also NN is opposite."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n = len(y)
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn1 = np.argmin(D, axis=1)
    knn = np.argpartition(D, kth=min(k, n - 2), axis=1)[:, :k]
    row = np.arange(n)[:, None]
    knn = knn[row, np.argsort(D[row, knn], axis=1)]
    opp_nn = (y[nn1] != y).astype(float)
    opp_k = (y[knn] != y[:, None]).mean(axis=1)
    real = y == 0
    ai = y == 1
    return {
        "frac_nn_opposite": float(opp_nn.mean()),
        "frac_nn_ai_among_real": float(opp_nn[real].mean()) if real.any() else float("nan"),
        "frac_nn_real_among_ai": float(opp_nn[ai].mean()) if ai.any() else float("nan"),
        "mean_k5_opp_frac": float(opp_k.mean()),
        "mean_k5_ai_frac_among_real": float(opp_k[real].mean()) if real.any() else float("nan"),
        "mean_k5_real_frac_among_ai": float(opp_k[ai].mean()) if ai.any() else float("nan"),
    }


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


def audit_artifacts() -> dict[str, Any]:
    existing = sorted(OUT.glob("v2_10e_r1_*_fold*_v1.npz"))
    cache_csvs = sorted(CACHE10D.glob("fold*_*.csv")) if CACHE10D.is_dir() else []
    # V2-10D cache stores predictions only
    sample = None
    if cache_csvs:
        sample = pd.read_csv(cache_csvs[0], nrows=1).columns.tolist()
    has_r1_in_10d = False
    if sample:
        has_r1_in_10d = any(c.startswith("r1") or c == "embedding" for c in sample)
    inv = {
        "clean_r1_present": all((OUT / f"v2_10a_lora_features_fold{f}_v1.npz").is_file() for f in [1, 2, 3, 4]),
        "h2_models_present": all(
            (MODELS / f"v2_10b_h2_balanced_logreg_fold{f}_v1.joblib").is_file() for f in [1, 2, 3, 4]
        ),
        "v2_10d_predictions_present": ROB_PRED.is_file(),
        "v2_10d_cache_csvs": len(cache_csvs),
        "v2_10d_cache_columns_sample": sample,
        "v2_10d_stored_transformed_r1": has_r1_in_10d,
        "preexisting_v2_10e_r1_npz": [p.name for p in existing],
        "need_feature_inference": len(existing) < 16,  # 4 folds × 4 transforms
    }
    if not inv["clean_r1_present"]:
        stop("missing clean V2-10A R1")
    if not inv["h2_models_present"]:
        stop("missing H2 models")
    if not inv["v2_10d_predictions_present"]:
        stop("missing V2-10D predictions")
    if inv["v2_10d_cache_csvs"] < 16:
        stop("incomplete V2-10D transform cache CSVs")
    return inv


def load_clean_r1(fold: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10a_lora_features_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def extract_transformed_r1(man: pd.DataFrame, rob: pd.DataFrame) -> dict[str, Any]:
    """Inference-only R1 for each fold×transform; on-the-fly transforms (no image regen)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta: dict[str, Any] = {
        "performed": False,
        "device": str(device),
        "method": "on_the_fly_transform_same_as_v2_10d_no_saved_images",
        "v2_10d_cache_reused_for_images": False,
        "v2_10d_cache_reused_for_prob_check": True,
        "folds": {},
    }
    man_ix = man.set_index("image_id", drop=False)

    for fold in [1, 2, 3, 4]:
        ef = rob[rob["fold"] == fold].reset_index(drop=True)
        if len(ef) != EXPECTED[fold]:
            stop(f"fold {fold} rob n={len(ef)}")
        rows = []
        for iid in ef["image_id"].astype(str):
            r = man_ix.loc[iid]
            path = resolve_path(r)
            if not path.is_file():
                stop(f"missing {iid}: {path}")
            rows.append({"image_id": iid, "path": str(path)})

        ckpt_path = MODELS / f"clip_lora_fold{fold}_best_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})

        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        cls1 = list(h2.classes_).index(1)

        for cond in TRANSFORMS:
            out_npz = OUT / f"v2_10e_r1_{cond}_fold{fold}_v1.npz"
            cache_csv = CACHE10D / f"fold{fold}_{cond}_v1.csv"
            if not cache_csv.is_file():
                stop(f"missing V2-10D cache {cache_csv}")
            cache = pd.read_csv(cache_csv)
            cache["image_id"] = cache["image_id"].astype(str)
            if list(cache["image_id"]) != list(ef["image_id"].astype(str)):
                stop(f"fold {fold} {cond} cache ID order != rob preds")

            if out_npz.is_file():
                z = np.load(out_npz, allow_pickle=False)
                if z["r1"].shape != (EXPECTED[fold], 512):
                    stop(f"bad existing {out_npz}")
                # verify H2 probs vs cache
                p = h2.predict_proba(z["r1"])[:, cls1]
                max_abs = float(np.max(np.abs(p - cache["p_h2"].to_numpy(dtype=float))))
                meta["folds"][f"fold_{fold}_{cond}"] = {
                    "reused": True,
                    "n": int(z["r1"].shape[0]),
                    "D_r1": 512,
                    "path": str(out_npz.relative_to(PROJECT_ROOT)),
                    "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                    "max_abs_p_vs_v2_10d_cache": max_abs,
                }
                if max_abs > 0.05:
                    print(f"[V2-10E] WARNING fold{fold}/{cond} p mismatch {max_abs:.4f}", flush=True)
                continue

            meta["performed"] = True
            print(f"[V2-10E] Extracting R1 fold{fold} {cond} on {device} n={len(rows)} ...", flush=True)
            t0 = time.perf_counter()
            model = ClipLoRAModel(cfg, device)
            incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
            crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
            if crit:
                stop(f"critical missing {crit[:5]}")
            model.eval()
            for p_ in model.parameters():
                p_.requires_grad = False

            ds = TransformDataset(rows, model.preprocess, TRANSFORM_FNS[cond])
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
            order = [pos[i] for i in ef["image_id"].astype(str)]
            r1 = r1[order]
            ids_arr = ids_arr[order]
            if not np.isfinite(r1).all() or r1.shape != (EXPECTED[fold], 512):
                stop(f"bad R1 fold{fold}/{cond}")

            p = h2.predict_proba(r1)[:, cls1]
            max_abs = float(np.max(np.abs(p - cache["p_h2"].to_numpy(dtype=float))))
            if max_abs > 0.05:
                print(
                    f"[V2-10E] WARNING fold{fold}/{cond} p vs V2-10D cache max_abs={max_abs:.4f}",
                    flush=True,
                )

            np.savez_compressed(
                out_npz,
                image_ids=ids_arr,
                r1=r1,
                y=ef["y"].to_numpy(dtype=np.int64),
                fold=np.array([fold]),
                condition=np.array([cond]),
                layer_r1="lora_visual_embedding_l2_normalized",
                checkpoint=np.array([str(ckpt_path.relative_to(PROJECT_ROOT))]),
                p_h2_from_r1=p.astype(np.float64),
                max_abs_p_vs_v2_10d=np.array([max_abs]),
            )
            meta["folds"][f"fold_{fold}_{cond}"] = {
                "reused": False,
                "n": int(len(ids_arr)),
                "D_r1": 512,
                "path": str(out_npz.relative_to(PROJECT_ROOT)),
                "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                "max_abs_p_vs_v2_10d_cache": max_abs,
                "seconds": float(time.perf_counter() - t0),
            }
            print(f"[V2-10E] Wrote {out_npz.name} max_abs_p={max_abs:.4g}", flush=True)

            # free model between transforms to reduce memory pressure
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    if any(not v.get("reused") for v in meta["folds"].values()):
        meta["performed"] = True
    return meta


def load_tf_r1(fold: int, cond: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10e_r1_{cond}_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32)


def h2_weights(fold: int) -> tuple[np.ndarray, float]:
    h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
    # classes [0,1] — coef for class 1 relative to 0 for binary
    w = h2.coef_.ravel().astype(np.float64)
    b = float(h2.intercept_[0])
    return w, b


def classify_transform(evidence: dict[str, Any]) -> str:
    """Per-transform mechanism from geometry + margin evidence."""
    purity = evidence["geom_tf"]["knn_purity"]
    purity_clean = evidence["geom_clean"]["knn_purity"]
    sep = evidence["geom_tf"]["centroid_euclidean"]
    sep_clean = evidence["geom_clean"]["centroid_euclidean"]
    bw = evidence["geom_tf"]["between_over_within"]
    bw_clean = evidence["geom_clean"]["between_over_within"]
    nn_mix = evidence["geom_tf"]["frac_nn_opposite"]
    nn_mix_clean = evidence["geom_clean"]["frac_nn_opposite"]
    # margin: |mean delta_z| large relative to |mean z|
    mean_dz_abs = abs(evidence["margin"]["delta_z_mean"])
    align_real = evidence["alignment"]["cos_d_real_w"]
    align_ai = evidence["alignment"]["cos_d_ai_w"]
    # strong geometry retention
    geom_ok = (
        purity >= 0.90
        or (purity >= purity_clean - 0.05 and purity >= 0.85)
    ) and (sep >= 0.5 * sep_clean or sep >= 0.25)
    geom_collapse = purity < 0.80 or (sep < 0.5 * sep_clean and nn_mix > nn_mix_clean + 0.10)
    # strong margin shift / alignment
    margin_dom = mean_dz_abs >= 0.5 or abs(align_real) >= 0.3 or abs(align_ai) >= 0.3

    if geom_ok and margin_dom and not geom_collapse:
        return "HEAD_SHIFT_DOMINANT"
    if geom_collapse and not (geom_ok and margin_dom):
        return "REPRESENTATION_DEGRADATION_DOMINANT"
    if geom_collapse and margin_dom:
        return "MIXED"
    if margin_dom:
        return "HEAD_SHIFT_DOMINANT"
    if geom_collapse:
        return "REPRESENTATION_DEGRADATION_DOMINANT"
    return "MIXED"


def make_figures(payload: dict[str, Any]) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    conds = TRANSFORMS

    # 1. k5 purity
    fig, ax = plt.subplots(figsize=(7, 4))
    clean_p = payload["geometry_macro"]["CLEAN"]["knn_purity"]
    vals = [clean_p] + [payload["geometry_macro"][c]["knn_purity"] for c in conds]
    ax.bar(["CLEAN"] + conds, vals, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("k=5 LOO neighbour purity")
    ax.set_title("R1 label purity CLEAN vs transforms")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = FIG / "v2_10e_geometry_knn_purity_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 2. centroid separation
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [payload["geometry_macro"]["CLEAN"]["centroid_euclidean"]] + [
        payload["geometry_macro"][c]["centroid_euclidean"] for c in conds
    ]
    ax.bar(["CLEAN"] + conds, vals, color="#55A868")
    ax.set_ylabel("Real–AI centroid Euclidean")
    ax.set_title("R1 class separation")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = FIG / "v2_10e_geometry_centroid_sep_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 3. delta_z by class
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(conds))
    w = 0.35
    ax.bar(x, [payload["margin_macro"][c]["delta_z_real_mean"] for c in conds], width=w, label="Real")
    ax.bar(
        x + w,
        [payload["margin_macro"][c]["delta_z_ai_mean"] for c in conds],
        width=w,
        label="AI",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("mean Δz (transform − clean)")
    ax.set_title("H2 logit margin shift")
    ax.legend()
    fig.tight_layout()
    p = FIG / "v2_10e_margin_shift_by_class_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 4. drift alignment
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, [payload["alignment_macro"][c]["cos_d_real_w"] for c in conds], width=w, label="cos(d_real,w)")
    ax.bar(
        x + w,
        [payload["alignment_macro"][c]["cos_d_ai_w"] for c in conds],
        width=w,
        label="cos(d_ai,w)",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("cosine alignment with H2 w")
    ax.set_title("Mean drift alignment with H2 decision direction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "v2_10e_feature_drift_alignment_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 5. feature drift magnitude
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, [payload["drift_macro"][c]["euc_real_mean"] for c in conds], width=w, label="Real euc drift")
    ax.bar(
        x + w,
        [payload["drift_macro"][c]["euc_ai_mean"] for c in conds],
        width=w,
        label="AI euc drift",
    )
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("mean ||x_tf − x_clean||")
    ax.set_title("Feature drift magnitude")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "v2_10e_feature_drift_magnitude_v1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10E — Transform-Shift Representation & Margin Diagnostic")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Analysis only. No training. FINAL_V2_MODEL NOT SELECTED.")
    lines.append("")
    lines.append("Artifact audit")
    for k, v in payload["artifact_audit"].items():
        lines.append(f"  {k}: {v}")
    inf = payload["feature_inference"]
    lines.append(
        f"NEW_TRANSFORMED_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}"
    )
    lines.append(f"  method: {inf.get('method')}")
    lines.append(f"  V2-10D image files regenerated: NO (on-the-fly transforms)")
    lines.append(f"  V2-10D prediction cache used for integrity check: YES")
    lines.append("")
    lines.append("Integrity")
    lines.append(f"  {payload['integrity']}")
    lines.append("")
    lines.append("Per-transform feature drift (macro mean folds; Euclidean)")
    for c in TRANSFORMS:
        d = payload["drift_macro"][c]
        lines.append(
            f"  {c}: overall={d['euc_overall_mean']:.4f} Real={d['euc_real_mean']:.4f} "
            f"AI={d['euc_ai_mean']:.4f} cos_sim_overall={d['cos_sim_overall_mean']:.4f}"
        )
    lines.append("")
    lines.append("H2 margin shift Δz (macro)")
    for c in TRANSFORMS:
        m = payload["margin_macro"][c]
        lines.append(
            f"  {c}: meanΔz={m['delta_z_mean']:+.4f} Real={m['delta_z_real_mean']:+.4f} "
            f"AI={m['delta_z_ai_mean']:+.4f} recon_err={m['recon_max_abs']:.2e}"
        )
    lines.append("")
    lines.append("Drift alignment with H2 w (macro)")
    for c in TRANSFORMS:
        a = payload["alignment_macro"][c]
        lines.append(
            f"  {c}: cos(d_real,w)={a['cos_d_real_w']:+.4f} ||d_real||={a['norm_d_real']:.4f} "
            f"cos(d_ai,w)={a['cos_d_ai_w']:+.4f} ||d_ai||={a['norm_d_ai']:.4f}"
        )
    lines.append("")
    lines.append("R1 geometry CLEAN vs transforms (macro)")
    for c in ["CLEAN"] + TRANSFORMS:
        g = payload["geometry_macro"][c]
        lines.append(
            f"  {c}: cent_euc={g['centroid_euclidean']:.4f} bw={g['between_over_within']:.4f} "
            f"k5_purity={g['knn_purity']:.4f} nn_opp={g['frac_nn_opposite']:.4f}"
        )
    lines.append("")
    lines.append("Hard generators (JPEG vs CLEAN geometry highlights)")
    for h in payload["hard_generators"]:
        lines.append(
            f"  {h['generator_id']}: CLEAN distReal={h['CLEAN_cent_to_real']:.4f} "
            f"nnReal={h['CLEAN_frac_nn_real']:.3f}; "
            f"jpeg distReal={h['jpeg_q50_cent_to_real']:.4f} nnReal={h['jpeg_q50_frac_nn_real']:.3f} "
            f"Δz_jpeg={h['jpeg_q50_delta_z_mean']:+.3f}"
        )
    lines.append("")
    lines.append("Real domains under resize_112")
    for r in payload["real_domains"]:
        lines.append(
            f"  {r['real_domain']}: CLEAN distAI={r['CLEAN_dist_to_ai']:.4f} k5AIfrac={r['CLEAN_k5_ai_frac']:.3f}; "
            f"resize distAI={r['resize_112_dist_to_ai']:.4f} k5AIfrac={r['resize_112_k5_ai_frac']:.3f} "
            f"Δz={r['resize_112_delta_z_mean']:+.3f}"
        )
    lines.append("")
    lines.append("Error-transition summary (macro; CLEAN correct→TRANSFORM wrong)")
    for c in TRANSFORMS:
        e = payload["error_transitions"][c]
        lines.append(
            f"  {c}: n_c2w={e['n_correct_to_wrong']} mean_euc={e['c2w_euc_mean']:.4f} "
            f"meanΔz={e['c2w_delta_z_mean']:+.4f} tf_k5_opp={e['c2w_tf_k5_opp_mean']:.3f}"
        )
    lines.append("")
    lines.append("Per-transform mechanism")
    for c, m in payload["per_transform_mechanism"].items():
        lines.append(f"  {c}: {m}")
    lines.append("")
    lines.append(f"OVERALL DECISION: {payload['decision']}")
    lines.append(f"Recommended intervention: {payload['recommended_intervention']}")
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
    if "## Stage V2-10E" in text:
        print("[V2-10E] research_log already contains V2-10E; not rewriting.", flush=True)
        return
    mech = "; ".join(f"{c}={m}" for c, m in payload["per_transform_mechanism"].items())
    entry = f"""
## Stage V2-10E — Transform-Shift Representation & Margin Diagnostic

**Date:** 2026-09-07  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Analysis only. No LoRA/CLIP/H2 training. No threshold/calibrator/selective changes. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation:** V2-10D = ROBUSTNESS_NOT_ACCEPTABLE. Diagnose whether StrongRobust failures are H2 margin/head-shift vs LoRA R1 geometry collapse.

**Artifacts:** Clean R1 from V2-10A; H2 from V2-10B; predictions/metrics from V2-10D. V2-10D cache stores probabilities only (no R1).  
**NEW_TRANSFORMED_FEATURE_INFERENCE_PERFORMED = {'YES' if payload['feature_inference'].get('performed') else 'NO'}** — on-the-fly transforms (same as V2-10D); no regenerated image files; R1 D=512; integrity vs V2-10D p_h2 checked.

**Per-transform mechanisms:** {mech}

**Overall decision:** **{payload['decision']}**  
**Recommended intervention class:** **{payload['recommended_intervention']}**

**Outputs:** `src/analyze_v2_10e_transform_shift_geometry_v1.py`; `results/v2/v2_10e_*`; figures `figures/v2/v2_10e_*.png`.

**Integrity:** NTIRE=NO; fal=NO; V1 unmodified; folds unchanged; no weight updates; transform suite unchanged; thr unchanged; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10E] Appended research_log.md", flush=True)


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

    print("[V2-10E] Artifact audit ...", flush=True)
    inv = audit_artifacts()
    print(f"[V2-10E] need_feature_inference={inv['need_feature_inference']}", flush=True)

    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)
    rob = pd.read_csv(ROB_PRED)
    rob["image_id"] = rob["image_id"].astype(str)

    feat_meta = extract_transformed_r1(man, rob)

    # Accumulate analyses
    drift_rows = []
    margin_rows = []
    geom_rows = []
    gen_rows = []
    real_rows = []
    per_fold_cond: dict[str, dict] = {}

    integrity = {"folds": {}, "ok": True, "max_p_mismatch": 0.0}

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10E] Analyzing fold {fold} ...", flush=True)
        ef = rob[rob["fold"] == fold].reset_index(drop=True)
        y = ef["y"].to_numpy(dtype=int)
        gens = ef["generator_id"].fillna("").astype(str).to_numpy()
        domains = ef["real_domain"].fillna("").astype(str).to_numpy()
        ids_c, Xc = load_clean_r1(fold)
        if list(ids_c) != list(ef["image_id"]):
            # align clean to ef order
            pos = {i: j for j, i in enumerate(ids_c)}
            try:
                order = [pos[i] for i in ef["image_id"]]
            except KeyError as e:
                stop(f"fold {fold} clean R1 missing ID {e}")
            Xc = Xc[order]
            ids_c = np.asarray(list(ef["image_id"]), dtype=str)
        if Xc.shape != (EXPECTED[fold], 512) or not np.isfinite(Xc).all():
            stop(f"fold {fold} bad clean R1")

        w, b = h2_weights(fold)
        zc = Xc.astype(np.float64) @ w + b

        # CLEAN geometry once
        g_clean = class_geometry(Xc, y)
        k_clean = knn_loo_purity(Xc, y, k=K_NN)
        opp_clean = local_opp_frac(Xc, y, k=K_NN)
        geom_rows.append(
            {
                "fold": fold,
                "condition": "CLEAN",
                **{f"g_{k}": v for k, v in g_clean.items()},
                "knn_purity": k_clean["mean_neighbour_label_purity"],
                **opp_clean,
            }
        )

        integrity["folds"][f"fold_{fold}"] = {"clean_ok": True, "transforms": {}}

        for cond in TRANSFORMS:
            ids_t, Xt = load_tf_r1(fold, cond)
            if list(ids_t) != list(ef["image_id"]):
                stop(f"fold {fold} {cond} ID mismatch")
            if not np.isfinite(Xt).all():
                stop(f"fold {fold} {cond} nonfinite")
            # p integrity already in feat_meta
            key = f"fold_{fold}_{cond}"
            mm = feat_meta["folds"].get(key, {}).get("max_abs_p_vs_v2_10d_cache", 0.0)
            integrity["max_p_mismatch"] = max(integrity["max_p_mismatch"], float(mm))
            integrity["folds"][f"fold_{fold}"]["transforms"][cond] = {
                "n": int(len(ids_t)),
                "D": 512,
                "ids_match": True,
                "max_abs_p_vs_cache": float(mm),
            }

            Xt64 = Xt.astype(np.float64)
            Xc64 = Xc.astype(np.float64)
            delta = Xt64 - Xc64
            euc = np.linalg.norm(delta, axis=1)
            csim = cosine_sim(Xc64, Xt64)
            cdists = 1.0 - csim

            zt = Xt64 @ w + b
            dz = zt - zc
            dz_recon = delta @ w
            recon_err = float(np.max(np.abs(dz - dz_recon)))
            if recon_err > 1e-5:
                stop(f"fold {fold} {cond} Δz recon err {recon_err}")

            real = y == 0
            ai = y == 1
            d_real = delta[real].mean(axis=0) if real.any() else np.zeros(512)
            d_ai = delta[ai].mean(axis=0) if ai.any() else np.zeros(512)
            align = {
                "cos_d_real_w": cos_vec(d_real, w),
                "cos_d_ai_w": cos_vec(d_ai, w),
                "norm_d_real": float(np.linalg.norm(d_real)),
                "norm_d_ai": float(np.linalg.norm(d_ai)),
            }

            g_tf = class_geometry(Xt, y)
            k_tf = knn_loo_purity(Xt, y, k=K_NN)
            opp_tf = local_opp_frac(Xt, y, k=K_NN)

            drift_rows.append(
                {
                    "fold": fold,
                    "condition": cond,
                    "euc_overall": summarize(euc),
                    "euc_real": summarize(euc[real]),
                    "euc_ai": summarize(euc[ai]),
                    "cos_sim_overall": summarize(csim),
                    "cos_dist_overall": summarize(cdists),
                }
            )
            # flatten for CSV later
            margin_rows.append(
                {
                    "fold": fold,
                    "condition": cond,
                    "delta_z_mean": float(dz.mean()),
                    "delta_z_real_mean": float(dz[real].mean()) if real.any() else float("nan"),
                    "delta_z_ai_mean": float(dz[ai].mean()) if ai.any() else float("nan"),
                    "delta_z_real_median": float(np.median(dz[real])) if real.any() else float("nan"),
                    "delta_z_ai_median": float(np.median(dz[ai])) if ai.any() else float("nan"),
                    "recon_max_abs": recon_err,
                    **align,
                }
            )
            geom_rows.append(
                {
                    "fold": fold,
                    "condition": cond,
                    **{f"g_{k}": v for k, v in g_tf.items()},
                    "knn_purity": k_tf["mean_neighbour_label_purity"],
                    **opp_tf,
                }
            )

            # generators
            for gid in HARD:
                mask = gens == gid
                if not mask.any():
                    continue
                Xg_c, Xg_t = Xc64[mask], Xt64[mask]
                X_real = Xc64[real]
                # centroid to Real (clean Real centroid for CLEAN; transform Real for tf)
                # For transform geometry: distance of generator centroid (in transform space) to transform Real centroid
                c_real_tf = Xt64[real].mean(0)
                c_g_tf = Xg_t.mean(0)
                c_real_c = Xc64[real].mean(0)
                c_g_c = Xg_c.mean(0)
                # NN Real fraction in full set
                D_c = pairwise_distances(Xg_c, Xc64, metric="euclidean")
                # exclude self: generator samples are AI so ok
                nn_c = np.argmin(D_c, axis=1)
                frac_nn_real_c = float((y[nn_c] == 0).mean())
                D_t = pairwise_distances(Xg_t, Xt64, metric="euclidean")
                nn_t = np.argmin(D_t, axis=1)
                frac_nn_real_t = float((y[nn_t] == 0).mean())
                # k5 real frac
                knn_t = np.argpartition(D_t, kth=min(K_NN, D_t.shape[1] - 1), axis=1)[:, :K_NN]
                k5_real_t = float((y[knn_t] == 0).mean())
                knn_c = np.argpartition(D_c, kth=min(K_NN, D_c.shape[1] - 1), axis=1)[:, :K_NN]
                k5_real_c = float((y[knn_c] == 0).mean())
                gen_rows.append(
                    {
                        "fold": fold,
                        "condition": cond,
                        "generator_id": gid,
                        "n": int(mask.sum()),
                        "cent_to_real": float(np.linalg.norm(c_g_tf - c_real_tf)),
                        "cent_to_real_clean_space": float(np.linalg.norm(c_g_c - c_real_c)),
                        "within_disp": float(np.mean(np.linalg.norm(Xg_t - c_g_tf, axis=1))),
                        "frac_nn_real": frac_nn_real_t,
                        "k5_real_frac": k5_real_t,
                        "CLEAN_cent_to_real": float(np.linalg.norm(c_g_c - c_real_c)),
                        "CLEAN_frac_nn_real": frac_nn_real_c,
                        "CLEAN_k5_real_frac": k5_real_c,
                        "euc_drift_mean": float(euc[mask].mean()),
                        "delta_z_mean": float(dz[mask].mean()),
                    }
                )

            # real domains
            for dom in REAL_DOMAINS:
                mask = real & (domains == dom)
                if not mask.any():
                    continue
                Xd_c, Xd_t = Xc64[mask], Xt64[mask]
                c_ai_c = Xc64[ai].mean(0)
                c_ai_t = Xt64[ai].mean(0)
                c_d_c = Xd_c.mean(0)
                c_d_t = Xd_t.mean(0)
                D_c = pairwise_distances(Xd_c, Xc64)
                nn_c = np.argmin(D_c, axis=1)
                D_t = pairwise_distances(Xd_t, Xt64)
                nn_t = np.argmin(D_t, axis=1)
                knn_t = np.argpartition(D_t, kth=min(K_NN, D_t.shape[1] - 1), axis=1)[:, :K_NN]
                knn_c = np.argpartition(D_c, kth=min(K_NN, D_c.shape[1] - 1), axis=1)[:, :K_NN]
                real_rows.append(
                    {
                        "fold": fold,
                        "condition": cond,
                        "real_domain": dom,
                        "n": int(mask.sum()),
                        "dist_to_ai": float(np.linalg.norm(c_d_t - c_ai_t)),
                        "CLEAN_dist_to_ai": float(np.linalg.norm(c_d_c - c_ai_c)),
                        "frac_nn_ai": float((y[nn_t] == 1).mean()),
                        "CLEAN_frac_nn_ai": float((y[nn_c] == 1).mean()),
                        "k5_ai_frac": float((y[knn_t] == 1).mean()),
                        "CLEAN_k5_ai_frac": float((y[knn_c] == 1).mean()),
                        "euc_drift_mean": float(euc[mask].mean()),
                        "delta_z_mean": float(dz[mask].mean()),
                    }
                )

            # Fold3 MLLM special stored in real_rows with note via fold==3

            # Error transitions using V2-10D probs
            pc = ef["p_CLEAN"].to_numpy(dtype=float)
            pt = ef[f"p_{cond}"].to_numpy(dtype=float)
            pred_c = (pc >= THR).astype(int)
            pred_t = (pt >= THR).astype(int)
            correct_c = pred_c == y
            wrong_c = ~correct_c
            c2w = correct_c & (pred_t != y)
            # k5 opp on transform for c2w samples
            opp_k = local_opp_frac(Xt, y, k=K_NN)
            # per-sample k5 opp for c2w
            D = pairwise_distances(Xt64, metric="euclidean")
            np.fill_diagonal(D, np.inf)
            knn = np.argpartition(D, kth=min(K_NN, len(y) - 2), axis=1)[:, :K_NN]
            row = np.arange(len(y))[:, None]
            knn = knn[row, np.argsort(D[row, knn], axis=1)]
            k5_opp = (y[knn] != y[:, None]).mean(axis=1)

            per_fold_cond[f"{fold}_{cond}"] = {
                "geom_clean": {
                    "knn_purity": k_clean["mean_neighbour_label_purity"],
                    "centroid_euclidean": g_clean["centroid_euclidean"],
                    "between_over_within": g_clean["between_over_within"],
                    "frac_nn_opposite": opp_clean["frac_nn_opposite"],
                },
                "geom_tf": {
                    "knn_purity": k_tf["mean_neighbour_label_purity"],
                    "centroid_euclidean": g_tf["centroid_euclidean"],
                    "between_over_within": g_tf["between_over_within"],
                    "frac_nn_opposite": opp_tf["frac_nn_opposite"],
                },
                "margin": {"delta_z_mean": float(dz.mean())},
                "alignment": align,
                "error_transitions": {
                    "n_correct_to_wrong": int(c2w.sum()),
                    "c2w_euc_mean": float(euc[c2w].mean()) if c2w.any() else float("nan"),
                    "c2w_delta_z_mean": float(dz[c2w].mean()) if c2w.any() else float("nan"),
                    "c2w_tf_k5_opp_mean": float(k5_opp[c2w].mean()) if c2w.any() else float("nan"),
                    "c2w_real_n": int((c2w & real).sum()),
                    "c2w_ai_n": int((c2w & ai).sum()),
                },
                "drift": {
                    "euc_overall_mean": float(euc.mean()),
                    "euc_real_mean": float(euc[real].mean()) if real.any() else float("nan"),
                    "euc_ai_mean": float(euc[ai].mean()) if ai.any() else float("nan"),
                    "cos_sim_overall_mean": float(csim.mean()),
                },
            }

    # Macro aggregates
    def mean_key(rows, cond, key):
        vals = [r[key] for r in rows if r.get("condition") == cond and key in r]
        return float(np.nanmean(vals)) if vals else float("nan")

    geometry_macro = {}
    for cond in ["CLEAN"] + TRANSFORMS:
        sub = [r for r in geom_rows if r["condition"] == cond]
        geometry_macro[cond] = {
            "centroid_euclidean": float(np.mean([r["g_centroid_euclidean"] for r in sub])),
            "between_over_within": float(np.mean([r["g_between_over_within"] for r in sub])),
            "knn_purity": float(np.mean([r["knn_purity"] for r in sub])),
            "frac_nn_opposite": float(np.mean([r["frac_nn_opposite"] for r in sub])),
            "disp_real": float(np.mean([r["g_disp_real"] for r in sub])),
            "disp_ai": float(np.mean([r["g_disp_ai"] for r in sub])),
        }

    drift_macro = {}
    margin_macro = {}
    alignment_macro = {}
    for cond in TRANSFORMS:
        # from per_fold_cond
        keys = [f"{f}_{cond}" for f in [1, 2, 3, 4]]
        drift_macro[cond] = {
            "euc_overall_mean": float(np.mean([per_fold_cond[k]["drift"]["euc_overall_mean"] for k in keys])),
            "euc_real_mean": float(np.mean([per_fold_cond[k]["drift"]["euc_real_mean"] for k in keys])),
            "euc_ai_mean": float(np.mean([per_fold_cond[k]["drift"]["euc_ai_mean"] for k in keys])),
            "cos_sim_overall_mean": float(
                np.mean([per_fold_cond[k]["drift"]["cos_sim_overall_mean"] for k in keys])
            ),
        }
        mr = [r for r in margin_rows if r["condition"] == cond]
        margin_macro[cond] = {
            "delta_z_mean": float(np.mean([r["delta_z_mean"] for r in mr])),
            "delta_z_real_mean": float(np.mean([r["delta_z_real_mean"] for r in mr])),
            "delta_z_ai_mean": float(np.mean([r["delta_z_ai_mean"] for r in mr])),
            "recon_max_abs": float(np.max([r["recon_max_abs"] for r in mr])),
        }
        alignment_macro[cond] = {
            "cos_d_real_w": float(np.mean([r["cos_d_real_w"] for r in mr])),
            "cos_d_ai_w": float(np.mean([r["cos_d_ai_w"] for r in mr])),
            "norm_d_real": float(np.mean([r["norm_d_real"] for r in mr])),
            "norm_d_ai": float(np.mean([r["norm_d_ai"] for r in mr])),
        }

    # Hard gen macro
    hard_out = []
    for gid in HARD:
        row = {"generator_id": gid}
        for cond in ["CLEAN"] + TRANSFORMS:
            if cond == "CLEAN":
                # use jpeg row's CLEAN fields averaged, or from gen_rows any cond
                sub = [r for r in gen_rows if r["generator_id"] == gid]
                if not sub:
                    continue
                row["CLEAN_cent_to_real"] = float(np.mean([r["CLEAN_cent_to_real"] for r in sub]))
                row["CLEAN_frac_nn_real"] = float(np.mean([r["CLEAN_frac_nn_real"] for r in sub]))
                row["CLEAN_k5_real_frac"] = float(np.mean([r["CLEAN_k5_real_frac"] for r in sub]))
            else:
                sub = [r for r in gen_rows if r["generator_id"] == gid and r["condition"] == cond]
                if not sub:
                    continue
                row[f"{cond}_cent_to_real"] = float(np.mean([r["cent_to_real"] for r in sub]))
                row[f"{cond}_frac_nn_real"] = float(np.mean([r["frac_nn_real"] for r in sub]))
                row[f"{cond}_k5_real_frac"] = float(np.mean([r["k5_real_frac"] for r in sub]))
                row[f"{cond}_delta_z_mean"] = float(np.mean([r["delta_z_mean"] for r in sub]))
                row[f"{cond}_euc_drift_mean"] = float(np.mean([r["euc_drift_mean"] for r in sub]))
        hard_out.append(row)

    # Real domain macro
    real_out = []
    for dom in REAL_DOMAINS:
        row = {"real_domain": dom}
        for cond in ["CLEAN"] + TRANSFORMS:
            if cond == "CLEAN":
                sub = [r for r in real_rows if r["real_domain"] == dom]
                if not sub:
                    continue
                row["CLEAN_dist_to_ai"] = float(np.mean([r["CLEAN_dist_to_ai"] for r in sub]))
                row["CLEAN_k5_ai_frac"] = float(np.mean([r["CLEAN_k5_ai_frac"] for r in sub]))
                row["CLEAN_frac_nn_ai"] = float(np.mean([r["CLEAN_frac_nn_ai"] for r in sub]))
            else:
                sub = [r for r in real_rows if r["real_domain"] == dom and r["condition"] == cond]
                if not sub:
                    continue
                row[f"{cond}_dist_to_ai"] = float(np.mean([r["dist_to_ai"] for r in sub]))
                row[f"{cond}_k5_ai_frac"] = float(np.mean([r["k5_ai_frac"] for r in sub]))
                row[f"{cond}_frac_nn_ai"] = float(np.mean([r["frac_nn_ai"] for r in sub]))
                row[f"{cond}_delta_z_mean"] = float(np.mean([r["delta_z_mean"] for r in sub]))
                row[f"{cond}_euc_drift_mean"] = float(np.mean([r["euc_drift_mean"] for r in sub]))
        real_out.append(row)

    # error transitions macro
    err_trans = {}
    for cond in TRANSFORMS:
        keys = [f"{f}_{cond}" for f in [1, 2, 3, 4]]
        err_trans[cond] = {
            "n_correct_to_wrong": int(
                np.sum([per_fold_cond[k]["error_transitions"]["n_correct_to_wrong"] for k in keys])
            ),
            "c2w_euc_mean": float(
                np.nanmean([per_fold_cond[k]["error_transitions"]["c2w_euc_mean"] for k in keys])
            ),
            "c2w_delta_z_mean": float(
                np.nanmean([per_fold_cond[k]["error_transitions"]["c2w_delta_z_mean"] for k in keys])
            ),
            "c2w_tf_k5_opp_mean": float(
                np.nanmean([per_fold_cond[k]["error_transitions"]["c2w_tf_k5_opp_mean"] for k in keys])
            ),
        }

    # Per-transform mechanism
    per_mech = {}
    for cond in TRANSFORMS:
        # aggregate evidence across folds for classify
        keys = [f"{f}_{cond}" for f in [1, 2, 3, 4]]
        evidence = {
            "geom_clean": {
                "knn_purity": float(np.mean([per_fold_cond[k]["geom_clean"]["knn_purity"] for k in keys])),
                "centroid_euclidean": float(
                    np.mean([per_fold_cond[k]["geom_clean"]["centroid_euclidean"] for k in keys])
                ),
                "between_over_within": float(
                    np.mean([per_fold_cond[k]["geom_clean"]["between_over_within"] for k in keys])
                ),
                "frac_nn_opposite": float(
                    np.mean([per_fold_cond[k]["geom_clean"]["frac_nn_opposite"] for k in keys])
                ),
            },
            "geom_tf": {
                "knn_purity": float(np.mean([per_fold_cond[k]["geom_tf"]["knn_purity"] for k in keys])),
                "centroid_euclidean": float(
                    np.mean([per_fold_cond[k]["geom_tf"]["centroid_euclidean"] for k in keys])
                ),
                "between_over_within": float(
                    np.mean([per_fold_cond[k]["geom_tf"]["between_over_within"] for k in keys])
                ),
                "frac_nn_opposite": float(
                    np.mean([per_fold_cond[k]["geom_tf"]["frac_nn_opposite"] for k in keys])
                ),
            },
            "margin": {
                "delta_z_mean": float(np.mean([per_fold_cond[k]["margin"]["delta_z_mean"] for k in keys]))
            },
            "alignment": {
                "cos_d_real_w": alignment_macro[cond]["cos_d_real_w"],
                "cos_d_ai_w": alignment_macro[cond]["cos_d_ai_w"],
            },
        }
        per_mech[cond] = classify_transform(evidence)

    mechs = set(per_mech.values())
    if mechs == {"HEAD_SHIFT_DOMINANT"}:
        decision = "ROBUSTNESS_HEAD_LIMITED"
        intervention = "OPTION_A_FROZEN_R1_ROBUST_HEAD"
        next_rec = (
            "Authorize ONE predeclared robustness-aware class-balanced linear head on FROZEN "
            "LoRA R1 using transformed TRAIN features — do not retrain LoRA. Do not auto-start."
        )
    elif mechs == {"REPRESENTATION_DEGRADATION_DOMINANT"}:
        decision = "ROBUSTNESS_REPRESENTATION_LIMITED"
        intervention = "OPTION_B_TARGETED_REPRESENTATION_ROBUSTNESS"
        next_rec = (
            "Authorize ONE tightly scoped robustness-aware LoRA adaptation focused on worst "
            "transforms — not a PEFT grid. Do not auto-start."
        )
    elif "HEAD_SHIFT_DOMINANT" in mechs and "REPRESENTATION_DEGRADATION_DOMINANT" in mechs:
        decision = "ROBUSTNESS_MIXED"
        intervention = "OPTION_C_MIXED_MINIMAL_INTERVENTION"
        next_rec = (
            "Use the smallest controlled mixed intervention: first frozen-R1 robust head for "
            "head-shift-dominant transforms; only if residual representation collapse remains, "
            "one targeted LoRA robustness adaptation. Avoid grid search. Do not auto-start."
        )
    elif mechs == {"MIXED"} or "MIXED" in mechs:
        decision = "ROBUSTNESS_MIXED"
        intervention = "OPTION_C_MIXED_MINIMAL_INTERVENTION"
        next_rec = (
            "Mixed transform mechanisms: prefer smallest controlled plan — frozen-R1 robust head "
            "first where geometry holds; targeted representation fix only where geometry collapses. "
            "Do not auto-start."
        )
    else:
        decision = "ROBUSTNESS_DIAGNOSTIC_INCONCLUSIVE"
        intervention = "OPTION_D_NO_JUSTIFIED_INTERVENTION"
        next_rec = "Need minimal clarifying diagnostic before authorizing training. Do not auto-start."

    # Scientific Qs
    jpeg = "jpeg_q50"
    resize = "resize_112"
    blur = "blur_sigma2"
    screen = "screenshot_strong"
    # largest real domain geom failure under resize: max k5_ai_frac
    worst_real = max(real_out, key=lambda r: r.get(f"{resize}_k5_ai_frac", -1))
    # largest AI drift under jpeg
    worst_ai = max(hard_out, key=lambda h: h.get(f"{jpeg}_euc_drift_mean", -1))

    interpretation = {
        "q1_jpeg_geometry_or_margin": (
            f"mech={per_mech[jpeg]}; purity CLEAN={geometry_macro['CLEAN']['knn_purity']:.3f}→"
            f"{geometry_macro[jpeg]['knn_purity']:.3f}; meanΔz_AI={margin_macro[jpeg]['delta_z_ai_mean']:+.3f}; "
            f"cos(d_ai,w)={alignment_macro[jpeg]['cos_d_ai_w']:+.3f}."
        ),
        "q2_jpeg_ai_realward": (
            f"AI meanΔz={margin_macro[jpeg]['delta_z_ai_mean']:+.3f} (negative→Real side of H2); "
            f"alignment cos(d_ai,w)={alignment_macro[jpeg]['cos_d_ai_w']:+.3f}."
        ),
        "q3_resize_destroys_geometry": (
            f"mech={per_mech[resize]}; purity {geometry_macro['CLEAN']['knn_purity']:.3f}→"
            f"{geometry_macro[resize]['knn_purity']:.3f}; cent_sep "
            f"{geometry_macro['CLEAN']['centroid_euclidean']:.3f}→"
            f"{geometry_macro[resize]['centroid_euclidean']:.3f}; nn_opp "
            f"{geometry_macro['CLEAN']['frac_nn_opposite']:.3f}→"
            f"{geometry_macro[resize]['frac_nn_opposite']:.3f}."
        ),
        "q4_resize_real_aiward": (
            f"Real meanΔz={margin_macro[resize]['delta_z_real_mean']:+.3f}; "
            f"cos(d_real,w)={alignment_macro[resize]['cos_d_real_w']:+.3f}."
        ),
        "q5_blur": f"mech={per_mech[blur]}; purity→{geometry_macro[blur]['knn_purity']:.3f}; RealΔz={margin_macro[blur]['delta_z_real_mean']:+.3f}.",
        "q6_screenshot": f"mech={per_mech[screen]}; purity→{geometry_macro[screen]['knn_purity']:.3f}; meanΔz={margin_macro[screen]['delta_z_mean']:+.3f}.",
        "q7_jpeg_hard_still_ai_like": (
            "; ".join(
                f"{h['generator_id'].split('::')[-1]} jpeg_nnReal={h.get('jpeg_q50_frac_nn_real', float('nan')):.3f} "
                f"(CLEAN={h.get('CLEAN_frac_nn_real', float('nan')):.3f})"
                for h in hard_out
            )
        ),
        "q8_resize_real_still_real_like": (
            "; ".join(
                f"{r['real_domain']} resize_k5AI={r.get('resize_112_k5_ai_frac', float('nan')):.3f} "
                f"(CLEAN={r.get('CLEAN_k5_ai_frac', float('nan')):.3f})"
                for r in real_out
            )
        ),
        "q9_worst_real_domain": f"{worst_real['real_domain']} under resize_112 k5_ai_frac={worst_real.get('resize_112_k5_ai_frac', float('nan')):.3f}",
        "q10_largest_ai_drift": f"{worst_ai['generator_id']} jpeg euc_drift={worst_ai.get('jpeg_q50_euc_drift_mean', float('nan')):.4f}",
        "q11_h2_main_problem": f"Head-shift dominant transforms: {[c for c,m in per_mech.items() if m=='HEAD_SHIFT_DOMINANT']}",
        "q12_r1_main_problem": f"Representation-degradation dominant: {[c for c,m in per_mech.items() if m=='REPRESENTATION_DEGRADATION_DOMINANT']}",
        "q13_mixed": f"overall={decision}; per_transform={per_mech}",
        "q14_intervention": intervention,
        "q15_h2_still_baseline": (
            "YES — clean H2 remains the primary CLEAN development baseline; "
            "robustness freeze blocked until the recommended intervention is authorized."
        ),
    }

    figures = make_figures(
        {
            "geometry_macro": geometry_macro,
            "margin_macro": margin_macro,
            "alignment_macro": alignment_macro,
            "drift_macro": drift_macro,
        }
    )

    # Write CSVs (flattened)
    pd.DataFrame(
        [
            {
                "fold": r["fold"],
                "condition": r["condition"],
                "euc_overall_mean": r["euc_overall"]["mean"],
                "euc_real_mean": r["euc_real"]["mean"],
                "euc_ai_mean": r["euc_ai"]["mean"],
                "cos_sim_overall_mean": r["cos_sim_overall"]["mean"],
            }
            for r in drift_rows
        ]
    ).to_csv(DRIFT_CSV, index=False)
    pd.DataFrame(margin_rows).to_csv(MARGIN_CSV, index=False)
    pd.DataFrame(geom_rows).to_csv(GEOM_CSV, index=False)
    pd.DataFrame(gen_rows).to_csv(GEN_CSV, index=False)
    pd.DataFrame(real_rows).to_csv(REAL_CSV, index=False)

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
        "TRANSFORM_SUITE_CHANGED": "NO",
        "EVALUATION_MEMBERSHIP_CHANGED": "NO",
        "THRESHOLD_CHANGED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_TRANSFORMED_FEATURE_INFERENCE_PERFORMED": "YES" if feat_meta.get("performed") else "NO",
    }

    payload = {
        "stage": "V2-10E",
        "status": "COMPLETE",
        "decision": decision,
        "recommended_intervention": intervention,
        "per_transform_mechanism": per_mech,
        "artifact_audit": inv,
        "feature_inference": feat_meta,
        "integrity": integrity,
        "drift_macro": drift_macro,
        "margin_macro": margin_macro,
        "alignment_macro": alignment_macro,
        "geometry_macro": geometry_macro,
        "hard_generators": hard_out,
        "real_domains": real_out,
        "error_transitions": err_trans,
        "figures": figures,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "clean_h2_still_primary_baseline": True,
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Decision: {decision} | intervention: {intervention}")


if __name__ == "__main__":
    main()
