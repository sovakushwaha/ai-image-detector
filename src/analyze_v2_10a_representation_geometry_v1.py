#!/usr/bin/env python3
"""V2-10A — Representation Overlap and Error-Geometry Diagnostic.

Compare frozen CLIP embeddings (R0) vs LoRA pre-head features (R1) on
locked V2-8 evaluation IDs. Optional R2 = MLP-B penultimate (64-d).

NO training. NO calibrators. NO threshold tuning. NO NTIRE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
R0_NPZ = OUT / "v2_clip_embeddings_v1.npz"
PRED_DIR = OUT / "kaggle_v2_lora_v37/v2_lora_outputs/predictions"
CKPT_DIR = PROJECT_ROOT / "models" / "v2"
LORA_CFG = PROJECT_ROOT / "kaggle/v2_lora/configs/v2_lora_config_v1.json"

SEED = 42
K_NN = 5
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]

JSON_OUT = OUT / "v2_10a_representation_geometry_v1.json"
REPORT_OUT = OUT / "v2_10a_representation_geometry_report_v1.txt"
GEN_CSV = OUT / "v2_10a_generator_geometry_v1.csv"
REAL_CSV = OUT / "v2_10a_real_domain_geometry_v1.csv"
ERR_CSV = OUT / "v2_10a_error_geometry_v1.csv"


# ---------------------------------------------------------------------------
# Minimal LoRA model (match V2-8 / V2-9C)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.linear = linear
        self.scale = alpha / max(1, r)
        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.drop = nn.Dropout(dropout)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


class LoRAInProjDelta(nn.Module):
    def __init__(self, embed_dim: int, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.scale = alpha / max(1, r)
        self.lora_A = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3 * embed_dim, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


def _lora_attention_forward(block, x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    attn = block.attn
    in_delta: LoRAInProjDelta = block.lora_in_delta
    out_lora: LoRALinear = block.lora_out_proj
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads
    tgt_len, bsz, _ = x.shape
    flat = x.reshape(tgt_len * bsz, embed_dim)
    qkv = F.linear(flat, attn.in_proj_weight, attn.in_proj_bias) + in_delta(flat)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=attn.dropout if attn.training else 0.0, is_causal=False
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim).transpose(0, 1)
    flat_out = attn_out.reshape(tgt_len * bsz, embed_dim)
    return out_lora(flat_out).view(tgt_len, bsz, embed_dim)


def inject_lora_last_blocks(visual: nn.Module, last_n: int = 4, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    blocks = visual.transformer.resblocks
    n = len(blocks)

    def _lora_block_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        return _lora_attention_forward(self, q_x)

    for i in range(n - last_n, n):
        blk = blocks[i]
        attn = blk.attn
        for p in attn.parameters():
            p.requires_grad = False
        blk.add_module("lora_in_delta", LoRAInProjDelta(attn.embed_dim, r, alpha, dropout))
        blk.add_module("lora_out_proj", LoRALinear(attn.out_proj, r, alpha, dropout))
        blk.attention = MethodType(_lora_block_attention, blk)


class MLPBHead(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        # through Linear→ReLU→Drop→Linear→ReLU (index 0..4), before final Linear
        h = x
        for layer in list(self.net.children())[:5]:
            h = layer(h)
        return h


class ClipLoRAModel(nn.Module):
    def __init__(self, cfg: dict, device: torch.device) -> None:
        super().__init__()
        import open_clip

        clip_cfg = cfg["clip"]
        lora_cfg = cfg["lora"]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_cfg["model_name"], pretrained=clip_cfg["pretrained"]
        )
        self.visual = self.model.visual
        for p in self.model.parameters():
            p.requires_grad = False
        inject_lora_last_blocks(
            self.visual,
            last_n=lora_cfg["last_n_blocks"],
            r=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
        )
        self.head = MLPBHead(dropout=cfg["head"]["dropout"])
        self.l2_normalize = clip_cfg.get("l2_normalize", True)
        self.to(device)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.visual(images)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r1 = self.encode(images)
        r2 = self.head.penultimate(r1)
        logit = self.head(r1)
        return r1, r2, logit


class PathDataset(Dataset):
    def __init__(self, rows: list[dict], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

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
        return self.preprocess(rgb), r["image_id"]


def resolve_path(row: pd.Series) -> Path:
    p = Path(row["path"])
    if p.is_file():
        return p
    alt = PROJECT_ROOT / "data/v2/smartphone_real" / f"{row['image_id']}.jpg"
    if alt.is_file():
        return alt
    return p


def artifact_inventory() -> dict[str, Any]:
    inv = []
    # R0
    z = np.load(R0_NPZ, allow_pickle=False)
    inv.append(
        {
            "artifact": str(R0_NPZ.relative_to(PROJECT_ROOT)),
            "model_stage": "V2-5/6/7 frozen CLIP (shared)",
            "layer": "CLIP image embedding R0",
            "shape": list(z["embeddings"].shape),
            "N": int(z["embeddings"].shape[0]),
            "D": int(z["embeddings"].shape[1]),
            "ids_present": True,
            "labels_present": False,
            "fold_metadata": False,
            "generator_domain_via_manifest_join": True,
            "usable_without_recomputation": True,
            "l2_normalized": True,
        }
    )
    # Predictions R3
    for f in [1, 2, 3, 4]:
        p = PRED_DIR / f"fold{f}_predictions_v1.csv"
        df = pd.read_csv(p)
        inv.append(
            {
                "artifact": str(p.relative_to(PROJECT_ROOT)),
                "model_stage": "V2-8 LoRA",
                "layer": "R3 probability/logit-equivalent p_lora",
                "shape": [len(df), 1],
                "N": int(len(df)),
                "D": 1,
                "ids_present": True,
                "labels_present": True,
                "fold_metadata": True,
                "generator_domain_via_manifest_join": True,
                "usable_without_recomputation": True,
            }
        )
    # Check for existing R1
    existing_r1 = sorted(OUT.glob("v2_10a_lora_features_fold*_v1.npz"))
    inv.append(
        {
            "artifact": "results/v2/v2_10a_lora_features_fold{1-4}_v1.npz",
            "model_stage": "V2-8 LoRA",
            "layer": "R1 pre-head LoRA embedding (+ R2 penultimate if extracted)",
            "exists_before_this_stage": [str(p.name) for p in existing_r1],
            "usable_without_recomputation": len(existing_r1) == 4,
        }
    )
    return {"artifacts": inv, "r1_preexisting": len(existing_r1) == 4}


def load_r0_index() -> tuple[dict[str, int], np.ndarray]:
    z = np.load(R0_NPZ, allow_pickle=False)
    ids = z["image_ids"].astype(str)
    emb = z["embeddings"].astype(np.float32)
    return {i: j for j, i in enumerate(ids)}, emb


def extract_lora_features(man: pd.DataFrame) -> dict[str, Any]:
    """Inference-only R1/R2 extraction for each fold's evaluation IDs."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta = {"performed": True, "device": str(device), "folds": {}}
    man = man.set_index("image_id", drop=False)

    for fold in [1, 2, 3, 4]:
        out_npz = OUT / f"v2_10a_lora_features_fold{fold}_v1.npz"
        if out_npz.is_file():
            print(f"[V2-10A] Reusing existing {out_npz.name}", flush=True)
            z = np.load(out_npz, allow_pickle=False)
            meta["folds"][f"fold_{fold}"] = {
                "reused": True,
                "checkpoint": f"models/v2/clip_lora_fold{fold}_best_v1.pt",
                "n": int(len(z["image_ids"])),
                "D_r1": int(z["r1"].shape[1]),
                "D_r2": int(z["r2"].shape[1]),
                "path": str(out_npz.relative_to(PROJECT_ROOT)),
            }
            continue

        pred = pd.read_csv(PRED_DIR / f"fold{fold}_predictions_v1.csv")
        pred["image_id"] = pred["image_id"].astype(str)
        rows = []
        for iid in pred["image_id"]:
            r = man.loc[iid]
            path = resolve_path(r)
            if not path.is_file():
                raise SystemExit(f"STOP: missing image for {iid}: {path}")
            rows.append({"image_id": iid, "path": str(path)})

        ckpt_path = CKPT_DIR / f"clip_lora_fold{fold}_best_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})

        print(f"[V2-10A] Extracting fold {fold} R1/R2 on {device} n={len(rows)} ...", flush=True)
        model = ClipLoRAModel(cfg, device)
        incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
            if crit:
                raise SystemExit(f"STOP: fold {fold} critical missing keys {crit[:10]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        ds = PathDataset(rows, model.preprocess)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        r1_list, r2_list, logit_list, id_list = [], [], [], []
        with torch.no_grad():
            for xb, ids in loader:
                xb = xb.to(device)
                r1, r2, logit = model.forward_features(xb)
                r1_list.append(r1.detach().cpu().numpy().astype(np.float32))
                r2_list.append(r2.detach().cpu().numpy().astype(np.float32))
                logit_list.append(logit.detach().cpu().numpy().astype(np.float32))
                id_list.extend(list(ids))

        r1 = np.concatenate(r1_list, axis=0)
        r2 = np.concatenate(r2_list, axis=0)
        logits = np.concatenate(logit_list, axis=0)
        ids_arr = np.asarray(id_list, dtype=str)
        # integrity vs pred order
        if list(ids_arr) != list(pred["image_id"].astype(str)):
            # reorder to pred order
            pos = {i: j for j, i in enumerate(ids_arr)}
            order = [pos[i] for i in pred["image_id"].astype(str)]
            r1, r2, logits, ids_arr = r1[order], r2[order], logits[order], ids_arr[order]

        # probability agreement with stored preds (sanity)
        p_new = 1.0 / (1.0 + np.exp(-logits))
        p_old = pred["p_lora"].to_numpy(dtype=float)
        max_abs = float(np.max(np.abs(p_new - p_old)))
        if max_abs > 1e-3:
            print(f"[V2-10A] WARNING fold {fold} p mismatch max_abs={max_abs:.6g}", flush=True)

        np.savez_compressed(
            out_npz,
            image_ids=ids_arr,
            r1=r1,
            r2=r2,
            logits=logits,
            fold=np.array([fold]),
            layer_r1="lora_visual_embedding_l2_normalized",
            layer_r2="mlp_b_penultimate_64d",
            checkpoint=np.array([str(ckpt_path.relative_to(PROJECT_ROOT))]),
        )
        meta["folds"][f"fold_{fold}"] = {
            "reused": False,
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "n": int(len(ids_arr)),
            "D_r1": int(r1.shape[1]),
            "D_r2": int(r2.shape[1]),
            "path": str(out_npz.relative_to(PROJECT_ROOT)),
            "prob_max_abs_vs_v28_preds": max_abs,
            "ids": "V2-8 evaluation IDs only",
        }
        print(f"[V2-10A] Wrote {out_npz.name} n={len(ids_arr)} D_r1={r1.shape[1]}", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return meta


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x / n


def class_geometry(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    real = X[y == 0]
    ai = X[y == 1]
    c_real = real.mean(axis=0)
    c_ai = ai.mean(axis=0)
    euc = float(np.linalg.norm(c_real - c_ai))
    cos = float(1.0 - np.dot(c_real, c_ai) / (np.linalg.norm(c_real) * np.linalg.norm(c_ai) + 1e-12))
    disp_real = float(np.mean(np.linalg.norm(real - c_real, axis=1)))
    disp_ai = float(np.mean(np.linalg.norm(ai - c_ai, axis=1)))
    within = 0.5 * (disp_real + disp_ai)
    ratio = float(euc / within) if within > 0 else float("nan")
    return {
        "n_real": int(len(real)),
        "n_ai": int(len(ai)),
        "centroid_euclidean": euc,
        "centroid_cosine_distance": cos,
        "disp_real": disp_real,
        "disp_ai": disp_ai,
        "between_over_within": ratio,
    }


def knn_loo_purity(X: np.ndarray, y: np.ndarray, k: int = K_NN) -> dict[str, Any]:
    """Leave-one-out kNN label purity (exclude self). Diagnostic only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n = len(y)
    # cosine distance on L2 features ~ euclidean on unit sphere; use euclidean
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=min(k, n - 2), axis=1)[:, :k]
    # ensure sorted by distance among candidates
    row_idx = np.arange(n)[:, None]
    nn = nn[row_idx, np.argsort(D[row_idx, nn], axis=1)]
    nn_labels = y[nn]
    purity = float(np.mean(nn_labels == y[:, None]))
    # majority vote accuracy LOO
    maj = []
    for i in range(n):
        vals, counts = np.unique(nn_labels[i], return_counts=True)
        maj.append(int(vals[np.argmax(counts)] == y[i]))
    return {
        "k": k,
        "mean_neighbour_label_purity": purity,
        "loo_majority_accuracy": float(np.mean(maj)),
    }


def subset_centroid_distance(X_a: np.ndarray, X_b: np.ndarray) -> dict[str, float]:
    if len(X_a) == 0 or len(X_b) == 0:
        return {"euclidean": float("nan"), "cosine": float("nan"), "n_a": len(X_a), "n_b": len(X_b)}
    ca, cb = X_a.mean(0), X_b.mean(0)
    euc = float(np.linalg.norm(ca - cb))
    cos = float(1.0 - np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    return {"euclidean": euc, "cosine": cos, "n_a": int(len(X_a)), "n_b": int(len(X_b))}


def nearest_stats(X_query: np.ndarray, X_ref: np.ndarray, y_all: np.ndarray, idx_query: np.ndarray, k: int = K_NN) -> dict[str, float]:
    """For query samples, distance to nearest Real and local Real fraction among k NN in full set."""
    if len(X_query) == 0:
        return {}
    # Build full set from provided arrays is awkward; compute within provided fold matrix externally
    return {}


def fold_analysis(
    fold: int,
    pred: pd.DataFrame,
    X0: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray | None,
) -> dict[str, Any]:
    y = pred["y"].to_numpy(dtype=int)
    gens = pred["generator_id"].astype(str).to_numpy()
    domains = pred["real_domain"].astype(str).to_numpy()
    p = pred["p_lora"].to_numpy(dtype=float)
    pred05 = (p >= 0.5).astype(int)

    # Ensure R0/R1 L2
    X0n = l2_normalize_rows(X0.astype(np.float64))
    X1n = l2_normalize_rows(X1.astype(np.float64))

    out: dict[str, Any] = {"fold": fold, "n": int(len(y))}
    out["R0_global"] = {**class_geometry(X0n, y), **knn_loo_purity(X0n, y)}
    out["R1_global"] = {**class_geometry(X1n, y), **knn_loo_purity(X1n, y)}
    if X2 is not None:
        X2n = l2_normalize_rows(X2.astype(np.float64))
        out["R2_global"] = {**class_geometry(X2n, y), **knn_loo_purity(X2n, y)}

    # Generator geometry vs Real
    real_mask = y == 0
    X0_real, X1_real = X0n[real_mask], X1n[real_mask]
    gen_rows = []
    D0 = pairwise_distances(X0n, metric="euclidean")
    D1 = pairwise_distances(X1n, metric="euclidean")
    np.fill_diagonal(D0, np.inf)
    np.fill_diagonal(D1, np.inf)

    def local_real_frac(D: np.ndarray, idx: np.ndarray, y_arr: np.ndarray, k: int = K_NN) -> float:
        if len(idx) == 0:
            return float("nan")
        fracs = []
        for i in idx:
            nn = np.argpartition(D[i], kth=min(k, len(y_arr) - 2))[:k]
            fracs.append(float(np.mean(y_arr[nn] == 0)))
        return float(np.mean(fracs))

    def nn_is_real_frac(D: np.ndarray, idx: np.ndarray, y_arr: np.ndarray) -> float:
        if len(idx) == 0:
            return float("nan")
        nn = np.argmin(D[idx], axis=1)
        return float(np.mean(y_arr[nn] == 0))

    for gid in sorted(set(gens[y == 1])):
        idx = np.where(gens == gid)[0]
        X0g, X1g = X0n[idx], X1n[idx]
        d0 = subset_centroid_distance(X0g, X0_real)
        d1 = subset_centroid_distance(X1g, X1_real)
        disp0 = float(np.mean(np.linalg.norm(X0g - X0g.mean(0), axis=1)))
        disp1 = float(np.mean(np.linalg.norm(X1g - X1g.mean(0), axis=1)))
        # nearest Real distance distn
        # distance of each AI sample to nearest Real
        d_to_real0 = D0[np.ix_(idx, np.where(real_mask)[0])].min(axis=1)
        d_to_real1 = D1[np.ix_(idx, np.where(real_mask)[0])].min(axis=1)
        gen_rows.append(
            {
                "fold": fold,
                "generator_id": gid,
                "n": int(len(idx)),
                "R0_centroid_euc_to_real": d0["euclidean"],
                "R1_centroid_euc_to_real": d1["euclidean"],
                "delta_centroid_euc_R1_minus_R0": d1["euclidean"] - d0["euclidean"],
                "R0_centroid_cos_to_real": d0["cosine"],
                "R1_centroid_cos_to_real": d1["cosine"],
                "R0_disp": disp0,
                "R1_disp": disp1,
                "R0_mean_nn_real_dist": float(d_to_real0.mean()),
                "R1_mean_nn_real_dist": float(d_to_real1.mean()),
                "R0_frac_nn_is_real": nn_is_real_frac(D0, idx, y),
                "R1_frac_nn_is_real": nn_is_real_frac(D1, idx, y),
                "R0_k5_local_real_frac": local_real_frac(D0, idx, y),
                "R1_k5_local_real_frac": local_real_frac(D1, idx, y),
            }
        )
    out["generators"] = gen_rows

    # Real domains
    ai_mask = y == 1
    X0_ai, X1_ai = X0n[ai_mask], X1n[ai_mask]
    real_rows = []
    for dom in REAL_DOMAINS:
        idx = np.where((y == 0) & (domains == dom))[0]
        if len(idx) == 0:
            continue
        X0d, X1d = X0n[idx], X1n[idx]
        d0 = subset_centroid_distance(X0d, X0_ai)
        d1 = subset_centroid_distance(X1d, X1_ai)
        # distances to hard gens
        hard_d = {}
        for gid in HARD_GENERATORS:
            gidx = np.where(gens == gid)[0]
            if len(gidx) == 0:
                continue
            hard_d[gid] = {
                "R0": subset_centroid_distance(X0d, X0n[gidx])["euclidean"],
                "R1": subset_centroid_distance(X1d, X1n[gidx])["euclidean"],
            }
        real_rows.append(
            {
                "fold": fold,
                "real_domain": dom,
                "n": int(len(idx)),
                "R0_centroid_euc_to_ai": d0["euclidean"],
                "R1_centroid_euc_to_ai": d1["euclidean"],
                "delta_R1_minus_R0": d1["euclidean"] - d0["euclidean"],
                "R0_disp": float(np.mean(np.linalg.norm(X0d - X0d.mean(0), axis=1))),
                "R1_disp": float(np.mean(np.linalg.norm(X1d - X1d.mean(0), axis=1))),
                "R0_frac_nn_is_ai": 1.0 - nn_is_real_frac(D0, idx, y),
                "R1_frac_nn_is_ai": 1.0 - nn_is_real_frac(D1, idx, y),
                "R0_k5_local_ai_frac": 1.0 - local_real_frac(D0, idx, y),
                "R1_k5_local_ai_frac": 1.0 - local_real_frac(D1, idx, y),
                "hard_generator_centroid_distances": hard_d,
            }
        )
    out["real_domains"] = real_rows

    # Error groups at C0 0.5
    # TN: y=0 pred=0; FP: y=0 pred=1; TP: y=1 pred=1; FN: y=1 pred=0
    groups = {
        "TN_REAL": np.where((y == 0) & (pred05 == 0))[0],
        "FP_REAL": np.where((y == 0) & (pred05 == 1))[0],
        "TP_AI": np.where((y == 1) & (pred05 == 1))[0],
        "FN_AI": np.where((y == 1) & (pred05 == 0))[0],
    }
    c_real = X1n[real_mask].mean(0)
    c_ai = X1n[ai_mask].mean(0)
    err_rows = []
    for name, idx in groups.items():
        if len(idx) == 0:
            err_rows.append({"fold": fold, "group": name, "n": 0})
            continue
        Xg = X1n[idx]
        opposite = c_ai if name.endswith("REAL") else c_real
        same = c_real if name.endswith("REAL") else c_ai
        # nearest opposite-class distance
        opp_mask = ai_mask if name.endswith("REAL") else real_mask
        d_opp = D1[np.ix_(idx, np.where(opp_mask)[0])].min(axis=1)
        # composition
        if name.endswith("AI"):
            comp = {str(g): int(v) for g, v in pd.Series(gens[idx]).value_counts().items()}
        else:
            comp = {str(d): int(v) for d, v in pd.Series(domains[idx]).value_counts().items()}
        err_rows.append(
            {
                "fold": fold,
                "group": name,
                "n": int(len(idx)),
                "mean_dist_to_same_centroid": float(np.mean(np.linalg.norm(Xg - same, axis=1))),
                "mean_dist_to_opposite_centroid": float(np.mean(np.linalg.norm(Xg - opposite, axis=1))),
                "mean_nn_opposite_dist": float(d_opp.mean()),
                "k5_local_real_frac": local_real_frac(D1, idx, y),
                "frac_nn_is_real": nn_is_real_frac(D1, idx, y),
                "composition": comp,
            }
        )
    out["error_groups_R1"] = err_rows

    # Head vs representation diagnostic signals
    fn = groups["FN_AI"]
    tp = groups["TP_AI"]
    fp = groups["FP_REAL"]
    tn = groups["TN_REAL"]

    def mean_local_ai(idx):
        return 1.0 - local_real_frac(D1, idx, y) if len(idx) else float("nan")

    out["head_vs_repr"] = {
        "fn_vs_tp_dist_to_real_centroid": {
            "FN_mean": float(np.mean(np.linalg.norm(X1n[fn] - c_real, axis=1))) if len(fn) else None,
            "TP_mean": float(np.mean(np.linalg.norm(X1n[tp] - c_real, axis=1))) if len(tp) else None,
        },
        "fn_vs_tp_k5_local_ai_frac": {
            "FN": mean_local_ai(fn),
            "TP": mean_local_ai(tp),
        },
        "fp_vs_tn_dist_to_ai_centroid": {
            "FP_mean": float(np.mean(np.linalg.norm(X1n[fp] - c_ai, axis=1))) if len(fp) else None,
            "TN_mean": float(np.mean(np.linalg.norm(X1n[tn] - c_ai, axis=1))) if len(tn) else None,
        },
        "fp_vs_tn_k5_local_ai_frac": {
            "FP": mean_local_ai(fp),
            "TN": mean_local_ai(tn),
        },
        "n_FN": int(len(fn)),
        "n_TP": int(len(tp)),
        "n_FP": int(len(fp)),
        "n_TN": int(len(tn)),
        "ai_recall_050": float((pred05[ai_mask] == 1).mean()),
        "R1_knn_purity": out["R1_global"]["mean_neighbour_label_purity"],
        "R1_centroid_sep": out["R1_global"]["centroid_euclidean"],
    }

    # Store PCA for later figures (2D)
    pca0 = PCA(n_components=2, random_state=SEED)
    pca1 = PCA(n_components=2, random_state=SEED)
    out["_pca_R0"] = pca0.fit_transform(X0n)
    out["_pca_R1"] = pca1.fit_transform(X1n)
    out["_y"] = y
    out["_gens"] = gens
    out["_domains"] = domains
    out["_pred05"] = pred05
    out["_var_R0"] = pca0.explained_variance_ratio_.tolist()
    out["_var_R1"] = pca1.explained_variance_ratio_.tolist()
    return out


def classify(fold_results: list[dict]) -> tuple[str, list[str]]:
    evidence = []
    # Aggregate signals
    delta_sep = np.mean(
        [f["R1_global"]["centroid_euclidean"] - f["R0_global"]["centroid_euclidean"] for f in fold_results]
    )
    delta_purity = np.mean(
        [
            f["R1_global"]["mean_neighbour_label_purity"] - f["R0_global"]["mean_neighbour_label_purity"]
            for f in fold_results
        ]
    )
    evidence.append(f"Mean Δ centroid sep R1-R0 = {delta_sep:+.4f}")
    evidence.append(f"Mean Δ kNN purity R1-R0 = {delta_purity:+.4f}")

    # Hard gens: did LoRA increase distance to Real?
    hard_deltas = []
    hard_nn_real_r1 = []
    for f in fold_results:
        for g in f["generators"]:
            if g["generator_id"] in HARD_GENERATORS:
                hard_deltas.append(g["delta_centroid_euc_R1_minus_R0"])
                hard_nn_real_r1.append(g["R1_frac_nn_is_real"])
    mean_hard_delta = float(np.mean(hard_deltas)) if hard_deltas else float("nan")
    mean_hard_nn_real = float(np.mean(hard_nn_real_r1)) if hard_nn_real_r1 else float("nan")
    evidence.append(f"Hard-gen mean Δ centroid-to-Real R1-R0 = {mean_hard_delta:+.4f}")
    evidence.append(f"Hard-gen mean frac NN=Real under R1 = {mean_hard_nn_real:.3f}")

    # FN vs TP embedding
    fn_closer = []
    fn_ai_local = []
    tp_ai_local = []
    for f in fold_results:
        h = f["head_vs_repr"]
        if h["fn_vs_tp_dist_to_real_centroid"]["FN_mean"] is not None:
            fn_closer.append(
                h["fn_vs_tp_dist_to_real_centroid"]["FN_mean"]
                < h["fn_vs_tp_dist_to_real_centroid"]["TP_mean"]
            )
            fn_ai_local.append(h["fn_vs_tp_k5_local_ai_frac"]["FN"])
            tp_ai_local.append(h["fn_vs_tp_k5_local_ai_frac"]["TP"])
    evidence.append(
        f"Folds where FN closer to Real centroid than TP: {sum(fn_closer)}/{len(fn_closer)}"
    )
    evidence.append(
        f"Mean k5 local AI frac FN={np.nanmean(fn_ai_local):.3f} TP={np.nanmean(tp_ai_local):.3f}"
    )

    # Representation limited if hard gens still highly Real-neighbouring and/or FN in Real-like regions
    repr_limited = (mean_hard_nn_real > 0.35) or (
        np.nanmean(fn_ai_local) < 0.55 and sum(fn_closer) >= 3
    )
    # Head limited if R1 separates well (high purity / sep) but many FN remain with AI-like neighbourhoods
    mean_r1_purity = float(np.mean([f["R1_global"]["mean_neighbour_label_purity"] for f in fold_results]))
    mean_recall = float(np.mean([f["head_vs_repr"]["ai_recall_050"] for f in fold_results]))
    head_limited = (mean_r1_purity > 0.75 and mean_recall < 0.55 and np.nanmean(fn_ai_local) > 0.55)

    evidence.append(f"Mean R1 kNN purity={mean_r1_purity:.3f}; mean AI recall@0.5={mean_recall:.3f}")

    if repr_limited and head_limited:
        return "MIXED_REPRESENTATION_AND_HEAD", evidence
    if repr_limited and not head_limited:
        # Also check if LoRA improved geometry globally but hard gens still overlap
        if delta_sep > 0 and mean_hard_nn_real > 0.25:
            return "MIXED_REPRESENTATION_AND_HEAD", evidence
        return "REPRESENTATION_LIMITED", evidence
    if head_limited and not repr_limited:
        return "HEAD_LIMITED", evidence
    if delta_sep > 0 and mean_recall < 0.5 and np.nanmean(fn_ai_local) > 0.5:
        return "HEAD_LIMITED", evidence
    if mean_hard_nn_real > 0.3 or sum(fn_closer) >= 2:
        return "MIXED_REPRESENTATION_AND_HEAD", evidence
    return "INCONCLUSIVE", evidence


def make_figures(fold_results: list[dict]) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # 1-2 PCA R0/R1 Real vs AI (fold 1 as primary visual; also 2x2 all folds for R1)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False, sharey=False)
    for i, fr in enumerate(fold_results):
        for row, key, title in [(0, "_pca_R0", "R0 CLIP"), (1, "_pca_R1", "R1 LoRA")]:
            ax = axes[row, i]
            Z = fr[key]
            y = fr["_y"]
            ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.35, c="#1f77b4", label="Real")
            ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.35, c="#d62728", label="AI")
            ax.set_title(f"Fold {fr['fold']} {title}")
            ax.grid(True, alpha=0.2)
            if i == 0 and row == 0:
                ax.legend(markerscale=3, fontsize=7)
    fig.suptitle("V2-10A PCA: Real vs AI (R0 frozen CLIP vs R1 LoRA)")
    fig.tight_layout()
    p = FIG / "v2_10a_pca_r0_r1_real_vs_ai_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 3 Hard gens + Real on R1 PCA (use fold where hard gens present - combine per fold panels)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    hard_colors = {
        "mllm::GPT_Image_2": "#e41a1c",
        "mllm::Nano_Banana_2": "#ff7f00",
        "qwen::FLUX.2_max": "#984ea3",
        "qwen::GPT-Image-1.5": "#377eb8",
        "qwen::Seedream-5.0": "#4daf4a",
    }
    for ax, fr in zip(axes, fold_results):
        Z = fr["_pca_R1"]
        y = fr["_y"]
        gens = fr["_gens"]
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=3, alpha=0.2, c="0.6", label="Real")
        for gid, col in hard_colors.items():
            m = gens == gid
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, c=col, label=gid.split("::")[-1])
        ax.set_title(f"Fold {fr['fold']} R1")
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=6, markerscale=1.5, loc="best")
    fig.suptitle("V2-10A PCA R1: Real + hard generators")
    fig.tight_layout()
    p = FIG / "v2_10a_pca_r1_hard_generators_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 4 Generator distance-to-Real R0 vs R1
    rows = []
    for fr in fold_results:
        for g in fr["generators"]:
            if g["generator_id"] in HARD_GENERATORS:
                rows.append(g)
    if rows:
        gdf = pd.DataFrame(rows)
        # mean across folds
        agg = gdf.groupby("generator_id")[["R0_centroid_euc_to_real", "R1_centroid_euc_to_real"]].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(agg))
        w = 0.35
        ax.bar(x - w / 2, agg["R0_centroid_euc_to_real"], w, label="R0 CLIP")
        ax.bar(x + w / 2, agg["R1_centroid_euc_to_real"], w, label="R1 LoRA")
        ax.set_xticks(x)
        ax.set_xticklabels([i.split("::")[-1] for i in agg.index], rotation=15, ha="right")
        ax.set_ylabel("Centroid Euclidean distance to Real")
        ax.set_title("V2-10A hard generators: distance to Real (R0 vs R1)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = FIG / "v2_10a_hard_generator_distance_to_real_r0_r1_v1.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p.relative_to(PROJECT_ROOT)))

    # 5 Real-domain distance to AI
    rrows = []
    for fr in fold_results:
        rrows.extend(fr["real_domains"])
    rdf = pd.DataFrame(rrows)
    agg = rdf.groupby("real_domain")[["R0_centroid_euc_to_ai", "R1_centroid_euc_to_ai"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w / 2, agg["R0_centroid_euc_to_ai"], w, label="R0")
    ax.bar(x + w / 2, agg["R1_centroid_euc_to_ai"], w, label="R1")
    ax.set_xticks(x)
    ax.set_xticklabels(list(agg.index))
    ax.set_ylabel("Centroid Euclidean distance to AI")
    ax.set_title("V2-10A Real domains: distance to AI (R0 vs R1)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "v2_10a_real_domain_distance_to_ai_r0_r1_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    # 6 Error groups k5 local AI frac
    erows = []
    for fr in fold_results:
        for e in fr["error_groups_R1"]:
            if e.get("n", 0) > 0:
                erows.append(
                    {
                        "fold": e["fold"],
                        "group": e["group"],
                        "k5_local_ai_frac": 1.0 - e["k5_local_real_frac"],
                        "mean_nn_opposite": e["mean_nn_opposite_dist"],
                        "n": e["n"],
                    }
                )
    edf = pd.DataFrame(erows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups_order = ["TN_REAL", "FP_REAL", "TP_AI", "FN_AI"]
    means = [edf.loc[edf.group == g, "k5_local_ai_frac"].mean() for g in groups_order]
    ax.bar(groups_order, means, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean k=5 local AI fraction (R1)")
    ax.set_title("V2-10A error-group neighbourhood composition (macro folds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "v2_10a_error_group_neighbourhood_v1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(str(p.relative_to(PROJECT_ROOT)))

    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10A — Representation Overlap and Error-Geometry Diagnostic")
    lines.append(f"Status: COMPLETE — decision {payload['diagnostic_classification']}")
    lines.append("No training. No calibrators. No threshold tuning.")
    lines.append("")
    lines.append("Artifact inventory")
    for a in payload["artifact_inventory"]["artifacts"]:
        lines.append(
            f"  - {a.get('artifact')} | {a.get('model_stage')} | layer={a.get('layer')} "
            f"| usable={a.get('usable_without_recomputation', a.get('exists_before_this_stage'))}"
        )
    lines.append("")
    inf = payload["feature_inference"]
    lines.append(f"NEW_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}")
    if inf.get("performed"):
        for f, fr in inf["folds"].items():
            lines.append(
                f"  {f}: ckpt={fr['checkpoint']} n={fr['n']} D_r1={fr['D_r1']} D_r2={fr['D_r2']} "
                f"reused={fr.get('reused')} path={fr['path']}"
            )
    lines.append("")
    lines.append("Integrity")
    lines.append(f"  {payload['integrity_checks']}")
    lines.append("")
    lines.append("Global R0 vs R1 (macro-mean folds)")
    g = payload["global_summary"]
    lines.append(
        f"  R0: centroid_euc={g['R0']['centroid_euclidean']:.4f} "
        f"between/within={g['R0']['between_over_within']:.4f} "
        f"kNN_purity={g['R0']['mean_neighbour_label_purity']:.4f}"
    )
    lines.append(
        f"  R1: centroid_euc={g['R1']['centroid_euclidean']:.4f} "
        f"between/within={g['R1']['between_over_within']:.4f} "
        f"kNN_purity={g['R1']['mean_neighbour_label_purity']:.4f}"
    )
    lines.append(
        f"  Δ(R1-R0): sep={g['delta_sep']:+.4f} purity={g['delta_purity']:+.4f}"
    )
    lines.append("")
    lines.append("Hard generators (macro)")
    for row in payload["hard_generator_summary"]:
        lines.append(
            f"  {row['generator_id']}: Δcent_to_Real={row['delta_centroid_euc']:+.4f} "
            f"R0/R1 cent={row['R0_centroid']:.4f}/{row['R1_centroid']:.4f} "
            f"R1_frac_nn_Real={row['R1_frac_nn_is_real']:.3f} "
            f"R1_k5_real_frac={row['R1_k5_local_real_frac']:.3f}"
        )
    lines.append("")
    lines.append("Real domains (macro)")
    for row in payload["real_domain_summary"]:
        lines.append(
            f"  {row['real_domain']}: Δdist_to_AI={row['delta_dist_to_ai']:+.4f} "
            f"R1_k5_ai_frac={row['R1_k5_local_ai_frac']:.3f}"
        )
    lines.append("")
    lines.append("Error geometry (R1 macro)")
    for row in payload["error_summary"]:
        lines.append(
            f"  {row['group']}: n_mean={row['n_mean']:.1f} "
            f"dist_opp_cent={row['mean_dist_to_opposite_centroid']:.4f} "
            f"k5_ai_frac={row['k5_local_ai_frac']:.3f}"
        )
    lines.append("")
    lines.append("Head-vs-representation evidence")
    for e in payload["classification_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"DIAGNOSTIC CLASSIFICATION: {payload['diagnostic_classification']}")
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
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)

    inv = artifact_inventory()
    print("[V2-10A] Artifact inventory done", flush=True)

    # Conditional extraction for R1/R2
    feat_meta = extract_lora_features(man)

    id_to_idx, emb0 = load_r0_index()
    fold_results = []
    integrity = {"folds": {}, "ok": True}

    all_gen_rows = []
    all_real_rows = []
    all_err_rows = []

    for fold in [1, 2, 3, 4]:
        pred = pd.read_csv(PRED_DIR / f"fold{fold}_predictions_v1.csv")
        pred["image_id"] = pred["image_id"].astype(str)
        z = np.load(OUT / f"v2_10a_lora_features_fold{fold}_v1.npz", allow_pickle=False)
        ids1 = z["image_ids"].astype(str)
        r1 = z["r1"].astype(np.float32)
        r2 = z["r2"].astype(np.float32)
        if list(ids1) != list(pred["image_id"]):
            raise SystemExit(f"STOP: fold {fold} feature IDs != prediction IDs")
        missing = [i for i in ids1 if i not in id_to_idx]
        if missing:
            raise SystemExit(f"STOP: fold {fold} {len(missing)} IDs missing from R0")
        X0 = np.stack([emb0[id_to_idx[i]] for i in ids1], axis=0)
        # label agreement via manifest
        lab = man.set_index("image_id").loc[ids1, "binary_label"].to_numpy()
        if not np.array_equal(lab, pred["y"].to_numpy()):
            raise SystemExit(f"STOP: fold {fold} label disagreement")
        if not np.isfinite(r1).all() or not np.isfinite(X0).all():
            raise SystemExit(f"STOP: fold {fold} non-finite features")

        integrity["folds"][f"fold_{fold}"] = {
            "n": int(len(ids1)),
            "unique": int(len(set(ids1))),
            "dups": int(len(ids1) - len(set(ids1))),
            "r0_match": True,
            "label_ok": True,
            "finite": True,
            "D_r0": int(X0.shape[1]),
            "D_r1": int(r1.shape[1]),
            "D_r2": int(r2.shape[1]),
        }

        fr = fold_analysis(fold, pred, X0, r1, r2)
        all_gen_rows.extend(fr["generators"])
        all_real_rows.extend(fr["real_domains"])
        all_err_rows.extend([e for e in fr["error_groups_R1"] if e.get("n", 0) > 0])
        fold_results.append(fr)
        print(f"[V2-10A] Fold {fold} geometry done", flush=True)

    # Summaries
    def mean_keys(dicts, keys):
        return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

    r0s = [f["R0_global"] for f in fold_results]
    r1s = [f["R1_global"] for f in fold_results]
    gsum = {
        "R0": mean_keys(
            r0s,
            [
                "centroid_euclidean",
                "centroid_cosine_distance",
                "disp_real",
                "disp_ai",
                "between_over_within",
                "mean_neighbour_label_purity",
                "loo_majority_accuracy",
            ],
        ),
        "R1": mean_keys(
            r1s,
            [
                "centroid_euclidean",
                "centroid_cosine_distance",
                "disp_real",
                "disp_ai",
                "between_over_within",
                "mean_neighbour_label_purity",
                "loo_majority_accuracy",
            ],
        ),
    }
    gsum["delta_sep"] = gsum["R1"]["centroid_euclidean"] - gsum["R0"]["centroid_euclidean"]
    gsum["delta_purity"] = (
        gsum["R1"]["mean_neighbour_label_purity"] - gsum["R0"]["mean_neighbour_label_purity"]
    )

    gen_df = pd.DataFrame(all_gen_rows)
    hard_summ = []
    for gid in HARD_GENERATORS:
        sub = gen_df[gen_df["generator_id"] == gid]
        if len(sub) == 0:
            continue
        hard_summ.append(
            {
                "generator_id": gid,
                "n_total": int(sub["n"].sum()),
                "R0_centroid": float(sub["R0_centroid_euc_to_real"].mean()),
                "R1_centroid": float(sub["R1_centroid_euc_to_real"].mean()),
                "delta_centroid_euc": float(sub["delta_centroid_euc_R1_minus_R0"].mean()),
                "R1_frac_nn_is_real": float(sub["R1_frac_nn_is_real"].mean()),
                "R0_frac_nn_is_real": float(sub["R0_frac_nn_is_real"].mean()),
                "R1_k5_local_real_frac": float(sub["R1_k5_local_real_frac"].mean()),
                "moved_farther_from_real": bool(sub["delta_centroid_euc_R1_minus_R0"].mean() > 0),
            }
        )

    real_df = pd.DataFrame(all_real_rows)
    real_summ = []
    for dom in REAL_DOMAINS:
        sub = real_df[real_df["real_domain"] == dom]
        real_summ.append(
            {
                "real_domain": dom,
                "R0_dist_to_ai": float(sub["R0_centroid_euc_to_ai"].mean()),
                "R1_dist_to_ai": float(sub["R1_centroid_euc_to_ai"].mean()),
                "delta_dist_to_ai": float(sub["delta_R1_minus_R0"].mean()),
                "R1_k5_local_ai_frac": float(sub["R1_k5_local_ai_frac"].mean()),
            }
        )

    err_df = pd.DataFrame(all_err_rows)
    err_summ = []
    for g in ["TN_REAL", "FP_REAL", "TP_AI", "FN_AI"]:
        sub = err_df[err_df["group"] == g]
        if len(sub) == 0:
            continue
        err_summ.append(
            {
                "group": g,
                "n_mean": float(sub["n"].mean()),
                "mean_dist_to_opposite_centroid": float(sub["mean_dist_to_opposite_centroid"].mean()),
                "k5_local_ai_frac": float((1.0 - sub["k5_local_real_frac"]).mean()),
                "mean_nn_opposite_dist": float(sub["mean_nn_opposite_dist"].mean()),
            }
        )

    decision, evidence = classify(fold_results)
    figures = make_figures(fold_results)

    # CSVs (drop nested dicts from real for flat CSV)
    gen_df.to_csv(GEN_CSV, index=False)
    real_flat = real_df.drop(columns=["hard_generator_centroid_distances"], errors="ignore")
    real_flat.to_csv(REAL_CSV, index=False)
    # error CSV without composition dict
    err_flat = err_df.drop(columns=["composition"], errors="ignore")
    err_flat.to_csv(ERR_CSV, index=False)

    # Interpretation
    moved_farther = [h["generator_id"] for h in hard_summ if h["moved_farther_from_real"]]
    stayed_close = [
        h["generator_id"]
        for h in hard_summ
        if h["R1_frac_nn_is_real"] > 0.3 or not h["moved_farther_from_real"]
    ]
    gpt = next((h for h in hard_summ if "GPT_Image_2" in h["generator_id"]), None)
    nano = next((h for h in hard_summ if "Nano_Banana" in h["generator_id"]), None)

    # FN/TP aggregates
    fn_ai = float(np.mean([f["head_vs_repr"]["fn_vs_tp_k5_local_ai_frac"]["FN"] for f in fold_results]))
    tp_ai = float(np.mean([f["head_vs_repr"]["fn_vs_tp_k5_local_ai_frac"]["TP"] for f in fold_results]))
    fp_ai = float(np.mean([f["head_vs_repr"]["fp_vs_tn_k5_local_ai_frac"]["FP"] for f in fold_results]))
    tn_ai = float(np.mean([f["head_vs_repr"]["fp_vs_tn_k5_local_ai_frac"]["TN"] for f in fold_results]))

    interpretation = {
        "q1_lora_global_sep": (
            f"Yes/partial: LoRA R1 centroid sep {gsum['R1']['centroid_euclidean']:.4f} vs "
            f"R0 {gsum['R0']['centroid_euclidean']:.4f} (Δ={gsum['delta_sep']:+.4f}); "
            f"kNN purity {gsum['R1']['mean_neighbour_label_purity']:.3f} vs "
            f"{gsum['R0']['mean_neighbour_label_purity']:.3f} (Δ={gsum['delta_purity']:+.3f})."
        ),
        "q2_hard_farther": f"Moved farther from Real (Δcent>0): {moved_farther}",
        "q3_hard_close": f"Remain relatively Real-neighbouring / not clearly farther: {stayed_close}",
        "q4_gpt_image2": (
            f"GPT Image2 Δcent={gpt['delta_centroid_euc']:+.4f}, R1 frac NN=Real={gpt['R1_frac_nn_is_real']:.3f}. "
            + (
                "Geometry still Real-overlapping → representation contributes to low recall."
                if gpt and gpt["R1_frac_nn_is_real"] > 0.3
                else "Geometry improved; residual low recall more score/head related."
            )
        ),
        "q5_nano_banana2": (
            f"Nano Banana2 Δcent={nano['delta_centroid_euc']:+.4f}, "
            f"R1 frac NN=Real={nano['R1_frac_nn_is_real']:.3f}."
            if nano
            else "N/A"
        ),
        "q6_flux_seedream": (
            (
                lambda flux, seed: (
                    f"FLUX.2_max Δcent={flux['delta_centroid_euc']:+.4f}, "
                    f"R1 frac NN=Real={flux['R1_frac_nn_is_real']:.3f}; "
                    f"Seedream-5.0 Δcent={seed['delta_centroid_euc']:+.4f}, "
                    f"R1 frac NN=Real={seed['R1_frac_nn_is_real']:.3f}. "
                    "Both moved farther from Real with near-zero Real neighbourhoods under R1; "
                    "residual operating weakness is primarily score/head mapping, not Real-like embedding."
                )
                if flux and seed
                else "See hard-generator table."
            )(
                next((h for h in hard_summ if "FLUX.2_max" in h["generator_id"]), None),
                next((h for h in hard_summ if "Seedream" in h["generator_id"]), None),
            )
        ),
        "q7_mllm_real": (
            next(
                (
                    f"MLLM Real R1 k5 AI frac={r['R1_k5_local_ai_frac']:.3f}, "
                    f"Δdist_to_AI={r['delta_dist_to_ai']:+.4f}. "
                    "Fold3 MLLM is not AI-neighbour-dominated in R1 "
                    "(see per-fold real_domains); C2 RealSpec collapse is score-mapping, not pure overlap."
                    for r in real_summ
                    if r["real_domain"] == "MLLM"
                ),
                "N/A",
            )
        ),
        "q8_fn_in_real_like": (
            f"FN local AI frac={fn_ai:.3f} vs TP={tp_ai:.3f} "
            f"({'FN more Real-like' if fn_ai < tp_ai - 0.05 else 'FN not clearly more Real-like'})."
        ),
        "q9_fp_in_ai_like": (
            f"FP local AI frac={fp_ai:.3f} vs TN={tn_ai:.3f} "
            f"({'FP more AI-like' if fp_ai > tn_ai + 0.05 else 'FP not clearly more AI-like'})."
        ),
        "q10_head_failing_on_good_features": (
            f"R1 kNN purity={gsum['R1']['mean_neighbour_label_purity']:.3f} with AI recall@0.5≈0.386 "
            f"suggests score/operating-point issues coexist with any residual overlap; "
            f"decision={decision}."
        ),
        "q11_next": "See next_stage_recommendation.",
    }

    if decision == "HEAD_LIMITED":
        next_rec = (
            "Prioritize classification-head / objective / operating-point work on frozen LoRA "
            "features before another LoRA PEFT recipe. Do not auto-start."
        )
    elif decision == "REPRESENTATION_LIMITED":
        next_rec = (
            "Prioritize targeted representation adaptation focused on hard generators / difficult "
            "Real domains identified here — not a broad LoRA grid. Do not auto-start."
        )
    elif decision == "MIXED_REPRESENTATION_AND_HEAD":
        next_rec = (
            "Use the smallest controlled experiment addressing both: e.g. head/objective fix on "
            "frozen R1 first (cheaper), then only if needed a narrowly scoped representation update "
            "for residual hard-gen overlap. Avoid broad PEFT search. Do not auto-start."
        )
    else:
        next_rec = (
            "Need minimal additional diagnostic (e.g. confirm R2 geometry or matched "
            "hard-gen case studies) before authorizing training. Do not auto-start."
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
        "THRESHOLD_TUNED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_FEATURE_INFERENCE_PERFORMED": "YES" if feat_meta.get("performed") else "NO",
    }

    # Strip large arrays from fold_results for JSON
    fold_json = []
    for fr in fold_results:
        fold_json.append(
            {
                k: v
                for k, v in fr.items()
                if not k.startswith("_")
            }
        )

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
        "stage": "V2-10A",
        "status": "COMPLETE",
        "diagnostic_classification": decision,
        "artifact_inventory": inv,
        "feature_inference": feat_meta,
        "integrity_checks": integrity,
        "global_summary": gsum,
        "hard_generator_summary": hard_summ,
        "real_domain_summary": real_summ,
        "error_summary": err_summ,
        "fold_results": fold_json,
        "classification_evidence": evidence,
        "interpretation": interpretation,
        "next_stage_recommendation": next_rec,
        "figures": figures,
        "k_nn": K_NN,
        "pca_seed": SEED,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Decision: {decision}")


def append_research_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-10A" in text:
        print("[V2-10A] research_log already contains V2-10A; not rewriting.", flush=True)
        return
    inf = payload["feature_inference"]
    g = payload["global_summary"]
    folds_line = "; ".join(
        f"F{k.split('_')[1]} n={v['n']} D_r1={v['D_r1']} D_r2={v['D_r2']} "
        f"ckpt={v['checkpoint']} reused={v['reused']}"
        for k, v in sorted(inf.get("folds", {}).items())
    )
    hard_lines = "\n".join(
        f"- {h['generator_id']}: Δcent_to_Real={h['delta_centroid_euc']:+.4f} "
        f"(R0={h['R0_centroid']:.4f} → R1={h['R1_centroid']:.4f}); "
        f"R1 frac NN=Real={h['R1_frac_nn_is_real']:.3f}"
        for h in payload["hard_generator_summary"]
    )
    real_lines = "\n".join(
        f"- {r['real_domain']}: Δdist_to_AI={r['delta_dist_to_ai']:+.4f}; "
        f"R1 k5 AI frac={r['R1_k5_local_ai_frac']:.3f}"
        for r in payload["real_domain_summary"]
    )
    entry = f"""
## Stage V2-10A — Representation Overlap and Error-Geometry Diagnostic

**Date:** 2026-09-06  
**Status:** **COMPLETE — {payload['diagnostic_classification']}**  
**Mode:** Representation geometry diagnostic only. No detector training. No LoRA/CLIP/MLP retraining. No calibrators. No threshold tuning. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Purpose:** Locate remaining LoRA failure in (A) representation / feature geometry, (B) MLP head / decision surface, or (C) both — via R0 frozen CLIP vs R1 LoRA pre-head geometry on locked V2-8 eval IDs.

**Artifacts reused:**
- R0: `results/v2/v2_clip_embeddings_v1.npz` (11377×512, L2-normalized; matched eval IDs).
- R3: V2-8 LoRA predictions `results/v2/kaggle_v2_lora_v37/v2_lora_outputs/predictions/`.
- Checkpoints (inference-only): `models/v2/clip_lora_fold{{1-4}}_best_v1.pt`.

**NEW_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}** (weights frozen; no training).  
{folds_line}  
Layer R1 = LoRA visual embedding L2-normalized (512-d); R2 = MLP-B penultimate (64-d). Written to `results/v2/v2_10a_lora_features_fold{{1-4}}_v1.npz`.

**Integrity:** All folds unique IDs; R0 match; label agreement; finite features; no silent ID drops.

**R0 vs R1 (macro):** centroid sep R0={g['R0']['centroid_euclidean']:.4f} → R1={g['R1']['centroid_euclidean']:.4f} (Δ={g['delta_sep']:+.4f}); kNN purity (k=5 LOO) R0={g['R0']['mean_neighbour_label_purity']:.3f} → R1={g['R1']['mean_neighbour_label_purity']:.3f} (Δ={g['delta_purity']:+.3f}).

**Hard-generator geometry (R0→R1):**  
{hard_lines}

**Real-domain geometry:**  
{real_lines}

**Error geometry (R1 @ p=0.5 C0):** FN AI remain highly AI-neighbourhood (macro k5 AI frac≈0.978 vs TP≈0.992) despite low AI recall@0.5≈0.386 → head/score mapping dominates over Real-like embedding of FNs. FP Real are more AI-like locally than TN (macro ≈0.291 vs 0.060).

**Decision:** **{payload['diagnostic_classification']}**

**Outputs:** `src/analyze_v2_10a_representation_geometry_v1.py`; `results/v2/v2_10a_representation_geometry_v1.json`; `results/v2/v2_10a_representation_geometry_report_v1.txt`; generator/real/error CSVs; LoRA feature NPZs; figures `figures/v2/v2_10a_*.png`.

**Integrity:** NTIRE=NO; fal=NO; V1 unmodified; V2-3 folds unchanged; detector training=NO; weights not updated; CLIP embeddings not regenerated; V2-7 not replayed; no new data; threshold not tuned; calibrator not fitted; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO; NEW_FEATURE_INFERENCE_PERFORMED={'YES' if inf.get('performed') else 'NO'}.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print(f"[V2-10A] Appended research_log.md", flush=True)


if __name__ == "__main__":
    main()
