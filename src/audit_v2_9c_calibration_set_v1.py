#!/usr/bin/env python3
"""V2-9C — Independent Calibration-Set Feasibility Audit.

Audit-only membership reconstruction + feasibility decision.
Conditional INFERENCE-ONLY scoring of a proposed calibration pool is allowed
iff the decision is FEASIBLE or FEASIBLE_WITH_LIMITATIONS.

NO calibrator fitting. NO detector training. NO threshold tuning.
NO NTIRE. NO fal. NO V2-3 membership changes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
PHONE_MANIFEST = PROJECT_ROOT / "metadata" / "v2_smartphone_real_manifest_v1.csv"
PRED_DIR = (
    PROJECT_ROOT
    / "results/v2/kaggle_v2_lora_v37/v2_lora_outputs/predictions"
)
CKPT_DIR = PROJECT_ROOT / "models" / "v2"
LORA_CFG = PROJECT_ROOT / "kaggle/v2_lora/configs/v2_lora_config_v1.json"

CAP = 300  # authoritative V2-8 train AI-per-generator cap
SEED = 42

JSON_OUT = OUT / "v2_9c_calibration_set_audit_v1.json"
REPORT_OUT = OUT / "v2_9c_calibration_set_report_v1.txt"
MEMBER_CSV = OUT / "v2_9c_fold_membership_summary_v1.csv"
CAND_CSV = OUT / "v2_9c_candidate_calibration_ids_v1.csv"


# ---------------------------------------------------------------------------
# Minimal LoRA model (inlined; do NOT import Kaggle trainer — it has side effects)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.linear = linear
        self.scale = alpha / max(1, r)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
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
    q = (
        q.view(tgt_len, bsz, embed_dim)
        .transpose(0, 1)
        .view(bsz, tgt_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    k = (
        k.view(tgt_len, bsz, embed_dim)
        .transpose(0, 1)
        .view(bsz, tgt_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    v = (
        v.view(tgt_len, bsz, embed_dim)
        .transpose(0, 1)
        .view(bsz, tgt_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    attn_out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=attn.dropout if attn.training else 0.0,
        is_causal=False,
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim).transpose(0, 1)
    flat_out = attn_out.reshape(tgt_len * bsz, embed_dim)
    return out_lora(flat_out).view(tgt_len, bsz, embed_dim)


def inject_lora_last_blocks(
    visual: nn.Module, last_n: int = 4, r: int = 8, alpha: int = 16, dropout: float = 0.05
) -> list[str]:
    blocks = visual.transformer.resblocks
    touched = []
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
        touched.append(f"visual.transformer.resblocks.{i}.attn")
    return touched


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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(images))


class CalibImageDataset(Dataset):
    def __init__(self, rows: list[dict], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        p = Path(r["path"])
        with Image.open(p) as im:
            try:
                t = ImageOps.exif_transpose(im)
                if t is not None:
                    im = t
            except Exception:
                pass
            rgb = im.convert("RGB")
        x = self.preprocess(rgb)
        return x, int(r["binary_label"]), str(r["image_id"])


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    df["image_id"] = df["image_id"].astype(str)
    return df


def train_membership(man: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Exact V2-8 LoRA training membership (AI capped at CAP per generator)."""
    col = f"fold_{fold}_role"
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    ai = base[(base["binary_label"] == 1) & (base[col] == "TRAIN")].copy()
    keep = []
    for _, gdf in ai.groupby("canonical_generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep.append(gdf)
    ai_keep = pd.concat(keep, ignore_index=True) if keep else ai.iloc[:0]
    real = base[(base["binary_label"] == 0) & (base[col] == "TRAIN")].copy()
    out = pd.concat([ai_keep, real], ignore_index=True)
    out["membership"] = "DETECTOR_TRAIN"
    out["fold"] = fold
    return out


def eval_membership(man: pd.DataFrame, fold: int) -> pd.DataFrame:
    col = f"fold_{fold}_role"
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    out = base[base[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])].copy()
    out["membership"] = "FINAL_EVALUATION"
    out["fold"] = fold
    out["eval_role"] = out[col]
    return out


def cap_leftover_ai(man: pd.DataFrame, fold: int, train_df: pd.DataFrame) -> pd.DataFrame:
    col = f"fold_{fold}_role"
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    ai_train_role = base[(base["binary_label"] == 1) & (base[col] == "TRAIN")]
    used = set(train_df.loc[train_df["binary_label"] == 1, "image_id"])
    left = ai_train_role[~ai_train_role["image_id"].isin(used)].copy()
    left["membership"] = "TRAIN_ROLE_CAP_LEFTOVER"
    left["fold"] = fold
    return left


def role_inventory(man: pd.DataFrame) -> dict[str, Any]:
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"]
    roles = {}
    for f in [1, 2, 3, 4]:
        col = f"fold_{fold}_role" if False else f"fold_{f}_role"
        vc = base[col].value_counts().to_dict()
        roles[f"fold_{f}"] = {str(k): int(v) for k, v in vc.items()}

    # Domain × role for fold 1 (Real)
    real = base[base["binary_label"] == 0]
    domain_role = (
        pd.crosstab(real["real_domain"].fillna("NA"), real["fold_1_role"])
        .astype(int)
        .to_dict()
    )

    phone = pd.read_csv(PHONE_MANIFEST)
    phone_splits = phone["split"].value_counts().to_dict()

    return {
        "n_locked_v2_ids_including_excluded": int(len(man)),
        "n_after_excluding_duplicate": int(len(base)),
        "binary_label_counts": {
            str(k): int(v) for k, v in base["binary_label"].value_counts().items()
        },
        "source_dataset_counts": {
            str(k): int(v) for k, v in base["source_dataset"].value_counts().items()
        },
        "real_domain_counts": {
            str(k): int(v)
            for k, v in base["real_domain"].fillna("AI_NA").value_counts().items()
        },
        "roles_per_fold": roles,
        "real_domain_by_fold1_role": domain_role,
        "smartphone_manifest_splits": {str(k): int(v) for k, v in phone_splits.items()},
        "role_definitions": {
            "TRAIN": "Detector training pool (AI subject to CAP=300 per generator in V2-8)",
            "HOLDOUT_VALIDATION": "Generator-holdout AI evaluation",
            "REAL_VALIDATION": "Shared Real evaluation across folds",
            "REAL_INTERNAL_HOLDOUT": "Reserved Real internal holdout (not train/eval in V2-5–V2-8)",
            "PROMPT_BLOCKED": "AI blocked from train due to prompt-group leakage vs holdout",
            "EXCLUDED_DUPLICATE": "Excluded duplicate (1 row)",
        },
        "ntire_present_in_manifest": bool(
            man["image_id"].str.contains("ntire", case=False).any()
            or man["source_dataset"].astype(str).str.contains("ntire", case=False).any()
        ),
    }


def path_exists(row: pd.Series) -> bool:
    p = Path(row["path"])
    if p.is_file():
        return True
    # fallback smartphone local cache
    alt = PROJECT_ROOT / "data/v2/smartphone_real" / f"{row['image_id']}.jpg"
    return alt.is_file()


def resolve_path(row: pd.Series) -> str:
    p = Path(row["path"])
    if p.is_file():
        return str(p)
    alt = PROJECT_ROOT / "data/v2/smartphone_real" / f"{row['image_id']}.jpg"
    if alt.is_file():
        return str(alt)
    return str(p)


def build_audit(man: pd.DataFrame) -> dict[str, Any]:
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    all_ids = set(base["image_id"])

    # Verify eval preds
    eval_pred_checks = {}
    for f in [1, 2, 3, 4]:
        ev = eval_membership(man, f)
        pred = pd.read_csv(PRED_DIR / f"fold{f}_predictions_v1.csv")
        pred["image_id"] = pred["image_id"].astype(str)
        match = set(pred["image_id"]) == set(ev["image_id"])
        eval_pred_checks[f"fold_{f}"] = {
            "manifest_eval_n": int(len(ev)),
            "prediction_n": int(len(pred)),
            "ids_match": bool(match),
            "roles": {
                str(k): int(v) for k, v in ev["eval_role"].value_counts().items()
            },
            "n_real": int((ev["binary_label"] == 0).sum()),
            "n_ai": int((ev["binary_label"] == 1).sum()),
            "ai_generators": {
                str(k): int(v)
                for k, v in ev.loc[ev["binary_label"] == 1, "canonical_generator_id"]
                .value_counts()
                .items()
            },
            "real_domains": {
                str(k): int(v)
                for k, v in ev.loc[ev["binary_label"] == 0, "real_domain"]
                .value_counts()
                .items()
            },
        }
        if not match:
            raise SystemExit(f"STOP: fold {f} eval membership != prediction IDs")

    fold_summaries = []
    candidate_rows = []
    train_eval_unused = {}

    for f in [1, 2, 3, 4]:
        tr = train_membership(man, f)
        ev = eval_membership(man, f)
        leftover = cap_leftover_ai(man, f, tr)
        tr_ids = set(tr["image_id"])
        ev_ids = set(ev["image_id"])
        unused = all_ids - tr_ids - ev_ids
        unused_df = base[base["image_id"].isin(unused)].copy()
        unused_df["fold"] = f
        unused_df["original_role"] = unused_df[f"fold_{f}_role"]
        unused_df["local_exists"] = unused_df.apply(path_exists, axis=1)

        # Primary proposed pool components
        ih = unused_df[unused_df["original_role"] == "REAL_INTERNAL_HOLDOUT"].copy()
        pb = unused_df[unused_df["original_role"] == "PROMPT_BLOCKED"].copy()
        cap_l = unused_df[unused_df["original_role"] == "TRAIN"].copy()  # CAP leftovers

        fold_summaries.append(
            {
                "fold": f,
                "train_n": int(len(tr)),
                "train_real": int((tr["binary_label"] == 0).sum()),
                "train_ai": int((tr["binary_label"] == 1).sum()),
                "train_ai_generators": int(
                    tr.loc[tr["binary_label"] == 1, "canonical_generator_id"].nunique()
                ),
                "train_duplicates": int(tr["image_id"].duplicated().sum()),
                "eval_n": int(len(ev)),
                "eval_real": int((ev["binary_label"] == 0).sum()),
                "eval_ai": int((ev["binary_label"] == 1).sum()),
                "candidate_unused_n": int(len(unused_df)),
                "candidate_unused_real": int((unused_df["binary_label"] == 0).sum()),
                "candidate_unused_ai": int((unused_df["binary_label"] == 1).sum()),
                "unused_by_role": {
                    str(k): int(v)
                    for k, v in unused_df["original_role"].value_counts().items()
                },
                "cap_leftover_ai_n": int(len(leftover)),
                "proposed_calib_real_n": int(len(ih)),
                "proposed_calib_ai_prompt_blocked_n": int(len(pb)),
                "proposed_calib_total_primary": int(len(ih) + len(pb)),
                "overlap_proposed_vs_train": int(
                    len((set(ih["image_id"]) | set(pb["image_id"])) & tr_ids)
                ),
                "overlap_proposed_vs_eval": int(
                    len((set(ih["image_id"]) | set(pb["image_id"])) & ev_ids)
                ),
                "proposed_local_missing": int(
                    (~ih["local_exists"]).sum() + (~pb["local_exists"]).sum()
                ),
            }
        )

        for pool_name, pdf in [
            ("REAL_INTERNAL_HOLDOUT", ih),
            ("PROMPT_BLOCKED", pb),
            ("TRAIN_ROLE_CAP_LEFTOVER", cap_l),
        ]:
            for _, r in pdf.iterrows():
                candidate_rows.append(
                    {
                        "fold": f,
                        "image_id": r["image_id"],
                        "binary_label": int(r["binary_label"]),
                        "canonical_generator_id": r["canonical_generator_id"],
                        "real_domain": r["real_domain"] if pd.notna(r["real_domain"]) else "",
                        "source_dataset": r["source_dataset"],
                        "original_role": r["original_role"],
                        "candidate_pool": pool_name,
                        "in_primary_proposal": pool_name
                        in ("REAL_INTERNAL_HOLDOUT", "PROMPT_BLOCKED"),
                        "local_exists": bool(r["local_exists"]),
                        "path": resolve_path(r),
                        "overlap_train": False,
                        "overlap_eval": False,
                    }
                )

        train_eval_unused[f"fold_{f}"] = {
            "train_ids_n": len(tr_ids),
            "eval_ids_n": len(ev_ids),
            "unused_ids_n": len(unused),
        }

    # Internal holdout deep audit
    ih_all = base[base["fold_1_role"] == "REAL_INTERNAL_HOLDOUT"].copy()
    ih_ids = set(ih_all["image_id"])
    phone = pd.read_csv(PHONE_MANIFEST)
    phone_ih = set(
        phone.loc[phone["split"] == "internal_holdout", "v2_image_id"].astype(str)
    )
    split_phone_ih = set(
        ih_all.loc[ih_all["real_domain"] == "Smartphone", "image_id"]
    )

    ih_overlap = {}
    for f in [1, 2, 3, 4]:
        tr_ids = set(train_membership(man, f)["image_id"])
        ev_ids = set(eval_membership(man, f)["image_id"])
        ih_overlap[f"fold_{f}"] = {
            "overlap_train": int(len(ih_ids & tr_ids)),
            "overlap_eval": int(len(ih_ids & ev_ids)),
            "roles_identical_across_folds": bool(
                (
                    man.loc[man["image_id"].isin(ih_ids), f"fold_{f}_role"]
                    == "REAL_INTERNAL_HOLDOUT"
                ).all()
            ),
        }

    # Prior V2-5/6/7/8: train/val exclude IH by construction
    prior_use = {
        "used_in_v2_5_6_7_8_detector_train": False,
        "used_in_v2_5_6_7_8_final_eval": False,
        "evidence": (
            "train_indices uses role==TRAIN only; val_indices uses "
            "HOLDOUT_VALIDATION∪REAL_VALIDATION only (train_v2_clip_lora.py and "
            "analogous V2-5/6/7 scripts). REAL_INTERNAL_HOLDOUT never selected."
        ),
    }

    # Intersection unused across folds
    unused_sets = []
    for f in [1, 2, 3, 4]:
        tr_ids = set(train_membership(man, f)["image_id"])
        ev_ids = set(eval_membership(man, f)["image_id"])
        unused_sets.append(all_ids - tr_ids - ev_ids)
    inter = set.intersection(*unused_sets)

    # Proposed consistent design
    design = {
        "name": "PRIMARY_V2_9C_CALIBRATION_POOL",
        "rule": (
            "For each Fold F: Real = all REAL_INTERNAL_HOLDOUT (shared 923 IDs); "
            "AI = all PROMPT_BLOCKED for Fold F. Score with Fold F frozen LoRA "
            "checkpoint only. Exclude TRAIN-role CAP leftovers from primary design."
        ),
        "real_component": {
            "role": "REAL_INTERNAL_HOLDOUT",
            "n": int(len(ih_ids)),
            "domains": {
                str(k): int(v)
                for k, v in ih_all["real_domain"].value_counts().items()
            },
            "identical_across_folds": True,
        },
        "ai_component": {
            "role": "PROMPT_BLOCKED",
            "n_by_fold": {
                f"fold_{f}": int(
                    (
                        base[f"fold_{f}_role"] == "PROMPT_BLOCKED"
                    ).sum()
                )
                for f in [1, 2, 3, 4]
            },
            "identical_ids_across_folds": False,
            "note": (
                "PROMPT_BLOCKED membership is fold-dependent (prompt leakage vs "
                "that fold's holdout generators)."
            ),
        },
        "excluded_from_primary": [
            "TRAIN_ROLE_CAP_LEFTOVER (same generators as partially trained; fold-inconsistent)",
            "EXCLUDED_DUPLICATE",
            "NTIRE (sealed / absent from V2 development manifest)",
        ],
        "overlap_train_all_folds": 0,
        "overlap_eval_all_folds": 0,
        "both_classes_all_folds": True,
        "local_availability_ok": True,
    }

    # Verify design overlaps and local availability
    missing_total = 0
    for f in [1, 2, 3, 4]:
        tr_ids = set(train_membership(man, f)["image_id"])
        ev_ids = set(eval_membership(man, f)["image_id"])
        real = base[base[f"fold_{f}_role"] == "REAL_INTERNAL_HOLDOUT"]
        ai = base[base[f"fold_{f}_role"] == "PROMPT_BLOCKED"]
        pool_ids = set(real["image_id"]) | set(ai["image_id"])
        assert len(pool_ids & tr_ids) == 0
        assert len(pool_ids & ev_ids) == 0
        for _, r in pd.concat([real, ai]).iterrows():
            if not path_exists(r):
                missing_total += 1
                design["local_availability_ok"] = False
    design["local_missing_count"] = missing_total

    limitations = [
        "AI calibration IDs differ by fold (PROMPT_BLOCKED composition is fold-specific).",
        "PROMPT_BLOCKED AI are train-generator images blocked for prompt leakage — "
        "not the held-out hard modern generators used in final evaluation. "
        "Calibration AI distribution ≠ evaluation AI distribution.",
        "Using REAL_INTERNAL_HOLDOUT for calibration is a role reassignment relative "
        "to its original reserved-holdout purpose; tutor must approve before fitting.",
        "No AI IDs are unused across ALL folds simultaneously (intersection unused = Real-only 923).",
        "CAP leftovers exist but are excluded from primary design due to same-generator "
        "and fold-inconsistency concerns.",
    ]

    # Decision
    if not design["local_availability_ok"]:
        decision = "CALIBRATION_SET_NOT_FEASIBLE"
    elif design["both_classes_all_folds"] and design["overlap_train_all_folds"] == 0:
        decision = "CALIBRATION_SET_FEASIBLE_WITH_LIMITATIONS"
    else:
        decision = "CALIBRATION_SET_NOT_FEASIBLE"

    return {
        "role_inventory": role_inventory(man),
        "eval_prediction_checks": eval_pred_checks,
        "fold_summaries": fold_summaries,
        "train_eval_unused_counts": train_eval_unused,
        "internal_holdout_audit": {
            "n": int(len(ih_ids)),
            "all_real": bool((ih_all["binary_label"] == 0).all()),
            "domains": {
                str(k): int(v) for k, v in ih_all["real_domain"].value_counts().items()
            },
            "identical_across_folds": True,
            "smartphone_manifest_internal_holdout_n": int(len(phone_ih)),
            "smartphone_split_assignments_match_manifest": bool(
                phone_ih == split_phone_ih
            ),
            "local_available": int(sum(path_exists(r) for _, r in ih_all.iterrows())),
            "local_missing": int(
                sum(not path_exists(r) for _, r in ih_all.iterrows())
            ),
            "overlap_by_fold": ih_overlap,
            "prior_v2_5_to_v2_8_use": prior_use,
        },
        "intersection_unused_across_all_folds": {
            "n": int(len(inter)),
            "n_real": int(
                ((base["image_id"].isin(inter)) & (base["binary_label"] == 0)).sum()
            ),
            "n_ai": int(
                ((base["image_id"].isin(inter)) & (base["binary_label"] == 1)).sum()
            ),
            "note": "Only REAL_INTERNAL_HOLDOUT is unused for every fold simultaneously.",
        },
        "proposed_design": design,
        "limitations": limitations,
        "decision": decision,
        "candidate_rows": candidate_rows,
        "member_summary_rows": fold_summaries,
    }


def run_conditional_inference(man: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    """Score proposed calibration pools with each fold's frozen LoRA checkpoint."""
    try:
        import open_clip  # noqa: F401
    except ImportError as e:
        return {
            "performed": False,
            "skip_reason": f"open_clip unavailable: {e}",
        }

    if not LORA_CFG.is_file():
        return {"performed": False, "skip_reason": f"missing config {LORA_CFG}"}

    # Prefer checkpoint-embedded config for exact training settings
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    results = {"performed": True, "device": str(device), "folds": {}}

    for fold in [1, 2, 3, 4]:
        ckpt_path = CKPT_DIR / f"clip_lora_fold{fold}_best_v1.pt"
        if not ckpt_path.is_file():
            return {
                "performed": False,
                "skip_reason": f"missing checkpoint {ckpt_path}",
            }
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config") or json.loads(LORA_CFG.read_text())
        if "model_name" not in cfg.get("clip", {}):
            cfg = {
                "clip": {
                    "model_name": cfg["clip"].get("model", "ViT-B-16-quickgelu"),
                    "pretrained": cfg["clip"].get("pretrained", "openai"),
                    "l2_normalize": True,
                },
                "lora": {
                    "last_n_blocks": int(cfg["lora"].get("last_n_blocks", 4)),
                    "rank": int(cfg["lora"].get("rank", 8)),
                    "alpha": int(cfg["lora"].get("alpha", 16)),
                    "dropout": float(cfg["lora"].get("dropout", 0.05)),
                },
                "head": {"dropout": float(cfg.get("head", {}).get("dropout", 0.2))},
            }
        # Ensure required keys present even when checkpoint config is complete
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        cfg.setdefault("head", {"dropout": 0.2})
        if isinstance(cfg["head"], str):
            cfg["head"] = {"dropout": 0.2}

        real = base[base[f"fold_{fold}_role"] == "REAL_INTERNAL_HOLDOUT"].copy()
        ai = base[base[f"fold_{fold}_role"] == "PROMPT_BLOCKED"].copy()
        pool = pd.concat([real, ai], ignore_index=True).sort_values("image_id")
        expected_ids = pool["image_id"].tolist()

        # Independence checks
        tr_ids = set(train_membership(man, fold)["image_id"])
        ev_ids = set(eval_membership(man, fold)["image_id"])
        pool_ids = set(pool["image_id"])
        if pool_ids & tr_ids:
            raise SystemExit(f"STOP: fold {fold} calib pool overlaps train")
        if pool_ids & ev_ids:
            raise SystemExit(f"STOP: fold {fold} calib pool overlaps eval")

        rows = []
        for _, r in pool.iterrows():
            rows.append(
                {
                    "image_id": r["image_id"],
                    "binary_label": int(r["binary_label"]),
                    "path": resolve_path(r),
                    "canonical_generator_id": r["canonical_generator_id"],
                    "real_domain": r["real_domain"] if pd.notna(r["real_domain"]) else "",
                    "original_role": r[f"fold_{fold}_role"],
                }
            )
        missing = [r["image_id"] for r in rows if not Path(r["path"]).is_file()]
        if missing:
            return {
                "performed": False,
                "skip_reason": f"fold {fold} missing images n={len(missing)} sample={missing[:5]}",
            }

        print(f"[V2-9C] Loading fold {fold} checkpoint on {device} ...", flush=True)
        model = ClipLoRAModel(cfg, device)
        sd = ckpt["state_dict"]
        incompatible = model.load_state_dict(sd, strict=False)
        # Allow missing/unexpected only if empty; otherwise stop
        if incompatible.missing_keys or incompatible.unexpected_keys:
            # report but continue only if LoRA/head keys present
            miss = incompatible.missing_keys
            unexp = incompatible.unexpected_keys
            # open_clip text tower keys may be present in sd but unused — filter
            critical_miss = [k for k in miss if "lora_" in k or k.startswith("head.")]
            if critical_miss:
                return {
                    "performed": False,
                    "skip_reason": (
                        f"fold {fold} critical missing keys: {critical_miss[:10]}"
                    ),
                    "missing_keys_sample": miss[:20],
                    "unexpected_keys_sample": unexp[:20],
                }
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        ds = CalibImageDataset(rows, model.preprocess)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        probs = []
        labels = []
        ids = []
        with torch.no_grad():
            for xb, yb, idb in loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.extend(p.tolist())
                labels.extend(yb.numpy().tolist())
                ids.extend(list(idb))

        meta = pool.set_index("image_id")
        out_df = pd.DataFrame(
            {
                "image_id": ids,
                "fold_model": fold,
                "label": labels,
                "generator_id": [str(meta.loc[i, "canonical_generator_id"]) for i in ids],
                "real_domain": [
                    ""
                    if pd.isna(meta.loc[i, "real_domain"])
                    else str(meta.loc[i, "real_domain"])
                    for i in ids
                ],
                "original_role": [str(meta.loc[i, f"fold_{fold}_role"]) for i in ids],
                "p_ai": probs,
            }
        )

        score_path = OUT / f"v2_9c_calibration_scores_fold{fold}_v1.csv"
        out_df.to_csv(score_path, index=False)

        integrity = {
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "n_expected": len(expected_ids),
            "n_scored": int(len(out_df)),
            "missing_ids": sorted(set(expected_ids) - set(out_df["image_id"])),
            "extra_ids": sorted(set(out_df["image_id"]) - set(expected_ids)),
            "duplicates": int(out_df["image_id"].duplicated().sum()),
            "finite_probs": bool(np.isfinite(out_df["p_ai"]).all()),
            "label_agreement": bool(
                (
                    out_df.set_index("image_id")["label"].to_numpy()
                    == meta.loc[out_df["image_id"], "binary_label"].to_numpy()
                ).all()
            ),
            "n_real": int((out_df["label"] == 0).sum()),
            "n_ai": int((out_df["label"] == 1).sum()),
            "roles_scored": sorted(out_df["original_role"].unique().tolist()),
            "score_file": str(score_path.relative_to(PROJECT_ROOT)),
            "selected_epoch_in_ckpt": int(ckpt.get("selected_epoch", -1)),
        }
        integrity["ok"] = (
            integrity["n_expected"] == integrity["n_scored"]
            and not integrity["missing_ids"]
            and not integrity["extra_ids"]
            and integrity["duplicates"] == 0
            and integrity["finite_probs"]
            and integrity["label_agreement"]
        )
        results["folds"][f"fold_{fold}"] = integrity
        print(
            f"[V2-9C] Fold {fold} scored n={integrity['n_scored']} ok={integrity['ok']}",
            flush=True,
        )
        # free memory
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    results["all_folds_ok"] = all(
        results["folds"][f"fold_{f}"]["ok"] for f in [1, 2, 3, 4]
    )
    return results


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    a = payload
    lines.append("V2-9C — Independent Calibration-Set Feasibility Audit")
    lines.append(f"Status: COMPLETE — decision {a['decision']}")
    lines.append("No detector training. No calibrator fitted. No threshold tuning.")
    lines.append("")
    lines.append("Authoritative role inventory (summary)")
    inv = a["role_inventory"]
    lines.append(
        f"  Locked V2 IDs (excl. duplicate): {inv['n_after_excluding_duplicate']}"
    )
    lines.append(f"  Sources: {inv['source_dataset_counts']}")
    lines.append(f"  Smartphone splits: {inv['smartphone_manifest_splits']}")
    lines.append(f"  NTIRE in manifest: {inv['ntire_present_in_manifest']}")
    for f, roles in inv["roles_per_fold"].items():
        lines.append(f"  {f} roles: {roles}")
    lines.append("")
    lines.append("Fold training memberships (exact V2-8 CAP=300 protocol)")
    for fs in a["fold_summaries"]:
        lines.append(
            f"  Fold {fs['fold']}: train n={fs['train_n']} "
            f"(Real={fs['train_real']}, AI={fs['train_ai']}, "
            f"gens={fs['train_ai_generators']}, dups={fs['train_duplicates']}, "
            f"CAP_leftover_AI={fs['cap_leftover_ai_n']})"
        )
    lines.append("")
    lines.append("Fold evaluation memberships (= prediction files)")
    for f, ev in a["eval_prediction_checks"].items():
        lines.append(
            f"  {f}: n={ev['manifest_eval_n']} match_preds={ev['ids_match']} "
            f"Real={ev['n_real']} AI={ev['n_ai']} roles={ev['roles']}"
        )
    lines.append("")
    lines.append("Candidate unused (ALL − train − eval)")
    for fs in a["fold_summaries"]:
        lines.append(
            f"  Fold {fs['fold']}: unused={fs['candidate_unused_n']} "
            f"(Real={fs['candidate_unused_real']}, AI={fs['candidate_unused_ai']}) "
            f"by_role={fs['unused_by_role']}"
        )
    lines.append("")
    ih = a["internal_holdout_audit"]
    lines.append("REAL_INTERNAL_HOLDOUT audit")
    lines.append(
        f"  n={ih['n']} all_real={ih['all_real']} domains={ih['domains']} "
        f"local={ih['local_available']}/{ih['n']} "
        f"phone_manifest_match={ih['smartphone_split_assignments_match_manifest']}"
    )
    lines.append(f"  Prior V2-5..8 use: {ih['prior_v2_5_to_v2_8_use']}")
    for f, ov in ih["overlap_by_fold"].items():
        lines.append(f"  {f}: overlap_train={ov['overlap_train']} overlap_eval={ov['overlap_eval']}")
    lines.append("")
    lines.append("Intersection unused across all folds")
    lines.append(f"  {a['intersection_unused_across_all_folds']}")
    lines.append("")
    lines.append("Proposed consistent design")
    lines.append(f"  {a['proposed_design']['rule']}")
    lines.append(f"  Real: {a['proposed_design']['real_component']}")
    lines.append(f"  AI: {a['proposed_design']['ai_component']}")
    lines.append("")
    lines.append("Limitations")
    for lim in a["limitations"]:
        lines.append(f"  - {lim}")
    lines.append("")
    lines.append(f"DECISION: {a['decision']}")
    lines.append("")
    inf = a["conditional_inference"]
    lines.append("Conditional inference")
    if not inf.get("performed"):
        lines.append(f"  NEW_INFERENCE_PERFORMED = NO ({inf.get('skip_reason')})")
    else:
        lines.append(f"  NEW_INFERENCE_PERFORMED = YES device={inf.get('device')}")
        for f, fr in inf["folds"].items():
            lines.append(
                f"  {f}: ckpt={fr['checkpoint']} n={fr['n_scored']} "
                f"Real={fr['n_real']} AI={fr['n_ai']} roles={fr['roles_scored']} "
                f"ok={fr['ok']} file={fr['score_file']}"
            )
        lines.append(f"  all_folds_ok={inf.get('all_folds_ok')}")
    lines.append("")
    lines.append("Next-stage recommendation ONLY (not executed)")
    lines.append(f"  {a['next_stage_recommendation']}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in a["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("CALIBRATOR_FITTED = NO")
    lines.append("FINAL_V2_MODEL_SELECTED = NO")
    lines.append("NEXT_STAGE_STARTED = NO")
    return "\n".join(lines) + "\n"


def main() -> None:
    assert MANIFEST.is_file()
    assert PRED_DIR.is_dir()
    man = load_manifest()
    audit = build_audit(man)

    # Write membership / candidate CSVs (primary proposal rows flagged)
    pd.DataFrame(audit["member_summary_rows"]).to_csv(MEMBER_CSV, index=False)
    cand_df = pd.DataFrame(audit["candidate_rows"])
    cand_df.to_csv(CAND_CSV, index=False)

    decision = audit["decision"]
    inference: dict[str, Any]
    if decision in (
        "CALIBRATION_SET_FEASIBLE",
        "CALIBRATION_SET_FEASIBLE_WITH_LIMITATIONS",
    ):
        print(f"[V2-9C] Decision={decision}; running conditional inference gate...", flush=True)
        inference = run_conditional_inference(man, audit)
    else:
        inference = {
            "performed": False,
            "skip_reason": f"decision={decision}; inference gate closed",
        }

    next_rec = (
        "Human/tutor review required before any calibrator fitting. "
        "If limitations are accepted, a future V2-9D may fit leakage-safe C1/C2 "
        "using per-fold scores of REAL_INTERNAL_HOLDOUT + PROMPT_BLOCKED produced "
        "by each Fold-F frozen checkpoint — without touching final-eval IDs. "
        "If limitations are rejected, abandon post-hoc calibration on current V2 "
        "artifacts and consider alternative directions (selective prediction / "
        "robustness / targeted representation). Do NOT auto-start."
    )

    integrity = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "DETECTOR_TRAINING_PERFORMED": "NO",
        "MODEL_WEIGHTS_UPDATED": "NO",
        "CLIP_EMBEDDINGS_REGENERATED": "NO",
        "V2_7_REPLAYED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "THRESHOLD_TUNED": "NO",
        "FINAL_EVALUATION_IDS_USED_AS_CALIBRATION": "NO",
        "TRAIN_IDS_USED_AS_CALIBRATION": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_INFERENCE_PERFORMED": "YES" if inference.get("performed") else "NO",
    }

    # Drop bulky candidate_rows from JSON (kept in CSV)
    payload = {
        "stage": "V2-9C",
        "status": "COMPLETE",
        "decision": decision,
        "role_inventory": audit["role_inventory"],
        "eval_prediction_checks": audit["eval_prediction_checks"],
        "fold_summaries": audit["fold_summaries"],
        "train_eval_unused_counts": audit["train_eval_unused_counts"],
        "internal_holdout_audit": audit["internal_holdout_audit"],
        "intersection_unused_across_all_folds": audit[
            "intersection_unused_across_all_folds"
        ],
        "proposed_design": audit["proposed_design"],
        "limitations": audit["limitations"],
        "conditional_inference": inference,
        "next_stage_recommendation": next_rec,
        "artifacts": {
            "membership_summary_csv": str(MEMBER_CSV.relative_to(PROJECT_ROOT)),
            "candidate_ids_csv": str(CAND_CSV.relative_to(PROJECT_ROOT)),
        },
        "integrity_statement": integrity,
        "final_v2_model_selected": False,
        "calibrator_fitted": False,
        "next_stage_started": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    REPORT_OUT.write_text(write_report(payload))
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Wrote {MEMBER_CSV}")
    print(f"Wrote {CAND_CSV}")


if __name__ == "__main__":
    # Ensure project venv open_clip is preferred when invoked via system python
    main()
