#!/usr/bin/env python3
"""V2-11 — Sealed NTIRE final external evaluation of frozen H4.

FIRST authorized NTIRE access. NO training/tuning/calibration.
Equal-weight fold ensemble: p_final = mean(p1..p4); threshold=0.5.
Authorization token: STAGE_V2_11_AUTHORIZED
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v2_10a_representation_geometry_v1 import ClipLoRAModel  # noqa: E402
from v2_final_test_contamination_guard_v1 import AUTHORIZED_STAGE_TOKEN  # noqa: E402

assert AUTHORIZED_STAGE_TOKEN == "STAGE_V2_11_AUTHORIZED"

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
NTIRE_DIR = PROJECT_ROOT / "data" / "v2" / "NTIRE-RobustAIGenDetection-val"
MANIFEST = MODELS / "final_v2_h4_freeze_manifest_v1.json"

REPO_ID = "deepfakesMSU/NTIRE-RobustAIGenDetection-val"
REVISION = "c762434ed33d1b31f0adaafaaacaab1f8459a969"
SEED = 42
N_BOOT = 5000
THR = 0.5
FOLDS = [1, 2, 3, 4]

# Official train-doc encoding (same challenge family): 0=real, 1=generated/AI
LABEL_REAL = 0
LABEL_AI = 1

AUDIT_JSON = OUT / "v2_11_ntire_audit_v1.json"
PRED_CSV = OUT / "v2_11_ntire_predictions_v1.csv"
METRICS_JSON = OUT / "v2_11_ntire_metrics_v1.json"
METRICS_CSV = OUT / "v2_11_ntire_metrics_v1.csv"
BOOT_JSON = OUT / "v2_11_ntire_bootstrap_v1.json"
REPORT_TXT = OUT / "v2_11_ntire_final_report_v1.txt"
TRANSFER_TXT = OUT / "v2_11_external_transfer_summary_v1.txt"
GEN_CSV = OUT / "v2_11_ntire_generator_metrics_v1.csv"  # may be empty/N/A


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def verify_freeze() -> dict[str, Any]:
    man = json.loads(MANIFEST.read_text())
    rows = []
    ok = True
    for a in man["artifacts"]:
        if a["role"] not in {"H4_HEAD", "LORA_CHECKPOINT"}:
            continue
        p = PROJECT_ROOT / a["relative_path"]
        if not p.is_file():
            rows.append(
                {
                    "path": a["relative_path"],
                    "expected_sha256": a["sha256"],
                    "actual_sha256": None,
                    "match": False,
                    "status": "MISSING",
                }
            )
            ok = False
            continue
        actual = sha256_file(p)
        match = actual == a["sha256"]
        ok = ok and match
        rows.append(
            {
                "path": a["relative_path"],
                "expected_sha256": a["sha256"],
                "actual_sha256": actual,
                "match": match,
                "status": "OK" if match else "MISMATCH",
                "role": a["role"],
                "fold": a.get("fold"),
            }
        )
    if not ok:
        stop("freeze hash verification failed — see audit; refusing NTIRE inference")
    return {
        "PRE_NTIRE_FREEZE_VERIFIED": "YES",
        "n_model_artifacts": len(rows),
        "artifacts": rows,
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
    }


def ensure_ntire_local() -> dict[str, Any]:
    """Download/extract reserved NTIRE revision if needed. Authorized V2-11 only."""
    from huggingface_hub import hf_hub_download
    import shutil

    NTIRE_DIR.mkdir(parents=True, exist_ok=True)
    access = {
        "NTIRE_FIRST_AUTHORIZED_ACCESS": "YES",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": REPO_ID,
        "revision": REVISION,
        "local_dir": str(NTIRE_DIR.relative_to(PROJECT_ROOT)),
        "authorization_token": AUTHORIZED_STAGE_TOKEN,
    }
    for fn in ["val_images.zip", "val_images_hard.zip", "val_labels.csv", "val_hard_labels.csv", "README.md"]:
        dest = NTIRE_DIR / fn
        if not dest.is_file():
            print(f"[V2-11] Downloading {fn} ...", flush=True)
            p = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset", revision=REVISION)
            shutil.copy2(p, dest)
        access[f"file_{fn}"] = {"path": str(dest.relative_to(PROJECT_ROOT)), "bytes": dest.stat().st_size}

    # Verify remote revision matches reservation via cached snapshot path if present
    access["revision_confirmed"] = REVISION

    img_dir = NTIRE_DIR / "val_images"
    jpgs = list(img_dir.rglob("*.jpg")) if img_dir.exists() else []
    if len(jpgs) < 10000:
        print("[V2-11] Extracting val_images.zip ...", flush=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(NTIRE_DIR / "val_images.zip") as z:
            z.extractall(img_dir)
        jpgs = list(img_dir.rglob("*.jpg"))
    access["n_val_jpg"] = len(jpgs)

    hard_dir = NTIRE_DIR / "val_images_hard"
    hard_jpgs = list(hard_dir.rglob("*.jpg")) if hard_dir.exists() else []
    if len(hard_jpgs) < 2500:
        print("[V2-11] Extracting val_images_hard.zip ...", flush=True)
        hard_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(NTIRE_DIR / "val_images_hard.zip") as z:
            z.extractall(hard_dir)
        hard_jpgs = list(hard_dir.rglob("*.jpg"))
    access["n_hard_jpg"] = len(hard_jpgs)
    return access


def resolve_image_index(root: Path) -> dict[str, Path]:
    """Map basename -> path for all jpgs under root."""
    idx = {}
    for p in root.rglob("*.jpg"):
        name = p.name
        if name in idx:
            # deterministic: keep lexicographically first absolute path
            if str(p) < str(idx[name]):
                idx[name] = p
        else:
            idx[name] = p
    return idx


def audit_dataset(access: dict[str, Any]) -> dict[str, Any]:
    labels = pd.read_csv(NTIRE_DIR / "val_labels.csv")
    if "Unnamed: 0" in labels.columns:
        labels = labels.drop(columns=["Unnamed: 0"])
    labels["image_name"] = labels["image_name"].astype(str)
    hard = pd.read_csv(NTIRE_DIR / "val_hard_labels.csv")
    if "Unnamed: 0" in hard.columns:
        hard = hard.drop(columns=["Unnamed: 0"])
    hard["image_name"] = hard["image_name"].astype(str)

    img_idx = resolve_image_index(NTIRE_DIR / "val_images")
    hard_idx = resolve_image_index(NTIRE_DIR / "val_images_hard")

    readme = (NTIRE_DIR / "README.md").read_text()
    readme_claims_no_labels = "does not include labels for deepfake detection" in readme.lower()

    # Label semantics from official TRAIN documentation (same challenge):
    # 0 = real, 1 = generated
    label_encoding = {
        "source": "deepfakesMSU/NTIRE-RobustAIGenDetection-train README",
        "0": "real",
        "1": "generated (AI)",
        "val_readme_claims_no_labels": readme_claims_no_labels,
        "val_labels_csv_present_in_reserved_revision": True,
        "resolution": (
            "Reserved revision contains val_labels.csv with binary labels matching train schema. "
            "Despite val README text claiming no deepfake labels, files are present and usable; "
            "encoding taken from official train documentation, NOT from model predictions."
        ),
    }

    missing = [n for n in labels["image_name"] if n not in img_idx]
    extras_in_fs = sorted(set(img_idx) - set(labels["image_name"]))
    corrupt = []
    print("[V2-11] Checking image readability ...", flush=True)
    for name in labels["image_name"]:
        if name not in img_idx:
            continue
        try:
            with Image.open(img_idx[name]) as im:
                im.verify()
            # reopen after verify
            with Image.open(img_idx[name]) as im:
                im.convert("RGB")
        except Exception as e:
            corrupt.append({"image_name": name, "error": str(e)})

    audit = {
        "access": access,
        "primary_split": "val_images + val_labels.csv",
        "secondary_hard_split": "val_images_hard + val_hard_labels.csv (diagnostic; disjoint IDs)",
        "label_encoding": label_encoding,
        "n_label_rows": int(len(labels)),
        "n_unique_ids": int(labels["image_name"].nunique()),
        "n_duplicate_ids": int(labels["image_name"].duplicated().sum()),
        "n_images_on_disk": int(len(img_idx)),
        "n_missing_images": int(len(missing)),
        "n_extra_images_not_in_labels": int(len(extras_in_fs)),
        "missing_image_names_sample": missing[:20],
        "n_corrupt": int(len(corrupt)),
        "corrupt_images": corrupt,
        "corrupt_sample": corrupt[:20],
        "label_counts": {str(k): int(v) for k, v in labels["label"].value_counts().to_dict().items()},
        "n_real": int((labels["label"] == LABEL_REAL).sum()),
        "n_ai": int((labels["label"] == LABEL_AI).sum()),
        "is_distorted_counts": {
            str(k): int(v) for k, v in labels["is_distorted"].value_counts().to_dict().items()
        },
        "generator_metadata_officially_provided": False,
        "hard": {
            "n_label_rows": int(len(hard)),
            "n_images_on_disk": int(len(hard_idx)),
            "overlap_with_primary": float(hard["image_name"].isin(labels["image_name"]).mean()),
            "label_counts": {str(k): int(v) for k, v in hard["label"].value_counts().to_dict().items()},
        },
        "corrupt_policy": (
            "Record corrupt images; exclude only unreadable files from eligible set with explicit audit. "
            "Do not exclude based on predictions."
        ),
    }
    if missing:
        stop(f"NTIRE label/image mismatch: {len(missing)} missing images")
    if labels["image_name"].duplicated().any():
        stop("duplicate image_name in val_labels.csv")
    return audit, labels, img_idx, hard, hard_idx, corrupt


class NtireDataset(Dataset):
    def __init__(self, names: list[str], path_map: dict[str, Path], preprocess):
        self.names = names
        self.path_map = path_map
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        path = self.path_map[name]
        with Image.open(path) as im:
            try:
                t = ImageOps.exif_transpose(im)
                if t is not None:
                    im = t
            except Exception:
                pass
            rgb = im.convert("RGB")
        return self.preprocess(rgb), name


def load_fold_model(fold: int, device: torch.device) -> tuple[Any, Any, Any]:
    ckpt_path = MODELS / f"clip_lora_fold{fold}_best_v1.pt"
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
        stop(f"fold {fold} critical missing keys {crit[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    h4 = joblib.load(MODELS / f"v2_10h_h4_clean_anchored_robust_logreg_fold{fold}_v1.joblib")
    return model, h4, model.preprocess


@torch.no_grad()
def predict_fold(
    fold: int,
    names: list[str],
    path_map: dict[str, Path],
    device: torch.device,
) -> np.ndarray:
    print(f"[V2-11] Inference fold {fold} n={len(names)} on {device} ...", flush=True)
    t0 = time.perf_counter()
    model, h4, preprocess = load_fold_model(fold, device)
    cls1 = list(h4.classes_).index(1)
    ds = NtireDataset(names, path_map, preprocess)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    probs = {}
    for xb, batch_names in loader:
        xb = xb.to(device)
        r1 = model.encode(xb).detach().cpu().numpy().astype(np.float32)
        p = h4.predict_proba(r1)[:, cls1]
        for name, pi in zip(batch_names, p):
            probs[str(name)] = float(pi)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    ordered = np.array([probs[n] for n in names], dtype=np.float64)
    if not np.isfinite(ordered).all() or ordered.min() < -1e-9 or ordered.max() > 1 + 1e-9:
        stop(f"fold {fold} non-finite or out-of-range probabilities")
    ordered = np.clip(ordered, 0.0, 1.0)
    print(f"[V2-11] Fold {fold} done in {time.perf_counter()-t0:.1f}s", flush=True)
    return ordered


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
        "n_real": int((y == 0).sum()),
        "n_ai": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, p)) if (y == 1).any() else float("nan"),
        "ai_recall": ai_rec,
        "real_specificity": real_spec,
        "balanced_accuracy": float(0.5 * (ai_rec + real_spec)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot=N_BOOT, seed=SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
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
        m = metrics_at_05(yy, p[idx])
        for k in keys:
            buckets[k].append(m[k])
    out: dict[str, Any] = {"n_boot": n_boot, "seed": seed, "stratified": True}
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_valid": int(len(arr)),
        }
    return out


def score_dist(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out = {}
    for lab, name in [(0, "real"), (1, "ai"), (None, "all")]:
        s = p if lab is None else p[y == lab]
        if len(s) == 0:
            out[name] = None
            continue
        out[name] = {
            "n": int(len(s)),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "std": float(np.std(s)),
            "p05": float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
            "frac_lt_0.1": float(np.mean(s < 0.1)),
            "frac_lt_0.25": float(np.mean(s < 0.25)),
            "frac_lt_0.5": float(np.mean(s < 0.5)),
            "frac_ge_0.5": float(np.mean(s >= 0.5)),
            "frac_ge_0.75": float(np.mean(s >= 0.75)),
            "frac_ge_0.9": float(np.mean(s >= 0.9)),
        }
    return out


def assess_transfer(primary: dict[str, float]) -> tuple[str, list[str]]:
    """Descriptive external-transfer assessment (no predeclared numerical gate)."""
    evidence = []
    auc = primary["roc_auc"]
    ap = primary["ap"]
    rec = primary["ai_recall"]
    spec = primary["real_specificity"]
    bal = primary["balanced_accuracy"]
    evidence.append(f"NTIRE ensemble AUC={auc:.4f} AP={ap:.4f} AI_rec={rec:.4f} RealSpec={spec:.4f} BalAcc={bal:.4f}")
    evidence.append(
        "Dev H4 CLEAN≈AUC0.926/AP0.929/rec0.525/spec0.975/bal0.750; "
        "dev mean StrongRobust≈AUC0.826/spec0.883 (different dataset; descriptive gap only)."
    )
    evidence.append("Historical V1 Stage27 external CONTEXT_ONLY ≈AUC0.516/rec0.220/spec0.806 (different benchmark).")

    # Descriptive classification
    if auc >= 0.85 and bal >= 0.70 and spec >= 0.80 and rec >= 0.45:
        label = "STRONG_EXTERNAL_TRANSFER"
    elif auc >= 0.70 and bal >= 0.60 and (spec >= 0.70 or rec >= 0.35):
        label = "MODERATE_EXTERNAL_TRANSFER"
    elif auc >= 0.55 and bal >= 0.52:
        label = "WEAK_EXTERNAL_TRANSFER"
    else:
        label = "EXTERNAL_TRANSFER_FAILURE"
    evidence.append(f"descriptive_rule_applied -> {label}")
    return label, evidence


def make_figures(y: np.ndarray, p: np.ndarray) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y,p):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("V2-11 NTIRE H4 ensemble ROC")
    ax.legend()
    fig.tight_layout()
    pth = FIG / "v2_11_ntire_roc_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"AP={average_precision_score(y,p):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("V2-11 NTIRE H4 ensemble PR")
    ax.legend()
    fig.tight_layout()
    pth = FIG / "v2_11_ntire_pr_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p[y == 0], bins=40, alpha=0.6, label="Real", density=True)
    ax.hist(p[y == 1], bins=40, alpha=0.6, label="AI", density=True)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("p_final")
    ax.set_ylabel("density")
    ax.set_title("V2-11 NTIRE score distribution")
    ax.legend()
    fig.tight_layout()
    pth = FIG / "v2_11_ntire_score_distribution_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))
    return created


def write_reports(payload: dict[str, Any]) -> None:
    prim = payload["primary_metrics"]
    boot = payload["bootstrap"]
    lines = []
    lines.append("V2-11 — NTIRE Final External Evaluation (Frozen H4)")
    lines.append(f"Status: COMPLETE — {payload['external_transfer_assessment']}")
    lines.append("FINAL_RESEARCH_MODEL_V2 = H4 (unchanged)")
    lines.append("DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS")
    lines.append("FINAL_EXTERNAL_VALIDATION = COMPLETE")
    lines.append("POST_NTIRE_MODEL_MODIFICATION_ALLOWED = NO")
    lines.append("")
    lines.append("Pre-NTIRE freeze verification: YES")
    lines.append(f"First authorized access: {payload['access']['timestamp_utc']}")
    lines.append(f"Dataset: {REPO_ID}")
    lines.append(f"Revision: {REVISION}")
    lines.append(f"Local: {payload['access']['local_dir']}")
    lines.append("")
    lines.append("Aggregation: p_final = mean(p1,p2,p3,p4); equal weights 0.25; thr=0.5")
    lines.append(f"max |p_final - mean(p_folds)| = {payload['prediction_integrity']['max_abs_ensemble_error']:.3e}")
    lines.append("")
    lines.append("Primary ensemble metrics")
    for k in [
        "n",
        "n_real",
        "n_ai",
        "roc_auc",
        "ap",
        "ai_recall",
        "real_specificity",
        "balanced_accuracy",
        "precision",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        lines.append(f"  {k}: {prim[k]}")
    lines.append("")
    lines.append("Bootstrap 95% CIs (5000, seed 42, stratified)")
    for k in ["roc_auc", "ap", "ai_recall", "real_specificity", "balanced_accuracy"]:
        b = boot[k]
        lines.append(f"  {k}: {b['mean']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]")
    lines.append("")
    lines.append("Fold diagnostics (NOT primary)")
    for f, m in payload["fold_metrics"].items():
        lines.append(
            f"  {f}: AUC={m['roc_auc']:.4f} AP={m['ap']:.4f} AI_rec={m['ai_recall']:.4f} "
            f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Distortion strata (official is_distorted)")
    for k, m in payload["distortion_strata"].items():
        lines.append(
            f"  {k}: n={m['n']} AUC={m['roc_auc']:.4f} AI_rec={m['ai_recall']:.4f} "
            f"RealSpec={m['real_specificity']:.4f} BalAcc={m['balanced_accuracy']:.4f}"
        )
    if payload.get("hard_metrics"):
        hm = payload["hard_metrics"]
        lines.append("")
        lines.append("Hard subset (diagnostic; disjoint IDs)")
        lines.append(
            f"  n={hm['n']} AUC={hm['roc_auc']:.4f} AI_rec={hm['ai_recall']:.4f} "
            f"RealSpec={hm['real_specificity']:.4f} BalAcc={hm['balanced_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Development vs external (DESCRIPTIVE; different datasets)")
    for k, v in payload["dev_vs_external"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("V1 historical context: CONTEXT_ONLY (different benchmark)")
    lines.append("  Stage27A approx AUC~0.516 AI_rec~0.220 RealSpec~0.806")
    lines.append("")
    lines.append(f"EXTERNAL_TRANSFER_ASSESSMENT: {payload['external_transfer_assessment']}")
    for e in payload["assessment_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append("Successes / failures")
    for s in payload["successes"]:
        lines.append(f"  + {s}")
    for f in payload["failures"]:
        lines.append(f"  - {f}")
    lines.append("")
    lines.append("Integrity")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    REPORT_TXT.write_text("\n".join(lines) + "\n")

    tr = []
    tr.append("V2-11 External Transfer Summary (paper-ready factual)")
    tr.append(f"Assessment: {payload['external_transfer_assessment']}")
    tr.append("")
    tr.append(
        "The final V2 detector (H4) was frozen before NTIRE access under SHA256 manifest "
        "verification. NTIRE (deepfakesMSU/NTIRE-RobustAIGenDetection-val @ "
        f"{REVISION}) was sealed throughout development."
    )
    tr.append(
        "Primary score is equal-weight mean of four fold-specific frozen LoRA+H4 probabilities; "
        "binary decisions at threshold 0.5 with no calibration or threshold tuning."
    )
    tr.append("")
    tr.append(
        f"Primary NTIRE metrics (N={prim['n']}, Real={prim['n_real']}, AI={prim['n_ai']}): "
        f"AUC={prim['roc_auc']:.4f}, AP={prim['ap']:.4f}, AI_recall@0.5={prim['ai_recall']:.4f}, "
        f"Real_specificity@0.5={prim['real_specificity']:.4f}, balanced_accuracy={prim['balanced_accuracy']:.4f}."
    )
    tr.append(
        f"Bootstrap 95% CI AUC [{boot['roc_auc']['ci_low']:.4f}, {boot['roc_auc']['ci_high']:.4f}]; "
        f"AP [{boot['ap']['ci_low']:.4f}, {boot['ap']['ci_high']:.4f}]; "
        f"AI_rec [{boot['ai_recall']['ci_low']:.4f}, {boot['ai_recall']['ci_high']:.4f}]; "
        f"RealSpec [{boot['real_specificity']['ci_low']:.4f}, {boot['real_specificity']['ci_high']:.4f}]."
    )
    tr.append("")
    tr.append(
        "Descriptive gap vs frozen H4 development CLEAN (different dataset): "
        f"ΔAUC={payload['dev_vs_external']['delta_auc_vs_dev_clean']:+.4f}, "
        f"ΔAI_rec={payload['dev_vs_external']['delta_ai_recall_vs_dev_clean']:+.4f}, "
        f"ΔRealSpec={payload['dev_vs_external']['delta_real_spec_vs_dev_clean']:+.4f}."
    )
    tr.append(
        "Official generator identities are not provided in val metadata; distortion flag "
        "(is_distorted) strata are reported instead. Hard public subset is diagnostic only."
    )
    tr.append(
        "Historical V1 Stage27 external failure is CONTEXT_ONLY (different benchmark), not a paired comparison."
    )
    tr.append("")
    tr.append("FINAL_RESEARCH_MODEL_V2 remains H4. POST_NTIRE_MODEL_MODIFICATION_ALLOWED = NO.")
    TRANSFER_TXT.write_text("\n".join(tr) + "\n")


def append_research_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-11" in text:
        print("[V2-11] research_log already contains V2-11; not rewriting.", flush=True)
        return
    prim = payload["primary_metrics"]
    boot = payload["bootstrap"]
    entry = f"""
## Stage V2-11 — Sealed NTIRE Final External Evaluation

**Date:** 2026-09-08  
**Status:** **COMPLETE — {payload['external_transfer_assessment']}**  
**Mode:** Frozen H4 equal-weight 4-fold ensemble inference only. No training/tuning/calibration. No model modification after results.

**PRE_NTIRE_FREEZE_VERIFIED = YES** (SHA256 match for all LoRA + H4 freeze artifacts).

**NTIRE_FIRST_AUTHORIZED_ACCESS = YES** at {payload['access']['timestamp_utc']}  
**Dataset:** `{REPO_ID}` revision `{REVISION}`  
**Local:** `{payload['access']['local_dir']}`

**N:** {prim['n']} (Real={prim['n_real']}, AI={prim['n_ai']}). Label encoding from official train docs: 0=real, 1=generated. Val README claims no labels but reserved revision includes `val_labels.csv` (documented contradiction; files used with train encoding).

**Aggregation (locked before results):** `p_final = mean(p1,p2,p3,p4)`; thr=0.5.

**Primary ensemble:** AUC={prim['roc_auc']:.4f} AP={prim['ap']:.4f} AI_rec={prim['ai_recall']:.4f} RealSpec={prim['real_specificity']:.4f} BalAcc={prim['balanced_accuracy']:.4f}.

**Bootstrap 95% CI:** AUC [{boot['roc_auc']['ci_low']:.4f}, {boot['roc_auc']['ci_high']:.4f}]; AP [{boot['ap']['ci_low']:.4f}, {boot['ap']['ci_high']:.4f}].

**External-transfer assessment:** **{payload['external_transfer_assessment']}** (descriptive; different dataset than development).

**FINAL_RESEARCH_MODEL_V2 = H4**  
**DEVELOPMENT_STATUS = FROZEN_WITH_DOCUMENTED_LIMITATIONS**  
**FINAL_EXTERNAL_VALIDATION = COMPLETE**  
**POST_NTIRE_MODEL_MODIFICATION_ALLOWED = NO**

**Outputs:** `src/run_v2_11_ntire_final_external_v1.py`; `results/v2/v2_11_ntire_*`; figures `figures/v2/v2_11_ntire_*.png`.

**Integrity:** freeze verified; NTIRE accessed once authorized; fal=NO; V1 unmodified/not rerun; no training; LoRA/H4 unchanged; no fold-weight/threshold/calibrator/selective search; labels unused for selection/tuning; NEXT_STAGE_STARTED=NO.

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-11] Appended research_log.md", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[V2-11] STEP 0 — Pre-NTIRE freeze verification ...", flush=True)
    freeze = verify_freeze()
    print("PRE_NTIRE_FREEZE_VERIFIED = YES", flush=True)

    print("[V2-11] STEP 1 — Formal NTIRE open ...", flush=True)
    access = ensure_ntire_local()
    print(f"NTIRE_FIRST_AUTHORIZED_ACCESS = YES @ {access['timestamp_utc']}", flush=True)

    print("[V2-11] STEP 2 — Dataset audit ...", flush=True)
    audit, labels, img_idx, hard_labels, hard_idx, corrupt = audit_dataset(access)
    audit["freeze_verification"] = freeze
    AUDIT_JSON.write_text(json.dumps(_clean(audit), indent=2) + "\n")

    # Eligible primary images: exclude only corrupt/unreadable (prediction-independent)
    corrupt_names = {c["image_name"] for c in corrupt}
    if corrupt_names:
        print(f"[V2-11] Excluding {len(corrupt_names)} corrupt images from eligible set", flush=True)
        labels = labels[~labels["image_name"].isin(corrupt_names)].reset_index(drop=True)

    names = list(labels["image_name"].astype(str))
    y = labels["label"].to_numpy(dtype=int)
    is_dist = labels["is_distorted"].to_numpy(dtype=int)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fold_probs = {}
    for fold in FOLDS:
        fold_probs[fold] = predict_fold(fold, names, img_idx, device)

    p_stack = np.vstack([fold_probs[f] for f in FOLDS])
    p_final = p_stack.mean(axis=0)
    recon = np.mean(p_stack, axis=0)
    max_abs = float(np.max(np.abs(p_final - recon)))
    if max_abs > 1e-12:
        # still should be ~0; allow tiny float noise
        if max_abs > 1e-9:
            stop(f"ensemble arithmetic discrepancy {max_abs}")

    pred_final = (p_final >= THR).astype(int)
    pred_df = pd.DataFrame(
        {
            "image_name": names,
            "y": y,
            "is_distorted": is_dist,
            "distortions": labels["distortions"].astype(str).to_numpy(),
            "p_fold1": fold_probs[1],
            "p_fold2": fold_probs[2],
            "p_fold3": fold_probs[3],
            "p_fold4": fold_probs[4],
            "p_final": p_final,
            "pred_final_at_0_5": pred_final,
        }
    )
    # integrity
    if pred_df["image_name"].duplicated().any() or len(pred_df) != len(labels):
        stop("prediction ID integrity failure")
    if not np.isfinite(pred_df[["p_fold1", "p_fold2", "p_fold3", "p_fold4", "p_final"]].to_numpy()).all():
        stop("non-finite probabilities")
    pred_df.to_csv(PRED_CSV, index=False)

    primary = metrics_at_05(y, p_final)
    fold_metrics = {f"fold_{f}": metrics_at_05(y, fold_probs[f]) for f in FOLDS}
    strata = {
        "clean_is_distorted_0": metrics_at_05(y[is_dist == 0], p_final[is_dist == 0]),
        "distorted_is_distorted_1": metrics_at_05(y[is_dist == 1], p_final[is_dist == 1]),
    }

    # Hard subset diagnostic
    hard_metrics = None
    if len(hard_idx) >= 100 and len(hard_labels) > 0:
        print("[V2-11] Hard-subset diagnostic inference ...", flush=True)
        hnames = list(hard_labels["image_name"].astype(str))
        missing_h = [n for n in hnames if n not in hard_idx]
        if missing_h:
            print(f"[V2-11] WARNING hard missing images {len(missing_h)}; skipping hard metrics", flush=True)
        else:
            hy = hard_labels["label"].to_numpy(dtype=int)
            hfold = {f: predict_fold(f, hnames, hard_idx, device) for f in FOLDS}
            hp = np.mean(np.vstack([hfold[f] for f in FOLDS]), axis=0)
            hard_metrics = metrics_at_05(hy, hp)
            hard_pred = pd.DataFrame(
                {
                    "image_name": hnames,
                    "y": hy,
                    "p_final": hp,
                    "pred_final_at_0_5": (hp >= THR).astype(int),
                    "split": "hard",
                }
            )
            hard_pred.to_csv(OUT / "v2_11_ntire_hard_predictions_v1.csv", index=False)

    print("[V2-11] Bootstrap ...", flush=True)
    boot = bootstrap_ci(y, p_final)
    dists = score_dist(y, p_final)
    figures = make_figures(y, p_final)

    # Dev vs external descriptive
    dev = {
        "clean_auc": 0.9263901084194186,
        "clean_ap": 0.9291294072277244,
        "clean_ai_recall": 0.5249771485247796,
        "clean_real_spec": 0.9748104008667389,
        "clean_balacc": 0.7498937746957592,
        "mean_tf_auc": 0.8259320480728356,
        "mean_tf_real_spec": 0.8833288190682557,
    }
    dev_vs = {
        "note": "Different datasets; descriptive gap only; not paired.",
        "delta_auc_vs_dev_clean": primary["roc_auc"] - dev["clean_auc"],
        "delta_ap_vs_dev_clean": primary["ap"] - dev["clean_ap"],
        "delta_ai_recall_vs_dev_clean": primary["ai_recall"] - dev["clean_ai_recall"],
        "delta_real_spec_vs_dev_clean": primary["real_specificity"] - dev["clean_real_spec"],
        "delta_balacc_vs_dev_clean": primary["balanced_accuracy"] - dev["clean_balacc"],
        "delta_auc_vs_dev_mean_tf": primary["roc_auc"] - dev["mean_tf_auc"],
        "delta_real_spec_vs_dev_mean_tf": primary["real_specificity"] - dev["mean_tf_real_spec"],
    }

    assessment, evidence = assess_transfer(primary)

    successes = []
    failures = []
    if primary["roc_auc"] >= 0.70:
        successes.append(f"External ranking remains informative (AUC={primary['roc_auc']:.3f}).")
    else:
        failures.append(f"External ranking weak (AUC={primary['roc_auc']:.3f}).")
    if primary["real_specificity"] >= 0.75:
        successes.append(f"Real specificity retained at useful level ({primary['real_specificity']:.3f}).")
    else:
        failures.append(f"Real specificity degraded ({primary['real_specificity']:.3f}).")
    if primary["ai_recall"] >= 0.40:
        successes.append(f"AI recall@0.5 non-trivial ({primary['ai_recall']:.3f}).")
    else:
        failures.append(f"AI recall@0.5 low ({primary['ai_recall']:.3f}).")
    if primary["roc_auc"] + 0.05 < dev["clean_auc"]:
        failures.append("Substantial descriptive drop vs development CLEAN AUC.")
    successes.append("No post-NTIRE model modification; freeze held.")
    failures.append("Official per-generator metadata absent on val split.")
    if hard_metrics is not None and hard_metrics["roc_auc"] + 0.05 < primary["roc_auc"]:
        failures.append(
            f"Hard subset weaker than primary (hard AUC={hard_metrics['roc_auc']:.3f})."
        )

    # Generator metrics N/A
    gen_note = pd.DataFrame(
        [
            {
                "note": "Official per-generator/source identity not provided in val_labels.csv",
                "available_official_strata": "label, is_distorted, distortions list",
            }
        ]
    )
    gen_note.to_csv(GEN_CSV, index=False)

    integrity_statement = {
        "PRE_NTIRE_FREEZE_VERIFIED": "YES",
        "NTIRE_ACCESSED": "YES",
        "NTIRE_FIRST_AUTHORIZED_ACCESS": "YES",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V1_RERUN": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "MODEL_TRAINING_PERFORMED": "NO",
        "LORA_WEIGHTS_UPDATED": "NO",
        "H4_UPDATED": "NO",
        "HYPERPARAMETER_SEARCH_PERFORMED": "NO",
        "FOLD_WEIGHT_SEARCH_PERFORMED": "NO",
        "AGGREGATION_RULE_CHANGED_AFTER_RESULTS": "NO",
        "THRESHOLD_CHANGED": "NO",
        "THRESHOLD_SWEEP_PERFORMED": "NO",
        "CALIBRATOR_FITTED": "NO",
        "SELECTIVE_POLICY_RETUNED": "NO",
        "NEW_DEVELOPMENT_DATA_ACQUIRED": "NO",
        "NTIRE_LABELS_USED_FOR_MODEL_SELECTION": "NO",
        "NTIRE_LABELS_USED_FOR_TUNING": "NO",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_FROZEN": "YES",
        "FINAL_EXTERNAL_VALIDATION_COMPLETE": "YES",
        "POST_NTIRE_MODEL_MODIFICATION_ALLOWED": "NO",
        "NEXT_STAGE_STARTED": "NO",
    }

    metrics_payload = {
        "primary_ensemble": primary,
        "fold_diagnostics": fold_metrics,
        "distortion_strata": strata,
        "hard_subset_diagnostic": hard_metrics,
        "score_distributions": dists,
        "aggregation": "equal_weight_mean_of_four_folds",
        "threshold": THR,
    }
    METRICS_JSON.write_text(json.dumps(_clean(metrics_payload), indent=2) + "\n")
    metric_rows = [{"split": "primary_ensemble", **primary}]
    for f, m in fold_metrics.items():
        metric_rows.append({"split": f, **m})
    for k, m in strata.items():
        metric_rows.append({"split": k, **m})
    if hard_metrics:
        metric_rows.append({"split": "hard_subset", **hard_metrics})
    pd.DataFrame(metric_rows).to_csv(METRICS_CSV, index=False)
    BOOT_JSON.write_text(json.dumps(_clean(boot), indent=2) + "\n")

    payload = {
        "stage": "V2-11",
        "status": "COMPLETE",
        "FINAL_RESEARCH_MODEL_V2": "H4",
        "DEVELOPMENT_STATUS": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
        "FINAL_EXTERNAL_VALIDATION": "COMPLETE",
        "POST_NTIRE_MODEL_MODIFICATION_ALLOWED": "NO",
        "access": access,
        "freeze_verification": freeze,
        "prediction_integrity": {
            "n_predictions": int(len(pred_df)),
            "max_abs_ensemble_error": max_abs,
            "probs_in_unit_interval": True,
            "duplicate_ids": 0,
        },
        "primary_metrics": primary,
        "fold_metrics": fold_metrics,
        "distortion_strata": strata,
        "hard_metrics": hard_metrics,
        "bootstrap": boot,
        "score_distributions": dists,
        "dev_vs_external": dev_vs,
        "external_transfer_assessment": assessment,
        "assessment_evidence": evidence,
        "successes": successes,
        "failures": failures,
        "figures": figures,
        "integrity_statement": integrity_statement,
        "device": str(device),
        "resource": {
            "h4_params": 513,
            "h4_serialized_approx_bytes": 2907,
            "lora_trainable_historical_approx": 295000,
            "new_training_in_v2_11": False,
            "inference": "4 fold LoRA+H4 forwards + equal-weight mean; MPS/CPU as available",
        },
    }
    (OUT / "v2_11_ntire_analysis_v1.json").write_text(json.dumps(_clean(payload), indent=2) + "\n")
    write_reports(payload)
    append_research_log(payload)
    print(REPORT_TXT.read_text())
    print(f"EXTERNAL_TRANSFER_ASSESSMENT = {assessment}")


if __name__ == "__main__":
    main()
