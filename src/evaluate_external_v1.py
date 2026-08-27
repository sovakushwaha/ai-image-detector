#!/usr/bin/env python3
"""
SUPERSEDED — DO NOT RUN FOR STAGE 27A
Replaced by Stage 27A V2 public-dataset evaluation (src/evaluate_external_v2.py).

Stage 27A — post-readiness external evaluation for FINAL_RESEARCH_MODEL_V1 (fal v1.1; historical).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import transforms
from tqdm import tqdm

from mobilenet_v3_small_binary_v1 import MobileNetV3SmallBinaryV1

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external_v1"
META = EXT / "metadata"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER_TABLES = ROOT / "paper" / "tables"

FINAL_POINTER = ROOT / "models" / "FINAL_MODEL_V1.json"
AI_MANIFEST = META / "external_ai_generation_manifest_v1.csv"
REAL_PARTIAL = META / "external_manifest_real_partial_v1.csv"
PROMPTS = ROOT / "metadata" / "external_prompt_set_v1.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SOURCE_SIZE = 224
CANVAS_SIZE = 512
CANVAS_RGB = (32, 32, 32)
JPEG_SUB = 0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

GENERATOR_KEYS = [
    "gpt_image_2",
    "gemini_3_1_flash_image",
    "stable_diffusion_3_5_large",
    "seedream_5_pro",
]
GEN_LABELS = {
    "none": "Real",
    "gpt_image_2": "GPT Image 2",
    "gemini_3_1_flash_image": "Gemini 3.1 Flash Image",
    "stable_diffusion_3_5_large": "Stable Diffusion 3.5 Large",
    "seedream_5_pro": "Seedream 5.0 Pro",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ece_15(probs: np.ndarray, labels: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    n = len(probs)
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        if i == 14:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def nll_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def jpeg_reencode(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=JPEG_SUB)
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        return im.convert("RGB")


def apply_jpeg_q50(image: Image.Image) -> Image.Image:
    return jpeg_reencode(image, 50)


def apply_resize_112(image: Image.Image) -> Image.Image:
    small = image.resize((112, 112), Image.Resampling.LANCZOS)
    return small.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS)


def apply_blur_sigma2(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=2.0))


def apply_screenshot_strong(image: Image.Image) -> Image.Image:
    decoded = jpeg_reencode(image, 65)
    displayed = decoded.resize((384, 384), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_RGB)
    offset = (CANVAS_SIZE - 384) // 2
    canvas.paste(displayed, (offset, offset))
    return canvas.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS).convert("RGB")


TRANSFORMS = {
    "jpeg_q50": apply_jpeg_q50,
    "resize_112": apply_resize_112,
    "blur_sigma2": apply_blur_sigma2,
    "screenshot_strong": apply_screenshot_strong,
}


def controlled_preprocess(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        if w <= h:
            new_w = 256
            new_h = int(round(h * (256 / w)))
        else:
            new_h = 256
            new_w = int(round(w * (256 / h)))
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (im.width - 224) // 2
        top = (im.height - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        return im.convert("RGB")


def assert_gate() -> None:
    ready = json.loads((RESULTS / "external_data_readiness_v1.json").read_text())
    if not ready.get("complete"):
        raise SystemExit("STOP: data readiness gate incomplete — no detector inference")
    counts = ready.get("counts") or {}
    required = {
        "real": 400,
        "gpt_image_2": 100,
        "gemini_3_1_flash_image": 100,
        "stable_diffusion_3_5_large": 100,
        "seedream_5_pro": 100,
        "total": 800,
    }
    for k, n in required.items():
        if int(counts.get(k, 0)) < n:
            raise SystemExit(f"STOP: readiness count {k}={counts.get(k)} < {n}")
    if ready.get("midjourney_required"):
        raise SystemExit("STOP: readiness JSON still requires Midjourney (protocol v1.1 expected)")


def build_unified_manifest() -> pd.DataFrame:
    real = pd.read_csv(REAL_PARTIAL)
    ai = pd.read_csv(AI_MANIFEST)
    ai = ai[ai["generation_status"] == "success"].copy()
    rows = []
    for _, r in real.iterrows():
        rows.append(
            {
                "external_image_id": r["external_image_id"],
                "label": 0,
                "generator": "none",
                "generator_version": "COCO_val2017",
                "prompt_id": "",
                "native_path": r["native_path"],
                "sha256": r["sha256"],
            }
        )
    for _, r in ai.iterrows():
        rows.append(
            {
                "external_image_id": r["external_image_id"],
                "label": 1,
                "generator": r["generator_key"],
                "generator_version": r["actual_model_endpoint"],
                "prompt_id": r["prompt_id"],
                "native_path": r["native_path"],
                "sha256": r["sha256"],
            }
        )
    df = pd.DataFrame(rows).sort_values("external_image_id").reset_index(drop=True)
    if len(df) != 800:
        raise SystemExit(f"STOP: unified manifest has {len(df)} rows, expected 800")
    df.to_csv(META / "external_manifest_v1.csv", index=False)
    return df


def build_controlled(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    ctrl_root = EXT / "controlled"
    for _, r in tqdm(df.iterrows(), total=len(df), desc="controlled"):
        src = ROOT / r["native_path"]
        img = controlled_preprocess(src)
        if r["label"] == 0:
            dest = ctrl_root / "real" / f"{r['external_image_id']}.jpg"
        else:
            dest = ctrl_root / "ai" / r["generator"] / f"{r['external_image_id']}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="JPEG", quality=96, subsampling=0)
        out_rows.append(
            {
                "external_image_id": r["external_image_id"],
                "label": r["label"],
                "generator": r["generator"],
                "prompt_id": r["prompt_id"],
                "source_native_path": r["native_path"],
                "controlled_path": str(dest.relative_to(ROOT)),
                "width": 224,
                "height": 224,
                "mode": "RGB",
                "format": "JPEG",
                "sha256": sha256_file(dest),
            }
        )
    cdf = pd.DataFrame(out_rows)
    cdf.to_csv(META / "external_controlled_manifest_v1.csv", index=False)
    return cdf


def build_transforms(cdf: pd.DataFrame) -> None:
    for cond, fn in TRANSFORMS.items():
        for _, r in tqdm(cdf.iterrows(), total=len(cdf), desc=cond):
            src = ROOT / r["controlled_path"]
            with Image.open(src) as im:
                im.load()
                rgb = im.convert("RGB")
            out = fn(rgb)
            dest = EXT / "transformed" / cond / f"{r['external_image_id']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.save(dest, format="PNG")


def load_final_model(device: torch.device):
    pointer = json.loads(FINAL_POINTER.read_text())
    ckpt_path = ROOT / pointer["checkpoint"]
    if not str(ckpt_path).endswith("mobilenet_resize_jpeg_aug_selected_v1.pt"):
        raise SystemExit("STOP: FINAL_MODEL_V1 pointer does not match expected checkpoint")
    temp = json.loads((ROOT / pointer["temperature_config"]).read_text())
    policy = json.loads((ROOT / pointer["selective_policy_config"]).read_text())
    frozen = json.loads((ROOT / pointer["frozen_config"]).read_text())
    model = MobileNetV3SmallBinaryV1()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, float(temp["temperature"]), policy, float(frozen["threshold"])


def tf_controlled():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def tf_native_deploy():
    # deployment decode path without JPEG standardization
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: ImageOps.exif_transpose(im).convert("RGB")),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.inference_mode()
def predict_paths(
    model,
    paths: list[Path],
    device,
    transform,
    *,
    controlled_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    for p in tqdm(paths, desc="infer"):
        with Image.open(p) as im:
            im.load()
            if controlled_rgb:
                tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
            else:
                tensor = transform(im).unsqueeze(0).to(device)
        logit = model(tensor).squeeze().float().cpu().item()
        logits.append(logit)
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def selective_label(p: float, lower: float, upper: float) -> str:
    if p <= lower:
        return "REAL"
    if p >= upper:
        return "AI-GENERATED"
    return "UNCERTAIN"


def metrics_block(labels, raw_p, cal_p, hist_thr, lower, upper) -> dict:
    hist_pred = (raw_p >= hist_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, hist_pred, labels=[0, 1]).ravel()
    sel = np.array([selective_label(p, lower, upper) for p in cal_p])
    accepted = sel != "UNCERTAIN"
    if accepted.any():
        acc_pred = np.array([1 if s == "AI-GENERATED" else 0 for s in sel[accepted]])
        acc_y = labels[accepted]
        accepted_acc = float(np.mean(acc_pred == acc_y))
        selective_risk = 1.0 - accepted_acc
        accepted_bal = float(balanced_accuracy_score(acc_y, acc_pred)) if len(np.unique(acc_y)) > 1 else float("nan")
    else:
        accepted_acc = selective_risk = accepted_bal = float("nan")
    # AURC: risk-coverage under confidence ordering
    conf = np.maximum(cal_p, 1 - cal_p)
    order = np.argsort(-conf)
    y_ord = labels[order]
    p_ord = cal_p[order]
    risks = []
    for k in range(1, len(labels) + 1):
        pred = (p_ord[:k] >= 0.5).astype(int)
        risks.append(1.0 - float(np.mean(pred == y_ord[:k])))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(trapz(risks, dx=1.0 / len(labels)))
    return {
        "n": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, cal_p)),
        "average_precision": float(average_precision_score(labels, cal_p)),
        "accuracy": float(accuracy_score(labels, hist_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, hist_pred)),
        "precision": float(precision_score(labels, hist_pred, zero_division=0)),
        "recall": float(recall_score(labels, hist_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(labels, hist_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "raw_nll": nll_binary(raw_p, labels),
        "calibrated_nll": nll_binary(cal_p, labels),
        "raw_brier": float(brier_score_loss(labels, raw_p)),
        "calibrated_brier": float(brier_score_loss(labels, cal_p)),
        "raw_ece15": ece_15(raw_p, labels),
        "calibrated_ece15": ece_15(cal_p, labels),
        "selective_coverage": float(accepted.mean()),
        "abstention_rate": float(1 - accepted.mean()),
        "accepted_accuracy": accepted_acc,
        "selective_risk": selective_risk,
        "accepted_balanced_accuracy": accepted_bal,
        "real_class_coverage": float(((sel == "REAL") | (sel == "AI-GENERATED"))[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        "ai_class_coverage": float(((sel == "REAL") | (sel == "AI-GENERATED"))[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "aurc": aurc,
    }


def main() -> None:
    assert_gate()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    print("Building unified manifest / controlled / transforms...")
    manifest = build_unified_manifest()
    controlled = build_controlled(manifest)
    build_transforms(controlled)

    # duplicate audits
    native_hashes = manifest["sha256"].tolist()
    exact_dups = len(native_hashes) - len(set(native_hashes))
    ctrl_hashes = controlled["sha256"].tolist()
    ctrl_dups = len(ctrl_hashes) - len(set(ctrl_hashes))
    print(f"Exact native SHA256 collisions: {exact_dups}")
    print(f"Exact controlled SHA256 collisions: {ctrl_dups}")

    model, T, policy, hist_thr = load_final_model(device)
    lower = float(policy["lower80"])
    upper = float(policy["upper80"])
    print(f"Frozen T={T:.6f} hist_thr={hist_thr:.6f} REAL<= {lower:.6f} AI>= {upper:.6f}")

    # Controlled original + transforms inference
    pred_rows = []
    conditions = ["original"] + list(TRANSFORMS.keys())
    for cond in conditions:
        paths = []
        meta_rows = []
        for _, r in controlled.iterrows():
            if cond == "original":
                path = ROOT / r["controlled_path"]
            else:
                path = EXT / "transformed" / cond / f"{r['external_image_id']}.png"
            paths.append(path)
            meta_rows.append(r)
        logits, raw_p = predict_paths(model, paths, device, tf_controlled(), controlled_rgb=True)
        cal_p = 1.0 / (1.0 + np.exp(-logits / T))
        for i, r in enumerate(meta_rows):
            sel = selective_label(cal_p[i], lower, upper)
            hist_pred = int(raw_p[i] >= hist_thr)
            accepted = sel != "UNCERTAIN"
            true_ai = int(r["label"]) == 1
            if sel == "AI-GENERATED":
                pred_ai = 1
            elif sel == "REAL":
                pred_ai = 0
            else:
                pred_ai = None
            is_correct = (pred_ai == true_ai) if accepted else ""
            pred_rows.append(
                {
                    "external_image_id": r["external_image_id"],
                    "label": int(r["label"]),
                    "generator": r["generator"],
                    "generator_version": "",
                    "prompt_id": r["prompt_id"],
                    "condition": cond,
                    "raw_logit": float(logits[i]),
                    "raw_probability": float(raw_p[i]),
                    "calibrated_probability": float(cal_p[i]),
                    "historical_binary_prediction": hist_pred,
                    "selective_prediction": sel,
                    "selective_confidence": float(max(cal_p[i], 1 - cal_p[i])),
                    "is_accepted": int(accepted),
                    "is_correct_if_accepted": is_correct,
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(RESULTS / "external_controlled_predictions_v1.csv", index=False)

    # Native secondary
    native_rows = []
    n_paths = [ROOT / p for p in manifest["native_path"]]
    n_logits, n_raw = predict_paths(model, n_paths, device, tf_native_deploy(), controlled_rgb=False)
    n_cal = 1.0 / (1.0 + np.exp(-n_logits / T))
    for i, r in manifest.iterrows():
        sel = selective_label(n_cal[i], lower, upper)
        accepted = sel != "UNCERTAIN"
        true_ai = int(r["label"]) == 1
        if sel == "AI-GENERATED":
            pred_ai = 1
        elif sel == "REAL":
            pred_ai = 0
        else:
            pred_ai = None
        is_correct = (pred_ai == true_ai) if accepted else ""
        native_rows.append(
            {
                "external_image_id": r["external_image_id"],
                "label": int(r["label"]),
                "generator": r["generator"],
                "generator_version": r["generator_version"],
                "prompt_id": r["prompt_id"],
                "condition": "native",
                "raw_logit": float(n_logits[i]),
                "raw_probability": float(n_raw[i]),
                "calibrated_probability": float(n_cal[i]),
                "historical_binary_prediction": int(n_raw[i] >= hist_thr),
                "selective_prediction": sel,
                "selective_confidence": float(max(n_cal[i], 1 - n_cal[i])),
                "is_accepted": int(accepted),
                "is_correct_if_accepted": is_correct,
            }
        )
    pd.DataFrame(native_rows).to_csv(RESULTS / "external_native_predictions_v1.csv", index=False)

    # Metrics tables
    metric_rows = []
    for cond in conditions + ["native"]:
        if cond == "native":
            sub = pd.DataFrame(native_rows)
            labels = sub["label"].to_numpy()
            raw_p = sub["raw_probability"].to_numpy()
            cal_p = sub["calibrated_probability"].to_numpy()
        else:
            sub = pred_df[pred_df["condition"] == cond]
            labels = sub["label"].to_numpy()
            raw_p = sub["raw_probability"].to_numpy()
            cal_p = sub["calibrated_probability"].to_numpy()
        m = metrics_block(labels, raw_p, cal_p, hist_thr, lower, upper)
        m["condition"] = cond if cond != "original" else "Controlled Original"
        if cond == "native":
            m["condition"] = "Native Original"
        metric_rows.append(m)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "external_evaluation_metrics_v1.csv", index=False)

    # Generator metrics on controlled original
    orig = pred_df[pred_df["condition"] == "original"].copy()
    gen_rows = []
    for gkey in ["none"] + GENERATOR_KEYS:
        if gkey == "none":
            sub = orig[orig["label"] == 0]
        else:
            sub = orig[orig["generator"] == gkey]
        if len(sub) == 0:
            continue
        p = sub["calibrated_probability"].to_numpy()
        raw = sub["raw_probability"].to_numpy()
        sel = sub["selective_prediction"].to_numpy()
        hist = sub["historical_binary_prediction"].to_numpy()
        y = sub["label"].to_numpy()
        accepted = sel != "UNCERTAIN"
        if gkey == "none":
            hist_rate = float((hist == 0).mean())  # real specificity proxy
            accepted_correct = float((sel[accepted] == "REAL").mean()) if accepted.any() else float("nan")
        else:
            hist_rate = float(hist.mean())
            accepted_correct = float((sel[accepted] == "AI-GENERATED").mean()) if accepted.any() else float("nan")
        gen_rows.append(
            {
                "generator": GEN_LABELS[gkey],
                "generator_key": gkey,
                "n": len(sub),
                "mean_raw_p_ai": float(raw.mean()),
                "mean_calibrated_p_ai": float(p.mean()),
                "median_calibrated_p_ai": float(np.median(p)),
                "historical_ai_rate": hist_rate,
                "selective_coverage": float(accepted.mean()),
                "selective_abstention": float((~accepted).mean()),
                "n_pred_real": int((sel == "REAL").sum()),
                "n_pred_ai": int((sel == "AI-GENERATED").sum()),
                "n_uncertain": int((sel == "UNCERTAIN").sum()),
                "accepted_target_rate": accepted_correct,
            }
        )
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(RESULTS / "external_generator_metrics_v1.csv", index=False)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(PAPER_TABLES / "external_validation_summary.csv", index=False)
    gen_df.to_csv(PAPER_TABLES / "external_generator_performance.csv", index=False)

    # Robustness delta vs controlled original
    primary = metrics_df[metrics_df["condition"] == "Controlled Original"].iloc[0]
    rob_rows = []
    for _, row in metrics_df.iterrows():
        if row["condition"] == "Controlled Original":
            continue
        rob_rows.append(
            {
                "condition": row["condition"],
                "roc_auc": row["roc_auc"],
                "delta_auc_vs_controlled": float(row["roc_auc"] - primary["roc_auc"]),
                "average_precision": row["average_precision"],
                "delta_ap_vs_controlled": float(row["average_precision"] - primary["average_precision"]),
                "selective_coverage": row["selective_coverage"],
                "selective_risk": row["selective_risk"],
                "calibrated_ece15": row["calibrated_ece15"],
            }
        )
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(RESULTS / "external_robustness_metrics_v1.csv", index=False)
    rob_df.to_csv(PAPER_TABLES / "external_robustness_summary.csv", index=False)

    # Stratified bootstrap on controlled original
    print("Running class/generator-stratified bootstrap...")
    boot_df = stratified_bootstrap(orig, hist_thr, lower, upper)
    boot_df.to_csv(RESULTS / "external_bootstrap_uncertainty_v1.csv", index=False)
    boot_df.to_csv(PAPER_TABLES / "external_bootstrap_summary.csv", index=False)
    (RESULTS / "external_bootstrap_uncertainty_v1.json").write_text(
        json.dumps(
            {
                "n_replicates": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
                "method": "class_and_generator_stratified",
                "metrics": boot_df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )

    write_figures(orig, metrics_df, gen_df, rob_df)
    write_report(
        exact_dups=exact_dups,
        ctrl_dups=ctrl_dups,
        T=T,
        hist_thr=hist_thr,
        lower=lower,
        upper=upper,
        metrics_df=metrics_df,
        gen_df=gen_df,
        rob_df=rob_df,
        boot_df=boot_df,
    )
    update_research_log(metrics_df, gen_df, boot_df)

    print("Primary controlled original AUC:", primary["roc_auc"])
    print("Stage 27A external evaluation COMPLETE.")
    (RESULTS / "external_evaluation_core_done_v1.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exact_native_duplicates": exact_dups,
                "exact_controlled_duplicates": ctrl_dups,
                "temperature": T,
                "historical_threshold": hist_thr,
                "lower80": lower,
                "upper80": upper,
                "primary_auc": float(primary["roc_auc"]),
                "primary_ap": float(primary["average_precision"]),
                "primary_selective_coverage": float(primary["selective_coverage"]),
                "primary_selective_risk": float(primary["selective_risk"]),
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )


def stratified_bootstrap(orig: pd.DataFrame, hist_thr: float, lower: float, upper: float) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = {}
    strata["real"] = orig[orig["label"] == 0].reset_index(drop=True)
    for g in GENERATOR_KEYS:
        strata[g] = orig[orig["generator"] == g].reset_index(drop=True)
    keys = ["roc_auc", "average_precision", "balanced_accuracy", "selective_coverage", "selective_risk", "calibrated_ece15"]
    store = {k: [] for k in keys}
    for _ in range(BOOTSTRAP_N):
        parts = []
        for name, sdf in strata.items():
            n = len(sdf)
            idx = rng.integers(0, n, size=n)
            parts.append(sdf.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        m = metrics_block(
            sample["label"].to_numpy(),
            sample["raw_probability"].to_numpy(),
            sample["calibrated_probability"].to_numpy(),
            hist_thr,
            lower,
            upper,
        )
        for k in keys:
            store[k].append(m[k])
    rows = []
    for k in keys:
        arr = np.asarray(store[k], dtype=np.float64)
        rows.append(
            {
                "metric": k,
                "point_estimate_mean": float(np.mean(arr)),
                "bootstrap_std": float(np.std(arr, ddof=1)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "n_replicates": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def write_figures(orig: pd.DataFrame, metrics_df: pd.DataFrame, gen_df: pd.DataFrame, rob_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # ROC
    y = orig["label"].to_numpy()
    p = orig["calibrated_probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#1f4e79", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("External validation ROC (controlled original)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_validation_roc_v1.png", dpi=160)
    plt.close(fig)

    # Generator mean calibrated p
    gplot = gen_df[gen_df["generator_key"] != "none"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(gplot["generator"], gplot["mean_calibrated_p_ai"], color="#2a6f97")
    ax.axhline(0.5, color="#888888", ls="--", lw=1)
    ax.set_ylabel("Mean calibrated P(AI)")
    ax.set_title("External generator scores (controlled original)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_generator_scores_v1.png", dpi=160)
    plt.close(fig)

    # Robustness AUC
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = ["Controlled Original"] + rob_df["condition"].tolist()
    aucs = [float(metrics_df[metrics_df["condition"] == "Controlled Original"].iloc[0]["roc_auc"])] + rob_df[
        "roc_auc"
    ].tolist()
    ax.bar(names, aucs, color="#3d5a80")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("External robustness AUC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_robustness_auc_v1.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    exact_dups: int,
    ctrl_dups: int,
    T: float,
    hist_thr: float,
    lower: float,
    upper: float,
    metrics_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    boot_df: pd.DataFrame,
) -> None:
    primary = metrics_df[metrics_df["condition"] == "Controlled Original"].iloc[0]
    lines = [
        "STAGE 27A — EXTERNAL VALIDATION REPORT (protocol v1.1)",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Frozen model: FINAL_RESEARCH_MODEL_V1 = C0",
        f"Temperature T: {T}",
        f"Historical Youden threshold: {hist_thr}",
        f"Selective 80%: REAL p<={lower} ; AI p>={upper}",
        "",
        f"Exact native SHA256 duplicates: {exact_dups}",
        f"Exact controlled SHA256 duplicates: {ctrl_dups}",
        "",
        "Primary (Controlled Original):",
        f"  ROC-AUC: {primary['roc_auc']:.6f}",
        f"  AP: {primary['average_precision']:.6f}",
        f"  Balanced accuracy (historical thr): {primary['balanced_accuracy']:.6f}",
        f"  Selective coverage: {primary['selective_coverage']:.6f}",
        f"  Selective risk: {primary['selective_risk']:.6f}",
        f"  Calibrated ECE-15: {primary['calibrated_ece15']:.6f}",
        "",
        "Bootstrap 95% CIs (class/generator stratified, n=5000):",
    ]
    for _, r in boot_df.iterrows():
        lines.append(
            f"  {r['metric']}: mean={r['point_estimate_mean']:.6f} "
            f"CI=[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
        )
    lines += ["", "Generator metrics:", gen_df.to_string(index=False), "", "Robustness:", rob_df.to_string(index=False)]
    lines += [
        "",
        "Integrity:",
        "  Protocol amendment before AI generation: YES",
        "  Protocol amendment before external model inference: YES",
        "  External results used to choose replacement: NO",
        "  Final detector / T / selective policy changed: NO",
        "  Generator-specific tuning: NO",
    ]
    (RESULTS / "external_validation_report_v1.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "paper" / "external_validation_report_v1.md").write_text(
        "# External Validation Report v1.1 (Stage 27A)\n\n```\n" + "\n".join(lines) + "\n```\n"
    )


def update_research_log(metrics_df: pd.DataFrame, gen_df: pd.DataFrame, boot_df: pd.DataFrame) -> None:
    primary = metrics_df[metrics_df["condition"] == "Controlled Original"].iloc[0]
    auc_ci = boot_df[boot_df["metric"] == "roc_auc"].iloc[0]
    block = f"""
### Stage 27A — External evaluation COMPLETE (protocol v1.1)

**Date:** {datetime.now(timezone.utc).date().isoformat()}  
**Gate:** 800/800 PASSED (COCO 400 + GPT/Gemini/SD3.5/Seedream 100 each via fal.ai)  
**Model:** FINAL_RESEARCH_MODEL_V1 unchanged; T/selective policy unchanged.

**Primary controlled-original:** ROC-AUC={primary['roc_auc']:.4f} (bootstrap 95% CI [{auc_ci['ci_low']:.4f}, {auc_ci['ci_high']:.4f}]); AP={primary['average_precision']:.4f}; selective coverage={primary['selective_coverage']:.4f}; selective risk={primary['selective_risk']:.4f}.

**Artifacts:** `results/external_evaluation_metrics_v1.csv`, `results/external_generator_metrics_v1.csv`, `results/external_bootstrap_uncertainty_v1.json`, `paper/external_validation_report_v1.md`.

"""
    log_path = ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    marker = "### External evaluation status"
    if marker in text and "External evaluation COMPLETE" not in text:
        text = text.replace(marker, block + marker, 1)
        log_path.write_text(text)


if __name__ == "__main__":
    main()
