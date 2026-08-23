"""Evaluate frozen MobileNet A0 on RQ3 validation robustness suite (Stage 23A).

Why this file exists
--------------------
Establishes the clean MobileNet development reference (RobustValAUC) on
validation transforms only. No training. No test inference.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_rq3_baseline_validation_v1.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cnn_dataset_v1 import PROJECT_ROOT, EXPECTED_SIZE, select_device, stop_if
from mobilenet_v3_small_binary_v1 import DEFAULT_WEIGHTS, MobileNetV3SmallBinaryV1
from rq3_augmentations_v1 import (
    CombinedRobustTransform,
    BlurAwareTransform,
    ResizeJPEGAwareTransform,
    SEED,
    force_seed,
)

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "mobilenet_v3_small_frozen_config_v1.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "mobilenet_v3_small_selected_v1.pt"

METRICS_CSV = PROJECT_ROOT / "results" / "rq3_baseline_validation_metrics_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "rq3_baseline_validation_report_v1.txt"
PROTOCOL_PATH = PROJECT_ROOT / "results" / "rq3_protocol_v1.txt"
AUG_QC_FIG = PROJECT_ROOT / "figures" / "rq3_training_augmentation_examples_v1.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
SELECTION_CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
ALL_CONDITIONS = SELECTION_CONDITIONS + ["screenshot_strong"]


class PathDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} bad size")
        return self.transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


@torch.no_grad()
def predict(model, rows: pd.DataFrame, device: torch.device) -> np.ndarray:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    model.eval()
    loader = DataLoader(PathDataset(rows, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probs = np.empty(len(rows), dtype=float)
    for images, _, indices in tqdm(loader, desc="MobileNet A0", leave=False):
        logits = model(images.to(device)).detach().cpu().numpy().reshape(-1)
        batch = 1.0 / (1.0 + np.exp(-logits))
        for i, idx in enumerate(indices.numpy()):
            probs[int(idx)] = float(batch[i])
    return probs


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_condition_frames(meta: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    original = meta[meta["split"] == "validation"][["image_id", "processed_path", "label", "generator"]].copy()
    original = original.rename(columns={"image_id": "source_image_id", "processed_path": "path"})
    original = original.sort_values("source_image_id").reset_index(drop=True)
    stop_if(len(original) != 456, f"validation original count {len(original)}")
    frames["original"] = original

    for condition in ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]:
        sub = manifest[manifest["condition"] == condition].copy()
        sub = sub.rename(columns={"output_path": "path"})
        sub = sub[["source_image_id", "path", "label", "generator"]].sort_values("source_image_id").reset_index(drop=True)
        stop_if(len(sub) != 456, f"{condition} count {len(sub)}")
        stop_if(
            not original["source_image_id"].astype(str).equals(sub["source_image_id"].astype(str)),
            f"{condition} source id alignment failed",
        )
        stop_if(not np.array_equal(original["label"].to_numpy(), sub["label"].to_numpy()), f"{condition} label mismatch")
        frames[condition] = sub
    return frames


def save_augmentation_qc(meta: pd.DataFrame) -> None:
    force_seed(SEED)
    train = meta[meta["split"] == "train"].sort_values("image_id").reset_index(drop=True)
    ids = train["image_id"].tolist()[:2]
    transforms_map = {
        "Original": None,
        "A1 Blur-aware": BlurAwareTransform(rng=random.Random(SEED)),
        "A2 Resize+JPEG": ResizeJPEGAwareTransform(rng=random.Random(SEED + 1)),
        "A3 Combined": CombinedRobustTransform(rng=random.Random(SEED + 2)),
    }
    fig, axes = plt.subplots(len(ids), len(transforms_map), figsize=(10, 5))
    for r, image_id in enumerate(ids):
        src_path = PROJECT_ROOT / train[train["image_id"] == image_id].iloc[0]["processed_path"]
        with Image.open(src_path) as image:
            base = image.convert("RGB")
        for c, (title, tfm) in enumerate(transforms_map.items()):
            ax = axes[r, c]
            out = base if tfm is None else tfm(base.copy())
            # Force each regime once for QC visibility when random may skip
            if title.startswith("A1"):
                from rq3_augmentations_v1 import apply_gaussian_blur

                out = apply_gaussian_blur(base.copy(), 1.5)
            elif title.startswith("A2"):
                from rq3_augmentations_v1 import apply_jpeg, apply_resize_degradation

                out = apply_jpeg(apply_resize_degradation(base.copy(), 140), 70)
            elif title.startswith("A3"):
                from rq3_augmentations_v1 import apply_gaussian_blur, apply_jpeg, apply_resize_degradation

                out = apply_gaussian_blur(base.copy(), 1.2)
                out = apply_resize_degradation(out, 128)
                out = apply_jpeg(out, 60)
            ax.imshow(out)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(image_id, fontsize=8)
    fig.suptitle("RQ3 training augmentation examples (QC; forced examples for visibility)")
    fig.tight_layout()
    AUG_QC_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUG_QC_FIG, dpi=150)
    plt.close(fig)


def write_protocol(baseline_row: dict) -> None:
    text = "\n".join(
        [
            "RQ3 Development Protocol — Stage 23A",
            "====================================",
            "",
            "## Sequential-study note",
            "RQ3 is a sequential follow-up experiment motivated by RQ2 findings on the same",
            "pilot benchmark. RQ2 identified robustness failures using the frozen test set.",
            "Therefore RQ3 augmentation choices are informed by prior RQ2 evidence and must",
            "not be described as independent confirmatory evidence on the same transformed",
            "test samples. Model development uses ONLY train + validation. Stronger",
            "confirmation will require external/new-generator evaluation.",
            "",
            "## Base model",
            "MobileNetV3-Small (A0 = existing clean-trained frozen baseline).",
            "Chosen as a lightweight (~1.52M params) intervention model with unseen ROC-AUC",
            "statistically indistinguishable from EfficientNet-B0 under RQ1 bootstrap analysis.",
            "Not claimed universally superior. EfficientNet remains frozen comparison only.",
            "",
            "## Regimes",
            "A0 mobilenet_clean_v1 — existing clean baseline (no retraining)",
            "A1 mobilenet_blur_aug_v1 — blur-aware training",
            "A2 mobilenet_resize_jpeg_aug_v1 — resize + JPEG-aware training",
            "A3 mobilenet_combined_aug_v1 — blur + resize + JPEG-aware training",
            "",
            "## Why these augmentations (RQ2 MobileNet unseen ΔAUC evidence)",
            "blur_sigma2 ≈ -0.166 (highest priority)",
            "resize_112 ≈ -0.078 (second)",
            "screenshot_strong ≈ -0.066 (kept OUT of augmentation as composite holdout)",
            "jpeg_q50 ≈ -0.041 (included as common deployment transform)",
            "crop_75 ≈ -0.001 (excluded; minimal measured degradation)",
            "",
            "## Augmentation parameters (online; validation never randomly augmented)",
            "A1: blur p=0.30, sigma ~ Uniform(0.5, 2.0)",
            "A2: resize p=0.30 intermediate side ~ UniformInt[112,192]; JPEG p=0.30 quality ~ UniformInt[50,90]",
            "A3: independent blur/resize/JPEG with same parameters",
            "Order: 1) blur 2) resize degradation 3) JPEG 4) ToTensor 5) ImageNet normalize",
            "Excluded: crop, screenshot, rotation, colour jitter, flip",
            "Seed: 42",
            "",
            "## Validation robustness suite",
            "456 validation sources × 4 deterministic transforms = 1824 images",
            "Conditions: original (controlled_v1), jpeg_q50, resize_112, blur_sigma2, screenshot_strong",
            "",
            "## Robust validation score",
            "RobustValAUC = (AUC_original + AUC_jpeg50 + AUC_resize112 + AUC_blur2) / 4",
            "screenshot_strong is NOT part of checkpoint selection",
            "",
            "## Model-selection rule",
            "1. highest RobustValAUC",
            "2. if tied, higher clean validation AUC",
            "3. if tied, higher mean AP across the four selection conditions",
            "4. if tied, earlier epoch",
            "No test data.",
            "",
            "## Screenshot policy",
            "Evaluated on validation as auxiliary held-out transform; excluded from augmentation and selection.",
            "",
            "## Test policy",
            "No RQ3 test evaluation during development.",
            "",
            "## Baseline A0 reference (Stage 23A)",
            f"RobustValAUC = {baseline_row['robust_val_auc']:.8f}",
            f"Original AUC = {baseline_row['auc_original']:.8f}",
            f"JPEG50 AUC = {baseline_row['auc_jpeg_q50']:.8f}",
            f"Resize112 AUC = {baseline_row['auc_resize_112']:.8f}",
            f"Blur2 AUC = {baseline_row['auc_blur_sigma2']:.8f}",
            f"ScreenshotStrong AUC = {baseline_row['auc_screenshot_strong']:.8f}",
            "",
            "Training performed in Stage 23A: NO",
        ]
    ) + "\n"
    PROTOCOL_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    print("STAGE 23A — RQ3 BASELINE VALIDATION EVALUATION")
    stop_if(not MANIFEST_PATH.exists(), f"missing manifest: {MANIFEST_PATH}")
    meta = pd.read_csv(SPLIT_META_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    stop_if(len(manifest) != 1824, f"manifest rows {len(manifest)} != 1824")
    stop_if((manifest["split"] != "validation").any(), "non-validation rows in RQ3 suite")

    frozen = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    threshold = float(frozen["threshold"])
    print(f"Frozen MobileNet threshold: {threshold:.12f}")

    frames = build_condition_frames(meta, manifest)
    device = select_device()
    print(f"Device: {device}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric_rows = []
    aucs = {}
    for condition in ALL_CONDITIONS:
        rows = frames[condition]
        probs = predict(model, rows, device)
        y_true = rows["label"].to_numpy(dtype=int)
        auc = float(roc_auc_score(y_true, probs))
        ap = float(average_precision_score(y_true, probs))
        thr = threshold_metrics(y_true, probs, threshold)
        aucs[condition] = auc
        metric_rows.append(
            {
                "model": "MobileNetV3-Small",
                "regime": "A0",
                "condition": condition,
                "num_samples": int(len(rows)),
                "roc_auc": auc,
                "average_precision": ap,
                "balanced_accuracy": thr["balanced_accuracy"],
                "recall": thr["recall"],
                "specificity": thr["specificity"],
                "f1": thr["f1"],
                "threshold": threshold,
                "used_in_robust_val_auc": condition in SELECTION_CONDITIONS,
            }
        )
        print(f"{condition}: AUC={auc:.6f} AP={ap:.6f}")

    robust_val_auc = float(np.mean([aucs[c] for c in SELECTION_CONDITIONS]))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    report = "\n".join(
        [
            "RQ3 Baseline Validation Report — Stage 23A",
            "=========================================",
            "",
            "Model: frozen MobileNetV3-Small (A0 / mobilenet_clean_v1)",
            f"Threshold: {threshold:.12f} (diagnostic only; not used for selection)",
            "Split: validation only",
            "Test inference: NO",
            "Training: NO",
            "",
            "ROC-AUC by condition:",
            *[f"- {r.condition}: AUC={r.roc_auc:.8f}, AP={r.average_precision:.8f}" for r in metrics_df.itertuples()],
            "",
            f"RobustValAUC = (orig+jpeg50+resize112+blur2)/4 = {robust_val_auc:.8f}",
            "screenshot_strong excluded from RobustValAUC",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Training performed: NO",
            "- Test inference performed: NO",
            "- Validation robustness inference: YES",
        ]
    ) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    baseline_summary = {
        "robust_val_auc": robust_val_auc,
        "auc_original": aucs["original"],
        "auc_jpeg_q50": aucs["jpeg_q50"],
        "auc_resize_112": aucs["resize_112"],
        "auc_blur_sigma2": aucs["blur_sigma2"],
        "auc_screenshot_strong": aucs["screenshot_strong"],
    }
    write_protocol(baseline_summary)
    save_augmentation_qc(meta)

    print("\nSTAGE 23A — RQ3 PROTOCOL LOCK COMPLETE")
    print("\nValidation sources: 456")
    print("Validation transformed images: 1824")
    print("\nConditions:\nOriginal\nJPEG50\nResize112\nBlur2\nScreenshotStrong")
    print("\nBASELINE MOBILE NET VALIDATION")
    print(f"Original AUC: {aucs['original']:.6f}")
    print(f"JPEG50 AUC: {aucs['jpeg_q50']:.6f}")
    print(f"Resize112 AUC: {aucs['resize_112']:.6f}")
    print(f"Blur2 AUC: {aucs['blur_sigma2']:.6f}")
    print(f"ScreenshotStrong AUC: {aucs['screenshot_strong']:.6f}")
    print(f"\nBaseline RobustValAUC: {robust_val_auc:.6f}")
    print("\nRQ3 REGIMES LOCKED")
    print("A0 Clean\nA1 Blur-aware\nA2 Resize+JPEG-aware\nA3 Combined Blur+Resize+JPEG")
    print("\nScreenshot augmentation: NO")
    print("Crop augmentation: NO")
    print("\nTraining performed: NO")
    print("Test inference: NO")
    print("\nRQ3 STATUS:\nDEVELOPMENT PROTOCOL LOCKED")
    print("\nSTOP BEFORE RQ3 TRAINING.")


if __name__ == "__main__":
    main()
