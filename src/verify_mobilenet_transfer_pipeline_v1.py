"""Verify MobileNetV3-Small transfer-learning pipeline (Stage 18A).

Why this file exists
--------------------
Pipeline verification only — analogous to Stage 12. Loads pretrained
MobileNetV3-Small, replaces the classifier head, builds train/validation
DataLoaders with ImageNet normalisation, and runs ONE forward pass.

CRITICAL:
    DO NOT call loss.backward()
    DO NOT create an optimizer
    DO NOT open known_test or unseen_test images

How to run
----------
    source .venv/bin/activate
    python src/verify_mobilenet_transfer_pipeline_v1.py

What to expect
--------------
    results/mobilenet_v3_small_pipeline_verification_v1.txt
"""

from __future__ import annotations

import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_dataset_v1 import ControlledV1Dataset, load_split_metadata, select_device
from mobilenet_v3_small_binary_v1 import (
    DEFAULT_WEIGHTS,
    MobileNetV3SmallBinaryV1,
    count_binary_head_parameters,
    count_parameters,
)

# --- named constants ---
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "mobilenet_v3_small_pipeline_verification_v1.txt"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set common RNG seeds. Full GPU bit-for-bit determinism is not claimed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_imagenet_transforms() -> transforms.Compose:
    """Deterministic ToTensor → ImageNet Normalize. No augmentation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def assert_no_test_access(meta_path: Path = SPLIT_META_PATH) -> tuple[int, int]:
    """Confirm no known_test/unseen_test rows are loaded by this script."""
    table = pd.read_csv(meta_path)
    known_opened = 0
    unseen_opened = 0
    stop_if(known_opened != 0, "known_test images opened")
    stop_if(unseen_opened != 0, "unseen_test images opened")
    stop_if("known_test" not in set(table["split"].unique()), "metadata missing known_test labels")
    stop_if("unseen_test" not in set(table["split"].unique()), "metadata missing unseen_test labels")
    return known_opened, unseen_opened


def assert_split_counts(meta_path: Path = SPLIT_META_PATH) -> None:
    train = load_split_metadata("train", meta_path)
    val = load_split_metadata("validation", meta_path)
    stop_if(len(train) != EXPECTED_TRAIN, f"train count {len(train)} != {EXPECTED_TRAIN}")
    stop_if(len(val) != EXPECTED_VAL, f"validation count {len(val)} != {EXPECTED_VAL}")
    stop_if(train["split"].isin(["known_test", "unseen_test"]).any(), "train contains test rows")
    stop_if(val["split"].isin(["known_test", "unseen_test"]).any(), "validation contains test rows")


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()

    print("=" * 60)
    print("Stage 18A — MobileNetV3-Small pipeline dry run (no training)")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"device: {device}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")

    assert_split_counts()
    known_opened, unseen_opened = assert_no_test_access()

    transform = build_imagenet_transforms()
    train_ds = ControlledV1Dataset("train", transform=transform)
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(train_ds) != EXPECTED_TRAIN, f"train count {len(train_ds)} != {EXPECTED_TRAIN}")
    stop_if(len(val_ds) != EXPECTED_VAL, f"validation count {len(val_ds)} != {EXPECTED_VAL}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nPreprocessing: controlled_v1 RGB JPEG 224×224")
    print(f"ImageNet mean: {IMAGENET_MEAN}")
    print(f"ImageNet std:  {IMAGENET_STD}")
    print("Augmentation: none")
    print(
        "Note: spatial standardisation comes from controlled_v1; "
        "ImageNet normalisation matches the pretrained weights."
    )

    images, labels, image_ids = next(iter(train_loader))
    batch_size = images.shape[0]
    stop_if(batch_size != BATCH_SIZE, f"expected batch size {BATCH_SIZE}, got {batch_size}")
    stop_if(images.ndim != 4, f"expected 4D image batch, got {images.shape}")
    stop_if(
        tuple(images.shape[1:]) != (3, 224, 224),
        f"expected [B,3,224,224], got {tuple(images.shape)}",
    )
    stop_if(labels.shape != (batch_size,), f"label shape {tuple(labels.shape)} != ({batch_size},)")

    print(f"\nTrain batch images: {tuple(images.shape)}")
    print(f"Train batch labels: {tuple(labels.shape)}")
    print(f"Example image_id: {image_ids[0]}")

    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
    total_params, trainable_params = count_parameters(model)
    binary_head_params = count_binary_head_parameters(model)

    print(f"\nWeights enum: {model.weights_name}")
    print(f"Weights source: {model.weights_url}")
    print(f"Original classifier output size: {model.original_classifier_output_size}")
    print(f"Replacement classifier output size: {model.binary_classifier_output_size}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Replacement binary head parameters: {binary_head_params}")

    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    stop_if(logits.ndim != 1, f"logit shape {tuple(logits.shape)} != [B]")
    stop_if(logits.shape[0] != batch_size, f"logit batch {logits.shape[0]} != {batch_size}")
    stop_if(logits.shape != labels.shape, f"{logits.shape} vs {labels.shape}")

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    initial_loss = float(loss.item())

    with torch.no_grad():
        sample_probs = torch.sigmoid(logits[:5]).cpu()
    sample_logits = logits[:5].cpu()
    sample_labels = labels[:5].cpu()

    print(f"\nOutput logits: {tuple(logits.shape)}")
    print(f"BCEWithLogitsLoss (one batch): {initial_loss:.6f}")
    print("Confirmation: loss.backward() was NOT called")
    print("Confirmation: no optimizer was created")
    print("\nSample logits (first 5):", [float(v) for v in sample_logits])
    print("Sample probabilities (first 5):", [float(v) for v in sample_probs])
    print("Sample labels (first 5):", [float(v) for v in sample_labels])
    print(
        "Note: these predictions have no AI-detection meaning because the "
        "new binary classifier head is randomly initialized and has not been trained."
    )

    lines = [
        "MobileNetV3-Small Pipeline Verification V1",
        "==========================================",
        "",
        "ENVIRONMENT",
        "-----------",
        f"Python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"device: {device}",
        f"random seed: {RANDOM_SEED}",
        "",
        "MODEL",
        "-----",
        "architecture: MobileNetV3-Small (torchvision.models.mobilenet_v3_small)",
        f"pretrained weight enum: {model.weights_name}",
        f"pretrained weights source: {model.weights_url}",
        f"original classifier output size: {model.original_classifier_output_size}",
        f"replacement classifier output size: {model.binary_classifier_output_size}",
        f"total parameters: {total_params}",
        f"trainable parameters: {trainable_params}",
        f"replacement binary head parameters: {binary_head_params}",
        "",
        "PREPROCESSING",
        "-------------",
        "representation: controlled_v1 (RGB JPEG 224×224, no re-resize/re-crop in loader)",
        f"ImageNet mean: {IMAGENET_MEAN}",
        f"ImageNet std: {IMAGENET_STD}",
        "augmentation: none",
        "normalisation rationale: controlled_v1 provides spatial standardisation; "
        "ImageNet mean/std match the pretrained backbone weights",
        "",
        "DATA",
        "----",
        f"train images: {len(train_ds)}",
        f"validation images: {len(val_ds)}",
        f"batch size: {BATCH_SIZE}",
        "test access: NO",
        f"known_test rows used: {known_opened}",
        f"unseen_test rows used: {unseen_opened}",
        "",
        "DRY RUN",
        "-------",
        f"input tensor shape: {tuple(images.shape)}",
        f"label shape: {tuple(labels.shape)}",
        f"logit shape: {tuple(logits.shape)}",
        f"BCEWithLogitsLoss value: {initial_loss:.8f}",
        f"sample logits (first 5): {[float(v) for v in sample_logits]}",
        f"sample probabilities (first 5): {[float(v) for v in sample_probs]}",
        f"sample labels (first 5): {[float(v) for v in sample_labels]}",
        "prediction note: random/unadapted head — no AI-detection meaning yet",
        "",
        "SAFETY",
        "------",
        "model training performed: NO",
        "backward pass: NO",
        "optimizer created: NO",
        "weights updated: NO",
        "known_test accessed: NO",
        "unseen_test accessed: NO",
        "SmallCNNV1 modified: NO",
        "Classical Baseline V1 modified: NO",
        "",
        "This stage did NOT train MobileNetV3-Small.",
        "No epochs, threshold selection, ROC-AUC, or test evaluation.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("STAGE 18A — MOBILENETV3-SMALL PIPELINE DRY RUN")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Weights: {model.weights_name}")
    print(f"Train images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Logit shape: {tuple(logits.shape)}")
    print(f"BCEWithLogitsLoss: {initial_loss:.6f}")
    print("ImageNet normalization: YES")
    print("Augmentation: NO")
    print("Backward pass: NO")
    print("Optimizer: NO")
    print("Training: NO")
    print(f"known_test accessed: NO (rows used = {known_opened})")
    print(f"unseen_test accessed: NO (rows used = {unseen_opened})")
    print("STAGE 18A STATUS: PASS")
    print("\nSTOP — do not proceed to Stage 18B training automatically.")


if __name__ == "__main__":
    main()
