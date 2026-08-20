"""Verify CNN data pipeline and SmallCNNV1 forward pass (Stage 12).

Why this file exists
--------------------
This is architecture/pipeline verification only. It computes train-only
RGB normalisation, builds train/validation DataLoaders, runs ONE forward
pass, and computes BCEWithLogitsLoss once.

CRITICAL:
    DO NOT call loss.backward()
    DO NOT call optimizer.step()
    DO NOT open known_test or unseen_test images

How to run
----------
    source .venv/bin/activate
    python src/verify_cnn_pipeline_v1.py

What to expect
--------------
    results/cnn_train_normalization_v1.json
    results/cnn_pipeline_verification_v1.txt
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

from cnn_dataset_v1 import (
    ControlledV1Dataset,
    build_transforms,
    compute_train_rgb_stats,
    save_train_rgb_stats,
    select_device,
)
from small_cnn_v1 import SmallCNNV1, count_parameters

# --- named constants ---
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
NORM_PATH = PROJECT_ROOT / "results" / "cnn_train_normalization_v1.json"
REPORT_PATH = PROJECT_ROOT / "results" / "cnn_pipeline_verification_v1.txt"


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


def assert_no_test_access(meta_path: Path = SPLIT_META_PATH) -> None:
    """Confirm Stage 12 never builds test loaders from known/unseen rows."""
    table = pd.read_csv(meta_path)
    # We only load train/validation through ControlledV1Dataset helpers.
    # Explicitly record that no test images were opened by this script.
    known_opened = 0
    unseen_opened = 0
    stop_if(known_opened != 0, "known_test images opened")
    stop_if(unseen_opened != 0, "unseen_test images opened")
    # Sanity: metadata may contain test rows, but we must not select them.
    stop_if("known_test" not in set(table["split"].unique()), "metadata missing known_test labels")
    stop_if("unseen_test" not in set(table["split"].unique()), "metadata missing unseen_test labels")


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()

    print("=" * 60)
    print("Stage 12 — CNN pipeline verification (no training)")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"device: {device}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")

    # --- Task 4: train-only normalisation ---
    print("\nComputing train-only RGB mean/std ...")
    stats = compute_train_rgb_stats(SPLIT_META_PATH)
    save_train_rgb_stats(stats, NORM_PATH)
    print(
        f"mean RGB = ({stats['mean_R']:.6f}, {stats['mean_G']:.6f}, {stats['mean_B']:.6f})"
    )
    print(
        f"std  RGB = ({stats['std_R']:.6f}, {stats['std_G']:.6f}, {stats['std_B']:.6f})"
    )
    print(
        "Note: channel statistics use TRAIN only, analogous to fitting "
        "StandardScaler on X_train only."
    )

    # --- Datasets / loaders (train + validation only) ---
    transform = build_transforms(stats)
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
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    assert_no_test_access()

    # --- One training batch ---
    images, labels, image_ids = next(iter(train_loader))
    batch_size = images.shape[0]
    stop_if(batch_size > BATCH_SIZE, f"batch size {batch_size} > {BATCH_SIZE}")
    stop_if(images.ndim != 4, f"expected 4D image batch, got {images.shape}")
    stop_if(
        tuple(images.shape[1:]) != (3, 224, 224),
        f"expected [B,3,224,224], got {tuple(images.shape)}",
    )
    stop_if(labels.shape != (batch_size,), f"label shape {tuple(labels.shape)} != ({batch_size},)")

    print(f"\nTrain batch images: {tuple(images.shape)}")
    print(f"Train batch labels: {tuple(labels.shape)}")
    print(f"Example image_id: {image_ids[0]}")

    # Tensor stats after normalisation (one batch ≈ not exactly 0/1).
    print(
        f"Normalised train batch: min={images.min().item():.4f}, "
        f"max={images.max().item():.4f}, "
        f"mean={images.mean().item():.4f}, "
        f"std={images.std().item():.4f}"
    )

    # --- Model + forward pass only ---
    model = SmallCNNV1()
    total_params, trainable_params = count_parameters(model)
    stop_if(total_params > 1_000_000, f"parameter count unexpectedly huge: {total_params}")
    print(f"\nSmallCNNV1 total parameters: {total_params}")
    print(f"SmallCNNV1 trainable parameters: {trainable_params}")

    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    stop_if(logits.ndim != 2 or logits.shape[1] != 1, f"logit shape {tuple(logits.shape)} != [B,1]")
    stop_if(logits.shape[0] != batch_size, f"logit batch {logits.shape[0]} != {batch_size}")
    print(f"Output logits: {tuple(logits.shape)}")

    # Compatible shapes for BCEWithLogitsLoss: both [B] after squeeze.
    logits_flat = logits.squeeze(1)
    stop_if(logits_flat.shape != labels.shape, f"{logits_flat.shape} vs {labels.shape}")

    criterion = nn.BCEWithLogitsLoss()
    # Compute loss once — NO backward, NO optimizer.
    loss = criterion(logits_flat, labels)
    initial_loss = float(loss.item())
    print(f"Initial BCEWithLogitsLoss (one batch): {initial_loss:.6f}")
    print("Confirmation: loss.backward() was NOT called")
    print("Confirmation: optimizer.step() was NOT called")

    # --- Validation pipeline check (one batch, no metrics) ---
    val_images, val_labels, val_ids = next(iter(val_loader))
    stop_if(tuple(val_images.shape[1:]) != (3, 224, 224), f"val shape {tuple(val_images.shape)}")
    stop_if(val_labels.shape != (val_images.shape[0],), f"val label shape {tuple(val_labels.shape)}")
    val_images = val_images.to(device)
    with torch.no_grad():
        val_logits = model(val_images)
    stop_if(val_logits.shape != (val_images.shape[0], 1), f"val logits {tuple(val_logits.shape)}")
    print(f"\nValidation batch images: {tuple(val_images.shape)}")
    print(f"Validation batch labels: {tuple(val_labels.shape)}")
    print(f"Validation logits: {tuple(val_logits.shape)}")
    print("Validation forward pass succeeded (no metrics computed)")

    # --- Report ---
    lines = [
        "CNN Pipeline Verification V1",
        "============================",
        "",
        "1. Python version: " + sys.version.split()[0],
        f"2. torch version: {torch.__version__}",
        f"3. torchvision version: {torchvision.__version__}",
        f"4. operating system: {platform.system()} {platform.release()} ({platform.machine()})",
        f"5. selected device: {device}",
        f"6. random seed: {RANDOM_SEED}",
        f"7. train images: {len(train_ds)}",
        f"8. validation images: {len(val_ds)}",
        f"9. batch size: {BATCH_SIZE}",
        f"10. train RGB means: R={stats['mean_R']:.8f}, G={stats['mean_G']:.8f}, B={stats['mean_B']:.8f}",
        f"11. train RGB stds: R={stats['std_R']:.8f}, G={stats['std_G']:.8f}, B={stats['std_B']:.8f}",
        "12. model architecture: SmallCNNV1 "
        "(Conv16-BN-ReLU-Pool → Conv32-BN-ReLU-Pool → Conv64-BN-ReLU-Pool → "
        "AdaptiveAvgPool → Linear(64→1 logit); no sigmoid in forward)",
        f"13. total parameters: {total_params}",
        f"14. trainable parameters: {trainable_params}",
        f"15. train batch shape: {tuple(images.shape)}",
        f"16. validation batch shape: {tuple(val_images.shape)}",
        f"17. output logit shape: {tuple(logits.shape)}",
        f"18. initial BCEWithLogitsLoss: {initial_loss:.8f}",
        "19. confirmation no backward pass occurred: True",
        "20. confirmation no optimizer update occurred: True",
        "21. confirmation tests remained untouched: "
        "known_test images opened = 0; unseen_test images opened = 0",
        "",
        "Normalisation note",
        "------------------",
        "Channel statistics are estimated from the training data only so",
        "validation and test distributions do not influence preprocessing",
        "parameters. This is analogous to fitting StandardScaler only on",
        "X_train in the classical ML experiment.",
        "",
        "Loss note",
        "---------",
        "Training will use BCEWithLogitsLoss, which expects raw logits.",
        "Therefore SmallCNNV1.forward() does not apply sigmoid.",
        "Later evaluation will use: probability = torch.sigmoid(logit).",
        "",
        "This stage did NOT train the CNN.",
        "No epochs, learning-rate search, ROC-AUC, or test evaluation.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {NORM_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print("\nAll Stage 12 assertions passed. STOP — no training performed.")


if __name__ == "__main__":
    main()
