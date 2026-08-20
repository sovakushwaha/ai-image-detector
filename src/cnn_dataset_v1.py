"""PyTorch Dataset for controlled_v1 images (CNN Baseline V1).

Why this file exists
--------------------
Stage 12 builds the neural-network data pipeline. Images are already
RGB JPEG 224×224 under controlled_v1. This Dataset loads train or
validation rows only, converts them to tensors, and applies
train-derived channel normalisation.

Generator identity is metadata only. It is never returned as a model
input.

How to use
----------
    from cnn_dataset_v1 import ControlledV1Dataset, build_transforms, select_device

Do not open known_test or unseen_test from this module during Stage 12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
NORM_PATH = PROJECT_ROOT / "results" / "cnn_train_normalization_v1.json"

ALLOWED_SPLITS = {"train", "validation"}
EXPECTED_SIZE = (224, 224)
REPRESENTATION = "controlled_v1"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_metadata(split: str, meta_path: Path = SPLIT_META_PATH) -> pd.DataFrame:
    """Load metadata for one allowed split. Refuse test splits."""
    stop_if(split not in ALLOWED_SPLITS, f"split '{split}' is not allowed in Stage 12")
    table = pd.read_csv(meta_path)
    subset = table[table["split"] == split].copy()
    stop_if(subset.empty, f"no rows found for split={split}")
    stop_if(
        subset["split"].isin(["known_test", "unseen_test"]).any(),
        "test rows leaked into requested split",
    )
    return subset.reset_index(drop=True)


def compute_train_rgb_stats(meta_path: Path = SPLIT_META_PATH) -> dict:
    """Compute RGB mean/std over ALL train pixels only.

    Validation and test images must not influence these statistics.
    This is analogous to fitting StandardScaler only on X_train.
    """
    train = load_split_metadata("train", meta_path)
    stop_if(len(train) != 1376, f"expected 1376 train images, found {len(train)}")

    # Accumulate per-channel sum and sum of squares over all pixels.
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for _, row in tqdm(train.iterrows(), total=len(train), desc="Train RGB stats"):
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            stop_if(image.format != "JPEG", f"{path} is {image.format}, expected JPEG")
            stop_if(image.mode != "RGB", f"{path} is mode {image.mode}, expected RGB")
            stop_if(image.size != EXPECTED_SIZE, f"{path} is {image.size}, expected {EXPECTED_SIZE}")
            rgb = np.asarray(image, dtype=np.float64) / 255.0

        channel_sum += rgb.sum(axis=(0, 1))
        channel_sq_sum += (rgb**2).sum(axis=(0, 1))
        pixel_count += rgb.shape[0] * rgb.shape[1]

    mean = channel_sum / pixel_count
    var = channel_sq_sum / pixel_count - mean**2
    # Numerical safety for tiny floating-point negatives.
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    stats = {
        "representation": REPRESENTATION,
        "source_split": "train",
        "sample_count": int(len(train)),
        "pixel_count": int(pixel_count),
        "mean_R": float(mean[0]),
        "mean_G": float(mean[1]),
        "mean_B": float(mean[2]),
        "std_R": float(std[0]),
        "std_G": float(std[1]),
        "std_B": float(std[2]),
        "note": (
            "Channel statistics are estimated from the training data only so "
            "validation and test distributions do not influence preprocessing "
            "parameters. This is analogous to fitting StandardScaler only on "
            "X_train in the classical ML experiment."
        ),
    }
    return stats


def save_train_rgb_stats(stats: dict, path: Path = NORM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_train_rgb_stats(path: Path = NORM_PATH) -> dict:
    stop_if(not path.exists(), f"missing normalisation file: {path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    stop_if(stats.get("source_split") != "train", "normalisation source_split must be train")
    stop_if(int(stats.get("sample_count", 0)) != 1376, "normalisation sample_count must be 1376")
    return stats


def build_transforms(stats: dict) -> transforms.Compose:
    """Deterministic convert → [0,1] → train-only Normalize. No augmentation."""
    mean = [stats["mean_R"], stats["mean_G"], stats["mean_B"]]
    std = [stats["std_R"], stats["std_G"], stats["std_B"]]
    return transforms.Compose(
        [
            transforms.ToTensor(),  # HWC uint8 → CHW float in [0, 1]
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class ControlledV1Dataset(Dataset):
    """Train/validation images for SmallCNNV1.

    Returns:
        image_tensor: float tensor [3, 224, 224]
        label: float tensor scalar (0.0 = Real, 1.0 = AI) for BCEWithLogitsLoss
        image_id: string identifier (not a model input)
    """

    def __init__(self, split: str, transform: transforms.Compose, meta_path: Path = SPLIT_META_PATH):
        stop_if(split not in ALLOWED_SPLITS, f"refusing split '{split}'")
        self.split = split
        self.transform = transform
        self.rows = load_split_metadata(split, meta_path)
        stop_if(
            self.rows["split"].isin(["known_test", "unseen_test"]).any(),
            "Dataset refused test rows",
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            stop_if(image.format != "JPEG", f"{path} is {image.format}, expected JPEG")
            stop_if(image.mode != "RGB", f"{path} is mode {image.mode}, expected RGB")
            stop_if(image.size != EXPECTED_SIZE, f"{path} is {image.size}, expected {EXPECTED_SIZE}")
            rgb = image.convert("RGB")

        image_tensor = self.transform(rgb)
        # Float label for BCEWithLogitsLoss; generator is intentionally omitted.
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        image_id = str(row["image_id"])
        return image_tensor, label, image_id
