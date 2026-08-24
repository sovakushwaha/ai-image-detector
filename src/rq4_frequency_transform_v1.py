"""RQ4 Stage 24A: locked FFT log-magnitude frequency representation.

Why this file exists
--------------------
Defines the single predefined FrequencyTransformV1 used by F1 and F2.
Computes train-only global frequency normalisation statistics and QC figures.
No model training or test access.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/rq4_frequency_transform_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, load_split_metadata, stop_if

IMAGE_SIZE = 224
GRAYSCALE_WEIGHTS = (0.299, 0.587, 0.114)
EXPECTED_TRAIN = 1376
SEED = 42

NORM_PATH = PROJECT_ROOT / "results" / "rq4_frequency_normalization_v1.json"
FIG_EXAMPLES = PROJECT_ROOT / "figures" / "rq4_frequency_examples_v1.png"
FIG_DIFF = PROJECT_ROOT / "figures" / "rq4_train_mean_frequency_difference_v1.png"


def _build_hann_window(size: int = IMAGE_SIZE) -> np.ndarray:
    """Fixed separable 2D Hann window (outer product of 1D hanning)."""
    w1d = np.hanning(size).astype(np.float64)
    return np.outer(w1d, w1d)


HANN_WINDOW = _build_hann_window(IMAGE_SIZE)


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8/float array in [0,255] or [0,1] to Y in [0,1] float32."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    stop_if(arr.ndim != 3 or arr.shape[2] != 3, f"expected HxWx3, got {arr.shape}")
    y = (
        GRAYSCALE_WEIGHTS[0] * arr[:, :, 0]
        + GRAYSCALE_WEIGHTS[1] * arr[:, :, 1]
        + GRAYSCALE_WEIGHTS[2] * arr[:, :, 2]
    )
    return y.astype(np.float32)


def compute_log_magnitude_spectrum(y: np.ndarray, hann: np.ndarray = HANN_WINDOW) -> np.ndarray:
    """Luminance [0,1] → mean-centred → Hann → FFT2 → fftshift → |F| → log1p."""
    stop_if(y.shape != (IMAGE_SIZE, IMAGE_SIZE), f"luminance shape {y.shape}")
    y64 = y.astype(np.float64)
    y_centered = y64 - float(y64.mean())
    y_windowed = y_centered * hann
    f = np.fft.fft2(y_windowed)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    return np.log1p(magnitude).astype(np.float32)


class FrequencyTransformV1:
    """Deterministic RGB→frequency pipeline with train-only global normalisation."""

    def __init__(
        self,
        train_mean: float,
        train_std: float,
        hann: np.ndarray | None = None,
    ) -> None:
        stop_if(train_std <= 0, "frequency_train_std must be > 0")
        self.train_mean = float(train_mean)
        self.train_std = float(train_std)
        self.hann = HANN_WINDOW if hann is None else hann

    @classmethod
    def from_json(cls, path: Path = NORM_PATH) -> "FrequencyTransformV1":
        with open(path) as f:
            payload = json.load(f)
        return cls(
            train_mean=float(payload["frequency_train_mean"]),
            train_std=float(payload["frequency_train_std"]),
        )

    def log_spectrum_from_pil(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"expected {EXPECTED_SIZE}, got {rgb.size}")
        arr = np.asarray(rgb)
        y = rgb_to_luminance(arr)
        return compute_log_magnitude_spectrum(y, self.hann)

    def __call__(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return normalised spectrum tensor shape [1, 224, 224]."""
        if isinstance(image, torch.Tensor):
            # Assume CHW float in [0,1] or ImageNet-denormalised RGB — prefer PIL path.
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.max() <= 1.0 + 1e-3:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0 + 1e-3:
                    image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image, mode="RGB")

        spectrum = self.log_spectrum_from_pil(image)
        normalised = (spectrum - self.train_mean) / self.train_std
        return torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0)

    def unnormalized_spectrum(self, image: Image.Image) -> np.ndarray:
        return self.log_spectrum_from_pil(image)


def compute_train_frequency_stats(meta_path: Path | None = None) -> dict:
    """Global mean/std over all log-spectrum pixels of clean training images only."""
    train = load_split_metadata("train", meta_path) if meta_path else load_split_metadata("train")
    stop_if(len(train) != EXPECTED_TRAIN, f"expected {EXPECTED_TRAIN} train, got {len(train)}")

    # Numerically stable accumulation: sum and sum-of-squares over all pixels
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    count = 0

    for _, row in tqdm(train.iterrows(), total=len(train), desc="Train frequency stats"):
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
            y = rgb_to_luminance(np.asarray(rgb))
            spectrum = compute_log_magnitude_spectrum(y).astype(np.float64)
        pixel_sum += float(spectrum.sum())
        pixel_sq_sum += float((spectrum * spectrum).sum())
        count += int(spectrum.size)

    stop_if(count != EXPECTED_TRAIN * IMAGE_SIZE * IMAGE_SIZE, f"pixel count {count}")
    mean = pixel_sum / count
    variance = pixel_sq_sum / count - mean * mean
    variance = max(float(variance), 0.0)
    std = float(np.sqrt(variance))
    stop_if(std <= 0, "degenerate frequency std")

    payload = {
        "transform_id": "FrequencyTransformV1",
        "representation": "controlled_v1",
        "split_protocol": "generator_protocol_v1",
        "image_size": IMAGE_SIZE,
        "grayscale_coefficients": {
            "R": GRAYSCALE_WEIGHTS[0],
            "G": GRAYSCALE_WEIGHTS[1],
            "B": GRAYSCALE_WEIGHTS[2],
        },
        "steps": [
            "rgb_to_luminance_[0,1]",
            "subtract_per_image_mean",
            "2d_hann_window",
            "fft2",
            "fftshift",
            "magnitude_abs",
            "log1p",
            "global_train_zscore",
        ],
        "hann_window": {
            "type": "separable_2d_outer_product",
            "numpy": "np.outer(np.hanning(224), np.hanning(224))",
            "fixed_for_all_images": True,
        },
        "fft_convention": "numpy.fft.fft2 + numpy.fft.fftshift",
        "magnitude": "abs(F_shifted)",
        "compression": "log1p",
        "phase_used": False,
        "per_image_zscore": False,
        "training_image_count": EXPECTED_TRAIN,
        "training_pixel_count": int(count),
        "frequency_train_mean": float(mean),
        "frequency_train_std": std,
        "output_tensor_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
        "seed_note": "deterministic transform; no randomness",
    }
    return payload


def save_normalization(payload: dict, path: Path = NORM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _pick_examples(train: pd.DataFrame, n_per_class: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    real = train[train["label"] == 0].sample(n=n_per_class, random_state=rng)
    ai = train[train["label"] == 1].sample(n=n_per_class, random_state=rng)
    return pd.concat([real, ai], ignore_index=True)


def create_qc_figures(transform: FrequencyTransformV1) -> None:
    train = load_split_metadata("train")
    examples = _pick_examples(train, n_per_class=3)

    fig, axes = plt.subplots(6, 3, figsize=(9, 14))
    for row_idx, (_, row) in enumerate(examples.iterrows()):
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        arr = np.asarray(rgb)
        y = rgb_to_luminance(arr)
        spectrum = transform.unnormalized_spectrum(rgb)
        label = "Real" if int(row["label"]) == 0 else f"AI ({row['generator']})"

        axes[row_idx, 0].imshow(arr)
        axes[row_idx, 0].set_title(f"{label}\nRGB", fontsize=8)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(y, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Luminance Y", fontsize=8)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(spectrum, cmap="magma")
        axes[row_idx, 2].set_title("log1p |FFT|", fontsize=8)
        axes[row_idx, 2].axis("off")

    fig.suptitle("RQ4 FrequencyTransformV1 QC (train examples only)", fontsize=11)
    fig.tight_layout()
    FIG_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_EXAMPLES, dpi=150)
    plt.close(fig)

    # Class-mean spectrum difference (descriptive only)
    real_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    ai_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    n_real = 0
    n_ai = 0
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Class mean spectra"):
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        spectrum = transform.unnormalized_spectrum(rgb).astype(np.float64)
        if int(row["label"]) == 0:
            real_sum += spectrum
            n_real += 1
        else:
            ai_sum += spectrum
            n_ai += 1
    stop_if(n_real != 688 or n_ai != 688, f"train class counts {n_real}/{n_ai}")
    real_mean = real_sum / n_real
    ai_mean = ai_sum / n_ai
    diff = ai_mean - real_mean

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(real_mean, cmap="magma")
    axes[0].set_title("Mean log|FFT| Real (train)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ai_mean, cmap="magma")
    axes[1].set_title("Mean log|FFT| AI (train)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    vmax = float(np.percentile(np.abs(diff), 99))
    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title("AI mean − Real mean (descriptive)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle("RQ4 train-only class frequency difference (not used for selection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIFF, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== Stage 24A.1–24A.3 FrequencyTransformV1 + train normalisation ===")
    payload = compute_train_frequency_stats()
    save_normalization(payload)
    print(f"Saved {NORM_PATH}")
    print(f"frequency_train_mean = {payload['frequency_train_mean']:.8f}")
    print(f"frequency_train_std  = {payload['frequency_train_std']:.8f}")

    transform = FrequencyTransformV1(
        train_mean=payload["frequency_train_mean"],
        train_std=payload["frequency_train_std"],
    )
    create_qc_figures(transform)
    print(f"Saved {FIG_EXAMPLES}")
    print(f"Saved {FIG_DIFF}")
    print("Stage 24A representation + QC COMPLETE (no training yet).")


if __name__ == "__main__":
    main()
