"""RQ3 online training augmentations (Stage 23A definitions).

Why this file exists
--------------------
Defines locked blur / resize / JPEG augmentation transforms for future RQ3
regimes A1–A3. No model training occurs here.

How to use
----------
Import transform classes into future training scripts. Validation images must
remain deterministic and never receive random augmentation.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
JPEG_SUBSAMPLING = 0
SOURCE_SIZE = 224

# Locked augmentation parameters (Stage 23A)
BLUR_PROB = 0.30
BLUR_SIGMA_MIN = 0.5
BLUR_SIGMA_MAX = 2.0

RESIZE_PROB = 0.30
RESIZE_SIDE_MIN = 112
RESIZE_SIDE_MAX = 192

JPEG_PROB = 0.30
JPEG_QUALITY_MIN = 50
JPEG_QUALITY_MAX = 90

SEED = 42


@dataclass(frozen=True)
class RQ3AugmentationConfig:
    regime: str
    blur_prob: float = 0.0
    blur_sigma_min: float = BLUR_SIGMA_MIN
    blur_sigma_max: float = BLUR_SIGMA_MAX
    resize_prob: float = 0.0
    resize_side_min: int = RESIZE_SIDE_MIN
    resize_side_max: int = RESIZE_SIDE_MAX
    jpeg_prob: float = 0.0
    jpeg_quality_min: int = JPEG_QUALITY_MIN
    jpeg_quality_max: int = JPEG_QUALITY_MAX
    order: tuple[str, ...] = ("blur", "resize", "jpeg")


REGIME_CONFIGS = {
    "A0": RQ3AugmentationConfig(regime="A0"),
    "A1": RQ3AugmentationConfig(regime="A1", blur_prob=BLUR_PROB),
    "A2": RQ3AugmentationConfig(
        regime="A2",
        resize_prob=RESIZE_PROB,
        jpeg_prob=JPEG_PROB,
    ),
    "A3": RQ3AugmentationConfig(
        regime="A3",
        blur_prob=BLUR_PROB,
        resize_prob=RESIZE_PROB,
        jpeg_prob=JPEG_PROB,
    ),
}


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_resize_degradation(image: Image.Image, intermediate_side: int) -> Image.Image:
    side = int(intermediate_side)
    small = image.resize((side, side), Image.Resampling.LANCZOS)
    return small.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS)


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        reloaded.load()
        return reloaded.convert("RGB")


class RobustnessAwarePILTransform:
    """Online PIL augmentations for RQ3 regimes. Call before ToTensor."""

    def __init__(self, config: RQ3AugmentationConfig, rng: random.Random | None = None):
        self.config = config
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, image: Image.Image) -> Image.Image:
        out = image.convert("RGB")
        for step in self.config.order:
            if step == "blur" and self.config.blur_prob > 0 and self.rng.random() < self.config.blur_prob:
                sigma = self.rng.uniform(self.config.blur_sigma_min, self.config.blur_sigma_max)
                out = apply_gaussian_blur(out, sigma)
            elif step == "resize" and self.config.resize_prob > 0 and self.rng.random() < self.config.resize_prob:
                side = self.rng.randint(self.config.resize_side_min, self.config.resize_side_max)
                out = apply_resize_degradation(out, side)
            elif step == "jpeg" and self.config.jpeg_prob > 0 and self.rng.random() < self.config.jpeg_prob:
                quality = self.rng.randint(self.config.jpeg_quality_min, self.config.jpeg_quality_max)
                out = apply_jpeg(out, quality)
        return out


class BlurAwareTransform(RobustnessAwarePILTransform):
    def __init__(self, rng: random.Random | None = None):
        super().__init__(REGIME_CONFIGS["A1"], rng=rng)


class ResizeJPEGAwareTransform(RobustnessAwarePILTransform):
    def __init__(self, rng: random.Random | None = None):
        super().__init__(REGIME_CONFIGS["A2"], rng=rng)


class CombinedRobustTransform(RobustnessAwarePILTransform):
    def __init__(self, rng: random.Random | None = None):
        super().__init__(REGIME_CONFIGS["A3"], rng=rng)


def build_train_transform(regime: str, seed: int | None = SEED) -> transforms.Compose:
    """Compose online augmentation + ImageNet normalization for a locked regime."""
    stop_if = regime not in REGIME_CONFIGS
    if stop_if:
        raise ValueError(f"unknown RQ3 regime: {regime}")
    rng = random.Random(seed) if seed is not None else random.Random()
    pil_aug = RobustnessAwarePILTransform(REGIME_CONFIGS[regime], rng=rng)
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: pil_aug(img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)))),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Deterministic evaluation transform (no random augmentation)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def force_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
