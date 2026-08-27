"""Shared constants and helpers for Stage 27A V2."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external_v2"
META = ROOT / "metadata"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER_TABLES = ROOT / "paper" / "tables"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SOURCE_SIZE = 224
CANVAS_SIZE = 512
CANVAS_RGB = (32, 32, 32)
JPEG_SUB = 0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
TRANSFORM_NAMES = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
STRONG_ROBUST_CONDITIONS = TRANSFORM_NAMES.copy()

MLLM_CLASS_MAP = {
    "real": {"label": 0, "generator": "Real", "generator_key": "real"},
    "GPT-Image2-fake": {"label": 1, "generator": "GPT Image 2", "generator_key": "gpt_image_2"},
    "Nano-Banana2-fake": {"label": 1, "generator": "Nano Banana 2", "generator_key": "nano_banana_2"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


TRANSFORM_FNS = {
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
        return im.crop((left, top, left + 224, top + 224)).convert("RGB")


def selective_label(p: float, lower: float, upper: float) -> str:
    if p <= lower:
        return "REAL"
    if p >= upper:
        return "AI-GENERATED"
    return "UNCERTAIN"


def ece_15(probs: np.ndarray, labels: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    n = len(probs)
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & ((probs <= hi) if i == 14 else (probs < hi))
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
    return float(ece)


def nll_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def load_frozen_config() -> dict:
    pointer = json.loads((ROOT / "models" / "FINAL_MODEL_V1.json").read_text())
    temp = json.loads((ROOT / pointer["temperature_config"]).read_text())
    policy = json.loads((ROOT / pointer["selective_policy_config"]).read_text())
    frozen = json.loads((ROOT / pointer["frozen_config"]).read_text())
    return {
        "pointer": pointer,
        "temperature": float(temp["temperature"]),
        "lower80": float(policy["lower80"]),
        "upper80": float(policy["upper80"]),
        "hist_thr": float(frozen["threshold"]),
    }
