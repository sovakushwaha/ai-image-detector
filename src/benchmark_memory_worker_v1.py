"""Isolated subprocess memory measurement for Stage 26A.

Usage (from benchmark_resources_v1.py):
    PYTHONPATH=src python src/benchmark_memory_worker_v1.py --model C0 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, load_split_metadata, stop_if
from mobilenet_v3_small_binary_v1 import MobileNetV3SmallBinaryV1
from rq3_augmentations_v1 import IMAGENET_MEAN, IMAGENET_STD
from rq4_frequency_transform_v1 import FrequencyTransformV1
from rq4_rgb_frequency_fusion_v1 import RGBFrequencyFusionV1

C0_CKPT = PROJECT_ROOT / "models/mobilenet_resize_jpeg_aug_selected_v1.pt"
C1_CKPT = PROJECT_ROOT / "models/rq4_F2_rgb_frequency_fusion_selected_v1.pt"
NORM_PATH = PROJECT_ROOT / "results/rq4_frequency_normalization_v1.json"


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def mps_mem_mib() -> tuple[float | None, float | None]:
    if not torch.backends.mps.is_available():
        return None, None
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    cur = float(current()) / (1024 * 1024) if current else None
    drv = float(driver()) / (1024 * 1024) if driver else None
    return cur, drv


def load_sample_path() -> Path:
    val = load_split_metadata("validation")
    row = val.sort_values("image_id").iloc[0]
    return PROJECT_ROOT / row["processed_path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["C0", "C1"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print(json.dumps({"error": "MPS unavailable"}))
        sys.exit(1)

    device = torch.device(args.device)
    baseline_rss = rss_mib()
    mps_before = mps_mem_mib() if args.device == "mps" else (None, None)

    rgb_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    freq_transform = FrequencyTransformV1.from_json(NORM_PATH)
    sample_path = load_sample_path()

    if args.model == "C0":
        model = MobileNetV3SmallBinaryV1().to(device)
        ckpt = torch.load(C0_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = RGBFrequencyFusionV1().to(device)
        ckpt = torch.load(C1_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    loaded_rss = rss_mib()
    if args.device == "mps":
        torch.mps.empty_cache()
    mps_loaded = mps_mem_mib() if args.device == "mps" else (None, None)

    peak_rss = loaded_rss
    peak_mps_cur = mps_loaded[0]
    peak_mps_drv = mps_loaded[1]

    with torch.inference_mode():
        for _ in range(60):
            with Image.open(sample_path) as image:
                image.load()
                rgb = image.convert("RGB")
            stop_if(rgb.size != EXPECTED_SIZE, "bad size")
            if args.model == "C0":
                x = rgb_transform(rgb).unsqueeze(0).to(device)
                if device.type == "mps":
                    torch.mps.synchronize()
                _ = torch.sigmoid(model(x)).cpu().item()
                if device.type == "mps":
                    torch.mps.synchronize()
            else:
                x_rgb = rgb_transform(rgb).unsqueeze(0).to(device)
                x_freq = freq_transform(rgb).unsqueeze(0).to(device)
                if device.type == "mps":
                    torch.mps.synchronize()
                _ = torch.sigmoid(model(x_rgb, x_freq)).cpu().item()
                if device.type == "mps":
                    torch.mps.synchronize()
            peak_rss = max(peak_rss, rss_mib())
            if args.device == "mps":
                cur, drv = mps_mem_mib()
                if cur is not None:
                    peak_mps_cur = max(peak_mps_cur or 0.0, cur)
                if drv is not None:
                    peak_mps_drv = max(peak_mps_drv or 0.0, drv)

    out = {
        "model": args.model,
        "device": args.device,
        "baseline_memory_mib": baseline_rss,
        "loaded_memory_mib": loaded_rss,
        "max_observed_memory_mib": peak_rss,
        "increment_mib": peak_rss - baseline_rss,
        "measurement_method": "psutil.Process().memory_info().rss isolated subprocess",
        "mps_current_allocated_mib_loaded": mps_loaded[0],
        "mps_driver_allocated_mib_loaded": mps_loaded[1],
        "mps_current_allocated_mib_max_observed": peak_mps_cur,
        "mps_driver_allocated_mib_max_observed": peak_mps_drv,
        "limitations": "RSS is approximate; MPS unified memory accounting is sampled not guaranteed peak",
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
