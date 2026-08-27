#!/usr/bin/env python3
"""CLI for FINAL_RESEARCH_MODEL_V1 local image prediction (Stage 28A).

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python src/predict_image_v1.py "/path/image.jpg"
    PYTHONPATH=src python src/predict_image_v1.py "/path/image.jpg" --device cpu --json
    PYTHONPATH=src python src/predict_image_v1.py "/path/image.jpg" --output /tmp/out.json

By default predictions are printed only (no automatic history / dataset writes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from final_inference_v1 import FinalImageDetectorV1


def format_human(result) -> str:
    name = Path(result.image_path).name
    return "\n".join(
        [
            result.model_id,
            "",
            "Image:",
            name,
            "",
            "Device:",
            result.device,
            "",
            "Raw model P(AI):",
            f"{result.raw_probability:.4f}",
            "",
            "Calibrated model P(AI):",
            f"{result.calibrated_probability:.4f}",
            "",
            "Final decision:",
            result.selective_decision,
            "",
            "Historical binary diagnostic:",
            result.historical_binary_diagnostic,
            "",
            "Warning:",
            result.warning,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FINAL_RESEARCH_MODEL_V1 local image prediction")
    parser.add_argument("image", type=str, help="Path to a local image file")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device (default: auto = CUDA → MPS → CPU)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the prediction JSON (only if explicitly requested)",
    )
    parser.add_argument(
        "--research-controlled-v1",
        action="store_true",
        help="Load an already-prepared controlled_v1 224×224 image (research reproduction)",
    )
    args = parser.parse_args(argv)

    try:
        detector = FinalImageDetectorV1(device=args.device)
        result = detector.predict(args.image, research_controlled_v1=args.research_controlled_v1)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = result.to_dict()
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_human(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
