"""Generate RQ3 validation robustness suite (Stage 23A).

Why this file exists
--------------------
Creates deterministic transformed copies of the 456 VALIDATION images only.
No training. No test-image access.

How to run
----------
    source .venv/bin/activate
    python src/generate_rq3_validation_v1.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "rq3_validation_v1"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "rq3_validation_generation_report_v1.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "rq3_validation_examples_v1.png"

SOURCE_SIZE = 224
CANVAS_SIZE = 512
CANVAS_RGB = (32, 32, 32)
JPEG_SUBSAMPLING = 0
EXPECTED_SOURCES = 456
EXPECTED_TRANSFORMED = 1824
CONDITIONS = ("jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong")
EXAMPLE_IDS = ["pilot_000015", "pilot_000017"]


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_validation_rows() -> pd.DataFrame:
    meta = pd.read_csv(SPLIT_META_PATH)
    rows = meta[meta["split"] == "validation"].copy()
    rows = rows.sort_values("image_id").reset_index(drop=True)
    stop_if(len(rows) != EXPECTED_SOURCES, f"validation count {len(rows)} != {EXPECTED_SOURCES}")
    return rows


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    stop_if(rgb.size != (SOURCE_SIZE, SOURCE_SIZE), f"{path} size {rgb.size}")
    return rgb


def jpeg_reencode(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=JPEG_SUBSAMPLING)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        reloaded.load()
        return reloaded.convert("RGB")


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
    final = canvas.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS)
    return final.convert("RGB")


TRANSFORMS = {
    "jpeg_q50": apply_jpeg_q50,
    "resize_112": apply_resize_112,
    "blur_sigma2": apply_blur_sigma2,
    "screenshot_strong": apply_screenshot_strong,
}


def generate_all(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    manifest_rows = []
    failures: list[str] = []
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="validation sources"):
        try:
            source = load_rgb(PROJECT_ROOT / row.processed_path)
        except SystemExit as exc:
            failures.append(f"{row.image_id}: {exc}")
            continue
        for condition, fn in TRANSFORMS.items():
            try:
                out = fn(source)
                stop_if(out.mode != "RGB" or out.size != (SOURCE_SIZE, SOURCE_SIZE), f"{row.image_id}/{condition} geometry")
                out_path = OUTPUT_ROOT / condition / f"{row.image_id}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out.save(out_path, format="PNG")
                manifest_rows.append(
                    {
                        "source_image_id": row.image_id,
                        "source_path": row.processed_path,
                        "split": "validation",
                        "label": int(row.label),
                        "generator": row.generator,
                        "condition": condition,
                        "output_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "output_format": "PNG",
                        "width": SOURCE_SIZE,
                        "height": SOURCE_SIZE,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{row.image_id}/{condition}: {exc}")
    return pd.DataFrame(manifest_rows), failures


def verify(manifest: pd.DataFrame, source_rows: pd.DataFrame) -> None:
    stop_if(len(manifest) != EXPECTED_TRANSFORMED, f"transformed {len(manifest)} != {EXPECTED_TRANSFORMED}")
    for cond in CONDITIONS:
        stop_if(int((manifest["condition"] == cond).sum()) != EXPECTED_SOURCES, f"{cond} count")
    variants = manifest.groupby("source_image_id")["condition"].nunique()
    stop_if((variants != 4).any(), "not exactly 4 variants per source")
    unreadable = 0
    for record in manifest.itertuples(index=False):
        path = PROJECT_ROOT / record.output_path
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (SOURCE_SIZE, SOURCE_SIZE):
                unreadable += 1
    stop_if(unreadable != 0, f"geometry/unreadable errors: {unreadable}")
    merged = manifest.merge(
        source_rows[["image_id", "label", "generator"]],
        left_on="source_image_id",
        right_on="image_id",
        how="left",
        validate="many_to_one",
    )
    stop_if((merged["label_x"] != merged["label_y"]).any(), "label mismatch")
    stop_if((merged["generator_x"] != merged["generator_y"]).any(), "generator mismatch")


def save_examples(source_rows: pd.DataFrame, manifest: pd.DataFrame) -> None:
    panels = ["original"] + list(CONDITIONS)
    fig, axes = plt.subplots(len(EXAMPLE_IDS), len(panels), figsize=(12, 5))
    for r, image_id in enumerate(EXAMPLE_IDS):
        src = source_rows[source_rows["image_id"] == image_id].iloc[0]
        for c, cond in enumerate(panels):
            ax = axes[r, c]
            if cond == "original":
                path = PROJECT_ROOT / src["processed_path"]
                title = "Original"
            else:
                path = PROJECT_ROOT / manifest[
                    (manifest["source_image_id"] == image_id) & (manifest["condition"] == cond)
                ].iloc[0]["output_path"]
                title = cond
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(image_id, fontsize=8)
    fig.suptitle("RQ3 validation robustness suite (validation images only)")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(failures: int) -> None:
    text = "\n".join(
        [
            "RQ3 Validation Robustness Suite — Stage 23A",
            "===========================================",
            "",
            "SOURCE: controlled_v1 validation only (456 images)",
            "TEST IMAGES ACCESSED: NO",
            "",
            "CONDITIONS:",
            "- jpeg_q50: JPEG q=50, subsampling=0",
            "- resize_112: 224→112→224 LANCZOS, PNG",
            "- blur_sigma2: Gaussian σ=2.0, PNG",
            "- screenshot_strong: Stage 22C strong digital screenshot-style approximation, PNG",
            "",
            f"Transformed images: {EXPECTED_TRANSFORMED}",
            f"Failures: {failures}",
            "Labels/generators/source IDs preserved: YES",
            "",
            "Screenshot is development evaluation only; not used for training augmentation.",
        ]
    ) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    print("STAGE 23A — RQ3 VALIDATION SUITE GENERATION")
    rows = load_validation_rows()
    print(f"Validation sources: {len(rows)}")
    manifest, failures = generate_all(rows)
    stop_if(bool(failures), f"generation failures: {len(failures)}; first={failures[0] if failures else 'n/a'}")
    verify(manifest, rows)
    manifest = manifest.sort_values(["condition", "source_image_id"]).reset_index(drop=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    save_examples(rows, manifest)
    write_report(0)
    print(f"Generated: {len(manifest)}")
    print("INTEGRITY: PASSED")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
