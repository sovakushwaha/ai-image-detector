"""Generate digital screenshot-style composite transforms (Stage 22C).

Why this file exists
--------------------
Creates reproducible software screenshot-style approximations from locked
controlled_v1 test sources. These are NOT physical LCD/camera recaptures.
No model training or inference is performed.

How to run
----------
    source .venv/bin/activate
    python src/generate_screenshot_v1.py

What to expect
--------------
    data/processed/screenshot_v1/
    metadata/screenshot_v1_manifest.csv
    results/screenshot_v1_generation_report.txt
    figures/screenshot_v1_examples.png
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "screenshot_v1"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "screenshot_v1_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "screenshot_v1_generation_report.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "screenshot_v1_examples.png"

SOURCE_SIZE = 224
CANVAS_SIZE = 512
CANVAS_RGB = (32, 32, 32)
JPEG_SUBSAMPLING = 0
TEST_SPLITS = {"known_test", "unseen_test"}
EXPECTED_SOURCE_COUNTS = {"known_test": 456, "unseen_test": 1712}
EXPECTED_TOTAL_SOURCES = 2168
EXPECTED_CONDITIONS = 2
EXPECTED_TOTAL_TRANSFORMED = 4336
CONTACT_SHEET_IMAGE_IDS = ["pilot_000006", "pilot_000007"]


@dataclass(frozen=True)
class ScreenshotSpec:
    condition: str
    severity: str
    jpeg_quality: int
    display_image_size: int


SPECS: tuple[ScreenshotSpec, ...] = (
    ScreenshotSpec("screenshot_mild", "mild", jpeg_quality=85, display_image_size=448),
    ScreenshotSpec("screenshot_strong", "strong", jpeg_quality=65, display_image_size=384),
)


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_source_rows() -> pd.DataFrame:
    meta = pd.read_csv(SPLIT_META_PATH)
    rows = meta[meta["split"].isin(TEST_SPLITS)].copy()
    rows = rows.sort_values(["split", "image_id"]).reset_index(drop=True)
    stop_if(len(rows) != EXPECTED_TOTAL_SOURCES, f"source count {len(rows)} != {EXPECTED_TOTAL_SOURCES}")
    for split, expected in EXPECTED_SOURCE_COUNTS.items():
        n = int((rows["split"] == split).sum())
        stop_if(n != expected, f"{split} source count {n} != {expected}")
    return rows


def load_source_image(processed_path: str) -> Image.Image:
    path = PROJECT_ROOT / processed_path
    stop_if(not path.exists(), f"missing source image: {path}")
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    stop_if(rgb.size != (SOURCE_SIZE, SOURCE_SIZE), f"{path} size {rgb.size} != (224, 224)")
    stop_if(rgb.mode != "RGB", f"{path} mode {rgb.mode} != RGB")
    return rgb


def jpeg_reencode(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=JPEG_SUBSAMPLING)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        reloaded.load()
        return reloaded.convert("RGB")


def render_virtual_display(image: Image.Image, display_size: int) -> Image.Image:
    stop_if(display_size > CANVAS_SIZE, f"display_size {display_size} > canvas {CANVAS_SIZE}")
    displayed = image.resize((display_size, display_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_RGB)
    offset = (CANVAS_SIZE - display_size) // 2
    canvas.paste(displayed, (offset, offset))
    return canvas


def apply_screenshot(image: Image.Image, spec: ScreenshotSpec) -> Image.Image:
    # Step 1: display/repost JPEG
    decoded = jpeg_reencode(image, spec.jpeg_quality)
    # Step 2–3: virtual display render treated as lossless software screenshot
    screenshot = render_virtual_display(decoded, spec.display_image_size)
    # Step 4: detector input standardisation
    final = screenshot.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS)
    return final.convert("RGB")


def output_path(split: str, condition: str, image_id: str) -> Path:
    return OUTPUT_ROOT / split / condition / f"{image_id}.png"


def generate_all(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    manifest_rows = []
    failures: list[str] = []

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="screenshot sources"):
        try:
            source = load_source_image(row.processed_path)
        except SystemExit as exc:
            failures.append(f"{row.image_id}: {exc}")
            continue

        for spec in SPECS:
            try:
                transformed = apply_screenshot(source, spec)
                stop_if(transformed.mode != "RGB", f"{row.image_id} {spec.condition} mode != RGB")
                stop_if(
                    transformed.size != (SOURCE_SIZE, SOURCE_SIZE),
                    f"{row.image_id} {spec.condition} size != 224x224",
                )
                out_path = output_path(row.split, spec.condition, row.image_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                transformed.save(out_path, format="PNG")
                manifest_rows.append(
                    {
                        "source_image_id": row.image_id,
                        "source_path": row.processed_path,
                        "split": row.split,
                        "label": int(row.label),
                        "true_label": int(row.label),
                        "generator": row.generator,
                        "condition": spec.condition,
                        "severity": spec.severity,
                        "output_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "output_format": "PNG",
                        "width": SOURCE_SIZE,
                        "height": SOURCE_SIZE,
                        "display_canvas_size": CANVAS_SIZE,
                        "display_image_size": spec.display_image_size,
                        "jpeg_quality": spec.jpeg_quality,
                        "jpeg_subsampling": JPEG_SUBSAMPLING,
                        "canvas_rgb": json.dumps(list(CANVAS_RGB)),
                        "final_resize_method": "LANCZOS",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{row.image_id}/{spec.condition}: {exc}")

    return pd.DataFrame(manifest_rows), failures


def verify_manifest(manifest: pd.DataFrame, source_rows: pd.DataFrame) -> dict:
    stop_if(len(manifest) != EXPECTED_TOTAL_TRANSFORMED, f"transformed count {len(manifest)} != {EXPECTED_TOTAL_TRANSFORMED}")

    unreadable = []
    geometry_errors = []
    for record in manifest.itertuples(index=False):
        path = PROJECT_ROOT / record.output_path
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB":
                    geometry_errors.append(f"{record.output_path}: mode {image.mode}")
                if image.size != (SOURCE_SIZE, SOURCE_SIZE):
                    geometry_errors.append(f"{record.output_path}: size {image.size}")
        except Exception as exc:  # noqa: BLE001
            unreadable.append(f"{record.output_path}: {exc}")

    stop_if(unreadable, f"unreadable outputs: {len(unreadable)}")
    stop_if(geometry_errors, f"geometry/mode errors: {len(geometry_errors)}")

    per_condition = manifest.groupby("condition")["source_image_id"].count().to_dict()
    for spec in SPECS:
        stop_if(per_condition.get(spec.condition, 0) != EXPECTED_TOTAL_SOURCES, f"{spec.condition} count wrong")

    known_n = int((manifest["split"] == "known_test").sum())
    unseen_n = int((manifest["split"] == "unseen_test").sum())
    stop_if(known_n != 456 * EXPECTED_CONDITIONS, f"known transformed count {known_n} != 912")
    stop_if(unseen_n != 1712 * EXPECTED_CONDITIONS, f"unseen transformed count {unseen_n} != 3424")

    variants = manifest.groupby("source_image_id")["condition"].nunique()
    stop_if((variants != EXPECTED_CONDITIONS).any(), "some sources do not have exactly 2 variants")

    merged = manifest.merge(
        source_rows[["image_id", "label", "generator", "split"]],
        left_on="source_image_id",
        right_on="image_id",
        how="left",
        validate="many_to_one",
    )
    stop_if(merged["label_y"].isna().any(), "manifest source linkage failed")
    stop_if((merged["true_label"] != merged["label_y"]).any(), "label mismatch in manifest")
    stop_if((merged["split_x"] != merged["split_y"]).any(), "split mismatch in manifest")
    stop_if((merged["generator_x"] != merged["generator_y"]).any(), "generator mismatch in manifest")

    return {
        "unreadable_outputs": unreadable,
        "geometry_errors": geometry_errors,
        "per_condition_counts": per_condition,
        "known_transformed": known_n,
        "unseen_transformed": unseen_n,
        "total_transformed": len(manifest),
    }


def save_contact_sheet(source_rows: pd.DataFrame, manifest: pd.DataFrame) -> None:
    panels = ["Original", "screenshot_mild", "screenshot_strong"]
    fig, axes = plt.subplots(len(CONTACT_SHEET_IMAGE_IDS), len(panels), figsize=(7.5, 5.0))
    if len(CONTACT_SHEET_IMAGE_IDS) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, image_id in enumerate(CONTACT_SHEET_IMAGE_IDS):
        source = source_rows[source_rows["image_id"] == image_id].iloc[0]
        for col_idx, label in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if label == "Original":
                path = PROJECT_ROOT / source["processed_path"]
                title = "Original"
            else:
                out = manifest[
                    (manifest["source_image_id"] == image_id) & (manifest["condition"] == label)
                ].iloc[0]["output_path"]
                path = PROJECT_ROOT / out
                title = "Screenshot Mild" if label.endswith("mild") else "Screenshot Strong"
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(image_id, fontsize=9)

    fig.suptitle("Controlled digital screenshot-style approximation")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(
    source_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    integrity: dict,
    failures: list[str],
) -> str:
    lines = [
        "Screenshot_v1 Generation Report — Stage 22C",
        "===========================================",
        "",
        "DEFINITION",
        "- software screenshot-style composite approximation",
        "- NOT physical LCD/smartphone camera recapture",
        "- no camera noise, perspective, moiré, sensor blur, or glare",
        "",
        "DATA SOURCE",
        "- representation: controlled_v1 (original locked test images)",
        "- NOT derived from robustness_v1 JPEG/resize/blur outputs",
        f"- known source count: {int((source_rows['split'] == 'known_test').sum())}",
        f"- unseen source count: {int((source_rows['split'] == 'unseen_test').sum())}",
        f"- total source count: {len(source_rows)}",
        "",
        "GENERATION PROTOCOL",
        f"- canvas: {CANVAS_SIZE}x{CANVAS_SIZE} RGB{CANVAS_RGB}",
        f"- JPEG subsampling: {JPEG_SUBSAMPLING}",
        f"- final resize: LANCZOS to {SOURCE_SIZE}x{SOURCE_SIZE}",
        f"- output format: PNG",
        "",
    ]
    for spec in SPECS:
        border = (CANVAS_SIZE - spec.display_image_size) // 2
        lines.append(
            f"- {spec.condition} ({spec.severity}): JPEG q={spec.jpeg_quality}, "
            f"display={spec.display_image_size}x{spec.display_image_size}, "
            f"border={border}px each side"
        )

    lines.extend(
        [
            "",
            "COUNTS",
            f"- expected transformed images: {EXPECTED_TOTAL_TRANSFORMED}",
            f"- actual transformed images: {integrity['total_transformed']}",
            f"- known transformed: {integrity['known_transformed']} (expected 912)",
            f"- unseen transformed: {integrity['unseen_transformed']} (expected 3424)",
            "",
            "PER-CONDITION COUNTS",
        ]
    )
    for condition, count in sorted(integrity["per_condition_counts"].items()):
        lines.append(f"- {condition}: {count} (expected 2168)")

    lines.extend(
        [
            "",
            "INTEGRITY",
            f"- unreadable outputs: {len(integrity['unreadable_outputs'])}",
            f"- geometry/mode errors: {len(integrity['geometry_errors'])}",
            f"- failures during generation: {len(failures)}",
            "- labels preserved: YES",
            "- generator preserved: YES",
            "- split preserved: YES",
            "- variants per source: 2",
            "",
            "LIMITATIONS",
            "- one fixed neutral canvas colour",
            "- one fixed rendering geometry per severity",
            "- no real app UI chrome",
            "- no platform-specific pipeline",
            "- no camera/display acquisition effects",
            "- not representative of every screenshot scenario",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training performed: NO",
            "- Physical screen recapture claimed: NO",
        ]
    )
    text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    print("STAGE 22C — SCREENSHOT-STYLE GENERATION")
    source_rows = load_source_rows()
    print(f"Sources: {len(source_rows)}")

    manifest, failures = generate_all(source_rows)
    stop_if(
        bool(failures),
        f"generation failures: {len(failures)}; first={failures[0] if failures else 'n/a'}",
    )

    integrity = verify_manifest(manifest, source_rows)
    stop_if(integrity["unreadable_outputs"] or integrity["geometry_errors"], "critical integrity failure")

    manifest = manifest.sort_values(["split", "condition", "source_image_id"]).reset_index(drop=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    save_contact_sheet(source_rows, manifest)
    report = write_report(source_rows, manifest, integrity, failures)

    print(report)
    print("GENERATION INTEGRITY: PASSED")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
