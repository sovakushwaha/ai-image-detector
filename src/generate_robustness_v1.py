"""Generate RQ2 robustness test-suite transformations (Stage 22A).

Why this file exists
--------------------
Locked known_test and unseen_test controlled_v1 images are transformed into
deterministic evaluation conditions for later robustness analysis. No model
training or inference is performed.

How to run
----------
    source .venv/bin/activate
    python src/generate_robustness_v1.py

What to expect
--------------
    data/processed/robustness_v1/
    metadata/robustness_v1_manifest.csv
    results/robustness_v1_generation_report.txt
    figures/robustness_v1_examples.png
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "robustness_v1"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "robustness_v1_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "robustness_v1_generation_report.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "robustness_v1_examples.png"

SOURCE_SIZE = 224
TEST_SPLITS = {"known_test", "unseen_test"}
EXPECTED_SOURCE_COUNTS = {"known_test": 456, "unseen_test": 1712}
EXPECTED_TOTAL_SOURCES = 2168
EXPECTED_CONDITIONS = 8
EXPECTED_TOTAL_TRANSFORMED = 17344
SANITY_SAMPLE_SIZE = 12
CONTACT_SHEET_IMAGE_IDS = ["pilot_000006", "pilot_000007"]

JPEG_SUBSAMPLING = 0


@dataclass(frozen=True)
class TransformSpec:
    condition: str
    severity: str
    transform_type: str
    output_format: str
    parameters: dict


TRANSFORMS: tuple[TransformSpec, ...] = (
    TransformSpec("jpeg_q75", "mild", "jpeg_compression", "JPEG", {"quality": 75}),
    TransformSpec("jpeg_q50", "strong", "jpeg_compression", "JPEG", {"quality": 50}),
    TransformSpec(
        "crop_90",
        "mild",
        "centre_crop_resize",
        "PNG",
        {"retain_fraction": 0.90, "crop_size": 202, "resize_to": 224, "resample": "LANCZOS"},
    ),
    TransformSpec(
        "crop_75",
        "strong",
        "centre_crop_resize",
        "PNG",
        {"retain_fraction": 0.75, "crop_size": 168, "resize_to": 224, "resample": "LANCZOS"},
    ),
    TransformSpec(
        "resize_160",
        "mild",
        "resize_degradation",
        "PNG",
        {"downsample_to": 160, "restore_to": 224, "resample": "LANCZOS"},
    ),
    TransformSpec(
        "resize_112",
        "strong",
        "resize_degradation",
        "PNG",
        {"downsample_to": 112, "restore_to": 224, "resample": "LANCZOS"},
    ),
    TransformSpec("blur_sigma1", "mild", "gaussian_blur", "PNG", {"sigma": 1.0}),
    TransformSpec("blur_sigma2", "strong", "gaussian_blur", "PNG", {"sigma": 2.0}),
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


def centre_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    stop_if(width != height, "centre crop expects square input")
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def apply_transform(image: Image.Image, spec: TransformSpec) -> Image.Image:
    if spec.transform_type == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(spec.parameters["quality"]),
            subsampling=JPEG_SUBSAMPLING,
        )
        buffer.seek(0)
        with Image.open(buffer) as reloaded:
            reloaded.load()
            return reloaded.convert("RGB")

    if spec.transform_type == "centre_crop_resize":
        cropped = centre_crop(image, int(spec.parameters["crop_size"]))
        return cropped.resize(
            (int(spec.parameters["resize_to"]), int(spec.parameters["resize_to"])),
            Image.Resampling.LANCZOS,
        )

    if spec.transform_type == "resize_degradation":
        down = int(spec.parameters["downsample_to"])
        up = int(spec.parameters["restore_to"])
        small = image.resize((down, down), Image.Resampling.LANCZOS)
        return small.resize((up, up), Image.Resampling.LANCZOS)

    if spec.transform_type == "gaussian_blur":
        sigma = float(spec.parameters["sigma"])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    raise ValueError(f"unknown transform type: {spec.transform_type}")


def output_extension(spec: TransformSpec) -> str:
    return "jpg" if spec.output_format == "JPEG" else "png"


def output_path(split: str, spec: TransformSpec, image_id: str) -> Path:
    return OUTPUT_ROOT / split / spec.condition / f"{image_id}.{output_extension(spec)}"


def save_transformed(image: Image.Image, path: Path, spec: TransformSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.output_format == "JPEG":
        image.save(path, format="JPEG", quality=int(spec.parameters["quality"]), subsampling=JPEG_SUBSAMPLING)
    else:
        image.save(path, format="PNG")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_all(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    manifest_rows = []
    failures: list[str] = []

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="sources"):
        try:
            source = load_source_image(row.processed_path)
        except SystemExit as exc:
            failures.append(f"{row.image_id}: {exc}")
            continue

        for spec in TRANSFORMS:
            try:
                transformed = apply_transform(source, spec)
                stop_if(transformed.mode != "RGB", f"{row.image_id} {spec.condition} mode != RGB")
                stop_if(
                    transformed.size != (SOURCE_SIZE, SOURCE_SIZE),
                    f"{row.image_id} {spec.condition} size != 224x224",
                )
                out_path = output_path(row.split, spec, row.image_id)
                save_transformed(transformed, out_path, spec)
                manifest_rows.append(
                    {
                        "source_image_id": row.image_id,
                        "source_path": row.processed_path,
                        "split": row.split,
                        "true_label": int(row.label),
                        "generator": row.generator,
                        "condition": spec.condition,
                        "severity": spec.severity,
                        "output_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "width": transformed.size[0],
                        "height": transformed.size[1],
                        "mode": transformed.mode,
                        "output_format": spec.output_format,
                        "transform_parameters": json.dumps(spec.parameters, sort_keys=True),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - collect per-image failures
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

    per_condition = manifest.groupby("condition")["source_image_id"].count().to_dict()
    for spec in TRANSFORMS:
        stop_if(per_condition.get(spec.condition, 0) != EXPECTED_TOTAL_SOURCES, f"{spec.condition} count wrong")

    known_n = int((manifest["split"] == "known_test").sum())
    unseen_n = int((manifest["split"] == "unseen_test").sum())
    stop_if(known_n != 456 * EXPECTED_CONDITIONS, f"known transformed count {known_n} != 3648")
    stop_if(unseen_n != 1712 * EXPECTED_CONDITIONS, f"unseen transformed count {unseen_n} != 13696")

    variants = manifest.groupby("source_image_id")["condition"].nunique()
    stop_if((variants != EXPECTED_CONDITIONS).any(), "some sources do not have exactly 8 variants")

    merged = manifest.merge(
        source_rows[["image_id", "label", "generator", "split"]],
        left_on="source_image_id",
        right_on="image_id",
        how="left",
        validate="many_to_one",
    )
    stop_if(merged["label"].isna().any(), "manifest source linkage failed")
    stop_if((merged["true_label"] != merged["label"]).any(), "label mismatch in manifest")
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


def pixel_sanity_check(manifest: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sample_ids = source_rows["image_id"].sample(
        n=min(SANITY_SAMPLE_SIZE, len(source_rows)), random_state=42
    ).tolist()
    rows = []
    source_lookup = source_rows.set_index("image_id")

    for image_id in sample_ids:
        src_path = PROJECT_ROOT / source_lookup.loc[image_id, "processed_path"]
        with Image.open(src_path) as image:
            source_arr = np.asarray(image.convert("RGB"), dtype=np.int16)

        subset = manifest[manifest["source_image_id"] == image_id]
        for record in subset.itertuples(index=False):
            with Image.open(PROJECT_ROOT / record.output_path) as image:
                transformed_arr = np.asarray(image.convert("RGB"), dtype=np.int16)
            mean_abs_diff = float(np.mean(np.abs(transformed_arr - source_arr)))
            max_abs_diff = int(np.max(np.abs(transformed_arr - source_arr)))
            rows.append(
                {
                    "source_image_id": image_id,
                    "condition": record.condition,
                    "mean_abs_pixel_diff": mean_abs_diff,
                    "max_abs_pixel_diff": max_abs_diff,
                    "nonzero_change": bool(max_abs_diff > 0),
                }
            )

    summary = pd.DataFrame(rows)
    for condition in [spec.condition for spec in TRANSFORMS]:
        group = summary[summary["condition"] == condition]
        stop_if(group.empty, f"no sanity-check samples for {condition}")
        stop_if(not group["nonzero_change"].all(), f"{condition} showed zero pixel change in sanity sample")
    return summary


def save_contact_sheet(source_rows: pd.DataFrame, manifest: pd.DataFrame) -> None:
    panels = [("Original", None)]
    panels.extend((spec.condition, spec) for spec in TRANSFORMS)

    fig, axes = plt.subplots(len(CONTACT_SHEET_IMAGE_IDS), len(panels), figsize=(2.2 * len(panels), 4.5 * len(CONTACT_SHEET_IMAGE_IDS)))
    if len(CONTACT_SHEET_IMAGE_IDS) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, image_id in enumerate(CONTACT_SHEET_IMAGE_IDS):
        source = source_rows[source_rows["image_id"] == image_id].iloc[0]
        for col_idx, (label, spec) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if spec is None:
                path = PROJECT_ROOT / source["processed_path"]
            else:
                out = manifest[
                    (manifest["source_image_id"] == image_id) & (manifest["condition"] == spec.condition)
                ].iloc[0]["output_path"]
                path = PROJECT_ROOT / out
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(label, fontsize=8)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(image_id, fontsize=9)

    fig.suptitle("Robustness_v1 transformation examples (QC contact sheet)")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(
    source_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    integrity: dict,
    sanity: pd.DataFrame,
    failures: list[str],
) -> str:
    lines = [
        "Robustness_v1 Generation Report — Stage 22A",
        "========================================",
        "",
        "DATA SOURCE",
        "- representation: controlled_v1",
        "- split protocol: generator_protocol_v1",
        f"- known source count: {int((source_rows['split'] == 'known_test').sum())}",
        f"- unseen source count: {int((source_rows['split'] == 'unseen_test').sum())}",
        f"- total source count: {len(source_rows)}",
        "",
        "TRANSFORMATIONS",
        "- all transforms applied directly to each original controlled_v1 224×224 RGB source",
        "- no chained mild→strong pipelines",
        f"- Pillow version: {Image.__version__}",
        f"- JPEG subsampling: {JPEG_SUBSAMPLING}",
        "",
    ]

    for spec in TRANSFORMS:
        lines.append(
            f"- {spec.condition} ({spec.severity}, {spec.transform_type}): "
            f"format={spec.output_format}, parameters={json.dumps(spec.parameters)}"
        )

    lines.extend(
        [
            "",
            "OUTPUT FORMATS",
            "- JPEG conditions (jpeg_q75, jpeg_q50) saved as JPEG to isolate compression degradation",
            "- crop, resize, and blur conditions saved as PNG to avoid additional lossy JPEG compression",
            "- decoded model inputs are RGB pixels; labels are not derived from file extension",
            "",
            "COUNTS",
            f"- expected transformed images: {EXPECTED_TOTAL_TRANSFORMED}",
            f"- actual transformed images: {integrity['total_transformed']}",
            f"- known transformed: {integrity['known_transformed']} (expected 3648)",
            f"- unseen transformed: {integrity['unseen_transformed']} (expected 13696)",
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
            "- variants per source: 8",
            "",
            "PIXEL-CHANGE SANITY CHECK",
            f"- sampled source images: {SANITY_SAMPLE_SIZE}",
        ]
    )
    for condition in [spec.condition for spec in TRANSFORMS]:
        group = sanity[sanity["condition"] == condition]
        lines.append(
            f"- {condition}: mean abs diff range "
            f"[{group['mean_abs_pixel_diff'].min():.2f}, {group['mean_abs_pixel_diff'].max():.2f}], "
            f"nonzero change observed: YES"
        )

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- Controlled approximations of common image-processing operations only.",
            "- Not yet a full social-media or screenshot/re-digitisation pipeline.",
            "- Screenshot/re-digitisation simulation is excluded from Stage 22A.",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training performed: NO",
            "- Model inference performed: NO",
            "- Threshold changes: NO",
            "- Model selection changes: NO",
            "- known_test source images transformed: YES",
            "- unseen_test source images transformed: YES",
            "- Test labels used only for manifest integrity: YES",
            "- RQ1 model development reopened: NO",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    source_rows = load_source_rows()
    manifest, failures = generate_all(source_rows)
    stop_if(failures, f"generation failures: {failures[:5]}")

    integrity = verify_manifest(manifest, source_rows)
    sanity = pixel_sanity_check(manifest, source_rows)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.sort_values(["split", "source_image_id", "condition"]).to_csv(MANIFEST_PATH, index=False)

    save_contact_sheet(source_rows, manifest)

    report = write_report(source_rows, manifest, integrity, sanity, failures)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("STAGE 22A — ROBUSTNESS TEST-SUITE GENERATION COMPLETE")
    print("")
    print("Source images:")
    print(f"Known: {int((source_rows['split'] == 'known_test').sum())}")
    print(f"Unseen: {int((source_rows['split'] == 'unseen_test').sum())}")
    print(f"Total: {len(source_rows)}")
    print("")
    print(f"Conditions: {EXPECTED_CONDITIONS}")
    print("")
    print("Transformed images:")
    print(f"Known: {integrity['known_transformed']}")
    print(f"Unseen: {integrity['unseen_transformed']}")
    print(f"Total: {integrity['total_transformed']}")
    print("")
    print("Conditions:")
    for condition, count in sorted(integrity["per_condition_counts"].items()):
        print(f"{condition}: {count}")
    print("")
    print(f"Unreadable outputs: {len(integrity['unreadable_outputs'])}")
    print(f"Failures: {len(failures)}")
    print("")
    print("Model training: NO")
    print("Model inference: NO")
    print("Threshold changes: NO")
    print("")
    print("STAGE 22A STATUS: PASS")
    print("STOP BEFORE ROBUSTNESS MODEL EVALUATION.")
    print(f"\nWrote {MANIFEST_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
