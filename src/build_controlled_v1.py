"""Build the bias-mitigated pilot representation: controlled_v1.

Why this file exists
--------------------
Stage 3 showed that Real and AI images differ by file format, aspect ratio,
resolution, and channel count. Those differences could become classifier
shortcuts. This script writes a *processed* copy with a shared pipeline so
those visible cues are reduced.

This is called controlled_v1, not "unbiased" or "bias-free". Preprocessing
can mitigate known biases; it cannot prove that every bias is gone.

Raw files under data/raw/ are never modified.

How to run
----------
    source .venv/bin/activate
    python src/build_controlled_v1.py

What to expect
--------------
    data/processed/controlled_v1/images/<generator>/<image_id>.jpg
    metadata/controlled_v1_metadata.csv
    results/controlled_v1_report.txt
    figures/controlled_v1_before_after.png
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- named constants (do not hide these) ---
RESIZE_SHORT_SIDE = 256
FINAL_SIZE = 224
JPEG_QUALITY = 96
JPEG_SUBSAMPLING = 0
PROCESSING_VERSION = "controlled_v1"
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "metadata" / "pilot_audit.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / PROCESSING_VERSION
IMAGE_DIR = PROCESSED_ROOT / "images"
METADATA_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_metadata.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "controlled_v1_report.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "controlled_v1_before_after.png"

GENERATOR_ORDER = [
    "Real",
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD15",
    "VQDM",
    "Wukong",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_adm_alpha(audit_df: pd.DataFrame) -> dict:
    """Inspect ADM alpha channels before any RGB conversion."""
    adm = audit_df[audit_df["generator"] == "ADM"]
    min_alpha = 255
    max_alpha = 0
    fully_opaque = 0
    has_transparency = 0
    transparent_pixels = 0
    total_pixels = 0

    for _, row in tqdm(adm.iterrows(), total=len(adm), desc="ADM alpha check"):
        with Image.open(PROJECT_ROOT / row["path"]) as image:
            if image.mode != "RGBA":
                raise ValueError(f"Expected RGBA for ADM, found {image.mode}: {row['path']}")
            alpha = np.asarray(image.split()[-1])
            a_min = int(alpha.min())
            a_max = int(alpha.max())
            n_transparent = int((alpha < 255).sum())
            min_alpha = min(min_alpha, a_min)
            max_alpha = max(max_alpha, a_max)
            total_pixels += int(alpha.size)
            transparent_pixels += n_transparent
            if a_min == 255:
                fully_opaque += 1
            else:
                has_transparency += 1

    return {
        "n_images": int(len(adm)),
        "min_alpha": min_alpha,
        "max_alpha": max_alpha,
        "fully_opaque": fully_opaque,
        "has_transparency": has_transparency,
        "transparent_pixels": transparent_pixels,
        "total_pixels": total_pixels,
        "fraction_transparent": (
            transparent_pixels / total_pixels if total_pixels else 0.0
        ),
    }


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert every image to RGB with the same rule, regardless of label."""
    if image.mode == "RGB":
        return image
    if image.mode == "L":
        return image.convert("RGB")
    if image.mode == "RGBA":
        # Safe only because the ADM alpha check confirmed full opacity.
        return image.convert("RGB")
    # Any other mode is converted the same way so we do not special-case a class.
    return image.convert("RGB")


def resize_shortest_side(image: Image.Image, target: int) -> Image.Image:
    """Scale so the shortest side equals target. Do not stretch to a square yet."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size {image.size}")
    if width <= height:
        new_width = target
        new_height = int(round(height * (target / width)))
        if new_height < target:
            new_height = target
    else:
        new_height = target
        new_width = int(round(width * (target / height)))
        if new_width < target:
            new_width = target
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def centre_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        raise ValueError(f"Cannot crop {image.size} to {size}x{size}")
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(raw_path: Path) -> Image.Image:
    """Shared pipeline for every label and generator."""
    with Image.open(raw_path) as image:
        image.load()
        rgb = to_rgb(image)
        resized = resize_shortest_side(rgb, RESIZE_SHORT_SIDE)
        cropped = centre_crop(resized, FINAL_SIZE)
        # Rebuild pixels so original EXIF/metadata is not carried forward.
        array = np.asarray(cropped.convert("RGB"))
        return Image.fromarray(array, mode="RGB")


def save_jpeg(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False,
    )


def process_all(audit_df: pd.DataFrame) -> pd.DataFrame:
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in tqdm(audit_df.iterrows(), total=len(audit_df), desc="Building controlled_v1"):
        raw_path = PROJECT_ROOT / row["path"]
        processed_rel = Path("data") / "processed" / PROCESSING_VERSION / "images" / str(row["generator"]) / f"{row['image_id']}.jpg"
        processed_path = PROJECT_ROOT / processed_rel
        record = {
            "image_id": row["image_id"],
            "raw_path": row["path"],
            "processed_path": processed_rel.as_posix(),
            "label": row["label"],
            "generator": row["generator"],
            "original_filename": row["original_filename"],
            "original_format": row["format"],
            "original_width": row["width"],
            "original_height": row["height"],
            "original_aspect_ratio": row["aspect_ratio"],
            "original_mode": row["image_mode"],
            "processed_format": pd.NA,
            "processed_width": pd.NA,
            "processed_height": pd.NA,
            "processed_mode": pd.NA,
            "processing_version": PROCESSING_VERSION,
            "raw_sha256": row["exact_sha256"],
            "processed_sha256": pd.NA,
            "processing_failed": False,
            "failure_reason": "",
        }
        try:
            processed = preprocess_image(raw_path)
            save_jpeg(processed, processed_path)
            record["processed_format"] = "JPEG"
            record["processed_width"] = processed.size[0]
            record["processed_height"] = processed.size[1]
            record["processed_mode"] = processed.mode
            record["processed_sha256"] = sha256_file(processed_path)
        except Exception as error:
            record["processing_failed"] = True
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            record["processed_path"] = ""
        rows.append(record)
    return pd.DataFrame(rows)


def validate_processed(meta: pd.DataFrame) -> dict:
    """Open processed files and confirm the shared representation."""
    ok = meta[meta["processing_failed"] == False]
    n_missing = 0
    n_unreadable = 0
    formats = Counter()
    modes = Counter()
    sizes = Counter()
    unexpected_alpha = 0

    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="Validating controlled_v1"):
        path = PROJECT_ROOT / row["processed_path"]
        if not path.is_file():
            n_missing += 1
            continue
        try:
            with Image.open(path) as image:
                image.load()
                formats[image.format] += 1
                modes[image.mode] += 1
                sizes[image.size] += 1
                if image.mode in {"RGBA", "LA", "PA"}:
                    unexpected_alpha += 1
        except Exception:
            n_unreadable += 1

    return {
        "n_success": int(len(ok)),
        "n_failed": int((meta["processing_failed"] == True).sum()),
        "n_missing": n_missing,
        "n_unreadable": n_unreadable,
        "formats": dict(formats),
        "modes": dict(modes),
        "sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()},
        "unexpected_alpha": unexpected_alpha,
        "label_counts": meta.loc[ok.index, "label"].value_counts().sort_index().to_dict() if len(ok) else {},
        "generator_counts": meta.loc[ok.index, "generator"].value_counts().to_dict() if len(ok) else {},
        "all_jpeg": formats == {"JPEG": len(ok)} and n_missing == 0 and n_unreadable == 0,
        "all_rgb": modes == {"RGB": len(ok)} and n_missing == 0 and n_unreadable == 0,
        "all_224": sizes == {(FINAL_SIZE, FINAL_SIZE): len(ok)} and n_missing == 0 and n_unreadable == 0,
    }


def processed_duplicate_summary(meta: pd.DataFrame) -> dict:
    ok = meta[(meta["processing_failed"] == False) & meta["processed_sha256"].notna()]
    counts = ok["processed_sha256"].value_counts()
    duplicated = counts[counts > 1]
    if duplicated.empty:
        return {
            "n_groups": 0,
            "n_files": 0,
            "cross_label_groups": 0,
            "cross_generator_groups": 0,
        }

    cross_label = 0
    cross_generator = 0
    for sha in duplicated.index:
        group = ok[ok["processed_sha256"] == sha]
        if group["label"].nunique() > 1:
            cross_label += 1
        if group["generator"].nunique() > 1:
            cross_generator += 1
    return {
        "n_groups": int(len(duplicated)),
        "n_files": int(duplicated.sum()),
        "cross_label_groups": cross_label,
        "cross_generator_groups": cross_generator,
    }


def save_before_after(meta: pd.DataFrame) -> None:
    """Display-only contact sheet. Raw and processed files are not rewritten."""
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(len(GENERATOR_ORDER), 2, figsize=(7, 2.4 * len(GENERATOR_ORDER)))

    for row_i, generator in enumerate(GENERATOR_ORDER):
        subset = meta[(meta["generator"] == generator) & (meta["processing_failed"] == False)]
        chosen = subset.iloc[int(rng.integers(0, len(subset)))]
        raw = Image.open(PROJECT_ROOT / chosen["raw_path"]).convert("RGB")
        processed = Image.open(PROJECT_ROOT / chosen["processed_path"])
        raw.thumbnail((160, 160))
        processed.thumbnail((160, 160))
        axes[row_i, 0].imshow(raw)
        axes[row_i, 0].set_title(f"RAW | {generator} | label={chosen['label']}", fontsize=8)
        axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(processed)
        axes[row_i, 1].set_title(
            f"CONTROLLED_V1 | {generator} | label={chosen['label']}", fontsize=8
        )
        axes[row_i, 1].axis("off")
        raw.close()
        processed.close()

    fig.suptitle(f"controlled_v1 before/after (seed={RANDOM_SEED})")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(audit_df: pd.DataFrame, meta: pd.DataFrame, alpha: dict, validation: dict, duplicates: dict) -> str:
    ok = meta[meta["processing_failed"] == False]
    lines = [
        "controlled_v1 processing report",
        "===============================",
        "",
        "This dataset is a bias-mitigated / controlled representation.",
        "It is not unbiased, bias-free, or leakage-free.",
        "",
        f"processing_version: {PROCESSING_VERSION}",
        f"RESIZE_SHORT_SIDE = {RESIZE_SHORT_SIDE}",
        f"FINAL_SIZE = {FINAL_SIZE}",
        f"JPEG_QUALITY = {JPEG_QUALITY}",
        f"JPEG_SUBSAMPLING = {JPEG_SUBSAMPLING}  (4:4:4)",
        "",
        "1. Number processed",
        f"   {validation['n_success']}",
        "",
        "2. Number failed",
        f"   {validation['n_failed']}",
        "",
        "3. Alpha-channel findings (ADM, before RGB conversion)",
        f"   ADM images: {alpha['n_images']}",
        f"   min alpha: {alpha['min_alpha']}",
        f"   max alpha: {alpha['max_alpha']}",
        f"   fully opaque images: {alpha['fully_opaque']}",
        f"   images containing transparency: {alpha['has_transparency']}",
        f"   transparent pixels: {alpha['transparent_pixels']} / {alpha['total_pixels']}",
        f"   fraction transparent: {alpha['fraction_transparent']}",
        "   All ADM alpha channels were fully opaque, so RGBA → RGB was used.",
        "",
        "4. Original format distribution",
        audit_df["format"].value_counts().to_string(),
        "",
        "5. Processed format distribution",
        str(validation["formats"]),
        "",
        "6. Original size distributions",
        "Width by label:",
        audit_df.groupby("label")["width"].describe().round(3).to_string(),
        "",
        "Height by label:",
        audit_df.groupby("label")["height"].describe().round(3).to_string(),
        "",
        "7. Processed size distribution",
        str(validation["sizes"]),
        "",
        "8. Original aspect-ratio differences",
        audit_df.groupby("label")["aspect_ratio"].describe().round(3).to_string(),
        "",
        "9. Processed aspect ratio",
        "   All successfully processed images are 224×224, so aspect ratio = 1.0.",
        "",
        "10. Original modes",
        audit_df["image_mode"].value_counts().to_string(),
        "",
        "11. Processed modes",
        str(validation["modes"]),
        "",
        "12. Processed duplicate findings (SHA-256 of JPEG bytes)",
        f"   duplicate groups: {duplicates['n_groups']}",
        f"   files in those groups: {duplicates['n_files']}",
        f"   cross-label groups: {duplicates['cross_label_groups']}",
        f"   cross-generator groups: {duplicates['cross_generator_groups']}",
        "   No groups were deleted automatically.",
        "",
        "Validation flags",
        f"   all JPEG: {validation['all_jpeg']}",
        f"   all RGB: {validation['all_rgb']}",
        f"   all 224×224: {validation['all_224']}",
        f"   missing processed files: {validation['n_missing']}",
        f"   unreadable processed files: {validation['n_unreadable']}",
        f"   unexpected alpha: {validation['unexpected_alpha']}",
        f"   processed label counts: {validation['label_counts']}",
        f"   processed generator counts: {validation['generator_counts']}",
        "",
        "KNOWN BIASES MITIGATED",
        "----------------------",
        "These are mitigations of *known visible* shortcuts, not a claim that",
        "the representation is unbiased.",
        "",
        "- Explicit file-format difference: original Real=JPEG and AI=PNG.",
        "  Every controlled_v1 file is re-encoded as JPEG quality 96.",
        "- Visible image dimensions: original AI sizes were generator-specific",
        "  squares (128/256/512/1024) while Real sizes varied. Every",
        "  controlled_v1 file is 224×224.",
        "- Visible aspect ratio: original AI images were all square and Real",
        "  images were mostly not. Centre-crop after shortest-side resize",
        "  removes that cue from the stored pixels. Direct stretch-to-square",
        "  was avoided because it would geometrically distort Real images.",
        "  Padding was avoided because almost only Real images would receive",
        "  borders, which could itself become a shortcut.",
        "- Channel-count difference: original ADM was RGBA and 10 Real images",
        "  were grayscale. Every controlled_v1 file is RGB.",
        "",
        "RESIDUAL / UNRESOLVED RISKS",
        "---------------------------",
        "- Real images were originally JPEG and may retain previous compression",
        "  artifacts. A shared final JPEG encode does NOT guarantee complete",
        "  removal of earlier JPEG traces. Compression bias is not claimed to",
        "  be eliminated.",
        "- Original generator resolution may leave different resampling artifacts",
        "  (BigGAN 128→256 vs Midjourney 1024→256, then crop to 224).",
        "- Content distribution cannot be perfectly verified from official",
        "  Tiny-GenImage metadata.",
        "- True source IDs are unavailable, so source-safe splits are still not",
        "  guaranteed by metadata.",
        "- Preprocessing cannot guarantee removal of all generator-specific",
        "  shortcuts (frequency traces, texture, semantic content, and so on).",
        "",
        "Raw originals remain under data/raw/ for later RAW vs controlled_v1",
        "comparisons using the same split assignments.",
    ]
    return "\n".join(lines)


def main() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(AUDIT_PATH)
    alpha = inspect_adm_alpha(audit_df)
    print("ADM alpha:", alpha)

    if alpha["has_transparency"] > 0 or alpha["min_alpha"] < 255:
        message = (
            "STOP: ADM images contain transparency. "
            "Do not choose a background without approval.\n"
            f"{alpha}"
        )
        REPORT_PATH.write_text(message, encoding="utf-8")
        raise SystemExit(message)

    meta = process_all(audit_df)
    meta.to_csv(METADATA_PATH, index=False)

    validation = validate_processed(meta)
    duplicates = processed_duplicate_summary(meta)
    save_before_after(meta)

    report = write_report(audit_df, meta, alpha, validation, duplicates)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("controlled_v1 complete")
    print(f"processed: {validation['n_success']}  failed: {validation['n_failed']}")
    print(f"all JPEG: {validation['all_jpeg']}  all RGB: {validation['all_rgb']}  all 224: {validation['all_224']}")
    print(f"processed duplicate groups: {duplicates['n_groups']}")
    print(f"Wrote {METADATA_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
