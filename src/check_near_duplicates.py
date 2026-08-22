"""Near-duplicate and source-leakage audit for the Tiny-GenImage pilot.

Why this file exists
--------------------
Exact SHA-256 hashes only catch byte-identical files. Two images can still
show the same source content after JPEG vs PNG encoding or a small visual
change. Perceptual hashes (pHash, dHash) are a cheap extra check before we
create train/validation/test splits.

This script does not modify raw images, create splits, or train a model.
A close perceptual hash is a *candidate* for manual inspection, not proof
of leakage.

How to run
----------
    source .venv/bin/activate
    python src/check_near_duplicates.py

What to expect
--------------
    results/source_metadata_report.txt
    metadata/pilot_similarity_metadata.csv
    metadata/near_duplicate_candidates.csv
    results/leakage_risk_summary.txt
    figures/near_duplicates/*.png
"""

from __future__ import annotations

import re
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- configuration ---
RANDOM_SEED = 42
PHASH_SIZE = 8
DHASH_SIZE = 8
# Diagnostic Hamming-distance cut-offs for a 64-bit pHash. Not scientific truth.
PHASH_BROAD_MAX = 8
MAX_PAIRS_PER_SHEET = 12
THUMBNAIL_SIZE = (128, 128)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "metadata" / "pilot_audit.csv"
SIMILARITY_PATH = PROJECT_ROOT / "metadata" / "pilot_similarity_metadata.csv"
CANDIDATES_PATH = PROJECT_ROOT / "metadata" / "near_duplicate_candidates.csv"
SOURCE_REPORT_PATH = PROJECT_ROOT / "results" / "source_metadata_report.txt"
LEAKAGE_REPORT_PATH = PROJECT_ROOT / "results" / "leakage_risk_summary.txt"
FIGURES_DIR = PROJECT_ROOT / "figures" / "near_duplicates"
PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "tiny-genimage" / "data"


def inspect_parquet_schema() -> str:
    """Report official Tiny-GenImage fields. Do not invent extra IDs."""
    import pyarrow.parquet as pq

    lines = [
        "Tiny-GenImage source metadata report",
        "====================================",
        "",
        "Official parquet columns found in the downloaded shards:",
    ]
    shard_paths = sorted(PARQUET_DIR.glob("*.parquet"))
    if not shard_paths:
        lines.append("No parquet shards found.")
        return "\n".join(lines)

    pf = pq.ParquetFile(shard_paths[0])
    lines.append(f"Inspected shard: {shard_paths[0].name}")
    lines.append(str(pf.schema_arrow))
    lines.append("")
    lines.append("Hugging Face feature metadata:")
    if pf.schema_arrow.metadata and b"huggingface" in pf.schema_arrow.metadata:
        lines.append(pf.schema_arrow.metadata[b"huggingface"].decode())
    lines.append("")
    lines.append("Official fields present:")
    lines.append("- image.bytes  (raw file bytes)")
    lines.append("- image.path   (original filename string)")
    lines.append("- label        (0=real, 1=fake)")
    lines.append("- generator    (class index: Real, ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM, Wukong)")
    lines.append("")
    lines.append("Official fields NOT present:")
    for missing in [
        "original sample ID / source_id",
        "ImageNet class name",
        "class index as a dedicated column",
        "category",
        "prompt",
        "source image ID",
        "seed",
        "URL",
        "parent/source reference",
    ]:
        lines.append(f"- {missing}")
    lines.append("")
    lines.append(
        "We already preserved image.path as original_filename in "
        "metadata/download_manifest.csv. No official parquet field was dropped."
    )
    lines.append("")
    lines.append("Filename-derived clues (NOT official source IDs):")
    lines.append(
        "- Real filenames match ImageNet style, e.g. n01440764_11602.JPEG. "
        "The n######## token looks like a WordNet/ImageNet synset. This is "
        "parsed from the filename only."
    )
    lines.append(
        "- Some AI filenames start with an integer (ADM, BigGAN, Midjourney, "
        "SD15, Wukong). That integer may be a class index, but Tiny-GenImage "
        "does not document it as a dataset field."
    )
    lines.append(
        "- GLIDE/VQDM filenames look like GLIDE_1000_200_00_001_glide_00086.png. "
        "Tokens after '00_' may be class/sample numbers, but this is uncertain."
    )
    lines.append("")
    lines.append(
        "Do not treat parsed filename tokens as invented source_id values. "
        "They are recorded separately as filename-derived metadata."
    )
    return "\n".join(lines)


def parse_filename_clues(original_filename: str, generator: str) -> dict:
    """Extract possible class tokens from the original filename only."""
    stem = Path(str(original_filename)).stem
    synset = pd.NA
    leading_index = pd.NA
    note = "No extra filename token parsed."

    synset_match = re.match(r"^(n\d{8})_", stem)
    if synset_match:
        synset = synset_match.group(1)
        note = "ImageNet-style synset parsed from original filename; not an official dataset field."
        return {
            "filename_imagenet_synset": synset,
            "filename_leading_index": leading_index,
            "filename_parse_note": note,
        }

    leading_match = re.match(r"^(\d+)_", stem)
    if leading_match and generator in {"ADM", "BigGAN", "Midjourney", "SD15", "Wukong"}:
        leading_index = int(leading_match.group(1))
        note = (
            "Leading integer parsed from original filename; may be a class "
            "index but is not an official source ID."
        )
        return {
            "filename_imagenet_synset": synset,
            "filename_leading_index": leading_index,
            "filename_parse_note": note,
        }

    glide_match = re.match(r"^(?:GLIDE|VQDM)_1000_200_00_(\d+)_", stem)
    if glide_match and generator in {"GLIDE", "VQDM"}:
        leading_index = int(glide_match.group(1))
        note = (
            "Integer token parsed from GLIDE/VQDM filename pattern; uncertain "
            "class index, not an official source ID."
        )

    return {
        "filename_imagenet_synset": synset,
        "filename_leading_index": leading_index,
        "filename_parse_note": note,
    }


def hash_image(path: Path) -> tuple[str, str]:
    """Compute pHash and dHash in memory. Do not write a new image file."""
    with Image.open(path) as image:
        # Convert only in memory so RGBA/grayscale images can be hashed.
        rgb = image.convert("RGB")
        phash = str(imagehash.phash(rgb, hash_size=PHASH_SIZE))
        dhash = str(imagehash.dhash(rgb, hash_size=DHASH_SIZE))
    return phash, dhash


def hex_to_uint64(hex_hash: str) -> np.uint64:
    return np.uint64(int(hex_hash, 16))


def pair_category(label_1: int, label_2: int, generator_1: str, generator_2: str) -> str:
    if {label_1, label_2} == {0}:
        return "A_real_real"
    if {label_1, label_2} == {1} and generator_1 == generator_2:
        return "B_ai_ai_same_generator"
    if {label_1, label_2} == {1} and generator_1 != generator_2:
        return "C_ai_ai_cross_generator"
    if {label_1, label_2} == {0, 1}:
        return "D_real_ai"
    return "other"


def build_similarity_table(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    readable = audit_df[audit_df["readable"] == True]
    for _, row in tqdm(readable.iterrows(), total=len(readable), desc="Computing perceptual hashes"):
        full_path = PROJECT_ROOT / row["path"]
        phash, dhash = hash_image(full_path)
        clues = parse_filename_clues(row["original_filename"], row["generator"])
        rows.append(
            {
                "image_id": row["image_id"],
                "path": row["path"],
                "label": row["label"],
                "generator": row["generator"],
                "original_filename": row["original_filename"],
                "exact_sha256": row["exact_sha256"],
                "phash": phash,
                "dhash": dhash,
                **clues,
            }
        )
    return pd.DataFrame(rows)


def find_phash_candidates(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Keep pairs with pHash Hamming distance <= 8. Do not dump all comparisons."""
    phash_ints = np.array([hex_to_uint64(h) for h in sim_df["phash"]], dtype=np.uint64)
    dhash_ints = np.array([hex_to_uint64(h) for h in sim_df["dhash"]], dtype=np.uint64)
    xor = np.bitwise_xor(phash_ints[:, None], phash_ints[None, :])
    phash_dist = np.bitwise_count(xor)

    i_idx, j_idx = np.triu_indices(len(sim_df), k=1)
    distances = phash_dist[i_idx, j_idx]
    keep = distances <= PHASH_BROAD_MAX
    i_keep = i_idx[keep]
    j_keep = j_idx[keep]
    p_keep = distances[keep]

    d_xor = np.bitwise_xor(dhash_ints[i_keep], dhash_ints[j_keep])
    d_keep = np.bitwise_count(d_xor)

    records = []
    for i, j, p_dist, d_dist in zip(i_keep, j_keep, p_keep, d_keep):
        left = sim_df.iloc[int(i)]
        right = sim_df.iloc[int(j)]
        records.append(
            {
                "path_1": left["path"],
                "path_2": right["path"],
                "image_id_1": left["image_id"],
                "image_id_2": right["image_id"],
                "label_1": int(left["label"]),
                "label_2": int(right["label"]),
                "generator_1": left["generator"],
                "generator_2": right["generator"],
                "phash_1": left["phash"],
                "phash_2": right["phash"],
                "phash_distance": int(p_dist),
                "dhash_distance": int(d_dist),
                "category": pair_category(
                    int(left["label"]),
                    int(right["label"]),
                    left["generator"],
                    right["generator"],
                ),
                "filename_imagenet_synset_1": left["filename_imagenet_synset"],
                "filename_imagenet_synset_2": right["filename_imagenet_synset"],
                "filename_leading_index_1": left["filename_leading_index"],
                "filename_leading_index_2": right["filename_leading_index"],
            }
        )
    candidates = pd.DataFrame(records)
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["phash_distance", "category", "path_1", "path_2"]
    ).reset_index(drop=True)


def save_pair_sheet(pairs: pd.DataFrame, title: str, output_path: Path) -> None:
    """Display-only contact sheet. Source files are not modified."""
    if pairs.empty:
        return
    pairs = pairs.head(MAX_PAIRS_PER_SHEET)
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.6 * n))
    if n == 1:
        axes = np.array([axes])

    for row_i, (_, pair) in enumerate(pairs.iterrows()):
        for col, path_key, label_key, gen_key in [
            (0, "path_1", "label_1", "generator_1"),
            (1, "path_2", "label_2", "generator_2"),
        ]:
            ax = axes[row_i, col]
            image = Image.open(PROJECT_ROOT / pair[path_key]).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE)
            ax.imshow(image)
            ax.set_title(
                f"{pair[gen_key]} | label={pair[label_key]} | pHash d={pair['phash_distance']}",
                fontsize=8,
            )
            ax.axis("off")
            image.close()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def class_distribution_text(sim_df: pd.DataFrame) -> str:
    lines = [
        "Content-class / semantic metadata",
        "---------------------------------",
        "",
        "Official ImageNet class/category columns are unavailable.",
        "The notes below use filename tokens only.",
        "",
    ]
    real = sim_df[sim_df["label"] == 0]
    synsets = real["filename_imagenet_synset"].dropna()
    lines.append(f"Real images with a parsed ImageNet-style synset: {len(synsets)}/{len(real)}")
    lines.append(f"Unique parsed Real synsets: {synsets.nunique()}")
    if len(synsets) > 0:
        top = synsets.value_counts().head(10)
        lines.append("Most common parsed Real synsets:")
        lines.append(top.to_string())
    lines.append("")

    ai = sim_df[sim_df["label"] == 1]
    lines.append("AI filename leading-index counts (unconfirmed class indices):")
    for generator, group in ai.groupby("generator"):
        parsed = group["filename_leading_index"].dropna()
        lines.append(
            f"- {generator}: parsed {len(parsed)}/{len(group)}, unique values {parsed.nunique()}"
        )
    lines.append("")
    lines.append(
        "Content-class balance cannot be verified from available official metadata."
    )
    lines.append(
        "We did not map synsets to names with an extra model or external class list "
        "in this stage, because the official dataset schema does not include class labels."
    )
    return "\n".join(lines)


def leakage_summary(sim_df: pd.DataFrame, candidates: pd.DataFrame) -> str:
    def count_at_most(max_dist: int) -> int:
        if candidates.empty:
            return 0
        return int((candidates["phash_distance"] <= max_dist).sum())

    def count_category(name: str, max_dist: int | None = None) -> int:
        if candidates.empty:
            return 0
        mask = candidates["category"] == name
        if max_dist is not None:
            mask &= candidates["phash_distance"] <= max_dist
        return int(mask.sum())

    n0 = count_at_most(0)
    n4 = count_at_most(4)
    n8 = count_at_most(8)
    real_ai = count_category("D_real_ai")
    cross = count_category("C_ai_ai_cross_generator")

    lines = [
        "Leakage-risk summary (pilot subset)",
        "===================================",
        "",
        "Language note: close perceptual hashes are candidate near-duplicates",
        "requiring manual inspection. They do not by themselves prove leakage.",
        "",
        "1. Exact SHA-256 duplicate groups",
        "   0 (from Stage 3; no byte-identical files).",
        "",
        "2. pHash distance = 0 candidates",
        f"   {n0} pairs. Very strong visual-similarity candidates.",
        "",
        "3. pHash distance <= 4 candidates",
        f"   {n4} pairs. Strict near-duplicate candidates.",
        "",
        "4. pHash distance <= 8 candidates",
        f"   {n8} pairs. Broader similarity candidates. Threshold is diagnostic only.",
        "",
        "5. Real ↔ AI candidates (pHash <= 8)",
        f"   {real_ai} pairs. These are the most important to inspect by hand,",
        "   because a later split could place similar content in both classes",
        "   or leak the same scene across train and test.",
        "",
        "6. Cross-generator AI candidates (pHash <= 8)",
        f"   {cross} pairs. These could indicate shared source content or a",
        "   generator-independent visual template.",
        "",
        "Category counts at pHash <= 8:",
        f"   A Real ↔ Real: {count_category('A_real_real')}",
        f"   B AI ↔ AI same generator: {count_category('B_ai_ai_same_generator')}",
        f"   C AI ↔ AI different generators: {cross}",
        f"   D Real ↔ AI: {real_ai}",
        "",
        "7. Source IDs",
        "   Official source_id / parent image ID: unavailable.",
        "   original_filename (from parquet image.path): available and already stored.",
        "",
        "8. Semantic/class metadata",
        "   Official class/category field: unavailable.",
        "   Content-class balance cannot be verified from available official metadata.",
        "",
        class_distribution_text(sim_df),
        "",
        "9. Anything suspicious",
    ]
    if real_ai > 0:
        lines.append(
            "   Real ↔ AI perceptual-hash candidates exist. These require manual "
            "review before split design. This does not prove leakage."
        )
    else:
        lines.append(
            "   No Real ↔ AI pairs at pHash <= 8. This does not prove there is no "
            "source overlap; perceptual hashing can miss crops and large edits."
        )
    if n0 > 0:
        lines.append(
            "   pHash distance 0 pairs exist. These are the strongest visual-similarity "
            "candidates and should be inspected first."
        )
    lines.append("")
    lines.append("10. Recommended manual review")
    lines.append("   - All pHash distance 0 pairs, if any.")
    lines.append("   - Real ↔ AI candidates, smallest distances first.")
    lines.append("   - Cross-generator AI candidates, smallest distances first.")
    lines.append("   Contact sheets are under figures/near_duplicates/.")
    lines.append("")
    lines.append("Important scientific limitation")
    lines.append("-------------------------------")
    lines.append(
        "Perceptual hashing can identify some visually similar images but cannot "
        "guarantee detection of all source-content overlap."
    )
    lines.append("It may miss:")
    lines.append("- substantial crops")
    lines.append("- large transformations")
    lines.append("- semantically related recreations")
    lines.append("- heavily altered images")
    lines.append(
        "Therefore this is an additional leakage check, not proof that the "
        "dataset contains no leakage."
    )
    return "\n".join(lines)


def print_brief(candidates: pd.DataFrame) -> None:
    print("Near-duplicate audit complete")
    if candidates.empty:
        print("No pHash <= 8 candidate pairs")
        return
    print("pHash d=0:", int((candidates["phash_distance"] == 0).sum()))
    print("pHash d<=4:", int((candidates["phash_distance"] <= 4).sum()))
    print("pHash d<=8:", len(candidates))
    print(candidates["category"].value_counts().sort_index().to_string())


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    source_report = inspect_parquet_schema()
    SOURCE_REPORT_PATH.write_text(source_report, encoding="utf-8")

    audit_df = pd.read_csv(AUDIT_PATH)
    sim_df = build_similarity_table(audit_df)
    sim_df.to_csv(SIMILARITY_PATH, index=False)

    candidates = find_phash_candidates(sim_df)
    if candidates.empty:
        # Keep a stable header so reruns always produce the file.
        candidates = pd.DataFrame(
            columns=[
                "path_1",
                "path_2",
                "image_id_1",
                "image_id_2",
                "label_1",
                "label_2",
                "generator_1",
                "generator_2",
                "phash_1",
                "phash_2",
                "phash_distance",
                "dhash_distance",
                "category",
                "filename_imagenet_synset_1",
                "filename_imagenet_synset_2",
                "filename_leading_index_1",
                "filename_leading_index_2",
            ]
        )
    candidates.to_csv(CANDIDATES_PATH, index=False)

    if not candidates.empty:
        save_pair_sheet(
            candidates[candidates["phash_distance"] == 0],
            "Candidate pairs with pHash distance 0",
            FIGURES_DIR / "phash_distance_0.png",
        )
        save_pair_sheet(
            candidates[candidates["category"] == "D_real_ai"],
            "Real ↔ AI candidate near-duplicates (smallest pHash first)",
            FIGURES_DIR / "real_ai_candidates.png",
        )
        save_pair_sheet(
            candidates[candidates["category"] == "C_ai_ai_cross_generator"],
            "Cross-generator AI candidate near-duplicates (smallest pHash first)",
            FIGURES_DIR / "cross_generator_candidates.png",
        )
        nonzero = candidates[candidates["phash_distance"] > 0]
        save_pair_sheet(
            nonzero,
            "Closest non-zero pHash candidate pairs",
            FIGURES_DIR / "closest_nonzero_candidates.png",
        )

    summary = leakage_summary(sim_df, candidates)
    LEAKAGE_REPORT_PATH.write_text(summary, encoding="utf-8")
    print_brief(candidates)
    print(f"Wrote {SOURCE_REPORT_PATH}")
    print(f"Wrote {SIMILARITY_PATH}")
    print(f"Wrote {CANDIDATES_PATH}")
    print(f"Wrote {LEAKAGE_REPORT_PATH}")


if __name__ == "__main__":
    main()
