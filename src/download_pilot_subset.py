"""Download a small Tiny-GenImage subset and save original image bytes.

Why this file exists
--------------------
Full GenImage is hundreds of gigabytes. Our first pilot only needs about
2,000-4,000 images. Tiny-GenImage on Hugging Face is a public 35,000-image
subset. Each parquet shard contains 2,000 images, so two train shards give
a 4,000-image pool without downloading the full 8.36 GB archive.

This script does not train a model and does not create our train/val/test
splits. It only obtains raw files and a download manifest.

The Tiny-GenImage parquet schema contains only image (bytes + original
filename), label, and generator. There is no official source_id. The
original filename is stored as original_filename so later audits can parse
filename tokens without modifying raw files.

How to run
----------
    source .venv/bin/activate
    python src/download_pilot_subset.py

What to expect
--------------
    data/raw/tiny-genimage/data/*.parquet
    data/raw/tiny-genimage/images/<generator>/<filename>
    metadata/download_manifest.csv
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- configuration (explicit, so the experiment can be repeated) ---
REPO_ID = "TheKernel01/Tiny-GenImage"
SHARDS = [
    "data/train-00000-of-00014.parquet",
    "data/train-00001-of-00014.parquet",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tiny-genimage"
IMAGE_DIR = RAW_DIR / "images"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "download_manifest.csv"

# Hugging Face class names from the dataset card.
# 0 = Real, 1 = AI-generated, matching our project labels.
GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}


def extension_from_bytes(data: bytes) -> str:
    """Guess a file extension from magic bytes, not from the label."""
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def unique_path(directory: Path, filename: str) -> Path:
    """Avoid overwriting if two rows share the same original filename."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def download_shards() -> list[Path]:
    """Download the chosen parquet shards. Existing files are reused."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in SHARDS:
        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )
        paths.append(Path(local_path))
        print(f"  saved {local_path}")
    return paths


def extract_images(shard_paths: list[Path]) -> pd.DataFrame:
    """Write original image bytes to disk. Do not re-encode or resize."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    image_id = 0

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        relative_shard = shard_path.relative_to(RAW_DIR).as_posix()
        # Tiny-GenImage uses Hugging Face split names in the filename.
        original_split = "train" if "train" in shard_path.name else "validation"

        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            records = table.to_pylist()
            for record in tqdm(
                records,
                desc=f"{shard_path.name} rg{row_group_index}",
                leave=False,
            ):
                image = record["image"]
                data = image.get("bytes") if isinstance(image, dict) else None
                original_name = image.get("path") if isinstance(image, dict) else None
                if not data:
                    raise ValueError(f"Missing image bytes in {relative_shard}")

                label = int(record["label"])
                generator_id = int(record["generator"])
                generator_name = GENERATOR_NAMES.get(generator_id, f"unknown_{generator_id}")

                if original_name:
                    filename = Path(original_name).name
                else:
                    filename = f"image_{image_id:06d}{extension_from_bytes(data)}"

                generator_dir = IMAGE_DIR / generator_name
                generator_dir.mkdir(parents=True, exist_ok=True)
                output_path = unique_path(generator_dir, filename)
                output_path.write_bytes(data)

                rows.append(
                    {
                        "image_id": f"pilot_{image_id:06d}",
                        "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                        "label": label,
                        "generator": generator_name,
                        "generator_id": generator_id,
                        "original_dataset": "Tiny-GenImage",
                        "original_split": original_split,
                        "original_filename": original_name,
                        "shard": relative_shard,
                        "file_size": len(data),
                    }
                )
                image_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    shard_paths = download_shards()
    manifest = extract_images(shard_paths)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print()
    print("Saved", len(manifest), "images")
    print("Manifest:", MANIFEST_PATH)
    print()
    print("Label counts (0=Real, 1=AI-generated):")
    print(manifest["label"].value_counts().sort_index().to_string())
    print()
    print("Generator counts:")
    print(manifest["generator"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
